"""
Unified model interface for Cpath.

"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import torch
from torch.nn import (
    Module,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
)

import logging

import signal
import sys
from argparse import Namespace

import numpy as np
import wandb

LOGGER = logging.getLogger(__name__)

from torch.utils.tensorboard import SummaryWriter


def configure_logging(verbose: bool = False) -> None:
    """
    Configure console logging.

    Parameters
    ----------
    verbose : bool, default=False
        Use informational logging when enabled and warnings otherwise.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def make_serializable(value: Any) -> Any:
    """
    Recursively convert configuration values into W&B-safe objects.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {key: make_serializable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [make_serializable(item) for item in value]

    return value


def get_current_learning_rates(model: BaseModel) -> list[float]:
    """
    Return the current learning rate of every optimizer parameter group.
    """
    learning_rates = []

    for optimizer in model.optimizers:
        learning_rates.extend(group["lr"] for group in optimizer.param_groups)

    return learning_rates


def write_tensorboard_metrics(
    writer,
    metrics: dict[str, float],
    epoch: int,
    prefix: str = "validation",
) -> None:
    """
    Write scalar metrics to TensorBoard.

    Non-finite values and non-scalar entries are ignored.
    """
    if writer is None:
        return

    for name, value in metrics.items():
        if not np.isscalar(value):
            continue

        numeric_value = float(value)

        if np.isfinite(numeric_value):
            writer.add_scalar(
                f"{prefix}/{name}",
                numeric_value,
                epoch,
            )


def initialize_logging_backends(
    model: BaseModel,
    args: Namespace,
    enabled: bool,
    project_name: str | None,
    job_name: str | None,
    dir_name: str | None,
) -> None:
    """
    Initialize W&B and TensorBoard logging.
    """
    is_primary_process = int(os.environ.get("RANK", "0")) == 0

    if enabled and is_primary_process:
        n_folds = args.n_folds or 1
        n_reps = args.n_reps or 1

        run_name = (
            f"fold_{args.test_fold + 1}-of-{n_folds}_"
            f"repeat_{args.repeat + 1}-of-{n_reps}"
        )

        wandb.init(
            project=project_name,
            group=job_name,
            name=run_name,
            dir=dir_name,
            config=make_serializable(vars(args)),
            reinit="finish_previous",
        )

        model.initialize_summary_writer(directory=args.job_dir / "tensorboard")
    else:
        wandb.init(mode="disabled")


def close_logging_backends(model: BaseModel) -> None:
    """
    Flush and close TensorBoard and W&B.
    """
    if model.writer is not None:
        model.writer.flush()
        model.writer.close()

    wandb.finish()


def register_exit_handlers(model: BaseModel) -> None:
    """
    Register graceful handlers for interruption and termination signals.
    """

    def handle_exit(
        signum: int,
        frame,
    ) -> None:
        LOGGER.warning(
            "Caught signal %d; closing logging backends.",
            signum,
        )

        close_logging_backends(model)
        sys.exit(128 + signum)

    signal.signal(
        signal.SIGTERM,
        handle_exit,
    )
    signal.signal(
        signal.SIGINT,
        handle_exit,
    )


class BaseModel(ABC):
    """
    Abstract interface shared by all models.

    Parameters
    ----------
    args
        Runtime and model configuration namespace.

    Attributes
    ----------
    network : torch.nn.Module
        Trainable neural network.

    optimizers : list[torch.optim.Optimizer]
        Optimizers associated with the model.

    schedulers : list
        Learning-rate schedulers associated with the optimizers.

    device : torch.device
        Device on which model operations are performed.

    counter : dict
        Epoch and optimization-step counters.
    """

    def __init__(self, args, verbose: bool = False) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.network: Module | None = None
        self.optimizers: list[Optimizer] = []
        self.schedulers: list[Any] = []
        self.counter = {
            "epoch": 0,
            "batch": 0,
        }
        self.writer: SummaryWriter | None = None

    @abstractmethod
    def train_batch(self, batch: dict[str, Any]) -> float:
        """
        Perform one optimization step and return the batch loss.
        """

    @abstractmethod
    def validate_batch(self, batch: dict[str, Any]) -> float:
        """
        Evaluate one validation batch and accumulate predictions.
        """

    @abstractmethod
    def predict_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Generate predictions for one batch.
        """

    @abstractmethod
    def create_checkpoint(self) -> dict[str, Any]:
        """
        Create a serializable checkpoint dictionary.
        """

    def initialize_summary_writer(
        self,
        directory: str | Path | None = None,
    ) -> None:
        """
        Initialize TensorBoard logging.

        If directory is not provided, the environment variable
        EVENTS_TF_FOLDER is used when available.
        """
        if directory is None:
            directory = os.environ.get("EVENTS_TF_FOLDER")

        self.writer = SummaryWriter(
            log_dir=None if directory is None else str(directory)
        )

    def update_learning_rate(
        self,
        monitored_value: float | None = None,
    ) -> None:
        """
        Advance every configured learning-rate scheduler after optional
        warm up.

        Plateau schedulers require a monitored validation value. Other
        schedulers are advanced without an argument.
        """
        for scheduler in self.schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                if monitored_value is None:
                    raise ValueError("ReduceLROnPlateau requires a monitored value.")
                scheduler.step(monitored_value)
            else:
                scheduler.step()

    @staticmethod
    def set_requires_grad(
        modules: Module | list[Module],
        requires_grad: bool,
    ) -> None:
        """
        Enable or disable gradient computation for one or more modules.
        """
        if not isinstance(modules, list):
            modules = [modules]

        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad

    def zero_optimizer_gradients(self) -> None:
        """
        Reset gradients for every configured optimizer.
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)


def writes_tensorboard_metrics(writer, to_write, epoch):
    """writes_metrics.
    Writes the validation metrics (and the train loss) in a Tensorboard Writer.

    Parameters
    ----------
    writer : Tensorboard Writer

    to_write : dict, scalars
        metrics to write.

    epoch : int
        time step.
    """
    for key in to_write:
        if type(to_write[key]) == dict:
            writer.add_scalars(key, to_write[key], epoch)
        else:
            writer.add_scalar(key, to_write[key], epoch)


class EarlyStopping:
    """
    Track a validation quantity and save latest and best checkpoints.

    Parameters
    ----------
    patience : int
        Number of consecutive non-improving epochs allowed.

    minimum_epochs : int
        Minimum number of epochs before stopping is permitted.

    mode : {"min", "max"}
        Whether lower or higher monitored values are considered better.

    checkpoint_path : str or pathlib.Path
        Path used to save the latest checkpoint.

    minimum_improvement : float
        Minimum change required to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int,
        minimum_epochs: int | None = None,
        mode: str = "min",
        checkpoint_path: str | Path = "model.pt.tar",
        minimum_improvement: float = 0.0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("`mode` must be either 'min' or 'max'.")

        self.patience = int(patience)
        self.minimum_epochs = int(minimum_epochs)
        self.mode = mode
        self.minimum_improvement = float(minimum_improvement)

        self.checkpoint_path = Path(checkpoint_path)
        self.best_checkpoint_path = self.checkpoint_path.with_name(
            f"best_{self.checkpoint_path.stem}" f"{self.checkpoint_path.suffix}"
        )

        self.epoch = 0
        self.num_bad_epochs = 0
        self.best_value: float | None = None

        self.stop = False
        self.is_best = False

    def __call__(
        self,
        monitored_value: float,
        checkpoint: dict[str, Any],
    ) -> None:
        """
        Update early-stopping state and save the checkpoint.
        """
        self.epoch += 1
        self.is_best = self._is_improvement(monitored_value)

        if self.is_best:
            self.best_value = float(monitored_value)
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        can_stop = self.minimum_epochs is None or self.epoch >= self.minimum_epochs

        if can_stop and self.num_bad_epochs >= self.patience:
            self.stop = True

        self.save_checkpoint(checkpoint)

    def _is_improvement(self, value: float) -> bool:
        """
        Determine whether a monitored value improves on the current best.
        """
        if self.best_value is None:
            return True

        if self.mode == "min":
            return value < self.best_value - self.minimum_improvement

        return value > self.best_value + self.minimum_improvement

    def save_checkpoint(
        self,
        checkpoint: dict[str, Any],
    ) -> None:
        """
        Save the latest checkpoint and copy it when it is the current best.
        """
        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            checkpoint,
            self.checkpoint_path,
        )

        if self.is_best:
            shutil.copyfile(
                self.checkpoint_path,
                self.best_checkpoint_path,
            )
