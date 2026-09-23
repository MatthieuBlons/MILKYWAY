"""
Training entry point for DeepMIL.

The script trains DeepMIL for classification, regression, or survival
analysis using the task-specific behavior implemented by the model class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

from arguments import get_arguments
from model import DeepMIL
from base_model import (
    configure_logging,
    initialize_logging_backends,
    get_current_learning_rates,
    write_tensorboard_metrics,
    register_exit_handlers,
    close_logging_backends,
)

LOGGER = logging.getLogger(__name__)


def train_one_epoch(
    model: DeepMIL,
    dataloader,
    verbose: bool = False,
    log_wandb: bool = False,
    log_steps: int = 1,
) -> float:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : DeepMIL
        Model being optimized.

    dataloader : torch.utils.data.DataLoader
        Training data loader.

    verbose : bool, default=False
        Display a batch-level progress bar.

    log_wandb : bool, default=False
        Log batch-level training quantities to W&B.

    log_every_n_steps : int, default=0
        Logging frequency in optimization steps. A value of zero disables
        batch-level logging.

    Returns
    -------
    float
        Mean training loss for the epoch.
    """
    batch_losses = []

    progress = tqdm(
        dataloader,
        desc="Training",
        unit="batch",
        leave=False,
        disable=not verbose,
    )

    criterion_name = model.args.criterion

    for batch in progress:
        loss = model.train_batch(batch)
        batch_losses.append(loss)

        progress.set_postfix(
            {
                f"{criterion_name}_loss": f"{loss:.4f}",
            },
            refresh=False,
        )

        should_log = (
            log_wandb and log_steps > 0 and model.counter["batch"] % log_steps == 0
        )

        if should_log:
            learning_rates = get_current_learning_rates(model)

            log_values = {
                "train_batch_loss": loss,
            }

            for index, learning_rate in enumerate(learning_rates):
                log_values[f"train_lr_group_{index}"] = learning_rate

            wandb.log(
                log_values,
                step=model.counter["batch"],
            )

    if not batch_losses:
        raise RuntimeError("The training DataLoader produced no batches.")

    mean_loss = float(np.mean(batch_losses))
    model.mean_train_loss = mean_loss

    return mean_loss


def validate_one_epoch(
    model: DeepMIL,
    dataloader,
    verbose: bool = False,
    log_wandb: bool = False,
    log_tensorboard: bool = False,
) -> dict[str, float]:
    """
    Evaluate the model over one validation epoch.

    Parameters
    ----------
    model : DeepMIL
        Model being evaluated.

    dataloader : torch.utils.data.DataLoader
        Validation data loader.

    verbose : bool, default=False
        Display a batch-level progress bar.

    log_wandb : bool, default=False
        Log epoch metrics to W&B.

    log_tensorboard : bool, default=False
        Log epoch metrics to TensorBoard.

    Returns
    -------
    dict
        Validation metrics, including mean training and validation losses.
    """
    batch_losses = []

    progress = tqdm(
        dataloader,
        desc="Validation",
        unit="batch",
        leave=False,
        disable=not verbose,
    )

    criterion_name = model.args.criterion

    for batch in progress:
        loss = model.validate_batch(batch)
        batch_losses.append(loss)

        progress.set_postfix(
            {
                f"{criterion_name}_loss": f"{loss:.4f}",
            },
            refresh=False,
        )

    if not batch_losses:
        raise RuntimeError("The validation DataLoader produced no batches.")

    model.mean_validation_loss = float(np.mean(batch_losses))

    validation_metrics = model.finalize_validation_epoch()

    if log_wandb:
        wandb.log(
            {
                **{
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                },
                "epoch": model.counter["epoch"],
            },
            step=model.counter["batch"],
        )

    if log_tensorboard:
        write_tensorboard_metrics(
            writer=model.writer,
            metrics=validation_metrics,
            epoch=model.counter["epoch"],
        )

    return validation_metrics


def step_learning_rate_scheduler(
    model: DeepMIL,
    validation_metrics: dict[str, float] | None,
) -> None:
    """
    Advance the configured scheduler after the warm-up period.
    """
    if not model.schedulers:
        return

    if model.counter["epoch"] <= model.args.lr_warm_up:
        return

    scheduler_name = model.args.lr_scheduler

    if scheduler_name == "plateau":
        if validation_metrics is None:
            raise RuntimeError("ReduceLROnPlateau requires validation metrics.")

        monitor_name = model.reference_metric

        if monitor_name not in validation_metrics:
            raise KeyError(
                f"Scheduler monitor '{monitor_name}' is unavailable. "
                f"Available metrics: "
                f"{sorted(validation_metrics)}"
            )

        model.update_learning_rate(validation_metrics[monitor_name])
    else:
        model.update_learning_rate()


def update_early_stopping(
    model: DeepMIL,
    validation_metrics: dict[str, float],
) -> None:
    """
    Update model-selection and early-stopping state.

    Early stopping is not evaluated until minimum_epochs have been
    completed.
    """
    monitored_value = validation_metrics[model.reference_metric]

    model.early_stopping(
        monitored_value=monitored_value,
        checkpoint=model.create_checkpoint(),
    )


def save_checkpoint_without_validation(
    model: DeepMIL,
) -> None:
    """Save the latest checkpoint when no validation set is used."""
    checkpoint_path = Path(model.args.job_dir) / "model.pt.tar"

    torch.save(
        model.create_checkpoint(),
        checkpoint_path,
    )


def format_epoch_summary(
    model: DeepMIL,
    validation_metrics: dict[str, float] | None,
    best_epoch: int | None,
) -> dict[str, str]:
    """Build compact progress-bar information for the current epoch."""
    summary = {
        "train_loss": f"{model.mean_train_loss:.4f}",
    }

    if np.isfinite(model.mean_validation_loss):
        summary["val_loss"] = f"{model.mean_validation_loss:.4f}"

    learning_rates = get_current_learning_rates(model)

    if learning_rates:
        summary["lr"] = f"{learning_rates[0]:.3e}"

    if validation_metrics is not None and model.best_reference_value is not None:
        summary[f"best_{model.reference_metric}"] = f"{model.best_reference_value:.4f}"

    if best_epoch is not None:
        summary["best_epoch"] = str(best_epoch)

    return summary


def main(
    project_name: str | None = None,
    job_name: str | None = None,
    known_args: Sequence[str] | None = None,
    verbose: bool = False,
    log: bool = False,
) -> int:
    """
    Train DeepMIL model.

    Parameters
    ----------
    project_name : str, optional
        W&B project name.

    job_name : str, optional
        W&B group name.

    known_args : sequence of str, optional
        Explicit command-line arguments.

    verbose : bool, default=False
        Display training and validation progress bars.

    log : bool, default=False
        Enable W&B and TensorBoard logging.

    Returns
    -------
    int
        Number of completed epochs.
    """
    configure_logging(verbose=verbose)

    args = get_arguments(
        known_args=known_args,
        train=True,
    )

    args.job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = DeepMIL(
        args=args,
        with_data=True,
    )

    initialize_logging_backends(
        model=model,
        args=args,
        enabled=log,
        project_name=project_name,
        job_name=job_name,
        dir_name=args.job_dir,
    )

    register_exit_handlers(model)

    n_folds = args.n_folds or 1
    n_reps = args.n_reps or 1

    epoch_progress = tqdm(
        total=args.epochs,
        desc=(
            f"Training "
            f"repeat={args.repeat + 1}/{n_reps}, "
            f"fold={args.test_fold + 1}/{n_folds}"
        ),
        unit="epoch",
        leave=True,
        disable=not verbose,
    )

    best_epoch = None

    try:
        while model.counter["epoch"] < args.epochs:

            train_one_epoch(
                model=model,
                dataloader=model.train_loader,
                verbose=verbose,
                log_wandb=log,
                log_steps=getattr(
                    args,
                    "log_steps",
                    1,
                ),
            )

            validation_metrics = None

            if args.use_val:
                previous_best = model.best_reference_value

                validation_metrics = validate_one_epoch(
                    model=model,
                    dataloader=model.val_loader,
                    verbose=verbose,
                    log_wandb=log,
                    log_tensorboard=log,
                )

                if model.best_reference_value != previous_best:
                    best_epoch = model.counter["epoch"]

                step_learning_rate_scheduler(
                    model=model,
                    validation_metrics=validation_metrics,
                )

                update_early_stopping(
                    model=model,
                    validation_metrics=validation_metrics,
                )

            else:
                step_learning_rate_scheduler(
                    model=model,
                    validation_metrics=None,
                )

                save_checkpoint_without_validation(model)

            epoch_progress.set_postfix(
                format_epoch_summary(
                    model=model,
                    validation_metrics=validation_metrics,
                    best_epoch=best_epoch,
                ),
                refresh=False,
            )
            epoch_progress.update()

            if args.use_val and model.early_stopping.stop:
                LOGGER.info(
                    "Early stopping at epoch %d.",
                    model.counter["epoch"],
                )
                break

    finally:
        epoch_progress.close()
        close_logging_backends(model)

    return int(model.counter["epoch"])


if __name__ == "__main__":
    main()
