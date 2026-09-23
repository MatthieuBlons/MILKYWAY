"""
Unified model interface for MIL.

MIL pipeline predicts patient-level clinical or molecular targets from
bags of encoded patches extracted for WSIs.

Supported tasks
---------------
survival
    Predict a continuous risk score from observed survival times and
    censoring indicators.

classification
    Predict a binary or multiclass categorical endpoint.

regression
    Predict one or multiple continuous targets.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import numpy as np
import torch
from sklearn import metrics
from sksurv.metrics import concordance_index_censored
from torch import Tensor
from torch.nn import (
    CrossEntropyLoss,
    HuberLoss,
    L1Loss,
    MSELoss,
    Module,
)
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    LinearLR,
)

from dataloader import DatasetHandler, SUPPORTED_TASKS
from networks import CustomMIL
from losses import CoxPartialLikelihoodLoss
from base_model import BaseModel, EarlyStopping


class DeepMIL(BaseModel):
    """
    DeepMIL training interface.

    The class supports survival analysis, categorical classification, and
    continuous regression through a common interface.

    Task-specific behavior is isolated in methods responsible for:

    - preparing targets
    - computing losses
    - converting network outputs into predictions
    - computing validation metrics

    Parameters
    ----------
    args
        Runtime configuration namespace.

        Important attributes include:

        task
            One of survival, classification, or regression.

        criterion
            Loss-function identifier.

        optimizer
            Optimizer identifier.

        device
            PyTorch device.

        ref_metric
            Validation metric used for model selection.

        metric_mode
            min or max, indicating whether the reference metric
            should decrease or increase.

    label_encoder : sklearn.preprocessing.LabelEncoder, optional
        Encoder used to convert encoded class indices back to original
        labels.

    with_data : bool, default=False
        Construct training and validation data loaders during model
        initialization.

    Raises
    ------
    ValueError
        If `args.task` is not in `SUPPORTED_TASKS`.
    """

    def __init__(
        self,
        args,
        label_encoder=None,
        with_data: bool = False,
    ) -> None:
        super().__init__(args)

        self.task = str(args.task).lower()

        if self.task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task '{self.task}'. "
                f"Expected one of {sorted(SUPPORTED_TASKS)}."
            )

        self.reference_metric = args.ref_metric
        self.metric_mode = getattr(args, "metric_mode", "max")
        self.network = self.build_network(args)
        self.optimizer = self.build_optimizer()
        self.optimizers = [self.optimizer]
        self.schedulers = self.build_schedulers()
        self.criterion = self.build_criterion()

        self.train_loader = None
        self.val_loader = None
        self.dataset_handler = None

        if with_data:
            self.get_data()

        if label_encoder is not None:
            self.label_encoder = label_encoder
        elif self.train_loader is not None:
            self.label_encoder = self.train_loader.dataset.label_encoder
        else:
            self.label_encoder = None

        self.best_metrics: dict[str, float] | None = None
        self.best_reference_value: float | None = None

        self.mean_train_loss = math.nan
        self.mean_validation_loss = math.nan

        self.validation_results = self._empty_validation_results()

        early_stopping_mode = getattr(
            args,
            "metric_mode",
            "max",
        )
        checkpoint_path = Path(args.job_dir) / "model.pt.tar"
        self.early_stopping = EarlyStopping(
            patience=args.patience,
            minimum_epochs=getattr(args, "minimum_epochs", 0),
            mode=early_stopping_mode,
            checkpoint_path=checkpoint_path,
            minimum_improvement=getattr(
                args,
                "minimum_improvement",
                0.0,
            ),
        )

    def build_network(self, args) -> Module:
        """
        Construct the neural network and move it to the configured device.

        The architecture itself is delegated to :class:`CustomMIL`.
        """

        network = CustomMIL(args, args.output_dim)
        return network.to(self.device)

    def get_data(self) -> None:
        """
        Construct training and validation data loaders.
        """
        self.dataset_handler = DatasetHandler(
            self.args,
            predict=False,
        )

        (
            self.train_loader,
            self.val_loader,
        ) = self.dataset_handler.get_loader(training=True)

    def build_optimizer(self) -> Optimizer:
        """
        Construct the configured optimizer.
        """
        optimizer_name = self.args.optimizer.lower()
        weight_decay = getattr(
            self.args,
            "weight_decay",
            1e-5,
        )

        if optimizer_name in {"adam", "adamw"}:
            return AdamW(
                self.network.trainable,
                lr=self.args.lr,
                weight_decay=weight_decay,
            )

        if optimizer_name == "sgd":
            return SGD(
                self.network.trainable,
                lr=self.args.lr,
                momentum=getattr(
                    self.args,
                    "momentum",
                    0.9,
                ),
                weight_decay=weight_decay,
            )

        raise ValueError(
            f"Unsupported optimizer '{self.args.optimizer}'. "
            "Expected 'adam', 'adamw' (same) or 'sgd'."
        )

    def build_schedulers(self) -> list[Any]:
        """
        Construct zero or more learning-rate schedulers.
        """
        scheduler_name = getattr(
            self.args,
            "lr_scheduler",
            "none",
        ).lower()

        if scheduler_name in {"none", ""}:
            return []

        if scheduler_name == "plateau":
            return [
                ReduceLROnPlateau(
                    optimizer,
                    mode=getattr(
                        self.args,
                        "metric_mode",
                        "min",
                    ),
                    patience=self.args.patience_lr,
                    factor=getattr(
                        self.args,
                        "lr_factor",
                        0.5,
                    ),
                )
                for optimizer in self.optimizers
            ]

        if scheduler_name == "cosine":
            return [
                CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=getattr(
                        self.args,
                        "restart_period",
                        1,
                    ),
                    T_mult=getattr(
                        self.args,
                        "restart_multiplier",
                        2,
                    ),
                )
                for optimizer in self.optimizers
            ]

        if scheduler_name == "linear":
            return [
                LinearLR(
                    optimizer,
                    start_factor=getattr(
                        self.args,
                        "start_factor",
                        1.0 / 10,
                    ),
                    total_iters=getattr(
                        self.args,
                        "scheduler_iter",
                        10,
                    ),
                )
                for optimizer in self.optimizers
            ]

        raise ValueError(f"Unsupported scheduler '{scheduler_name}'.")

    def build_criterion(self) -> Module:
        """
        Construct a loss function compatible with the configured task.

        Supported combinations
        ----------------------
        survival
            cox

        classification
            cross_entropy, nll (if logsoftmax ends network)

        regression
            mse, mae, l1, or huber
        """
        criterion_name = self.args.criterion.lower()

        if self.task == "survival":
            if criterion_name != "cox":
                raise ValueError(
                    "Survival analysis currently requires " "`criterion='cox'`."
                )

            return CoxPartialLikelihoodLoss().to(self.device)

        if self.task == "classification":
            if criterion_name not in {"cross_entropy", "nll"}:
                raise ValueError(
                    "Classification requires CrossEntropyLoss or  NLLLoss (Deprecated)."
                    "Use `criterion='cross_entropy'`."
                )

            class_weights = getattr(
                self.args,
                "class_weights",
                None,
            )

            if class_weights is not None:
                class_weights = torch.as_tensor(
                    class_weights,
                    dtype=torch.float32,
                    device=self.device,
                )

            return CrossEntropyLoss(weight=class_weights).to(self.device)

        if criterion_name == "mse":
            return MSELoss().to(self.device)

        if criterion_name in {"mae", "l1"}:
            return L1Loss().to(self.device)

        if criterion_name == "huber":
            return HuberLoss(
                delta=getattr(
                    self.args,
                    "huber_delta",
                    1.0,
                )
            ).to(self.device)

        raise ValueError(f"Unsupported regression criterion '{criterion_name}'.")

    def forward(self, tiles: Tensor, coords: Tensor | None) -> Tensor:
        """
        Apply the neural network to a batch of proteomic images.
        """
        return self.network(tiles, coords)

    def forward_without_gradients(self, tiles: Tensor, coords: Tensor | None) -> Tensor:
        """
        Perform inference without storing gradients.
        """
        self.network.eval()

        with torch.inference_mode():
            return self.forward(tiles, coords)

    def forward_batch(
        self,
        batch: dict[str, Any],
    ) -> Tensor:
        """
        Forward a batch containing either fixed-size or variable-size WSI bags.

        Returns
        -------
        torch.Tensor
            Patient-level outputs with shape [B, output_dim].
        """
        tiles = batch["tiles"]
        coords = batch["coords"]

        # Fixed-size bags: [B, N, F]
        if isinstance(tiles, Tensor):
            return self.forward(
                tiles,
                coords,
            )

        # Variable-size bags: list[Tensor[N_i, F]]
        outputs = []

        for index, tile_bag in enumerate(tiles):
            coord_bag = None if coords is None else coords[index]

            # CustomMIL expects [B = 1, N, F].
            tile_bag = tile_bag.unsqueeze(0)

            if coord_bag is not None:
                coord_bag = coord_bag.unsqueeze(0)

            output = self.forward(
                tile_bag,
                coord_bag,
            )

            outputs.append(output)

        return torch.cat(
            outputs,
            dim=0,
        )

    def forward_batch_without_gradients(self, batch: dict[str, Any]) -> Tensor:
        """
        Perform batch inference without storing gradients.
        """
        self.network.eval()

        tiles = batch["tiles"]
        coords = batch["coords"]

        # Fixed-size bags: [B, N, F]
        if isinstance(tiles, Tensor):
            return self.forward_without_gradients(
                tiles,
                coords,
            )

        # Variable-size bags: list[Tensor[N_i, F]]
        outputs = []

        for index, tile_bag in enumerate(tiles):
            coord_bag = None if coords is None else coords[index]

            # CustomMIL expects [B = 1, N, F].
            tile_bag = tile_bag.unsqueeze(0)

            if coord_bag is not None:
                coord_bag = coord_bag.unsqueeze(0)

            output = self.forward_without_gradients(
                tile_bag,
                coord_bag,
            )

            outputs.append(output)

        return torch.cat(
            outputs,
            dim=0,
        )

    def move_batch_to_device(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Move tensors in a batch to the model device.
        """
        prepared = {}

        for key, value in batch.items():
            if isinstance(value, Tensor):
                prepared[key] = value.to(
                    self.device,
                    non_blocking=True,
                )

            elif isinstance(value, list):
                prepared[key] = [
                    (
                        item.to(
                            self.device,
                            non_blocking=True,
                        )
                        if isinstance(item, Tensor)
                        else item
                    )
                    for item in value
                ]

            else:
                prepared[key] = value

        return prepared

    def compute_task_loss(
        self,
        outputs: Tensor,
        batch: dict[str, Any],
    ) -> Tensor:
        """
        Compute the task-specific training or validation loss.
        """
        if self.task == "survival":
            return self.criterion(
                outputs,
                batch["time"],
                batch["event"],
            )

        if self.task == "classification":
            targets = batch["target"].long().reshape(-1)
            return self.criterion(
                outputs,
                targets,
            )

        targets = batch["target"].float()

        if outputs.shape != targets.shape:
            raise ValueError(
                "Regression output and target shapes not equal: "
                f"output={tuple(outputs.shape)}, "
                f"target={tuple(targets.shape)}."
            )

        return self.criterion(
            outputs,
            targets,
        )

    def train_batch(
        self,
        batch: dict[str, Any],
    ) -> float:
        """
        Perform one optimization step.

        Supports both fixed-size and variable-size WSI bags.
        """
        self.network.train()
        batch = self.move_batch_to_device(batch)

        self.zero_optimizer_gradients()

        outputs = self.forward_batch(batch)

        loss = self.compute_task_loss(
            outputs,
            batch,
        )

        loss.backward()

        clip_grad_norm_(
            self.network.parameters(),
            max_norm=self.args.gradient_clip_norm,
        )

        self.optimizer.step()

        self.counter["batch"] += 1

        return loss.detach().item()

    def validate_batch(
        self,
        batch: dict[str, Any],  # forced to size 1
    ) -> float:
        """
        Evaluate one validation case and store its output and targets.
        """
        batch = self.move_batch_to_device(batch)

        outputs = self.forward_without_gradients(batch["tiles"], batch["coords"])

        loss = self.compute_task_loss(
            outputs,
            batch,
        )

        self.accumulate_validation_results(
            outputs=outputs,
            batch=batch,
        )

        return float(loss.detach().cpu())

    def accumulate_validation_results(
        self,
        outputs: Tensor,
        batch: dict[str, Any],
    ) -> None:
        """
        Store model outputs and task-specific targets for epoch metrics.
        """
        self.validation_results["outputs"].append(outputs.detach().cpu())

        self.validation_results["case_ids"].extend(batch["case_id"])

        if self.task == "survival":
            self.validation_results["times"].append(batch["time"].detach().cpu())
            self.validation_results["events"].append(batch["event"].detach().cpu())
        else:
            self.validation_results["targets"].append(batch["target"].detach().cpu())

    def finalize_validation_epoch(
        self,
    ) -> dict[str, float]:
        """
        Compute validation metrics and clear accumulated predictions.

        Returns
        -------
        dict
            Task-specific validation metrics together with mean training
            and validation losses.
        """
        outputs = torch.cat(
            self.validation_results["outputs"],
            dim=0,
        ).numpy()

        if self.task == "survival":
            times = torch.cat(
                self.validation_results["times"],
                dim=0,
            ).numpy()

            events = torch.cat(
                self.validation_results["events"],
                dim=0,
            ).numpy()

            validation_metrics = self.compute_survival_metrics(
                risk_scores=outputs,
                times=times,
                events=events,
            )

        else:
            targets = torch.cat(
                self.validation_results["targets"],
                dim=0,
            ).numpy()

            if self.task == "classification":
                validation_metrics = self.compute_classification_metrics(
                    logits=outputs,
                    targets=targets,
                )
            else:
                validation_metrics = self.compute_regression_metrics(
                    predictions=outputs,
                    targets=targets,
                )

        validation_metrics["mean_train_loss"] = float(self.mean_train_loss)
        validation_metrics["mean_val_loss"] = float(self.mean_validation_loss)

        self.update_best_metrics(validation_metrics)
        self.validation_results = self._empty_validation_results()

        self.counter["epoch"] += 1

        return validation_metrics

    def _empty_validation_results(
        self,
    ) -> dict[str, list]:
        """
        Return an empty task-compatible validation accumulator.
        """
        results = {
            "outputs": [],
            "case_ids": [],
        }

        if self.task == "survival":
            results.update(
                {
                    "times": [],
                    "events": [],
                }
            )
        else:
            results["targets"] = []

        return results

    def predict_batch(
        self,
        batch: dict[str, Any],  # forced to size 1
    ) -> dict[str, Any]:
        """
        Generate task-specific predictions for one batch.
        """
        prepared_batch = self.move_batch_to_device(batch)

        outputs = self.forward_batch_without_gradients(prepared_batch)

        result: dict[str, Any] = {
            "case_id": batch["case_id"],
        }

        if self.task == "survival":
            result["risk_score"] = outputs.reshape(-1).cpu().numpy()
            return result

        if self.task == "classification":
            probabilities = torch.softmax(
                outputs,
                dim=1,
            )
            predictions = outputs.argmax(dim=1)

            result["proba"] = probabilities.cpu().numpy()
            result["prediction"] = predictions.cpu().numpy()

            if self.label_encoder is not None:
                result["predicted_class"] = self.label_encoder.inverse_transform(
                    predictions.cpu().numpy()
                )
            else:
                result["predicted_class"] = predictions.cpu().numpy()

            return result

        result["prediction"] = outputs.cpu().numpy()
        return result

    @staticmethod
    def compute_classification_metrics(
        logits: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute binary or multiclass classification metrics.
        """
        targets = np.asarray(targets).reshape(-1)
        predicted_classes = logits.argmax(axis=1)
        probabilities = torch.softmax(
            torch.as_tensor(logits),
            dim=1,
        ).numpy()

        metric_values = {
            "accuracy": metrics.accuracy_score(
                targets,
                predicted_classes,
            ),
            "balanced_accuracy": (
                metrics.balanced_accuracy_score(
                    targets,
                    predicted_classes,
                )
            ),
            "precision_macro": metrics.precision_score(
                targets,
                predicted_classes,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": metrics.recall_score(
                targets,
                predicted_classes,
                average="macro",
                zero_division=0,
            ),
            "f1_macro": metrics.f1_score(
                targets,
                predicted_classes,
                average="macro",
                zero_division=0,
            ),
        }

        unique_classes = np.unique(targets)

        try:
            if probabilities.shape[1] == 2:
                metric_values["roc_auc"] = metrics.roc_auc_score(
                    targets,
                    probabilities[:, 1],
                )
            else:
                metric_values["roc_auc_ovr_macro"] = metrics.roc_auc_score(
                    targets,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(probabilities.shape[1]),
                )
        except ValueError:
            # AUROC is undefined if a validation fold lacks a class.
            (
                "AUROC could not be calculated because the validation "
                "targets do not contain all required classes: %s",
                unique_classes.tolist(),
            )

        return {key: float(value) for key, value in metric_values.items()}

    @staticmethod
    def compute_regression_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute aggregate metrics for single- or multi-output regression.
        """
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)

        if predictions.ndim == 1:
            predictions = predictions[:, None]

        if targets.ndim == 1:
            targets = targets[:, None]

        mse = metrics.mean_squared_error(
            targets,
            predictions,
        )

        metric_values = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": metrics.mean_absolute_error(
                targets,
                predictions,
            ),
            "r2": metrics.r2_score(
                targets,
                predictions,
                multioutput="uniform_average",
            ),
        }

        correlations = []

        for target_index in range(targets.shape[1]):
            true_values = targets[:, target_index]
            predicted_values = predictions[:, target_index]

            if np.std(true_values) == 0 or np.std(predicted_values) == 0:
                continue

            correlations.append(
                np.corrcoef(
                    true_values,
                    predicted_values,
                )[0, 1]
            )

        metric_values["pearson_mean"] = (
            float(np.mean(correlations)) if correlations else math.nan
        )

        return {key: float(value) for key, value in metric_values.items()}

    @staticmethod
    def compute_survival_metrics(
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute Harrell's concordance index for right-censored outcomes.

        Higher predicted scores are interpreted as higher risk and therefore
        shorter expected survival.
        """

        risk_scores = np.asarray(risk_scores).reshape(-1)
        times = np.asarray(times).reshape(-1)
        events = np.asarray(events).reshape(-1).astype(bool)

        (
            concordance_index,
            concordant,
            discordant,
            tied_risk,
            tied_time,
        ) = concordance_index_censored(
            event_indicator=events,
            event_time=times,
            estimate=risk_scores,
        )

        return {
            "concordance_index": float(concordance_index),
            "concordant_pairs": int(concordant),
            "discordant_pairs": int(discordant),
            "tied_risk_pairs": int(tied_risk),
            "tied_time_pairs": int(tied_time),
        }

    def update_best_metrics(
        self,
        current_metrics: dict[str, float],
    ) -> None:
        """
        Retain metrics from the best validation epoch.
        """
        if self.reference_metric not in current_metrics:
            raise KeyError(
                f"Reference metric '{self.reference_metric}' is absent. "
                f"Available metrics: {sorted(current_metrics)}"
            )

        current_value = current_metrics[self.reference_metric]

        if not np.isfinite(current_value):
            return

        if self.best_reference_value is None:
            improved = True
        elif self.metric_mode == "min":
            improved = current_value < self.best_reference_value
        elif self.metric_mode == "max":
            improved = current_value > self.best_reference_value
        else:
            raise ValueError("`metric_mode` must be either 'min' or 'max'.")

        if improved:
            self.best_reference_value = float(current_value)
            self.best_metrics = dict(current_metrics)

    def create_checkpoint(self) -> dict[str, Any]:
        """
        Create a complete checkpoint for resuming training or inference.
        """
        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dicts": [
                scheduler.state_dict() for scheduler in self.schedulers
            ],
            "counter": dict(self.counter),
            "args": self.args,
            "task": self.task,
            "best_metrics": self.best_metrics,
            "best_reference_value": self.best_reference_value,
            "label_encoder": self.label_encoder,
        }

        if self.train_loader is not None:
            checkpoint["target_table"] = self.train_loader.dataset.target_table
            checkpoint["target_names"] = self.train_loader.dataset.target_names
            checkpoint["output_dim"] = self.train_loader.dataset.output_dim

        return checkpoint
