"""
Command-line and YAML configuration utilities for MIL.

The configuration pipeline follows this precedence order:

    parser defaults < YAML configuration < explicit command-line arguments

The module also resolves task-dependent defaults and validates consistency
between the selected task, targets, loss, metrics, optimizer, scheduler,
data-loading options, and cross-validation configuration.
"""

from __future__ import annotations

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
    Namespace,
)
from pathlib import Path
from typing import Any, Sequence

import yaml

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
from pathlib import Path


from mil.dataloader import SUPPORTED_TASKS

TASK_CRITERIA = {
    "survival": {
        "cox",
    },
    "classification": {
        "cross_entropy",
    },
    "regression": {
        "mse",
        "l1",
        "huber",
    },
}

TASK_METRICS = {
    "survival": {
        "concordance_index": "max",
        "mean_val_loss": "min",
    },
    "classification": {
        "accuracy": "max",
        "balanced_accuracy": "max",
        "precision_macro": "max",
        "recall_macro": "max",
        "f1_macro": "max",
        "roc_auc": "max",
        "roc_auc_ovr_macro": "max",
        "mean_val_loss": "min",
    },
    "regression": {
        "mse": "min",
        "rmse": "min",
        "mae": "min",
        "r2": "max",
        "pearson_mean": "max",
        "mean_val_loss": "min",
    },
}

DEFAULT_CRITERIA = {
    "survival": "cox",
    "classification": "cross_entropy",
    "regression": "mse",
}

DEFAULT_REFERENCE_METRICS = {
    "survival": "concordance_index",
    "classification": "f1_macro",
    "regression": "pearson_mean",
}

ENCODER_DIM = {
    "conch_v1": 512,
    "conch_v15": 768,
    "ctranspath": 768,
    "hoptimus0": 1536,
    "hoptimus1": 1536,
    "musk": 1024,
    "phikon": 768,
    "phikon_v2": 1024,
    "prov_gigapath": 1536,
    "resnet50": 1024,
    "uni_v1": 1024,
    "uni_v2": 1536,
    "virchow": 2560,
    "virchow2": 2560,
}


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------


def load_yaml_config(
    config_path: str | Path | None,
) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str, pathlib.Path, or None
        Path to the YAML file. If None, return an empty configuration.

    Returns
    -------
    dict
        YAML configuration.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.

    TypeError
        If the YAML root object is not a mapping.
    """
    if config_path is None:
        return {}

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise TypeError("The YAML configuration must contain a mapping at its root.")

    return config


def validate_config_keys(
    parser: ArgumentParser,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Reject YAML keys that are not defined by the argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Complete argument parser.

    config : dict
        Parsed YAML configuration.

    Returns
    -------
    dict
        Unmodified configuration.

    Raises
    ------
    ValueError
        If the YAML file contains unknown argument names.
    """
    valid_keys = {action.dest for action in parser._actions if action.dest != "help"}

    unknown_keys = set(config).difference(valid_keys)

    if unknown_keys:
        raise ValueError(
            "Unknown YAML configuration arguments: " f"{sorted(unknown_keys)}"
        )

    return config


def _make_yaml_serializable(value: Any) -> Any:
    """
    Recursively convert configuration values into YAML-safe objects.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {key: _make_yaml_serializable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_make_yaml_serializable(item) for item in value]

    return value


def write_config(args: Namespace) -> Namespace:
    """
    Write the fully resolved configuration to job_dir/config.yaml.

    Parameters
    ----------
    args : argparse.Namespace
        Resolved configuration.

    Returns
    -------
    argparse.Namespace
        Unmodified configuration namespace.
    """
    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    config_path = job_dir / "config.yaml"

    config = {key: _make_yaml_serializable(value) for key, value in vars(args).items()}

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            default_flow_style=False,
        )

    return args


# -----------------------------------------------------------------------------
# Argument Configuration
# -----------------------------------------------------------------------------


def normalize_string_arguments(args: Namespace) -> Namespace:
    """
    Normalize selected string-valued arguments.

    Task, optimizer, criterion, scheduler, and metric names are converted to
    lowercase to avoid case-dependent behavior.
    """
    fields = [
        "task",
        "optimizer",
        "criterion",
        "lr_scheduler",
        "ref_metric",
        "metric_mode",
        "backbone",
    ]

    for field in fields:
        value = getattr(args, field, None)

        if isinstance(value, str):
            setattr(args, field, value.lower())

    return args


def normalize_path_arguments(args: Namespace) -> Namespace:
    """
    Convert configured path arguments into pathlib.Path objects.
    """
    path_fields = [
        "config",
        "enc_dir",
        "target_path",
        "job_dir",
        "model_path",
    ]

    for field in path_fields:
        value = getattr(args, field, None)

        if value is not None:
            setattr(args, field, Path(value).expanduser())

    return args


def normalize_target_names(args: Namespace) -> Namespace:
    """
    Normalize target_name into a list when it is provided.

    This ensures that:

    - classification receives a one-element list;
    - regression receives one or several target names;
    - survival does not rely on target_name but time and event.
    """
    if args.target_name is None:
        return args

    if isinstance(args.target_name, str):
        args.target_name = [args.target_name]
    else:
        args.target_name = list(args.target_name)

    return args


def configure_training_mode(
    args: Namespace,
    train: bool,
) -> Namespace:
    """
    Record whether the current command is training or inference.
    """
    args.train = bool(train)
    
    return args


def configure_task_targets(args: Namespace) -> Namespace:
    """
    Resolve and validate target arguments for the selected task.

    Survival
        Requires time_name and event_name. target_name is cleared.

    Classification
        Requires exactly one target_name. Survival column arguments are
        cleared.

    Regression
        Requires one or more target_name values. Survival column
        arguments are cleared.
    """
    if args.task == "survival":
        if not args.time_name:
            raise ValueError("`time_name` is required when task='survival'.")

        if not args.event_name:
            raise ValueError("`event_name` is required when task='survival'.")

        args.target_name = None

    elif args.task == "classification":
        if not args.target_name:
            raise ValueError("`target_name` is required when task='classification'.")

        if len(args.target_name) != 1:
            raise ValueError(
                "Classification requires exactly one target column. "
                f"Received: {args.target_name}"
            )

        args.time_name = None
        args.event_name = None

    elif args.task == "regression":
        if not args.target_name:
            raise ValueError("`target_name` is required when task='regression'.")

        args.time_name = None
        args.event_name = None

    else:
        raise ValueError(
            f"Unsupported task '{args.task}'. "
            f"Expected one of {sorted(SUPPORTED_TASKS)}."
        )

    return args


def configure_output_dimension(args: Namespace) -> Namespace:
    """
    Resolve the size of network outputs.

    Survival
        Cox:
            One continuous risk score.

        NLLSurv:
            One logit per discrete survival-time bin.

    Regression
        One output per target column.

    Classification
        One output per class.
    """
    if args.task == "survival":
        if args.criterion == "cox":
            args.output_dim = 1

        elif args.criterion == "nllsurv":
            if args.n_time_bins is None:
                raise ValueError("`n_time_bins` must be provided for NLL survival.")

            if args.n_time_bins < 2:
                raise ValueError("`n_time_bins` must be at least 2.")

            args.output_dim = args.n_time_bins

    elif args.task == "regression":
        args.output_dim = len(args.target_name)

    elif args.task == "classification":
        if args.n_classes is None:
            raise ValueError("`n_classes` must be provided for classification.")

        if args.n_classes < 2:
            raise ValueError("`n_classes` must be at least 2 for classification.")

        args.output_dim = args.n_classes

    return args


def configure_task_criterion(args: Namespace) -> Namespace:
    """
    Resolve and validate the loss function for the selected task.
    """
    if args.criterion is None:
        args.criterion = DEFAULT_CRITERIA[args.task]

    valid_criteria = TASK_CRITERIA[args.task]

    if args.criterion not in valid_criteria:
        raise ValueError(
            f"Criterion '{args.criterion}' is incompatible with "
            f"task='{args.task}'. Expected one of "
            f"{sorted(valid_criteria)}."
        )

    return args


def configure_reference_metric(args: Namespace) -> Namespace:
    """
    Resolve the validation metric and its optimization direction.

    If no metric is supplied, a task-specific default is selected. The metric
    direction is inferred automatically unless metric_mode is explicitly
    provided.
    """
    if args.ref_metric is None:
        args.ref_metric = DEFAULT_REFERENCE_METRICS[args.task]

    valid_metrics = TASK_METRICS[args.task]

    if args.ref_metric not in valid_metrics:
        raise ValueError(
            f"Reference metric '{args.ref_metric}' is incompatible with "
            f"task='{args.task}'. Expected one of "
            f"{sorted(valid_metrics)}."
        )

    inferred_mode = valid_metrics[args.ref_metric]

    if args.metric_mode is None:
        args.metric_mode = inferred_mode

    elif args.metric_mode != inferred_mode:
        raise ValueError(
            f"Metric '{args.ref_metric}' must use "
            f"metric_mode='{inferred_mode}', received "
            f"'{args.metric_mode}'."
        )

    return args


def configure_training_defaults(args: Namespace) -> Namespace:
    """
    Resolve generic training defaults.
    """
    if args.patience is None:
        args.patience = args.epochs

    if args.minimum_epochs is None:
        args.minimum_epochs = 0

    return args


def configure_lr_scheduler(args: Namespace) -> Namespace:
    """
    Resolve scheduler-specific parameters and clear unused ones.
    """
    if args.lr_scheduler == "none":
        args.patience_lr = None
        args.restart_period = None
        args.restart_multiplier = None
        args.linear_start_factor = None
        args.linear_total_iters = None

    elif args.lr_scheduler == "plateau":
        if args.patience_lr is None:
            args.patience_lr = max(
                1,
                args.patience // 2,
            )

        args.restart_period = None
        args.restart_multiplier = None
        args.linear_start_factor = None
        args.linear_total_iters = None

    elif args.lr_scheduler == "cosine":
        if args.restart_period < 1:
            raise ValueError("`restart_period` must be at least 1.")

        if args.restart_multiplier < 1:
            raise ValueError("`restart_multiplier` must be at least 1.")

        args.patience_lr = None
        args.linear_start_factor = None
        args.linear_total_iters = None

    elif args.lr_scheduler == "linear":
        if not 0.0 < args.linear_start_factor <= 1.0:
            raise ValueError("`linear_start_factor` must be in the interval (0, 1].")

        if args.linear_total_iters is None:
            args.linear_total_iters = args.epochs

        if args.linear_total_iters < 1:
            raise ValueError("`linear_total_iters` must be at least 1.")

        args.patience_lr = None
        args.restart_period = None
        args.restart_multiplier = None

    else:
        raise ValueError(f"Unknown learning-rate scheduler: {args.lr_scheduler}")

    return args


def configure_optimizer(args: Namespace) -> Namespace:
    """
    Validate optimizer-dependent arguments and clear unused values.
    """
    if args.optimizer == "sgd":
        if not 0.0 <= args.momentum < 1.0:
            raise ValueError("`momentum` must be in the interval [0, 1).")
    else:
        args.momentum = None

    return args


def configure_classifier_layers(args: Namespace) -> Namespace:
    """
    Normalize the prediction-head hidden-layer configuration.

    width_fe is stored as a list containing one width per hidden layer.
    The classifier depth is inferred from that list.
    """
    if args.width_fe is None:
        args.width_fe = []

    args.width_fe = list(args.width_fe)
    args.n_layers_classif = len(args.width_fe)

    return args


def validate_pooling_layers(args: Namespace) -> Namespace:
    """
    Validate pooling-dependent arguments and clear unused values.
    """
    if args.pooling in {"attention", "gated_attention"}:
        if args.attention_dim < 1:
            raise ValueError(
                f"`attention_dim` must be at least 1 with pooling: {args.pooling}."
            )
        if args.num_heads < 1:
            raise ValueError(
                f"`num_heads` must be at least 1 with pooling: {args.pooling}."
            )
        return args

    if args.pooling == "top_k":
        if args.attention_dim < 1:
            raise ValueError(
                f"`attention_dim` must be at least 1 with pooling: {args.pooling}."
            )
        if args.num_heads < 1:
            raise ValueError(
                f"`num_heads` must be at least 1 with pooling: {args.pooling}."
            )
        if args.top_k < 1:
            raise ValueError(
                f"`top_k` must be at least 1 with pooling: {args.pooling}."
            )
        return args

    if args.pooling in {"mean", "max"}:
        args.attention_dim = None
        args.num_heads = None
        args.top_k = None
        return args


def configure_encoder_dim(args):
    """
    Fetch the right encorder output dimension
    """
    args.encoder_dim = ENCODER_DIM[args.encoder]
    return args


def configure_input_size(args):
    """
    Ensure feature dimensions are consistent.

    encoder_dim
        original HDF5 embedding dimensionality

    feature_dim
        number of input embedding dimensions retained

    instance_dim
        representation dimension after MIL instance transformation
    """
    if args.feature_dim == 0:
        args.feature_dim = args.encoder_dim

    if args.encoder_dim < args.feature_dim:
        raise ValueError("feature_dim cannot exceed encoder_dim")

    if not args.instance_transf:
        args.instance_dim = args.feature_dim

    if args.instance_dim == 0:
        args.instance_dim = args.feature_dim

    return args


def configure_sampling_strat(args):
    """
    Define whether sampling produces constant-size batches.
    """
    if args.n_tiles == 0:
        args.sampling_strat = "all"

    if args.sampling_strat == "random":
        args.constant_size = True
    else:
        args.constant_size = False
    return args


def validate_required_paths(
    args: Namespace,
    train: bool,
) -> Namespace:
    """
    Validate required input and output paths.
    """
    required = [
        "enc_dir",
        "target_path",
        "job_dir",
        "task",
    ]

    if not train:
        required.append("model_path")

    missing = [field for field in required if getattr(args, field, None) is None]

    if missing:
        raise ValueError(
            "Missing required arguments: "
            + ", ".join(f"--{field}" for field in missing)
        )

    if not args.enc_dir.is_dir():
        raise FileNotFoundError(f"WSI directory does not exist: {args.enc_dir}")

    if not args.target_path.is_file():
        raise FileNotFoundError(f"Target table does not exist: {args.target_path}")

    if not train and not args.model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {args.model_path}")

    return args


def validate_cross_validation(args: Namespace) -> Namespace:
    """
    Validate fold and repetition indices.
    """
    if args.n_folds is not None:
        if args.n_folds < 2:
            raise ValueError("`n_folds` must be at least 2.")

        if not 0 <= args.test_fold < args.n_folds:
            raise ValueError(
                f"`test_fold` must be between 0 and "
                f"{args.n_folds - 1}, received {args.test_fold}."
            )

    elif args.test_fold != 0:
        raise ValueError("`test_fold` was provided without setting `n_folds`.")

    if args.n_reps is not None:
        if args.n_reps < 1:
            raise ValueError("`n_reps` must be at least 1.")

        if not 0 <= args.repeat < args.n_reps:
            raise ValueError(
                f"`repeat` must be between 0 and "
                f"{args.n_reps - 1}, received {args.repeat}."
            )

    elif args.repeat != 0:
        raise ValueError("`repeat` was provided without setting `n_reps`.")

    return args


def validate_data_loading(args: Namespace) -> Namespace:
    """
    Validate DataLoader and validation-split arguments.
    """
    if args.batch_size < 1:
        raise ValueError("`batch_size` must be at least 1.")

    if args.criterion == "cox" and args.batch_size < 2:
        raise ValueError("Cox loss requires a batch containing multiple patients.")

    if args.eval_batch_size < 1:
        raise ValueError("`eval_batch_size` must be at least 1.")

    if args.criterion == "cox" and args.batch_size < 2:
        raise ValueError("Cox loss requires a batch containing multiple patients.")

    if args.num_workers < 0:
        raise ValueError("`num_workers` cannot be negative.")

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("`val_fraction` must lie strictly between 0 and 1.")

    if not args.use_val:
        args.val_fraction = None

    if args.persistent_workers and args.num_workers == 0:
        args.persistent_workers = False

    return args


def validate_training_parameters(args: Namespace) -> Namespace:
    """
    Validate generic optimization and training parameters.
    """
    if args.epochs < 1:
        raise ValueError("`epochs` must be at least 1.")

    if args.lr <= 0:
        raise ValueError("`lr` must be strictly positive.")

    if args.lr_warm_up < 0:
        raise ValueError("`lr_warm_up` cannot be negative.")

    if args.lr_warm_up >= args.epochs:
        raise ValueError("`lr_warm_up` must be smaller than `epochs`.")

    if args.weight_decay < 0:
        raise ValueError("`weight_decay` cannot be negative.")

    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("`dropout` must be in the interval [0, 1).")

    if args.gradient_clip_norm is not None:
        if args.gradient_clip_norm <= 0:
            raise ValueError("`gradient_clip_norm` must be strictly positive.")

    if args.patience < 1:
        raise ValueError("`patience` must be at least 1.")

    if args.minimum_epochs < 0:
        raise ValueError("`minimum_epochs` cannot be negative.")

    if args.minimum_epochs > args.epochs:
        raise ValueError("`minimum_epochs` cannot exceed `epochs`.")

    return args


# =============================================================================
# Parser construction
# =============================================================================


def build_parser(
    train: bool = True,
) -> ArgumentParser:
    """
    Build the complete MIL command-line parser.

    Parameters
    ----------
    train : bool, default=True
        Include training arguments when True. 
        Include the required model checkpoint argument when False.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = ArgumentParser(
        description=("Train or evaluate MIL on WSI "),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--write_config",
        action="store_true",
        help="Write the resolved configuration to job_dir/config.yaml.",
    )

    # -------------------------------------------------------------------------
    # Data paths and map representation
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--enc_dir",
        type=Path,
        default=None,
        help="Directory containing one Encoded WSI HDF5 file per case.",
    )
    parser.add_argument(
        "--target_path",
        type=Path,
        default=None,
        help="Path to the clinical or molecular target table.",
    )
    parser.add_argument(
        "--job_dir",
        type=Path,
        default=None,
        help="Directory used to store model outputs and checkpoints.",
    )

    # -------------------------------------------------------------------------
    # Target table
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--id_name",
        type=str,
        default="ID",
        help="Target-table column containing case identifiers.",
    )
    parser.add_argument(
        "--stratif_name",
        type=str,
        default="stratif",
        help="Column containing train-validation stratification labels.",
    )
    parser.add_argument(
        "--fold_name",
        type=str,
        default="test",
        help="Column containing cross-validation test-fold indices.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=sorted(SUPPORTED_TASKS),
        default=None,
        help="Prediction task.",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Target column name. Classification requires one column; "
            "regression accepts one or more columns."
        ),
    )
    parser.add_argument(
        "--time_name",
        type=str,
        default=None,
        help="Observed survival-time column for survival analysis.",
    )
    parser.add_argument(
        "--event_name",
        type=str,
        default=None,
        help="Event-indicator column for survival analysis.",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=None,
        help="Number of classes for categorical classification.",
    )
    parser.add_argument(
        "--n_time_bins",
        type=int,
        default=None,
        help="Number of discretized survival-time bins for survival.",
    )
    # -------------------------------------------------------------------------
    # WSI encoded
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--wsi_format", type=str, default="h5", help="Format of WSI embeddings"
    )
    parser.add_argument(
        "--n_tiles",
        type=int,
        default=0,
        help="Number of tiles per WSI (0 = full slide)",
    )
    parser.add_argument("--encoder", type=str, default="hoptimus1", help="Encoder used")
    parser.add_argument(
        "--feature_dim", type=int, default=0, help="First N feat to consider"
    )
    parser.add_argument(
        "--sampling_strat",
        type=str,
        choices=[
            "all",
            "random",
            "random_strict",
            "niche",
        ],
        default="random",
        help="Tile sampling strategy",
    )

    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device, for example cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=29,
        help="Random seed used for splitting and sampling.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--pin_memory",
        action=BooleanOptionalAction,
        default=False,
        help="Use pinned CPU memory in DataLoaders.",
    )
    parser.add_argument(
        "--persistent_workers",
        action=BooleanOptionalAction,
        default=False,
        help="Keep DataLoader workers alive between epochs.",
    )

    # -------------------------------------------------------------------------
    # Cross-validation
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--n_folds",
        type=int,
        default=None,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--test_fold",
        type=int,
        default=0,
        help="Index of the held-out test fold.",
    )
    parser.add_argument(
        "--n_reps",
        type=int,
        default=None,
        help="Number of repeated runs per fold.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Current repetition index.",
    )

    # -------------------------------------------------------------------------
    # DataLoader and validation split
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--use_val",
        action=BooleanOptionalAction,
        default=True,
        help="Create a validation subset from the development folds.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of development cases assigned to validation.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--no_strat_sampling",
        action=BooleanOptionalAction,
        default=True,
        help="Disable weighted sampling of training stratification groups.",
    )
    parser.add_argument(
        "--sample_wr_whole_label",
        action=BooleanOptionalAction,
        default=False,
        help="Use inverse-frequency weighting of complete stratification labels.",
    )

    # -------------------------------------------------------------------------
    # Model selection
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--ref_metric",
        type=str,
        default=None,
        help="Validation metric used for best-model selection.",
    )
    parser.add_argument(
        "--metric_mode",
        type=str,
        choices=["min", "max"],
        default=None,
        help="Whether the reference metric should be minimized or maximized.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Number of non-improving epochs tolerated.",
    )
    parser.add_argument(
        "--minimum_epochs",
        type=int,
        default=None,
        help="Minimum number of epochs before early stopping is allowed.",
    )
    parser.add_argument(
        "--minimum_improvement",
        type=float,
        default=0.0,
        help="Minimum monitored improvement required to reset patience.",
    )

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--criterion",
        type=str,
        default=None,
        help="Task-specific loss function.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "sgd"],
        default="adamw",
        help="Optimizer.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum.",
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm; use null in YAML to disable clipping.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs.",
    )

    # -------------------------------------------------------------------------
    # Learning-rate scheduler
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--lr_warm_up",
        type=int,
        default=0,
        help="Number of warm up epochs before scheduler is turn on.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["none", "cosine", "linear", "plateau"],
        default="cosine",
        help="Learning-rate scheduler.",
    )
    parser.add_argument(
        "--patience_lr",
        type=int,
        default=None,
        help="Plateau scheduler patience.",
    )
    parser.add_argument(
        "--restart_period",
        type=int,
        default=1,
        help="Initial restart period for cosine warm restarts.",
    )
    parser.add_argument(
        "--restart_multiplier",
        type=int,
        default=2,
        help="Cosine restart-period multiplier.",
    )
    parser.add_argument(
        "--linear_start_factor",
        type=float,
        default=0.1,
        help="Initial learning-rate factor for a linear scheduler.",
    )
    parser.add_argument(
        "--linear_total_iters",
        type=int,
        default=None,
        help="Number of scheduler steps for linear learning-rate scaling.",
    )

    # -------------------------------------------------
    # Network
    # -------------------------------------------------
    parser.add_argument(
        "--model",
        type=str,
        default="abmil",
        choices=["abmil", "ibmil", "transmil"],
        help="Model architecture",
    )

    parser.add_argument(
        "--instance_transf",
        default="identity",
        choices=["identity", "linear", "transformer", "roformer", "rposbias"],
        type=str,
        help="Instance-level transformation before pooling",
    )
    parser.add_argument(
        "--instance_dim",
        type=int,
        default=0,
        help="Internal feature dimension after projection",
    )

    parser.add_argument(
        "--pooling",
        type=str,
        default="gated_attention",
        choices=[
            "attention",
            "gated_attention",
            "max_attention",
            "mean",
            "max",
            "top_k",
        ],
        help="Pooling function over tiles",
    )

    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k tile selection parameter"
    )

    parser.add_argument(
        "--attention_dim", type=int, default=256, help="Attention hidden dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=1, help="Number of attention heads"
    )

    parser.add_argument(
        "--width_fe",
        type=int,
        nargs="*",
        default=None,
        help="Hidden widths of the final prediction MLP.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout probability in the prediction head.",
    )

    # Resolved by configure_output_dimension().
    parser.set_defaults(
        output_dim=None,
        n_layers_classif=None,
    )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    if not train:
        parser.add_argument(
            "--model_path",
            type=Path,
            default=None,
            help="Path to a trained model checkpoint.",
        )
    else:
        parser.set_defaults(model_path=None)

    return parser


def get_arguments(
    known_args: Sequence[str] | None = None,
    train: bool = True,
    config: str | Path | None = None,
) -> Namespace:
    """
    Parse, resolve, and validate MIL arguments.

    Precedence
    ----------
    1. Parser defaults.
    2. YAML configuration values.
    3. Explicit command-line arguments.

    Parameters
    ----------
    known_args : sequence of str, optional
        Explicit argument list. When None, use sys.argv.

    train : bool, default=True
        Parse arguments for training when True and inference otherwise.

    config : str or pathlib.Path, optional
        Default YAML configuration path. An explicit --config command-line
        argument takes precedence.

    Returns
    -------
    argparse.Namespace
        Fully resolved and validated configuration.
    """
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=config,
    )

    config_args, _ = config_parser.parse_known_args(known_args)

    yaml_config = load_yaml_config(config_args.config)

    parser = build_parser(train=train)

    validate_config_keys(
        parser,
        yaml_config,
    )

    parser.set_defaults(**yaml_config)

    args = parser.parse_args(known_args)

    try:
        args = normalize_string_arguments(args)
        args = normalize_path_arguments(args)
        args = validate_required_paths(
            args,
            train=train,
        )
        args = normalize_target_names(args)

        args = configure_training_mode(
            args,
            train=train,
        )
        args = configure_task_targets(args)
        args = configure_sampling_strat(args)
        args = configure_task_criterion(args)
        args = configure_output_dimension(args)
        args = configure_reference_metric(args)

        args = configure_training_defaults(args)
        args = configure_lr_scheduler(args)
        args = configure_optimizer(args)

        args = configure_classifier_layers(args)
        args = validate_pooling_layers(args)
        args = configure_encoder_dim(args)
        args = configure_input_size(args)

        args = validate_cross_validation(args)
        args = validate_data_loading(args)
        args = validate_training_parameters(args)

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))

    if args.write_config:
        write_config(args)

    return args
