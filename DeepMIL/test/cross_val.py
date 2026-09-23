"""
Cross-validation training launcher for MIL.

The script orchestrates repeated cross-validation experiments by invoking
the single-run training entry point once for every fold and repetition.

Testing is done on the exclued fold with the best models if validation is
used. 

Ensembling of models can be invocted. 

Each run receives:

- a shared YAML configuration;
- a fold-specific test-fold index;
- a repetition index;
- a dedicated output directory;
- the resolved compute device.

"""

from __future__ import annotations

from datetime import datetime
import logging
import warnings
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
)
from pathlib import Path

import torch
from tqdm.auto import tqdm

from dtime.trackers import timetracker

import pandas as pd

from mil.arguments import load_yaml_config
from mil.train import main as train_single_run
from mil.test import (
    mean_dataframe,
    select_best_repeat,
    copy_best_to_root,
    test,
    compute_test_metrics,
    ensemble_results,
    results_to_dataframe,
)
from mil.utils import load_model
from mil.dataloader import DatasetHandler

warnings.filterwarnings("ignore")


LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """
    Configure console logging.

    Parameters
    ----------
    verbose : bool, default=False
        Use informational logging when enabled and warning-level logging
        otherwise.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def build_job_parser() -> ArgumentParser:
    """
    Build the cross-validation launcher argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
    """
    parser = ArgumentParser(
        description=(
            "Run repeated cross-validation training for ProteONET on "
            "multichannel spatial proteomic maps."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # Experiment tracking

    parser.add_argument(
        "--project",
        type=str,
        default="ProteONET",
        help="W&B project name.",
    )
    parser.add_argument(
        "--job",
        type=str,
        default=None,
        help=(
            "Experiment group and output-directory name. "
            "Defaults to a date-stamped name."
        ),
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable W&B and TensorBoard logging.",
    )

    # Cross-validation

    parser.add_argument(
        "--n_folds",
        type=int,
        required=True,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=1,
        help="Number of repeated training runs per fold.",
    )
    parser.add_argument(
        "--n_ensemble",
        type=int,
        default=1,
        help="Number of models to ensemble per fold.",
    )

    # Configuration and outputs

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration passed to each training run.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Parent directory used to store the cross-validation outputs.",
    )

    # Runtime

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "PyTorch device passed to each run, for example cpu, cuda, "
            "cuda:0, or mps. When omitted, the device is inferred."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Request a CUDA device when --device is not explicitly set. "
            "Falls back to CPU if CUDA is unavailable."
        ),
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=None,
        help="CUDA device index used with --gpu.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display detailed logging and progress bars.",
    )
    parser.add_argument(
        "--clock",
        action="store_true",
        help="Display total execution timing.",
    )

    return parser


def parse_job_arguments(
    known_args: list[str] | None = None,
) -> Namespace:
    """
    Parse and validate cross-validation launcher arguments.

    Parameters
    ----------
    known_args : list of str, optional
        Explicit argument sequence. When None, use command-line
        arguments.

    Returns
    -------
    argparse.Namespace
        Validated launcher configuration.
    """
    parser = build_job_parser()
    args = parser.parse_args(known_args)

    if args.n_folds < 2:
        parser.error("--n_folds must be at least 2.")

    if args.n_repeats < 1:
        parser.error("--n_repeats must be at least 1.")

    if not args.config.is_file():
        parser.error(f"Configuration file does not exist: {args.config}")

    if args.device is not None and args.gpu:
        parser.error("Use either --device or --gpu, not both.")

    return args


def resolve_device(args: Namespace) -> str:
    """
    Resolve the compute device passed to each training run.

    Explicit --device takes precedence. Otherwise, --gpu requests
    CUDA, optionally with a selected device index. If no device option is
    provided, CUDA is used when available and CPU otherwise.

    Parameters
    ----------
    args : argparse.Namespace
        Launcher arguments.

    Returns
    -------
    str
        Resolved PyTorch device string.
    """
    if args.device is not None:
        return args.device

    if args.gpu:
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA was requested but is unavailable; using CPU.")
            return "cpu"

        if args.device_id is None:
            return "cuda"

        if args.device_id >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {args.device_id} was requested, but only "
                f"{torch.cuda.device_count()} device(s) are available."
            )

        return f"cuda:{args.device_id}"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def resolve_job_name(job_name: str | None) -> str:
    """
    Resolve a date-stamped experiment name.

    Parameters
    ----------
    job_name : str or None
        User-provided experiment name.

    Returns
    -------
    str
        Resolved job name.
    """
    date_tag = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if job_name is None:
        return f"ProteONET_CV_{date_tag}"

    return f"{job_name}_{date_tag}"


def log_configuration(
    config: dict,
    title: str = "Resolved base configuration",
) -> None:
    """
    Log the contents of a configuration mapping.

    Parameters
    ----------
    config : dict
        Configuration values.

    title : str, default="Resolved base configuration"
        Heading written before the configuration entries.
    """
    LOGGER.info("%s:", title)

    for key, value in config.items():
        LOGGER.info("  %s: %s", key, value)


def build_run_directory(
    job_dir: Path,
    fold: int,
    repetition: int,
) -> Path:
    """
    Construct and create one fold/repetition output directory.

    Parameters
    ----------
    job_dir : pathlib.Path
        Parent experiment directory.

    fold : int
        Zero-based test-fold index.

    repetition : int
        Zero-based repetition index.

    Returns
    -------
    pathlib.Path
        Created run directory.
    """
    run_dir = job_dir / f"test_{fold}" / f"rep_{repetition}"

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_dir


def build_training_arguments(
    args: Namespace,
    device: str,
    run_dir: Path,
    fold: int,
    repetition: int,
) -> list[str]:
    """
    Build command-line arguments for one training run.

    Values provided here override matching YAML configuration values.

    Parameters
    ----------
    args : argparse.Namespace
        Cross-validation launcher configuration.

    device : str
        Resolved PyTorch device.

    run_dir : pathlib.Path
        Output directory for the current run.

    fold : int
        Zero-based test-fold index.

    rep : int
        Zero-based repetition index.

    Returns
    -------
    list[str]
        Argument list passed to the single-run training entry point.
    """
    return [
        "--config",
        str(args.config),
        "--device",
        device,
        "--num_workers",
        str(args.num_workers),
        "--job_dir",
        str(run_dir),
        "--n_reps",
        str(args.n_repeats),
        "--repeat",
        str(repetition),
        "--n_folds",
        str(args.n_folds),
        "--test_fold",
        str(fold),
        "--write_config",
    ]


def train_cross_validation(
    args: Namespace,
) -> str:
    """
    Train every cross-validation fold and repetition.

    Parameters
    ----------
    args : argparse.Namespace
        Validated launcher configuration.

    Returns
    -------
    dict
        Mapping (fold, repetition) to the number of completed epochs.
    """
    device = resolve_device(args)
    job_name = resolve_job_name(args.job)

    job_dir = args.output_dir.expanduser().resolve() / job_name
    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info(
        "Starting ProteONET cross-validation | folds=%d | repeats=%d",
        args.n_folds,
        args.n_repeats,
    )
    LOGGER.info("Device: %s", device)
    LOGGER.info("Output directory: %s", job_dir)

    base_config = load_yaml_config(args.config)
    log_configuration(base_config)

    run_indices = [
        (fold, repetition)
        for fold in range(args.n_folds)
        for repetition in range(args.n_repeats)
    ]

    completed_epochs: dict[tuple[int, int], int] = {}

    progress = tqdm(
        run_indices,
        desc="Cross-validation",
        unit="run",
        disable=not args.verbose,
    )

    for fold, repetition in progress:
        run_dir = build_run_directory(
            job_dir=job_dir,
            fold=fold,
            repetition=repetition,
        )

        progress.set_postfix(
            {
                "fold": f"{fold + 1}/{args.n_folds}",
                "repeat": (f"{repetition + 1}/{args.n_repeats}"),
            },
            refresh=False,
        )

        LOGGER.info(
            "Starting run | fold=%d/%d | repeat=%d/%d",
            fold + 1,
            args.n_folds,
            repetition + 1,
            args.n_repeats,
        )

        training_args = build_training_arguments(
            args=args,
            device=device,
            run_dir=run_dir,
            fold=fold,
            repetition=repetition,
        )

        try:
            stop_epoch = train_single_run(
                project_name=args.project,
                job_name=job_name,
                known_args=training_args,
                verbose=args.verbose,
                log=args.log,
            )

        except Exception:
            LOGGER.exception(
                "Training failed | fold=%d | repeat=%d | run_dir=%s",
                fold,
                repetition,
                run_dir,
            )
            raise

        completed_epochs[(fold, repetition)] = stop_epoch

        LOGGER.info(
            "Completed run | fold=%d/%d | repeat=%d/%d | epochs=%d",
            fold + 1,
            args.n_folds,
            repetition + 1,
            args.n_repeats,
            stop_epoch,
        )

        # LOG completed_epochs

    return job_dir


def test_cross_validation(
    job_dir: str,
    n_best: int = 1,
) -> None:

    model_list = list(Path(job_dir).rglob("*best_*.pt.tar"))
    val_results = []
    for m in model_list:
        try:
            state = torch.load(m, map_location="cpu", weights_only=False)
        except:
            continue
        args_m = state["args"]
        tags = {"test": args_m.test_fold, "repeat": args_m.repeat}
        metrics = state["best_metrics"]
        tags.update(metrics)
        val_results.append(tags)

    # make df
    df = pd.DataFrame(val_results)
    df.to_csv(Path(job_dir) / "all_validation_results.csv", index=False)
    df_mean_r = mean_dataframe(df)
    df_mean_r.to_csv(Path(job_dir) / "mean_validation_over_repeats.csv", index=False)

    ref_metric = (
        args_m.ref_metric,
        args_m.metric_mode,
    )  # extract the name of the reference from one of the loaded models args (last one)

    models_params = select_best_repeat(
        df=df, reference_metric=ref_metric[0], metric_mode=ref_metric[1], n_best=n_best
    )  # selection is done according to best metric on validation

    best_in_root = copy_best_to_root(job_dir, models_params)

    # test with seleceted best models
    test_results = []
    test_metrics = []
    ensemble_test_results = []
    ensemble_test_metrics = []

    for test_fold, repeats in best_in_root.items():
        fold_results = []
        fold_task = None
        fold_model = None
        for repeat, model_path in repeats.items():
            model_tag = {"test": test_fold, "repeat": repeat}
            try:
                m = load_model(model_path)
                m.network.print_summary()

                m_dataset = DatasetHandler(
                    m.args,
                    predict=False,
                )

                test_loader = m_dataset.get_loader(
                    training=False,
                )

                model_results = test(
                    m,
                    test_loader,
                    mcdo_passes=1,
                )

            except Exception:
                LOGGER.exception(
                    "Testing failed for test=%s, repeat=%s.",
                    test_fold,
                    repeat,
                )
                continue

            if fold_task is None:
                fold_task = m.task
            elif m.task != fold_task:
                raise ValueError(
                    f"Inconsistent tasks in test fold {test_fold}: "
                    f"{fold_task} != {m.task}."
                )

            model_results["test"] = test_fold
            model_results["repeat"] = repeat

            model_metrics = compute_test_metrics(
                model=m,
                result=model_results,
            )
            model_tag.update(model_metrics)
            
            fold_results.append(model_results)
            test_results.append(model_results)
            test_metrics.append(model_tag)

            fold_model = m

        if not fold_results:
            LOGGER.warning(
                "No valid models found for test fold %s.",
                test_fold,
            )
            continue

        fold_ensemble = ensemble_results(
            results=fold_results,
            task=fold_task,
        )
        fold_ensemble["test"] = test_fold

        fold_metrics = compute_test_metrics(
            model=fold_model,
            result=fold_ensemble,
        )
        fold_metrics["test"] = test_fold
        fold_metrics["n_models"] = len(fold_results)

        ensemble_test_results.append(fold_ensemble)
        ensemble_test_metrics.append(fold_metrics)

    individual_results_df = results_to_dataframe(
        results=test_results,
        task=fold_task,
        target_names=getattr(m.args, "target_name", None),
    )

    ensemble_results_df = results_to_dataframe(
        results=ensemble_test_results,
        task=fold_task,
        target_names=getattr(m.args, "target_name", None),
    )

    individual_results_df.to_csv(
        Path(job_dir) / "all_test_results.csv",
        index=False,
    )

    pd.DataFrame(test_metrics).to_csv(
        Path(job_dir) / "test_metrics.csv",
        index=False,
    )

    ensemble_results_df.to_csv(
        Path(job_dir) / "ensemble_results.csv",
        index=False,
    )

    pd.DataFrame(ensemble_test_metrics).to_csv(
        Path(job_dir) / "ensemble_metrics.csv",
        index=False,
    )


def main(
    known_args: list[str] | None = None,
) -> dict[tuple[int, int], int]:
    """
    Run repeated cross-validation training.

    Parameters
    ----------
    known_args : list of str, optional
        Explicit launcher arguments. When None, use command-line
        arguments.

    Returns
    -------
    dict
        Mapping (fold, repetition) to completed training epochs.
    """
    args = parse_job_arguments(known_args=known_args)

    configure_logging(verbose=args.verbose)

    timer = timetracker(
        name="MIL cross-validation",
        verbose=args.clock,
    )
    timer.tic()

    try:
        job_dir = train_cross_validation(args)
    finally:
        timer.toc()

    LOGGER.info("Cross-validation training completed successfully.")

    try:
        test_cross_validation(job_dir, n_best=args.n_ensemble)

    finally:
        LOGGER.info(f"Cross-validation results saved in: {job_dir}.")


if __name__ == "__main__":
    main()
