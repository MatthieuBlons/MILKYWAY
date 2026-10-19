"""
Grid-search launcher for MIL.

The script runs one repeated cross-validation per hyperparameter
configuration. Configurations are not written as separate YAML files: a
single base configuration (`--config`) is combined with a search-space YAML
(`--search_space`), and each grid point is passed to the training entry point
as command-line overrides. Every run still writes its fully resolved
configuration to `<run_dir>/config.yaml`.

Search-space format
-------------------
Each top-level key is one axis and maps to a list of values. The grid is the
cartesian product of all axes.

- A plain value (scalar or list) overrides the argument named by the axis.
- A mapping value overrides several arguments together (coupled
  parameters). The axis name is then only a label.

Outputs
-------
::

    output_dir/<job>_<date>/
        base_config.yaml        copy of --config
        search_space.yaml       copy of --search_space
        grid.yaml, grid.csv     config_id -> overrides
        cfg_<i>/test_<f>/rep_<r>/   one cross-validation per configuration
        failed_runs.csv         runs that raised during training, if any
        grid_summary.csv        validation and test metrics per configuration

Configurations are ranked on the mean validation reference metric over
folds. Test metrics are reported for every configuration, but only the
validation ranking should be used to select one.
"""

from __future__ import annotations

import itertools
import logging
import shutil
import warnings
from argparse import (
    ArgumentParser,
    BooleanOptionalAction,
    Namespace,
)
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm.auto import tqdm

from utils import timetracker, configure_logging

from cross_val import (
    build_job_parser,
    build_run_directory,
    build_training_arguments,
    parse_job_arguments,
    resolve_device,
    resolve_job_name,
    test_cross_validation,
)

from MIL.arguments import build_parser, load_yaml_config, validate_config_keys
from MIL.train import main as train_single_run

warnings.filterwarnings("ignore")


LOGGER = logging.getLogger(__name__)

# Arguments set by the launcher for every run; they cannot be searched.
RESERVED_ARGUMENTS = {
    "config",
    "device",
    "num_workers",
    "job_dir",
    "n_reps",
    "repeat",
    "n_folds",
    "test_fold",
    "seed",
    "write_config",
}

COMPLETED_MARKER = ".completed"


def build_grid_parser() -> ArgumentParser:
    """
    Build the grid-search launcher argument parser.

    Extends the cross-validation parser with the search space, resume and
    dry-run options.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
    """
    parser = build_job_parser() # parser is build on top of cross_val.build_job_parser -> Can pass any of its arguments
    parser.description = (
        "Run one repeated cross-validation per hyperparameter configuration "
        "of a grid search for MIL on encoded whole-slide images."
    )

    parser.add_argument(
        "--search_space",
        type=Path,
        required=True,
        help="YAML search space. See the module docstring for the format.",
    )
    parser.add_argument(
        "--resume_dir",
        type=Path,
        default=None,
        help=(
            "Existing grid-search directory to resume. Completed runs are "
            "skipped. The base config and search space must be unchanged."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Log the expanded grid and the per-configuration overrides, then exit.",
    )

    return parser


def parse_grid_arguments(
    known_args: list[str] | None = None,
) -> Namespace:
    """
    Parse and validate grid-search launcher arguments.

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
    parser = build_grid_parser()
    args = parse_job_arguments(
        known_args=known_args,
        parser=parser,
    )

    if not args.search_space.is_file():
        parser.error(f"Search space file does not exist: {args.search_space}")

    if args.resume_dir is not None and not args.resume_dir.is_dir():
        parser.error(f"`resume_dir` does not exist: {args.resume_dir}")

    return args


def expand_search_space(
    search_space: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Expand a search space into the list of its grid points.

    Parameters
    ----------
    search_space : dict
        Mapping axis name -> list of values. A mapping value overrides
        several arguments together; any other value overrides the argument
        named by the axis.

    Returns
    -------
    list of dict
        One mapping argument name -> value per grid point, in
        `itertools.product` order.

    Raises
    ------
    ValueError
        If an axis is not a non-empty list, or if two axes set the same
        argument.
    """
    axes = []

    for axis_name, values in search_space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Search space axis `{axis_name}` must be a non-empty list, "
                f"got: {values!r}"
            )

        axes.append(
            [
                dict(value) if isinstance(value, dict) else {axis_name: value}
                for value in values
            ]
        )

    grid = []

    for combination in itertools.product(*axes):
        overrides: dict[str, Any] = {}

        for bundle in combination:
            duplicated_keys = set(overrides).intersection(bundle)

            if duplicated_keys:
                raise ValueError(
                    "Several search space axes set the same arguments: "
                    f"{sorted(duplicated_keys)}"
                )

            overrides.update(bundle)

        grid.append(overrides)

    return grid


def validate_grid_keys(
    parser: ArgumentParser,
    grid: list[dict[str, Any]],
) -> None:
    """
    Reject grid arguments that are unknown or set by the launcher.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Training argument parser.

    grid : list of dict
        Expanded grid.

    Raises
    ------
    ValueError
        If a grid argument is unknown or reserved.
    """
    grid_keys = set().union(*grid)

    validate_config_keys(
        parser,
        dict.fromkeys(grid_keys),
    )

    reserved_keys = grid_keys.intersection(RESERVED_ARGUMENTS)

    if reserved_keys:
        raise ValueError(
            f"Search space sets launcher-managed arguments {sorted(reserved_keys)}. "
            f"Reserved arguments: {sorted(RESERVED_ARGUMENTS)}"
        )


def overrides_to_cli(
    parser: ArgumentParser,
    overrides: dict[str, Any],
) -> list[str]:
    """
    Convert one grid point into training command-line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Training argument parser, used to find each option string and how
        it consumes values.

    overrides : dict
        Mapping argument name -> value.

    Returns
    -------
    list of str
        Command-line arguments, for example
        `["--lr", "0.0001", "--width_fe", "512", "256"]`.

    Raises
    ------
    ValueError
        If a value cannot be expressed on the command line for its
        argument.
    """
    actions = {action.dest: action for action in parser._actions}
    cli_arguments = []

    for key, value in overrides.items():
        action = actions[key]
        flag = action.option_strings[0]

        if value is None:
            raise ValueError(
                f"Search space value for `{key}` is null. Set null values in "
                "the base config instead."
            )

        if isinstance(action, BooleanOptionalAction):
            if not isinstance(value, bool):
                raise ValueError(
                    f"`{key}` expects true or false in the search space, "
                    f"got: {value!r}"
                )

            cli_arguments.append(action.option_strings[0 if value else 1])

        elif action.nargs == 0:
            if value != action.const:
                raise ValueError(
                    f"`{key}` is a flag that can only be set to {action.const} "
                    f"from the search space, got: {value!r}"
                )

            cli_arguments.append(flag)

        elif isinstance(value, list):
            if action.nargs not in {"+", "*"}:
                raise ValueError(
                    f"`{key}` takes a single value, got the list: {value!r}"
                )

            if not value and action.nargs == "+":
                raise ValueError(f"`{key}` requires at least one value.")

            cli_arguments.extend([flag, *(str(item) for item in value)])

        else:
            cli_arguments.extend([flag, str(value)])

    return cli_arguments


def build_config_id(index: int) -> str:
    """
    Build the identifier and directory name of one grid configuration.

    Parameters
    ----------
    index : int
        Zero-based configuration index.

    Returns
    -------
    str
        Configuration identifier, for example `cfg_007`.
    """
    return f"cfg_{index:03d}"


def grid_to_dataframe(
    grid: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Tabulate the grid, one row per configuration.

    Parameters
    ----------
    grid : list of dict
        Expanded grid.

    Returns
    -------
    pandas.DataFrame
        Columns `config_id` and one column per searched argument. List
        values are written as space-separated strings.
    """
    rows = []

    for index, overrides in enumerate(grid):
        row = {"config_id": build_config_id(index)}

        for key, value in overrides.items():
            if isinstance(value, list):
                value = " ".join(str(item) for item in value)

            row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


def prepare_job_directory(
    args: Namespace,
    grid: list[dict[str, Any]],
) -> tuple[Path, str]:
    """
    Create a new grid-search directory, or check a resumed one.

    A new directory receives copies of the base config and the search
    space, and the expanded grid. A resumed directory must hold the same
    base config and grid.

    Parameters
    ----------
    args : argparse.Namespace
        Grid-search launcher configuration.

    grid : list of dict
        Expanded grid.

    Returns
    -------
    tuple of (pathlib.Path, str)
        Job directory and job name.

    Raises
    ------
    ValueError
        If a resumed directory was created with another base config or
        grid.
    """
    if args.resume_dir is not None:
        job_dir = args.resume_dir.expanduser().resolve()

        stored_config = load_yaml_config(job_dir / "base_config.yaml")
        stored_grid = load_yaml_config(job_dir / "grid.yaml")["grid"]

        if stored_config != load_yaml_config(args.config):
            raise ValueError(
                f"`config` differs from the base config stored in {job_dir}."
            )

        if stored_grid != grid:
            raise ValueError(
                f"`search_space` expands to another grid than the one stored "
                f"in {job_dir}."
            )

        LOGGER.info("Resuming grid search in: %s", job_dir)

        return job_dir, job_dir.name

    job_name = resolve_job_name(args.job)
    job_dir = args.output_dir.expanduser().resolve() / job_name
    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy(args.config, job_dir / "base_config.yaml")
    shutil.copy(args.search_space, job_dir / "search_space.yaml")

    with (job_dir / "grid.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {"grid": grid},
            file,
            sort_keys=False,
            default_flow_style=False,
        )

    grid_to_dataframe(grid).to_csv(
        job_dir / "grid.csv",
        index=False,
    )

    return job_dir, job_name


def run_grid_search(
    args: Namespace,
    grid: list[dict[str, Any]],
    grid_cli_arguments: list[list[str]],
    job_dir: Path,
    job_name: str,
) -> None:
    """
    Train and test every configuration of the grid.

    Each configuration is tested as soon as all its runs are trained.
    Runs holding a completion marker are skipped. A configuration is
    tested again only if one of its runs was trained in this session or
    its test results are missing.

    Parameters
    ----------
    args : argparse.Namespace
        Grid-search launcher configuration.

    grid : list of dict
        Expanded grid.

    grid_cli_arguments : list of list of str
        Command-line overrides of each configuration.

    job_dir : pathlib.Path
        Grid-search directory.

    job_name : str
        Grid-search name, used as W&B group prefix.
    """
    device = resolve_device(args)

    LOGGER.info(
        "Starting grid search | configs=%d | folds=%d | repeats=%d | runs=%d",
        len(grid),
        args.n_folds,
        args.n_repeats,
        len(grid) * args.n_folds * args.n_repeats,
    )
    LOGGER.info("Device: %s", device)
    LOGGER.info("Output directory: %s", job_dir)

    failed_runs = []

    progress = tqdm(
        total=len(grid) * args.n_folds * args.n_repeats,
        desc="Grid search",
        unit="run",
        disable=not args.verbose,
    )

    for index, cli_arguments in enumerate(grid_cli_arguments):
        config_id = build_config_id(index)
        config_dir = job_dir / config_id
        trained_runs = 0

        LOGGER.info("Configuration %s: %s", config_id, " ".join(cli_arguments))

        for fold, repetition in itertools.product(
            range(args.n_folds),
            range(args.n_repeats),
        ):
            progress.update()
            progress.set_postfix(
                {
                    "config": f"{index + 1}/{len(grid)}",
                    "fold": f"{fold + 1}/{args.n_folds}",
                    "repeat": f"{repetition + 1}/{args.n_repeats}",
                },
                refresh=False,
            )

            run_dir = build_run_directory(
                job_dir=config_dir,
                fold=fold,
                repetition=repetition,
            )

            if (run_dir / COMPLETED_MARKER).is_file():
                LOGGER.info(
                    "Skipping completed run | config=%s | fold=%d | repeat=%d",
                    config_id,
                    fold,
                    repetition,
                )
                continue

            training_args = build_training_arguments(
                args=args,
                device=device,
                run_dir=run_dir,
                fold=fold,
                repetition=repetition,
                overrides=cli_arguments,
            )

            try:
                stop_epoch = train_single_run(
                    project_name=args.project,
                    job_name=f"{job_name}_{config_id}",
                    known_args=training_args,
                    verbose=args.verbose,
                    log=args.log,
                )

            except (Exception, SystemExit):
                # get_arguments reports invalid configurations with
                # parser.error, which raises SystemExit.
                LOGGER.exception(
                    "Training failed | config=%s | fold=%d | repeat=%d | run_dir=%s",
                    config_id,
                    fold,
                    repetition,
                    run_dir,
                )
                failed_runs.append(
                    {
                        "config_id": config_id,
                        "test": fold,
                        "repeat": repetition,
                        "run_dir": str(run_dir),
                    }
                )
                continue

            (run_dir / COMPLETED_MARKER).touch()
            trained_runs += 1

            LOGGER.info(
                "Completed run | config=%s | fold=%d/%d | repeat=%d/%d | epochs=%d",
                config_id,
                fold + 1,
                args.n_folds,
                repetition + 1,
                args.n_repeats,
                stop_epoch,
            )

        already_tested = (config_dir / "ensemble_metrics.csv").is_file()

        if trained_runs == 0 and already_tested:
            continue

        try:
            test_cross_validation(
                config_dir,
                n_best=args.n_ensemble,
                n_mcdo=args.n_mcdo,
            )

        except Exception:
            LOGGER.exception("Testing failed | config=%s", config_id)

    progress.close()

    if failed_runs:
        pd.DataFrame(failed_runs).to_csv(
            job_dir / "failed_runs.csv",
            index=False,
        )
        LOGGER.warning(
            "%d run(s) failed. See %s",
            len(failed_runs),
            job_dir / "failed_runs.csv",
        )


def summarize_configuration(
    config_dir: Path,
) -> dict[str, Any] | None:
    """
    Summarize the validation and test metrics of one configuration.

    Parameters
    ----------
    config_dir : pathlib.Path
        Cross-validation directory of the configuration.

    Returns
    -------
    dict or None
        Reference metric name and mode, mean and std over folds of the
        validation reference metric, and mean and std over folds of each
        ensemble test metric. None if the configuration has no results.
    """
    validation_path = config_dir / "mean_validation_over_repeats.csv"
    run_config_path = next(config_dir.rglob("config.yaml"), None)

    if not validation_path.is_file() or run_config_path is None:
        return None

    run_config = load_yaml_config(run_config_path)
    ref_metric = run_config["ref_metric"]

    validation = pd.read_csv(validation_path)

    summary = {
        "ref_metric": ref_metric,
        "metric_mode": run_config["metric_mode"],
        "n_folds_done": len(validation),
        f"val_{ref_metric}_mean": validation[ref_metric].mean(),
        f"val_{ref_metric}_std": validation[ref_metric].std(),
    }

    test_path = config_dir / "ensemble_metrics.csv"

    if test_path.is_file():
        test_metrics = pd.read_csv(test_path).drop(
            columns=["test", "n_models"],
            errors="ignore",
        )

        for metric_name in test_metrics.select_dtypes("number").columns:
            summary[f"test_{metric_name}_mean"] = test_metrics[metric_name].mean()
            summary[f"test_{metric_name}_std"] = test_metrics[metric_name].std()

    return summary


def summarize_grid_search(
    job_dir: Path,
    grid: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Rank all configurations on the validation reference metric.

    Parameters
    ----------
    job_dir : pathlib.Path
        Grid-search directory.

    grid : list of dict
        Expanded grid.

    Returns
    -------
    pandas.DataFrame
        One row per configuration with its searched arguments, metrics,
        `rank` (1 is best on validation) and `selected`. Also written to
        `job_dir/grid_summary.csv`.
    """
    grid_df = grid_to_dataframe(grid)
    summaries = []

    for config_id in grid_df["config_id"]:
        summary = summarize_configuration(job_dir / config_id)

        if summary is None:
            LOGGER.warning("No results for configuration %s.", config_id)
            continue

        summary["config_id"] = config_id
        summaries.append(summary)

    if not summaries:
        LOGGER.warning("No configuration produced results.")
        return grid_df

    summary_df = grid_df.merge(
        pd.DataFrame(summaries),
        on="config_id",
        how="left",
    )

    ref_metric = summaries[0]["ref_metric"]
    metric_mode = summaries[0]["metric_mode"]

    summary_df["rank"] = (
        summary_df[f"val_{ref_metric}_mean"]
        .rank(ascending=metric_mode == "min", method="min")
        .astype("Int64")
    )
    summary_df["selected"] = summary_df["rank"] == 1
    summary_df = summary_df.sort_values("rank", na_position="last")

    summary_df.to_csv(
        job_dir / "grid_summary.csv",
        index=False,
    )

    best = summary_df.iloc[0]
    LOGGER.info(
        "Best configuration on validation: %s | %s=%.4f",
        best["config_id"],
        ref_metric,
        best[f"val_{ref_metric}_mean"],
    )

    return summary_df


def main(
    known_args: list[str] | None = None,
) -> None:
    """
    Run a grid search of repeated cross-validations.

    Parameters
    ----------
    known_args : list of str, optional
        Explicit launcher arguments. When None, use command-line
        arguments.
    """
    args = parse_grid_arguments(known_args=known_args)

    configure_logging(verbose=args.verbose or args.dry_run)

    training_parser = build_parser(train=True) # with default args
    grid = expand_search_space(load_yaml_config(args.search_space))
    validate_grid_keys(training_parser, grid)

    grid_cli_arguments = [
        overrides_to_cli(training_parser, overrides) for overrides in grid
    ]

    if args.dry_run:
        LOGGER.info(
            "Grid: %d configurations x %d folds x %d repeats = %d runs",
            len(grid),
            args.n_folds,
            args.n_repeats,
            len(grid) * args.n_folds * args.n_repeats,
        )

        for index, cli_arguments in enumerate(grid_cli_arguments):
            LOGGER.info("  %s: %s", build_config_id(index), " ".join(cli_arguments))

        return

    job_dir, job_name = prepare_job_directory(args, grid)

    timer = timetracker(
        name="MIL grid search",
        verbose=args.clock,
    )
    timer.tic()

    try:
        run_grid_search(
            args=args,
            grid=grid,
            grid_cli_arguments=grid_cli_arguments,
            job_dir=job_dir,
            job_name=job_name,
        )
    finally:
        timer.toc()

    summarize_grid_search(job_dir, grid)

    LOGGER.info("Grid-search results saved in: %s", job_dir)


if __name__ == "__main__":
    main()
