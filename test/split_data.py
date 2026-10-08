"""
Stratified k-fold test split for MIL target tables.

The script reads a csv target table and adds two columns:

- `stratif`: the stratification group of each case, made by joining the
  values of the balancing variables (`--stratif_vars`) and of the target
  (`--target_name`);
- `test`: the index of the test fold each case belongs to, in
  `[0, n_folds - 1]`. This is the default `fold_name` read by
  `DatasetHandler`.

Folds are built with a `StratifiedKFold` so that every fold keeps the same
proportion of each stratification group.

Small groups are handled as follows:

- groups with a single case raise an error;
- groups with two cases are forced into fold 0 and left out of the k-fold.

"""

from __future__ import annotations

import datetime
import logging
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
)
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from MIL.utils import make_new_var
from utils import timetracker, configure_logging

LOGGER = logging.getLogger(__name__)

STRATIF_COLUMN = "stratif"
FOLD_COLUMN = "test"
DOUBLET_FOLD = 0


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


def build_stratification_groups(
    table: pd.DataFrame,
    stratification_columns: list[str],
) -> pd.Series:
    """
    Build one stratification label per case.

    Parameters
    ----------
    table : pandas.DataFrame
        Target table.
    stratification_columns : list of str
        Columns whose values are joined into the label.

    Returns
    -------
    pandas.Series
        Stratification label of each row, e.g. `"male_1_"`.

    Raises
    ------
    KeyError
        If a stratification column is missing from the table.
    """
    missing_columns = [
        column for column in stratification_columns if column not in table.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Stratification columns {missing_columns} not found in the table. "
            f"Available columns: {list(table.columns)}."
        )
    return table[stratification_columns].apply(
        lambda row: make_new_var(row.tolist()),
        axis=1,
    )


def assign_test_folds(
    table: pd.DataFrame,
    stratification_vars: str | list[str] | None,
    target_name: str | None,
    n_folds: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Assign every case to a stratified test fold.

    Parameters
    ----------
    table : pandas.DataFrame
        Target table, one row per case.
    stratification_vars : str or list of str or None
        Variables to balance across folds, on top of the target.
    target_name : str or None
        Target column, also balanced across folds.
    n_folds : int
        Number of test folds.
    seed : int or None, default=None
        Seed of the fold shuffling. `None` gives a different split each run.

    Returns
    -------
    pandas.DataFrame
        Copy of `table` with the `stratif` and `test` columns added.

    Raises
    ------
    ValueError
        If no stratification variable is given, or if a stratification group
        contains a single case.

    Notes
    -----
    Groups with exactly two cases cannot be split across folds, so they are
    all placed in fold 0. Fold 0 can therefore be slightly larger than the
    others. Groups with fewer than `n_folds` cases will not appear in every
    fold (`StratifiedKFold` warns about it).
    """
    if isinstance(stratification_vars, str):
        stratification_vars = [stratification_vars]
    stratification_columns = [
        column
        for column in [*(stratification_vars or []), target_name]
        if column is not None
    ]
    if not stratification_columns:
        raise ValueError(
            "At least one of `--stratification_vars` or `--target_name` must be given."
        )

    table = table.reset_index(drop=True).copy()
    table[STRATIF_COLUMN] = build_stratification_groups(
        table,
        stratification_columns,
    )

    group_sizes = table[STRATIF_COLUMN].value_counts()
    singleton_groups = group_sizes[group_sizes == 1].index.tolist()
    if singleton_groups:
        raise ValueError(
            f"Stratification groups with only 1 case: {singleton_groups}. "
            "Merge or drop them before splitting."
        )

    doublet_groups = group_sizes[group_sizes == 2].index.tolist()
    is_doublet = table[STRATIF_COLUMN].isin(doublet_groups).to_numpy()
    if doublet_groups:
        LOGGER.info(
            "Forcing %s groups with 2 cases into fold %s: %s",
            len(doublet_groups),
            DOUBLET_FOLD,
            doublet_groups,
        )

    test_folds = np.full(len(table), -1, dtype=int)
    test_folds[is_doublet] = DOUBLET_FOLD

    kfold_positions = np.flatnonzero(~is_doublet)
    kfold_groups = table[STRATIF_COLUMN].to_numpy()[kfold_positions]
    splitter = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )
    for fold, (_, test_indices) in enumerate(
        splitter.split(kfold_positions, kfold_groups)
    ):
        test_folds[kfold_positions[test_indices]] = fold

    table[FOLD_COLUMN] = test_folds
    LOGGER.info(
        "Cases per test fold: %s",
        table[FOLD_COLUMN].value_counts().sort_index().to_dict(),
    )
    return table


def parse_arguments() -> Namespace:
    """
    Parse the command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = ArgumentParser(
        description=(
            "Add a stratified k-fold `test` column to a csv target table."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target_path",
        type=str,
        required=True,
        help="Path to the csv target table.",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default=None,
        help="Target column, balanced across folds.",
    )
    parser.add_argument(
        "--stratification_vars",
        dest="stratif_vars",
        type=str,
        default=None,
        nargs="+",
        help="Extra columns balanced across folds, on top of the target.",
    )
    parser.add_argument(
        "-k",
        "--n_folds",
        dest="n_folds",
        type=int,
        default=5,
        help="Number of test folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed of the fold shuffling. Random if not given.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="dst",
        type=str,
        default=None,
        help="Output directory. Defaults to the directory of the target table.",
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        default=False,
        help=(
            "Save as `<name>_split_<date>.csv` instead of overwriting "
            "`<name>.csv`."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Split a target table into stratified test folds and save it.
    """
    configure_logging()
    args = parse_arguments()

    target_path = Path(args.target_path)
    if not target_path.is_file():
        raise FileNotFoundError(f"`--target_path` not found: {target_path}")

    split_table = assign_test_folds(
        pd.read_csv(target_path),
        stratification_vars=args.stratif_vars,
        target_name=args.target_name,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    output_dir = target_path.parent if args.dst is None else Path(args.dst)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rename:
        date_tag = datetime.date.today().strftime("%Y_%m_%d")
        output_name = f"{target_path.stem}_split_{date_tag}{target_path.suffix}"
    else:
        output_name = target_path.name

    output_path = output_dir / output_name
    split_table.to_csv(output_path, index=False)
    LOGGER.info("Saved split table to %s", output_path)


if __name__ == "__main__":
    main()
