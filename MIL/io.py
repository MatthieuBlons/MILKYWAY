from __future__ import annotations

from typing import Any
from pathlib import Path

import pandas as pd

import numpy as np


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Read a tabular file with pandas.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the table.

    **kwargs
        Arguments passed to the selected pandas reader.

    Returns
    -------
    pandas.DataFrame
        Loaded table.

    Raises
    ------
    ValueError
        If the file extension is unsupported.
    """
    path = Path(path)
    extension = path.suffix.lower()

    readers = {
        ".csv": lambda p: pd.read_csv(p, **kwargs),
        ".tsv": lambda p: pd.read_csv(p, sep="\t", **kwargs),
        ".txt": lambda p: pd.read_csv(p, sep="\t", **kwargs),
        ".xlsx": lambda p: pd.read_excel(p, **kwargs),
        ".xls": lambda p: pd.read_excel(p, **kwargs),
        ".parquet": lambda p: pd.read_parquet(p, **kwargs),
        ".feather": lambda p: pd.read_feather(p, **kwargs),
        ".pkl": lambda p: pd.read_pickle(p, **kwargs),
        ".pickle": lambda p: pd.read_pickle(p, **kwargs),
    }

    try:
        reader = readers[extension]
    except KeyError as exc:
        supported = ", ".join(sorted(readers))
        raise ValueError(
            f"Unsupported table format '{extension}'. "
            f"Supported formats: {supported}."
        ) from exc

    return reader(path)


def parse_file_name(file: str | Path) -> tuple[Path, str, str]:
    """
    Split a file path into its parent directory, stem, and extension.

    Parameters
    ----------
    file : str or pathlib.Path
        Input file path.

    Returns
    -------
    parent : pathlib.Path
        Parent directory.

    stem : str
        File name without its extension.

    extension : str
        File extension, including the leading period.
    """
    path = Path(file)
    return path.parent, path.stem, path.suffix
