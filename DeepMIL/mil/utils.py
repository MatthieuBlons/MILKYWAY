from __future__ import annotations

from typing import Any, Generator
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch

import pandas as pd

from slide.utils import read_h5_features, read_h5_coords
from model import DeepMIL

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


def to_numpy(value: Any) -> np.ndarray:
    """Convert a tensor or array-like object to a NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def extend_case_ids(
    case_ids: list[str],
    batch_case_ids: Any,
) -> None:
    """Append case identifiers returned by a collated batch."""
    if isinstance(batch_case_ids, str):
        case_ids.append(batch_case_ids)
        return

    case_ids.extend(str(case_id) for case_id in batch_case_ids)


def nan_like(
    array: np.ndarray,
) -> np.ndarray:
    """Create a floating-point NaN array with the same shape as the input."""
    return np.full(
        np.asarray(array).shape,
        np.nan,
        dtype=float,
    )


@contextmanager
def enable_mc_dropout(
    network: torch.nn.Module,
    enabled: bool = False,
) -> Generator[None]:
    """Enable dropout during test-time forward pass."""
    network.eval()
    try:
        if enabled:
            for m in network.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
        yield

    finally:
        network.eval()


def prep_wsi(path: str | Path, args) -> dict[str, Any]:
    """Preprocess the encoded wsi."""
    _, feat_array = read_h5_features(path)
    coords_attrs, coords_array = read_h5_coords(path)
    slide_size = coords_attrs.get("level_size")
    if slide_size is None:
        raise ValueError(f"Missing 'level_size' attribute in '{path.name}'.")
    slide_size = np.asarray(
        slide_size,
        dtype=np.float32,
    )
    coords_array = coords_array[:, :2] / slide_size

    feats = feat_array[:, : args.feature_dim]
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ttensor = torch.from_numpy(feats)  # get all the tiles

    coords = np.ascontiguousarray(coords, dtype=np.float32)
    ctensor = torch.from_numpy(coords)

    sample: dict[str, Any] = {
        "tiles": ttensor,
        "coords": ctensor,
        "case_id": path.stem,
        "path": str(path),
    }

    return sample


def load_model(
    path: str | Path,
    device: str = "cpu",
    weights_only: bool = False,
    map_location: str = "cpu",
    with_data: bool = False,
    dropout: bool = False,
):
    """
    Loads and prepare a trained model for prediction.

    Parameters
    ----------
        model_path (str): path to the *.pt.tar model
    """
    device = device.lower()

    checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)

    args = checkpoint["args"]
    args.device = device

    model = DeepMIL(
        args=args,
        with_data=with_data,
    )

    model.network.load_state_dict(checkpoint["model_state_dict"])

    enable_mc_dropout(model.network, enabled=dropout)

    return model


def assert_same_case_order(
    results: list[dict[str, Any]],
) -> list[str]:
    """Verify that all ensemble members contain the same cases in the same order."""
    reference_case_ids = list(results[0]["case_id"])

    for model_index, result in enumerate(
        results[1:],
        start=1,
    ):
        if list(result["case_id"]) != reference_case_ids:
            raise ValueError(
                "Case ordering differs between ensemble members. "
                f"Mismatch found for ensemble member {model_index}."
            )

    return reference_case_ids


def assert_same_array(
    reference: np.ndarray,
    candidate: np.ndarray,
    name: str,
) -> None:
    """Verify that ground-truth arrays agree across ensemble members."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)

    if reference.shape != candidate.shape:
        raise ValueError(
            f"{name} shape differs between ensemble members: "
            f"{reference.shape} != {candidate.shape}."
        )

    if not np.array_equal(
        reference,
        candidate,
        equal_nan=True,
    ):
        raise ValueError(f"{name} differs between ensemble members.")


def mean_mcdo_std(
    values: np.ndarray,
) -> np.ndarray:
    """Average MCDO variability across ensemble members without warning when MCDO was disabled for every model."""
    values = np.asarray(values, dtype=float)

    if np.isnan(values).all():
        return np.full(
            values.shape[1:],
            np.nan,
            dtype=float,
        )

    return np.nanmean(
        values,
        axis=0,
    )
