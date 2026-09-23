from __future__ import annotations

from typing import Any, Generator
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch


from slide.utils import read_h5_features, read_h5_coords
from MIL.model import DeepMIL

import numpy as np

import matplotlib.pyplot as plt

from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.metrics import (
    RocCurveDisplay,
    auc,
)


def to_numpy(value: Any) -> np.ndarray:
    """Convert a tensor or array-like object to a NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)

def nan_like(
    array: np.ndarray,
) -> np.ndarray:
    """Create a floating-point NaN array with the same shape as the input."""
    return np.full(
        np.asarray(array).shape,
        np.nan,
        dtype=float,
    )

def extend_case_ids(
    case_ids: list[str],
    batch_case_ids: Any,
) -> None:
    """Append case identifiers returned by a collated batch."""
    if isinstance(batch_case_ids, str):
        case_ids.append(batch_case_ids)
        return

    case_ids.extend(str(case_id) for case_id in batch_case_ids)

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

    coords = np.ascontiguousarray(coords_array, dtype=np.float32)
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
        label_encoder=checkpoint.get("label_encoder"),
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


def survival_curve(
    df,
    metric,
    event,
    stratif=None,
    variability=False,
    time_pt=None,
    time_unit="days",
    ax=None,
    show=True,
):
    summary = {}
    if not ax:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, fig=fig)

    def plot_km_curve(time, survival_prob, conf_int, label):
        ax.step(time, survival_prob, label=label, linewidth=1.5, where="post")
        if variability:
            ax.fill_between(time, conf_int[0], conf_int[1], alpha=0.2, step="post")
        ax.set_ylim(bottom=0, top=1)

    tmp = df.dropna(axis=0, how="any", subset=[metric, event], inplace=False)
    if stratif is not None:  # Stratified survival curves
        # if stratif is list
        # new_var = results_table.apply(make_new_var(on_vars), axis=1)
        # stratif = "_".join(on_vars)
        # df[stratif] = new_var
        tmp = tmp.dropna(axis=0, subset=stratif, inplace=False)
        for group in tmp[stratif].unique():
            summary[group] = {}
            mask = tmp.loc[tmp[stratif] == group]
            time, survival_prob, conf_int = kaplan_meier_estimator(
                mask[event].astype("bool"),
                mask[metric],
                conf_level=0.95,
                conf_type="log-log",
            )
            summary[group]["unit"] = time_unit
            summary[group]["time"] = time
            summary[group]["proba"] = survival_prob
            summary[group]["confidence"] = conf_int
            # Median PFS
            cross = survival_prob <= 0.5
            summary[group]["median"] = time[cross][0] if cross.any() else np.nan
            # plot
            mask = [survival_prob[i] != 0 for i in range(len(survival_prob))]
            plot_km_curve(
                time[mask], survival_prob[mask], conf_int[:, mask], f"{group}"
            )

        ax.set_title(f"Kaplan Meier Estimates for {metric} with respect to {stratif}")

    else:  # Single survival curve
        summary["unit"] = time_unit
        time, survival_prob, conf_int = kaplan_meier_estimator(
            tmp[event].astype("bool"), tmp[metric], conf_level=0.95, conf_type="log-log"
        )
        summary["time"] = time
        summary["proba"] = survival_prob
        summary["confidence"] = conf_int
        # Median PFS
        cross = survival_prob <= 0.5
        summary["median"] = time[cross][0] if cross.any() else np.nan
        plot_km_curve(time, survival_prob, conf_int, metric)
        ax.set_title(f"{metric}")
    if time_pt:
        ax.vlines(
            x=time_pt,
            colors="red",
            linestyles="dashed",
            ymin=0,
            ymax=1,
            label=f"{time_pt} {time_unit}",
        )
    ax.set_xlabel(f"{time_unit}")
    ax.set_ylabel("Fraction of patients")
    ax.legend()

    if show:
        plt.show()

    return tmp, summary


def km_at_time(time, proba, conf, thr):
    """
    Return the Kaplan-Meier estimate and 95% CI at time t.

    Parameters
    ----------
    time : ndarray
        Event times.
    proba : ndarray
        KM survival probabilities.
    conf : ndarray
        Shape (2, n_times): lower and upper confidence limits.
    thr : float
        Time of interest.

    Returns
    -------
    metric : float
    ci_low : float
    ci_high : float
    """
    idx = np.searchsorted(time, thr, side="right") - 1

    if idx < 0:
        # before the first event
        return 1.0, 1.0, 1.0

    return (
        proba[idx],
        conf[0, idx],
        conf[1, idx],
    )


def roc_curve(
    y_true: list[np.ndarray],
    score_pos: list[np.ndarray],
    ax: plt.axis,
    show: bool = True,
):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # check y_true and score_pos have the same lenght = nb of test sets
    tests = len(y_true)
    for t in range(tests):
        viz = RocCurveDisplay.from_predictions(
            y_true[t],
            score_pos[t],
            pos_label=1,
            name=f"ROC fold {t}",
            alpha=0.5,
            lw=1,
            ax=ax,
            plot_chance_level=(t == tests - 1),
        )

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    # Mean over test
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    # Error over test
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability",
    )
    ax.legend(loc="lower right")

    if show:
        plt.show()
