"""
Test ProteONET.

The script contains functions to Test ProteONET for classification, regression, or survival
analysis using the task-specific behavior implemented by the model class.
"""

import os
import pandas as pd
from pathlib import Path
import shutil
import torch
from typing import Any

import numpy as np
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from DeepMiL.mil.utils import (
    to_numpy,
    extend_case_ids,
    assert_same_array,
    mean_mcdo_std,
    assert_same_case_order,
    enable_mc_dropout,
)
from predict import predict_batch


def extract_test_repeat(path: str | Path):
    """Extract test and repeat identifier from model directory."""
    path = Path(path)
    test, repeat = [int(s) for s in path.stem.split("_") if s.isdigit()]
    return {"test": test, "repeat": repeat}


def assert_identity(i1, i2):
    """Asserts that all indices are in the same sequence in the res list."""
    assert list(i1) == list(
        i2
    ), "the sequence of images are different between several models"
    return i2


def copy_best_to_root(
    path: str | Path, param: list[tuple[int, int]] | tuple[int, int]
) -> dict[str, Any]:
    """Copy the best models for every fold"""
    events_dir = Path(path) / "model_best_events"
    os.makedirs(events_dir, exist_ok=True)

    best_in_root = {}
    for p in param:
        t, r = p
        src_model = Path(path) / f"test_{t}/rep_{r}/best_model.pt.tar"
        dst_model = Path(path) / f"best_model_test_{t}_repeat_{r}.pt.tar"
        best_in_root.setdefault(t, {})
        best_in_root[t][r] = str(dst_model)
        shutil.copy(src_model, dst_model)
        try:
            src_event = Path(path) / f"test_{t}/rep_{r}/runs/"
            dst_event = events_dir / f"event_test_{t}_repeat_{r}"
            shutil.copytree(src_event, dst_event)
        except:
            print("No log to copy...")
            continue

    return best_in_root


def store_all(path: str | Path, param: list[tuple[int, int]] | tuple[int, int]) -> Path:
    """Copy all best models to storage"""
    store_dir = Path(path) / "store"
    os.makedirs(store_dir, exist_ok=True)

    for p in param:
        t, r = p
        src_model = Path(path) / f"test_{t}/rep_{r}/best_model.pt.tar"
        dst_model = Path(path) / f"best_model_test_{t}_repeat_{r}.pt.tar"
        shutil.copy(src_model, dst_model)

    return store_dir


def select_best_repeat(
    df: pd.DataFrame,
    reference_metric: str = "mean_val_loss",
    metric_mode: str = "min",
    n_best: int | None = None,
):
    """
    Selects models that yeild the best validation results = single run.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with all results in it
    sgn_metric : int
        1 or -1. Help select the 'best' sample. Defautl is considering
        the best as the lowest according to the ref_metric.
    ref_metric : str
        metric on which to select

    Returns
    -------
    list
        list containing tuples of parameters (test, repeat)
        that unequivocately leads to a model.
    """
    if not n_best:
        n_best = 1

    ascending = metric_mode == "min"  # other option is 'max'
    selection = []

    for test, df_t in df.groupby("test"):
        best = df_t.sort_values(
            reference_metric,
            ascending=ascending,
        ).head(n_best)

        selection.extend((int(test), int(repeat)) for repeat in best["repeat"])

    return selection


def mean_dataframe(df):
    """
    Computes mean metrics for a set of model.
    for a given config c and a given test set t, computes
    1/r sum(metrics) over the repetitions.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of results. Columns contain config, test and repeat.

    returns:
    ----------
    df_mean_r = pd.DataFrame
        mean dataframe, w/o column repeat.
    df_mean_rt = pdf.DataFrame
        mean dataframe over repeats and then over test sets.
    """
    tests = set(df["test"])
    rows_r = []
    for t in tests:
        dft = df[df["test"] == int(t)]
        dft_mean = dft.mean(axis=0)
        dft_mean = dft_mean.drop("repeat").to_frame().transpose()
        rows_r.append(dft_mean)
    df_mean_r = pd.concat(rows_r, ignore_index=True)
    df_mean_r["test"] = df_mean_r["test"].astype(int)
    return df_mean_r


def ensemble_regression(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ensemble regression predictions using arithmetic averaging."""
    reference_target = np.asarray(results[0]["target"])

    for result in results[1:]:
        assert_same_array(
            reference=reference_target,
            candidate=result["target"],
            name="Regression targets",
        )

    predictions = np.stack(
        [np.asarray(result["prediction"]) for result in results],
        axis=0,
    )

    mcdo_std = np.stack(
        [np.asarray(result["mcdo_std"]) for result in results],
        axis=0,
    )

    return {
        "case_id": list(results[0]["case_id"]),
        "target": reference_target,
        "prediction": predictions.mean(axis=0),
        "ensemble_std": predictions.std(axis=0),
        "mcdo_std": mean_mcdo_std(mcdo_std),
        "n_models": len(results),
    }


def ensemble_classification(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ensemble classification predictions by averaging probabilities."""
    reference_target = np.asarray(results[0]["target"])

    for result in results[1:]:
        assert_same_array(
            reference=reference_target,
            candidate=result["target"],
            name="Classification targets",
        )

    probabilities = np.stack(
        [np.asarray(result["proba"]) for result in results],
        axis=0,
    )

    mcdo_std = np.stack(
        [np.asarray(result["mcdo_std"]) for result in results],
        axis=0,
    )

    ensemble_proba = probabilities.mean(axis=0)
    predicted_indices = ensemble_proba.argmax(axis=1)

    label_encoder = results[0].get("label_encoder")

    ensemble_result: dict[str, Any] = {
        "case_id": list(results[0]["case_id"]),
        "target": reference_target,
        "proba": ensemble_proba,
        "prediction": predicted_indices,
        "ensemble_std": probabilities.std(axis=0),
        "mcdo_std": mean_mcdo_std(mcdo_std),
        "n_models": len(results),
        "label_encoder": label_encoder,
    }

    if label_encoder is not None:
        ensemble_result["predicted_class"] = label_encoder.inverse_transform(
            predicted_indices
        )
    else:
        ensemble_result["predicted_class"] = predicted_indices

    return ensemble_result


def ensemble_survival(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Ensemble Cox-model risk scores using average model-wise ranks.

    Risk ranks are used because independently trained Cox models may produce
    risk scores on different numerical scales even when their ordering is
    similar. Rank aggregation therefore prevents a model with a wider output
    scale from dominating the ensemble. We can understand ranking as
    scale-independent risk score before averaging on the ensemble of models.
    We note that: ranking throws away magnitude information.
    """
    reference_time = np.asarray(results[0]["time"])
    reference_event = np.asarray(results[0]["event"])

    for result in results[1:]:
        assert_same_array(
            reference=reference_time,
            candidate=result["time"],
            name="Survival times",
        )
        assert_same_array(
            reference=reference_event,
            candidate=result["event"],
            name="Survival events",
        )

    risk_scores = np.stack(
        [np.asarray(result["risk_score"]).reshape(-1) for result in results],
        axis=0,
    )

    ranked_risks = np.stack(
        [
            rankdata(
                scores,
                method="average",
            )
            for scores in risk_scores
        ],
        axis=0,
    )

    mcdo_std = np.stack(
        [np.asarray(result["mcdo_std"]).reshape(-1) for result in results],
        axis=0,
    )

    return {
        "case_id": list(results[0]["case_id"]),
        "time": reference_time,
        "event": reference_event,
        "risk_score": ranked_risks.mean(axis=0),
        "ensemble_std": ranked_risks.std(axis=0),
        "mcdo_std": mean_mcdo_std(mcdo_std),
        "n_models": len(results),
    }


def ensemble_results(
    results: list[dict[str, Any]],
    task: str,
) -> dict[str, Any]:
    """
    Ensemble predictions from models evaluated on the same test fold.

    The aggregation rule depends on the prediction space:
    regression predictions are averaged, classification probabilities are
    averaged, and survival risks are aggregated using mean model-wise ranks.
    """
    if len(results) == 0:
        raise ValueError("At least one test result is required for ensembling.")

    assert_same_case_order(results)

    if task == "regression":
        return ensemble_regression(results)

    if task == "classification":
        return ensemble_classification(results)

    if task == "survival":
        return ensemble_survival(results)

    raise ValueError(f"Unsupported task: {task}")


def compute_test_metrics(
    model: torch.nn.Module,
    result: dict[str, Any],
) -> dict[str, float]:
    """
    Compute task-specific metrics for a single model or an ensemble result.

    Classification probabilities are converted to log-probabilities before
    being passed to compute_classification_metrics. Since that method
    applies softmax internally, softmax(log(p)) recovers p and allows the
    existing model metric implementation to be reused unchanged.
    """
    if model.task == "regression":
        return model.compute_regression_metrics(
            predictions=result["prediction"],
            targets=result["target"],
        )

    if model.task == "survival":
        return model.compute_survival_metrics(
            risk_scores=result["risk_score"],
            times=result["time"],
            events=result["event"],
        )

    if model.task == "classification":
        probabilities = np.asarray(result["proba"])

        eps = np.finfo(
            probabilities.dtype
            if np.issubdtype(probabilities.dtype, np.floating)
            else np.float64
        ).eps

        log_probabilities = np.log(
            np.clip(
                probabilities,
                eps,
                1.0,
            )
        )

        return model.compute_classification_metrics(
            logits=log_probabilities,
            targets=result["target"],
        )

    raise ValueError(f"Unsupported task: {model.task}")


@torch.inference_mode()
def test(
    model: torch.nn.Module,
    dataloader: DataLoader,
    mcdo_passes: int = 1,
) -> dict[str, Any]:
    """
    Run inference on a complete test dataloader.

    Parameters
    ----------
    model
        ProteONET model exposing ``task`` and ``predict_batch``.
    dataloader
        Test dataloader yielding dictionaries containing ``image``,
        ``case_id`` and task-specific targets.
    mcdo_passes
        Number of Monte Carlo dropout passes. A value of 1 disables MCDO.

    Returns
    -------
    dict
        Task-specific predictions, ground truth, case identifiers and
        MCDO variability.
    """
    case_ids: list[str] = []
    mcdo_variability: list[np.ndarray] = []

    if model.task == "survival":
        risk_scores: list[np.ndarray] = []
        times: list[np.ndarray] = []
        events: list[np.ndarray] = []

    elif model.task == "classification":
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []

    elif model.task == "regression":
        predictions = []
        targets = []

    else:
        raise ValueError(f"Unsupported task: {model.task}")

    with enable_mc_dropout(
        network=model.network,
        enabled=mcdo_passes > 1,
    ):
        for batch in dataloader:
            batch_result = predict_batch(
                model=model,
                batch=batch,
                mcdo_passes=mcdo_passes,
            )

            extend_case_ids(
                case_ids,
                batch_result["case_id"],
            )

            mcdo_variability.append(np.asarray(batch_result["mcdo_std"]))

            if model.task == "survival":
                risk_scores.append(np.asarray(batch_result["risk_score"]).reshape(-1))
                times.append(to_numpy(batch["time"]).reshape(-1))
                events.append(to_numpy(batch["event"]).reshape(-1))

            elif model.task == "classification":
                probabilities.append(np.asarray(batch_result["proba"]))
                predictions.append(np.asarray(batch_result["prediction"]).reshape(-1))
                targets.append(to_numpy(batch["target"]).reshape(-1))

            else:
                predictions.append(np.asarray(batch_result["prediction"]))
                targets.append(to_numpy(batch["target"]))

    result: dict[str, Any] = {
        "case_id": case_ids,
        "mcdo_std": np.concatenate(
            mcdo_variability,
            axis=0,
        ),
        "mcdo_passes": mcdo_passes,
    }

    if model.task == "survival":
        result.update(
            {
                "risk_score": np.concatenate(
                    risk_scores,
                    axis=0,
                ),
                "time": np.concatenate(
                    times,
                    axis=0,
                ),
                "event": np.concatenate(
                    events,
                    axis=0,
                ),
            }
        )

    elif model.task == "classification":
        result.update(
            {
                "proba": np.concatenate(
                    probabilities,
                    axis=0,
                ),
                "prediction": np.concatenate(
                    predictions,
                    axis=0,
                ),
                "target": np.concatenate(
                    targets,
                    axis=0,
                ),
                "label_encoder": model.label_encoder,
            }
        )

    else:
        result.update(
            {
                "prediction": np.concatenate(
                    predictions,
                    axis=0,
                ),
                "target": np.concatenate(
                    targets,
                    axis=0,
                ),
            }
        )

    return result


def results_to_dataframe(
    results: list[dict[str, Any]],
    task: str,
    target_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert test or ensemble prediction results into a case-level DataFrame.

    Parameters
    ----------
    results
        List of result dictionaries returned by ``test`` or
        ``ensemble_results``.

    task
        Prediction task. One of:
        ``"survival"``, ``"classification"``, or ``"regression"``.

    target_names
        Optional names for regression targets. If omitted, generic names
        ``target_0``, ``target_1``, ... are used.

    Returns
    -------
    pd.DataFrame
        One row per tested case.
    """
    rows = []

    for result in results:
        case_ids = result["case_id"]
        n_cases = len(case_ids)

        test_fold = result.get("test")
        repeat = result.get("repeat")
        n_models = result.get("n_models")

        if task == "survival":
            times = np.asarray(result["time"]).reshape(-1)
            events = np.asarray(result["event"]).reshape(-1)
            risk_scores = np.asarray(result["risk_score"]).reshape(-1)
            mcdo_std = np.asarray(result["mcdo_std"]).reshape(-1)

            ensemble_std = result.get("ensemble_std")
            if ensemble_std is not None:
                ensemble_std = np.asarray(ensemble_std).reshape(-1)

            for index, case_id in enumerate(case_ids):
                row = {
                    "case_id": case_id,
                    "test": test_fold,
                    "time": times[index],
                    "event": events[index],
                    "risk_score": risk_scores[index],
                    "mcdo_std": mcdo_std[index],
                }

                if repeat is not None:
                    row["repeat"] = repeat

                if ensemble_std is not None:
                    row["ensemble_std"] = ensemble_std[index]

                if n_models is not None:
                    row["n_models"] = n_models

                rows.append(row)

        elif task == "classification":
            targets = np.asarray(result["target"]).reshape(-1)
            predictions = np.asarray(result["prediction"]).reshape(-1)
            probabilities = np.asarray(result["proba"])
            mcdo_std = np.asarray(result["mcdo_std"])

            ensemble_std = result.get("ensemble_std")
            if ensemble_std is not None:
                ensemble_std = np.asarray(ensemble_std)

            predicted_classes = result.get("predicted_class")
            if predicted_classes is not None:
                predicted_classes = np.asarray(predicted_classes).reshape(-1)

            label_encoder = result.get("label_encoder")

            if label_encoder is not None:
                true_classes = label_encoder.inverse_transform(targets.astype(int))
            else:
                true_classes = targets

            for index, case_id in enumerate(case_ids):
                row = {
                    "case_id": case_id,
                    "test": test_fold,
                    "target": targets[index],
                    "target_class": true_classes[index],
                    "prediction": predictions[index],
                }

                if predicted_classes is not None:
                    row["predicted_class"] = predicted_classes[index]

                if repeat is not None:
                    row["repeat"] = repeat

                if n_models is not None:
                    row["n_models"] = n_models

                for class_index in range(probabilities.shape[1]):
                    row[f"proba_class_{class_index}"] = probabilities[
                        index, class_index
                    ]

                    row[f"mcdo_std_class_{class_index}"] = mcdo_std[index, class_index]

                    if ensemble_std is not None:
                        row[f"ensemble_std_class_{class_index}"] = ensemble_std[
                            index, class_index
                        ]

                rows.append(row)

        elif task == "regression":
            targets = np.asarray(result["target"])
            predictions = np.asarray(result["prediction"])
            mcdo_std = np.asarray(result["mcdo_std"])

            if targets.ndim == 1:
                targets = targets[:, None]

            if predictions.ndim == 1:
                predictions = predictions[:, None]

            if mcdo_std.ndim == 1:
                mcdo_std = mcdo_std[:, None]

            ensemble_std = result.get("ensemble_std")
            if ensemble_std is not None:
                ensemble_std = np.asarray(ensemble_std)

                if ensemble_std.ndim == 1:
                    ensemble_std = ensemble_std[:, None]

            n_targets = predictions.shape[1]

            if target_names is None:
                names = [f"target_{index}" for index in range(n_targets)]
            else:
                if len(target_names) != n_targets:
                    raise ValueError(
                        f"Received {len(target_names)} target names for "
                        f"{n_targets} regression outputs."
                    )

                names = target_names

            for index, case_id in enumerate(case_ids):
                row = {
                    "case_id": case_id,
                    "test": test_fold,
                }

                if repeat is not None:
                    row["repeat"] = repeat

                if n_models is not None:
                    row["n_models"] = n_models

                for target_index, target_name in enumerate(names):
                    row[target_name] = targets[index, target_index]
                    row[f"{target_name}_pred"] = predictions[index, target_index]
                    row[f"{target_name}_mcdo_std"] = mcdo_std[index, target_index]

                    if ensemble_std is not None:
                        row[f"{target_name}_ensemble_std"] = ensemble_std[
                            index, target_index
                        ]

                rows.append(row)

        else:
            raise ValueError(f"Unsupported task: {task}")

    return pd.DataFrame(rows)
