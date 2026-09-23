from glob import glob
import torch
import os

from pathlib import Path

import numpy as np
from typing import Any
from MIL.utils import load_model, prep_wsi, nan_like


@torch.inference_mode()
def predict(
    model_path: str | Path,
    enc_path: str | Path | None = None,
    enc_dir: str | Path | None = None,
    wsi_list: list[str | Path] | None = None,
    device: str = "cpu",
    ext="h5",
):
    """
    Load a trained model and make a prediction on a single wsi if 'enc_path' is provided
    or on a batch of wsi if both 'enc_dir' or 'wsi_list' are provided
    
    """
    model = load_model(model_path, device)

    if enc_path:
        enc_paths = [enc_path]
    if not wsi_list:
        enc_paths = glob(os.path.join(enc_dir, f"*.{ext}"))
    else:
        enc_paths = wsi_list

    results = []

    for path in enc_paths:
        sample = prep_wsi(path, model.args)
        result = model.predict_batch(sample)
        results.append(result)  # is it the best way to accumulate result dict?

    return results


@torch.inference_mode()
def predict_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    mcdo_passes: int = 1,
) -> dict[str, Any]:
    """
    Generate task-specific predictions for one batch with optional MCDO.

    MCDO aggregation is performed in the natural prediction space:
    regression predictions, 
    classification probabilities, 
    or Cox risk scores.
    """
    if mcdo_passes < 1:
        raise ValueError(f"mcdo_passes must be >= 1, received {mcdo_passes}.")

    if mcdo_passes == 1:
        prediction = model.predict_batch(batch)

        if model.task == "survival":
            risk_score = np.asarray(prediction["risk_score"]).reshape(-1)

            return {
                "case_id": prediction["case_id"],
                "risk_score": risk_score,
                "mcdo_std": nan_like(risk_score),
            }

        if model.task == "classification":
            probabilities = np.asarray(prediction["proba"])
            predicted_indices = probabilities.argmax(axis=1)

            result = {
                "case_id": prediction["case_id"],
                "proba": probabilities,
                "prediction": predicted_indices,
                "mcdo_std": nan_like(probabilities),
            }

            if model.label_encoder is not None:
                result["predicted_class"] = model.label_encoder.inverse_transform(
                    predicted_indices
                )
            else:
                result["predicted_class"] = predicted_indices

            return result

        if model.task == "regression":
            predictions = np.asarray(prediction["prediction"])

            return {
                "case_id": prediction["case_id"],
                "prediction": predictions,
                "mcdo_std": nan_like(predictions),
            }

        raise ValueError(f"Unsupported task: {model.task}")

    if model.task == "survival":
        risk_scores = np.stack(
            [
                np.asarray(model.predict_batch(batch)["risk_score"]).reshape(-1)
                for _ in range(mcdo_passes)
            ],
            axis=0,
        )

        return {
            "case_id": batch["case_id"],
            "risk_score": risk_scores.mean(axis=0),
            "mcdo_std": risk_scores.std(axis=0),
        }

    if model.task == "classification":
        probabilities = np.stack(
            [
                np.asarray(model.predict_batch(batch)["proba"])
                for _ in range(mcdo_passes)
            ],
            axis=0,
        )

        mean_probabilities = probabilities.mean(axis=0)
        predicted_indices = mean_probabilities.argmax(axis=1)

        result = {
            "case_id": batch["case_id"],
            "proba": mean_probabilities,
            "prediction": predicted_indices,
            "mcdo_std": probabilities.std(axis=0),
        }

        if model.label_encoder is not None:
            result["predicted_class"] = model.label_encoder.inverse_transform(
                predicted_indices
            )
        else:
            result["predicted_class"] = predicted_indices

        return result

    if model.task == "regression":
        predictions = np.stack(
            [
                np.asarray(model.predict_batch(batch)["prediction"])
                for _ in range(mcdo_passes)
            ],
            axis=0,
        )

        return {
            "case_id": batch["case_id"],
            "prediction": predictions.mean(axis=0),
            "mcdo_std": predictions.std(axis=0),
        }

    raise ValueError(f"Unsupported task: {model.task}")

