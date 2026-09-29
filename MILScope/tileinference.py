"""
Tile-level inference with IBMIL regression models trained at the slide level.

IBMIL applies its `PredictionHead` to every tile before pooling, so the
per-tile outputs are a by-product of the slide forward pass. With a model
supervised on slide-level targets (e.g. bulk RNA expression), they give one
prediction per tile and per target (e.g. spatialised gene expression).

Classes
-------
IBMILVisualizer
    `BaseTileVisualizer` restricted to IBMIL regression models. Adds the
    per-tile outputs to `_run_network`.

TileInfer
    Runs `IBMILVisualizer` on slides and stores the per-tile predictions of
    every tile, one column per target in `summarise_storage`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from MILScope.base_visualizer import BaseTileVisualizer
from MILScope.utils import as_model_list

LOGGER = logging.getLogger(__name__)


class IBMILVisualizer(BaseTileVisualizer):
    """
    `BaseTileVisualizer` restricted to IBMIL regression models.

    Adds the per-tile inference (see the module docstring).
    """

    SUPPORTED_ARCHITECTURES = {"IBMIL"}
    SUPPORTED_TASKS = {"regression"}

    def __init__(
        self,
        model: str | Path,
        wsi_dir: str | Path,
        feat_dir: str | Path | None = None,
        target_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            model=model,
            wsi_dir=wsi_dir,
            feat_dir=feat_dir,
            target_path=target_path,
            device=device,
        )

    @torch.inference_mode()
    def _run(
        self,
        case_id: str,
    ) -> dict[str, np.ndarray | None]:
        """
        Slide outputs of `_run_network` plus the per-tile outputs.

        IBMIL already applies `PredictionHead` to each tile before pooling.
        `HookerMIL` captures it in `tiles_scores` and `reprewsi`.

        Returns
        -------
        dict
            Keys of `_run_network`, plus:

            tile_scores
                Prediction head applied to each tile, shape (N, output_dim).
                Raw regression outputs, in the units of the targets.
            tile_repr
                Input of the last linear layer for each tile, shape
                (N, hidden).
        """
        outputs = self._run_network(case_id)
        outputs["tile_scores"] = self.hooker.tiles_scores[0]
        outputs["tile_repr"] = self.hooker.reprewsi[0]

        return outputs


class IBMILMember(IBMILVisualizer):
    """Ensemble member: only provides `_run`."""

    def forward(
        self,
        case_id: str,
    ) -> dict[str, np.ndarray | None]:
        """Outputs of `IBMILVisualizer._run`."""
        return self._run(case_id)


def _check_same_target(
    members: Sequence[IBMILVisualizer],
) -> None:
    """Raise a `ValueError` if the members do not share the same target_names."""
    reference = members[0].target_name

    for member in members[1:]:
        if member.target_name != reference:
            raise ValueError(
                f"Target of `{member.model_path}` "
                f"({member.target_name}) differ from "
                f"`{members[0].model_path}` ({reference})."
            )


class TileInfer(IBMILVisualizer):
    """
    Per-tile predictions of an IBMIL regression model.

    Every tile of every processed slide is kept.

    Parameters
    ----------
    model, wsi_dir, feat_dir, target_path, device
        See `BaseTileVisualizer`.

    store : bool, default=False
        If True, `forward` updates the storage.

    dropout : bool, default=False
        Apply MC dropout to the tile inference. Not implemented yet.

    Notes
    -----
    `store_raw` and `store_feat` keep the `feature_dim` and `instance_dim`
    embeddings of every tile. On a large cohort this can take several GB of
    memory.
    """

    def __init__(
        self,
        model: str | Path,
        wsi_dir: str | Path,
        feat_dir: str | Path | None = None,
        target_path: str | Path | None = None,
        device: str = "cpu",
        dropout: bool = False,  # Apply MCD to the tile inference
        store: bool = False,
    ):
        super().__init__(
            model=model,
            wsi_dir=wsi_dir,
            feat_dir=feat_dir,
            target_path=target_path,
            device=device,
        )

        self.store = store
        self.dropout = dropout

        if self.dropout:
            LOGGER.warning("MC dropout is not implemented yet, `dropout` is ignored.")

        # lists that have to be filled
        self._reset_storage()

    @torch.inference_mode()
    def forward(
        self,
        case_id: str,
    ) -> dict[str, np.ndarray | None]:
        """
        Run the model on one slide and, if `store`, store its tiles.

        Parameters
        ----------
        case_id : str
            Case identifier, name of the `<case_id>.h5` file.

        Returns
        -------
        dict
            Outputs of `IBMILVisualizer._run`, plus `selection` (indices of
            the tiles kept, here all of them).
        """
        attrs, coords = self._get_info(case_id)
        outputs = self._run(case_id)

        return self._process(
            case_id,
            outputs,
            attrs,
            coords,
        )

    def _process(
        self,
        case_id: str,
        outputs: dict[str, np.ndarray | None],
        attrs: dict[str, Any],
        coords: np.ndarray,
    ) -> dict[str, np.ndarray | None]:
        """Select the tiles and update the storage."""
        outputs["selection"] = self._select_tiles(
            None,
            len(coords),
            None,
        )

        if self.store:
            self._store_inference(
                case_id,
                outputs,
                attrs,
                coords,
            )

        return outputs

    def forward_all(self) -> TileInfer:
        """Run `forward` on every case of the target table."""
        for case_id in self.case_ids:
            self.forward(case_id)

        return self

    def _store_inference(
        self,
        case_id: str,
        outputs: dict[str, np.ndarray | None],
        attrs: dict[str, Any],
        coords: np.ndarray,
    ) -> None:
        """
        Add the all tiles of one slide to the storage.

        Parameters
        ----------
        case_id : str
            Case identifier.

        outputs : dict
            Output of `forward` (needs `selection`).

        attrs : dict
            Attributes of the h5 `coords` dataset.

        coords : numpy.ndarray
            Raw tile coordinates (x, y, w, h), shape (N, 4).
        """
        selection = outputs["selection"]
        attention = outputs["attention"]

        self.store_info.append(self._tile_info(case_id, attrs))
        self.store_coords.append(coords[selection, :4])
        self.store_raw.append(outputs["tiles_encoding"][selection])
        self.store_feat.append(outputs["tiles_transf"][selection])
        self.store_attention.append(
            attention[selection]
            if attention is not None
            else np.full(len(selection), np.nan)
        )
        self.store_preclassif.append(outputs["tile_repr"][selection])
        self.store_score.append(outputs["tile_scores"][selection])
        self.store_slide_score.append(outputs["scores"])

    def summarise_storage(self) -> pd.DataFrame:
        """
        One row per stored tile.

        Returns
        -------
        pandas.DataFrame
            Tile metadata, box (x, y, w, h), image file name, attention,
            one column per target with the tile prediction, and one
            `slide <target>` column per target with the slide prediction.

        Raises
        ------
        ValueError
            If the number of target names does not match the model outputs.
        """
        if not self.store_info:
            return pd.DataFrame()

        target_names = list(self.target_name)
        output_dim = self.store_score[0].shape[-1]

        if len(target_names) != output_dim:
            raise ValueError(
                f"`target_name` has {len(target_names)} names but the model has "
                f"{output_dim} outputs."
            )

        n_tiles = [len(slide_coords) for slide_coords in self.store_coords]
        slide_index = np.repeat(
            np.arange(len(n_tiles)),
            n_tiles,
        )

        summary = pd.concat(
            [
                pd.DataFrame(self.store_info)
                .iloc[slide_index]
                .reset_index(
                    drop=True,
                ),
                pd.DataFrame(
                    np.concatenate(self.store_coords, axis=0),
                    columns=["x", "y", "w", "h"],
                ),
            ],
            axis=1,
        )
        summary.insert(
            loc=0,
            column="img_path",
            value=(
                summary["name"].astype(str)
                + "_"
                + summary["x"].astype(str)
                + "_"
                + summary["y"].astype(str)
                + "_"
                + summary["w"].astype(str)
                + "_"
                + summary["h"].astype(str)
                + ".jpeg"
            ),
        )
        summary["attention score"] = np.concatenate(self.store_attention)

        tile_scores = np.concatenate(self.store_score, axis=0)
        slide_scores = np.stack(self.store_slide_score, axis=0)[slide_index]

        for target_index, name in enumerate(target_names):
            summary[name] = tile_scores[:, target_index]

        for target_index, name in enumerate(target_names):
            summary[f"slide {name}"] = slide_scores[:, target_index]

        return summary

    def _storages(self) -> list[list]:
        """Per-slide storages, kept aligned by `_store_inference`."""
        return [
            self.store_info,
            self.store_coords,
            self.store_raw,
            self.store_feat,
            self.store_attention,
            self.store_preclassif,
            self.store_score,
            self.store_slide_score,
        ]

    def _reset_storage(self) -> None:
        """
        Empty the storage.

        Every `store_*` attribute is a list with one item per stored slide,
        unlike `TileSeeker` (class index -> one item per tile), since
        regression has no classes:

        - `store_info`: dict of slide metadata (`_tile_info`).
        - `store_coords`, `store_raw`, `store_feat`, `store_attention`,
          `store_preclassif`, `store_score`: arrays with one row per tile of
          the slide. `store_raw` holds the encoder embeddings (input of
          `InstanceTransform`), `store_feat` its output.
        - `store_slide_score`: slide prediction, shape (output_dim,).

        `store_image` is kept for symmetry with `TileSeeker` and not filled.
        """
        self.store_info = []
        self.store_coords = []
        self.store_image = []
        self.store_raw = []
        self.store_feat = []
        self.store_attention = []
        self.store_preclassif = []
        self.store_score = []
        self.store_slide_score = []


class ConsensusTileInfer(TileInfer):
    """
    `TileInfer` on the mean outputs of an ensemble of models.

    For each slide, only the models whose `test_fold` has the slide are
    used (all models are used if none matches, see
    `BaseTileVisualizer._select_ensemble_members`).

    Parameters
    ----------
    model : str or pathlib.Path or sequence of them
        Checkpoints of the ensemble. The first one also provides the target
        table and the storage.

    Other parameters
        See `TileInfer`.

    Raises
    ------
    ValueError
        If the models do not share the same target.
    """

    def __init__(
        self,
        model: str | Path | Sequence[str | Path],
        wsi_dir: str | Path,
        feat_dir: str | Path | None = None,
        target_path: str | Path | None = None,
        device: str = "cpu",
        dropout: bool = False,  # Apply MCD to the tile inference,
        store: bool = False,
    ) -> None:
        models = as_model_list(model)

        super().__init__(
            models[0],
            wsi_dir,
            feat_dir,
            target_path,
            device,
            dropout,
            store,
        )

        self.seekers = [
            self,
            *(
                IBMILMember(
                    member,
                    wsi_dir,
                    feat_dir,
                    target_path,
                    device,
                )
                for member in models[1:]
            ),
        ]
        _check_same_target(self.seekers)

    @torch.inference_mode()
    def forward(
        self,
        case_id: str,
    ) -> dict[str, np.ndarray | None]:
        """Same as `TileSeeker.forward`, on the mean ensemble outputs."""
        attrs, coords = self._get_info(case_id)
        members = self._select_ensemble_members(
            case_id,
            self.seekers,
        )
        outputs = self._average_outputs([member._run(case_id) for member in members])

        return self._process(
            case_id,
            outputs,
            attrs,
            coords,
        )
