"""
Tile-level interpretation of ABMIL classification models.

Tools
---------
TileSeeker
    Best tiles per class for one model.

ConsensusTileSeeker
    Best tiles per class for an ensemble (e.g. the cross-validation folds).

HeatmapMaker
    Attention and per-class heatmaps on the WSI thumbnail, for one model or
    an ensemble.

Per-tile class scores
---------------------
The ABMIL slide head is applied to each transformed tile. This needs the
pooled slide embedding to have the size of a tile embedding: one attention
head, or `mean` / `max` pooling. Otherwise (several attention heads), the
visualizers are *attention-only*:

- tiles are ranked for class i on `attention * slide_proba[i]`;
- heatmaps show the attention of every head instead of per-class maps.

IBMIL is not supported here (see `tileinference`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

from MILScope.base_visualizer import BaseTileVisualizer
from MILScope.utils import as_model_list


LOGGER = logging.getLogger(__name__)

HEATMAP_MODES = {
    "proba",
    "logits",
}



class ABMILVisualizer(BaseTileVisualizer):
    """
    `BaseTileVisualizer` restricted to ABMIL classification models.

    Adds the per-tile class scores (see the module docstring).
    """

    SUPPORTED_ARCHITECTURES = {"ABMIL"}
    SUPPORTED_TASKS = {"classification"}

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

        self.tile_head_available = (
            self.network._get_pooled_dim() == self.network.instance_dim
        )

        if not self.tile_head_available:
            LOGGER.info(
                "%s: %s attention heads, the slide head cannot score single "
                "tiles. Using attention-only mode.",
                self.model_path.name,
                self.num_heads,
            )

    @torch.inference_mode()
    def _run(
        self,
        case_id: str,
    ) -> dict[str, np.ndarray | None]:
        """
        Slide outputs of `_run_network` plus the per-tile outputs.

        Returns
        -------
        dict
            Keys of `_run_network`, plus (None in attention-only mode):

            tile_scores, tile_proba
                Slide head applied to each tile, shape (N, n_classes).
            tile_repr
                Input of the last linear layer for each tile, shape
                (N, hidden).
        """
        outputs = self._run_network(case_id)
        outputs["tile_scores"] = None
        outputs["tile_proba"] = None
        outputs["tile_repr"] = None

        if self.tile_head_available:
            tiles_transf = torch.from_numpy(outputs["tiles_transf"]).to(self.device)
            self.network.prediction_layer(tiles_transf.unsqueeze(0))
            outputs["tile_scores"] = self.hooker.scores[0]
            outputs["tile_proba"] = self.hooker.proba[0]
            outputs["tile_repr"] = self.hooker.reprewsi[0]

        return outputs

    @staticmethod
    def _ranking_scores(
        outputs: dict[str, np.ndarray | None],
    ) -> np.ndarray:
        """
        Score used to rank the tiles for each class, shape (N, n_classes).

        `tile_proba`, or `attention * proba` in attention-only mode.
        """
        if outputs["tile_proba"] is not None:
            return outputs["tile_proba"]

        return outputs["attention"][:, None] * outputs["proba"][None, :]


# ----------------------------------------------------------------------------
# Ensemble helpers
# ----------------------------------------------------------------------------


class ABMILMember(ABMILVisualizer):
    """Ensemble member: only provides `_run`."""

    def forward(
        self,
        case_id: str,
    ) -> dict[str, np.ndarray | None]:
        """Outputs of `ABMILVisualizer._run`."""
        return self._run(case_id)


def _check_same_classes(
    members: Sequence[ABMILVisualizer],
) -> None:
    """Raise a `ValueError` if the members do not share the same classes."""
    reference = members[0].classes

    for member in members[1:]:
        if member.classes != reference:
            raise ValueError(
                f"Classes of `{member.model_path}` "
                f"({member.classes}) differ from "
                f"`{members[0].model_path}` ({reference})."
            )


# ----------------------------------------------------------------------------
# Best tiles
# ----------------------------------------------------------------------------


class TileSeeker(ABMILVisualizer):
    """
    Decision-based extraction of the most predictive tiles of each class.

    For each slide, tiles are first filtered on attention (`att_thres`),
    then ranked per class (see `ABMILVisualizer._ranking_scores`). A slide
    contributes at most `max_per_slides` tiles per class, and the `n_best`
    best tiles over all the slides processed are kept.

    Parameters
    ----------
    model, wsi_dir, feat_dir, target_path, device
        See `BaseTileVisualizer`.

    n_best : int, default=1000
        Number of tiles kept per class.

    min_prob : bool, default=False
        If True, keep the tiles with the *lowest* ranking score.

    max_per_slides : int, default=300
        Maximum number of tiles a slide contributes per class.

    att_thres : {"otsu"} or int or None, default="otsu"
        Attention filter applied before ranking (see
        `BaseTileVisualizer._select_tiles`).

    store : bool, default=False
        If True, `forward` updates the storage.
    """

    def __init__(
        self,
        model: str | Path,
        wsi_dir: str | Path,
        feat_dir: str | Path | None = None,
        target_path: str | Path | None = None,
        device: str = "cpu",
        n_best: int = 1000,
        min_prob: bool = False,
        max_per_slides: int = 300,
        att_thres: str | int | None = "otsu",
        store: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            wsi_dir=wsi_dir,
            feat_dir=feat_dir,
            target_path=target_path,
            device=device,
        )

        # Fail early on an invalid threshold.
        self._select_tiles(
            np.zeros(1),
            1,
            att_thres,
        )

        self.n_best = n_best
        self.min_prob = min_prob
        self.max_per_slides = max_per_slides
        self.att_thres = att_thres
        self.store = store

        self._reset_storage()

    @torch.inference_mode()
    def forward(
        self,
        case_id: str,
    ) -> dict[str, np.ndarray | None]:
        """
        Run the model on one slide and, if `store`, update the best tiles.

        Parameters
        ----------
        case_id : str
            Case identifier, name of the `<case_id>.h5` file.

        Returns
        -------
        dict
            Outputs of `ABMILVisualizer._run`, plus `selection` (indices of
            the tiles kept by the attention filter).
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
            outputs["attention"],
            len(coords),
            self.att_thres,
        )

        if self.store:
            self.store_best(
                case_id,
                outputs,
                attrs,
                coords,
            )

        return outputs

    def forward_all(self) -> TileSeeker:
        """Run `forward` on every case of the target table."""
        for case_id in self.case_ids:
            self.forward(case_id)

        return self

    def store_best(
        self,
        case_id: str,
        outputs: dict[str, np.ndarray | None],
        attrs: dict[str, Any],
        coords: np.ndarray,
    ) -> None:
        """
        Add the best tiles of one slide to the storage, then keep the
        `n_best` best tiles per class.

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
        info = self._tile_info(case_id, attrs)
        ranking = self._ranking_scores(outputs)
        sign = -1 if self.min_prob else 1
        selection = outputs["selection"]
        attention = outputs["attention"]

        for class_index, _ in enumerate(self.classes):
            keys = sign * ranking[selection, class_index]
            order = np.argsort(keys)[::-1][: self.max_per_slides]

            for tile in selection[order]:
                self.store_info[class_index].append(dict(info))
                self.store_coords[class_index].append(coords[tile, :])
                self.store_raw[class_index].append(outputs["tiles_encoding"][tile])
                self.store_feat[class_index].append(outputs["tiles_transf"][tile])
                self.store_attention[class_index].append(
                    attention[tile] if attention is not None else np.nan
                )
                self.store_preclassif[class_index].append(
                    outputs["tile_repr"][tile]
                    if outputs["tile_repr"] is not None
                    else None
                )
                self.store_score[class_index].append(
                    outputs["tile_scores"][tile, class_index]
                    if outputs["tile_scores"] is not None
                    else np.nan
                )
                self.store_proba[class_index].append(
                    outputs["tile_proba"][tile, class_index]
                    if outputs["tile_proba"] is not None
                    else np.nan
                )
                self.store_rank[class_index].append(ranking[tile, class_index])

            # Keep the n_best tiles over all slides, best first.
            keep = np.argsort(sign * np.asarray(self.store_rank[class_index]))[::-1]
            keep = keep[: self.n_best]

            for storage in self._storages():
                storage[class_index] = [storage[class_index][k] for k in keep]

    def summarise_storage(self) -> pd.DataFrame:
        """
        One row per stored tile.

        Returns
        -------
        pandas.DataFrame
            Tile metadata, box (x, y, w, h), image file name, attention,
            class label, and class score / proba / rank score.
        """
        summaries = []

        for class_index, label in enumerate(self.classes):
            if not self.store_info[class_index]:
                continue

            summary = pd.concat(
                [
                    pd.DataFrame(self.store_info[class_index]),
                    pd.DataFrame(
                        np.asarray(self.store_coords[class_index])[:, :4],
                        columns=["x", "y", "w", "h"],
                    ),
                ],
                axis=1,
            )
            summary.insert(
                loc=0,
                column="img_path",
                value=[
                    f"{row['name']}_{row['x']}_{row['y']}_{row['w']}_{row['h']}.jpeg"
                    for _, row in summary.iterrows()
                ],
            )
            summary["attention score"] = self.store_attention[class_index]
            summary["class label"] = label
            summary["class score"] = self.store_score[class_index]
            summary["class proba"] = self.store_proba[class_index]
            summary["rank score"] = self.store_rank[class_index]
            summaries.append(summary)

        if not summaries:
            return pd.DataFrame()

        return pd.concat(
            summaries,
            axis=0,
            ignore_index=True,
        )

    def extract_images(self) -> dict[int, list]:
        """
        Read the image of every stored tile from the WSIs.

        Returns
        -------
        dict
            Class index -> list of tile images, in storage order.
        """
        slides = {}

        for class_index, _ in enumerate(self.classes):
            self.store_image[class_index] = []

            if not self.store_info[class_index]:
                LOGGER.warning("No tile stored for class %s.", class_index)
                continue

            for info, (x, y, w, h) in zip(
                self.store_info[class_index],
                np.asarray(self.store_coords[class_index])[:, :4],
            ):
                name = info["name"]

                if name not in slides:
                    slides[name] = self._get_slide(name)

                self.store_image[class_index].append(
                    slides[name].read_region(
                        location=(int(x), int(y)),
                        level=info["level"],
                        size=(int(w), int(h)),
                    )
                )

        return self.store_image

    def _storages(self) -> list[dict[int, list]]:
        """Per-tile storages, kept aligned by `store_best`."""
        return [
            self.store_info,
            self.store_coords,
            self.store_raw,
            self.store_feat,
            self.store_attention,
            self.store_preclassif,
            self.store_score,
            self.store_proba,
            self.store_rank,
        ]

    def _reset_storage(self) -> None:
        """
        Empty the storage.

        Every `store_*` attribute is a dict: class index -> list, one item
        per stored tile. `store_image` is only filled by `extract_images`.
        """
        self.store_info = {}
        self.store_coords = {}
        self.store_image = {}
        self.store_raw = {}
        self.store_feat = {}
        self.store_attention = {}
        self.store_preclassif = {}
        self.store_score = {}
        self.store_proba = {}
        self.store_rank = {}

        for class_index, _ in enumerate(self.classes):
            for storage in [*self._storages(), self.store_image]:
                storage[class_index] = []


class ConsensusTileSeeker(TileSeeker):
    """
    `TileSeeker` on the mean outputs of an ensemble of ABMIL models.

    For each slide, only the models whose `test_fold` is the slide fold are
    used (all models if none matches, see
    `BaseTileVisualizer._select_ensemble_members`).

    Parameters
    ----------
    model : str or pathlib.Path or sequence of them
        Checkpoints of the ensemble. The first one also provides the target
        table and the storage.

    Other parameters
        See `TileSeeker`.

    Raises
    ------
    ValueError
        If the models do not share the same classes.
    """

    def __init__(
        self,
        model: str | Path | Sequence[str | Path],
        wsi_dir: str | Path,
        feat_dir: str | Path | None = None,
        target_path: str | Path | None = None,
        device: str = "cpu",
        n_best: int = 1000,
        min_prob: bool = False,
        max_per_slides: int = 300,
        att_thres: str | int | None = "otsu",
        store: bool = False,
    ) -> None:
        models = as_model_list(model)

        super().__init__(
            models[0],
            wsi_dir,
            feat_dir,
            target_path,
            device,
            n_best,
            min_prob,
            max_per_slides,
            att_thres,
            store,
        )

        self.seekers = [
            self,
            *(
                ABMILMember(
                    member,
                    wsi_dir,
                    feat_dir,
                    target_path,
                    device,
                )
                for member in models[1:]
            ),
        ]
        _check_same_classes(self.seekers)

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


# ----------------------------------------------------------------------------
# Heatmaps
# ----------------------------------------------------------------------------


class HeatmapMaker(ABMILVisualizer):
    """
    Attention and per-class heatmaps of one ABMIL model or an ensemble.

    Parameters
    ----------
    model : str or pathlib.Path or sequence of them
        Checkpoint(s). With several, the outputs are averaged over the
        models whose `test_fold` is the slide fold.

    Other parameters
        See `BaseTileVisualizer`.
    """

    def __init__(
        self,
        model: str | Path | Sequence[str | Path],
        wsi_dir: str | Path,
        feat_dir: str | Path | None = None,
        target_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        models = as_model_list(model)

        super().__init__(
            models[0],
            wsi_dir,
            feat_dir,
            target_path,
            device,
        )

        self.seekers = [
            self,
            *(
                ABMILMember(
                    member,
                    wsi_dir,
                    feat_dir,
                    target_path,
                    device,
                )
                for member in models[1:]
            ),
        ]
        _check_same_classes(self.seekers)

    @torch.inference_mode()
    def forward(
        self,
        case_id: str,
    ) -> tuple[dict[str, np.ndarray | None], dict[str, Any], np.ndarray]:
        """
        Mean outputs of the ensemble members for one slide.

        Returns
        -------
        dict
            Averaged outputs of `ABMILVisualizer._run`.

        dict
            Attributes of the h5 `coords` dataset.

        numpy.ndarray
            Raw tile coordinates (x, y, w, h), shape (N, 4).
        """
        attrs, coords = self._get_info(case_id)
        members = self._select_ensemble_members(
            case_id,
            self.seekers,
        )
        outputs = self._average_outputs([member._run(case_id) for member in members])

        return outputs, attrs, coords

    def make_montage(
        self,
        case_id: str,
        by: str = "proba",
        weighted: bool = False,
        downsample: int = 1,
        smooth: float | None = 0,
        alpha: float = 0.5,
        save: str | Path | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Heatmaps of one slide blended on its thumbnail.

        Parameters
        ----------
        case_id : str
            Case identifier.

        by : {"proba", "logits"}, default="proba"
            Per-tile class score painted on the class maps, weighted by the
            attention when the pooling has attention.

        downsample : int, default=1
            Thumbnail downsampling from the tile extraction level.

        smooth : float, optional
            Gaussian sigma (thumbnail pixels). None or 0 disables it.

        alpha : float, default=0.5
            Heatmap opacity.

        save : str or pathlib.Path, optional
            If given, save `<save>/<key>/<case_id>_<key>_heatmap.png`.

        Returns
        -------
        dict
            `class:<label>` maps (not in attention-only mode), `attn` (mean
            attention) and, with several heads, `attn_head:<h>`.

        Raises
        ------
        ValueError
            If `by` is not one of `HEATMAP_MODES`.
        """
        if by not in HEATMAP_MODES:
            raise ValueError(
                f"`by` must be one of {sorted(HEATMAP_MODES)}, received {by!r}."
            )

        outputs, attrs, coords = self.forward(case_id)
        thumbnail, thumbnail_coords = self._thumbnail_and_coords(
            self._get_slide(case_id),
            attrs,
            coords,
            downsample=downsample,
        )
        size = (thumbnail.shape[1], thumbnail.shape[0])

        maps = {}
        attention = outputs["attention"]
        tile_values = outputs["tile_proba"] if by == "proba" else outputs["tile_scores"]

        if tile_values is not None:
            weights = attention if attention is not None else 1.0

            for class_index, label in enumerate(self.classes):
                maps[f"class:{label}"] = (
                    weights * tile_values[:, class_index]
                    if weighted
                    else tile_values[:, class_index]
                )
        else:
            LOGGER.info(
                "%s: attention-only mode, no per-class heatmaps.",
                case_id,
            )

        if attention is not None:
            maps["attn"] = attention

            if outputs["attention_heads"].shape[-1] > 1:
                for head in range(outputs["attention_heads"].shape[-1]):
                    maps[f"attn_head:{head}"] = outputs["attention_heads"][:, head]

        overlays = {}

        for key, scores in maps.items():
            heatmap, background = self.fill_heatmap(
                size=size,
                xywh=thumbnail_coords,
                scores=scores,
            )
            overlays[key] = self.overlay_heatmap_on_thumbnail(
                thumbnail,
                heatmap,
                background,
                smooth=smooth,
                alpha=alpha,
            )

        if save is not None:
            for key, overlay in overlays.items():
                save_dir = Path(save) / key
                save_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                Image.fromarray(overlay.astype("uint8"), "RGB").save(
                    save_dir / f"{case_id}_{key}_heatmap.png"
                )

        return overlays

    def make_all(
        self,
        dst: str | Path,
        downsample: int = 16,
        **kwargs,
    ) -> HeatmapMaker:
        """
        Save the heatmaps of every case of the target table.

        Parameters
        ----------
        dst : str or pathlib.Path
            Output directory (see `make_montage`, `save`).

        downsample : int, default=16
            Thumbnail downsampling.

        **kwargs
            Passed to `make_montage` (`by`, `smooth`, `alpha`).
        """
        for case_id in self.case_ids:
            self.make_montage(
                case_id,
                downsample=downsample,
                save=dst,
                **kwargs,
            )

        return self
