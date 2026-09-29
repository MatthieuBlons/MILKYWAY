"""
Shared base class for tile-level analysis of a trained DeepMIL model.

`BaseTileVisualizer` loads the model, the target table use during training, 
the encoded slides and the corresponding WSIs, runs one full forward pass with `HookerMIL` attached, and
provides the tile selection, ensemble and heatmap helpers. 

Subclasses set:

SUPPORTED_ARCHITECTURES
    Network class names they accept (`ABMIL`, `IBMIL`).

SUPPORTED_TASKS
    Tasks they accept (see `SUPPORTED_TASKS` in `dataloader.py`).

Use
-----
tilescope
    ABMIL: best tiles and heatmaps.

tileinference
    IBMIL tile-level prediction.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from skimage import filters

from MIL.utils import load_model, prep_wsi
from MIL.io import read_table
from MILScope.hooks import HookerMIL
from MILScope.utils import blend_images
from osfile.manager import findFile
from slide.tile import SlidePatcher
from slide.utils import get_slide_reader, get_x_y_to, read_h5_coords

LOGGER = logging.getLogger(__name__)

# Keys of the h5 `coords` attributes kept in the per-tile metadata.
TILE_INFO_KEYS = (
    "name",
    "level",
    "target_magnification",
)


class BaseTileVisualizer(ABC):
    """
    Load a trained MIL model and run it on encoded slides.

    Parameters
    ----------
    model : str or pathlib.Path
        Path to a `*.pt.tar` checkpoint.

    wsi_dir : str or pathlib.Path
        Directory of the raw WSIs (used for thumbnails and tile images).

    feat_dir : str or pathlib.Path, optional
        Directory of the `<case_id>.h5` encoded slides. Defaults to the
        `enc_dir` stored in the checkpoint args.

    target_path : str or pathlib.Path, optional
        Target table. Defaults to the table stored in the checkpoint. Its
        `id_name` / `fold_name` columns are renamed to `case_id` / `fold`,
        as in `DatasetHandler`.

    device : str, default="cpu"
        Device used for the forward pass.

    Raises
    ------
    ValueError
        If the network architecture or the task is not supported by the
        subclass, or if no target table is available.
    """

    SUPPORTED_ARCHITECTURES: set[str] = set()
    SUPPORTED_TASKS: set[str] = set()

    def __init__(
        self,
        model: str | Path,
        wsi_dir: str | Path,
        feat_dir: str | Path | None = None,
        target_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model_path = Path(model)
        self.model = load_model(
            self.model_path,
            self.device,
        )
        self.model.network.eval()
        self.args = self.model.args
        self.task = self.model.task

        if self.task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"`{type(self).__name__}` does not support task `{self.task}`. "
                f"Expected one of {sorted(self.SUPPORTED_TASKS)}."
            )

        self.hooker = HookerMIL(
            self.model,
            task=self.task,
        )
        self.network = self.hooker.network

        if self.hooker.architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"`{type(self).__name__}` does not support the "
                f"`{self.hooker.architecture}` architecture. "
                f"Expected one of {sorted(self.SUPPORTED_ARCHITECTURES)}."
            )

        # None when the training labels were already contiguous integers
        # starting at zero (see `_transform_classification_target`).
        self.label_encoder = self.model.label_encoder

        self.table = self._load_table(target_path)
        self.wsi_dir = Path(wsi_dir)
        self.feat_dir = Path(feat_dir) if feat_dir is not None else Path(
            self.args.enc_dir
        )
        self.model_name = self.args.model
        self.target_name = self.args.target_name
        self.n_classes = getattr(self.args, "n_classes", None)
        self.num_heads = self.hooker.num_heads


    def _load_table(
        self,
        target_path: str | Path | None,
    ) -> pd.DataFrame:
        """
        Load the target table with internal column names.

        Parameters
        ----------
        target_path : str or pathlib.Path or None
            Table to read. If None, use the table stored in the checkpoint,
            which already has the internal column names.

        Returns
        -------
        pandas.DataFrame
            Table with a `case_id` column (str) and, if present, a `fold`
            column.

        Raises
        ------
        ValueError
            If `target_path` is None and the checkpoint has no table.

        KeyError
            If the table has no case identifier column.
        """
        if target_path is None:
            table = getattr(self.model, "target_table", None)

            if table is None:
                raise ValueError(
                    f"Checkpoint `{self.model_path}` has no target table. "
                    "Pass `target_path`."
                )

            table = table.copy()

        else:
            table = read_table(target_path)
            rename_mapping = {
                self.args.id_name: "case_id",
            }
            fold_name = getattr(self.args, "fold_name", "test")

            if fold_name in table.columns:
                rename_mapping[fold_name] = "fold"

            table = table.rename(columns=rename_mapping)

        if "case_id" not in table.columns:
            raise KeyError(
                f"Case identifier column `{self.args.id_name}` was not found in "
                f"the target table."
            )

        table["case_id"] = table["case_id"].astype(str)

        return table

    @property
    def classes(self) -> list[Any]:
        """
        Class labels, in the order of the network outputs.

        Original labels when the checkpoint has a `label_encoder`, else the
        integer class indices `0, ..., n_classes - 1`.
        """
        if self.label_encoder is not None:
            return list(self.label_encoder.classes_)

        return list(range(self.n_classes))

    @property
    def case_ids(self) -> list[str]:
        """Case identifiers of the target table."""
        return self.table["case_id"].tolist()

    def _feature_path(
        self,
        case_id: str,
    ) -> Path:
        """Path of the encoded slide `<feat_dir>/<case_id>.h5`."""
        path = self.feat_dir / f"{case_id}.h5"

        if not path.is_file():
            raise FileNotFoundError(f"Encoded slide not found: {path}")

        return path

    def _get_info(
        self,
        case_id: str,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """
        Read the tile coordinates of an encoded slide.

        Returns
        -------
        dict
            Attributes of the h5 `coords` dataset.

        numpy.ndarray
            Raw tile coordinates (x, y, w, h), shape (N, 4).
        """
        return read_h5_coords(self._feature_path(case_id))

    def _tile_info(
        self,
        case_id: str,
        attrs: dict[str, Any],
    ) -> dict[str, Any]:
        """Per-tile metadata: `case_id` plus the `TILE_INFO_KEYS` attributes."""
        info = {key: attrs[key] for key in TILE_INFO_KEYS if key in attrs}
        info.setdefault("name", case_id)
        info["case_id"] = case_id

        return info

    def _get_slide(
        self,
        case_id: str,
    ):
        """
        Open the raw WSI of a case.

        Raises
        ------
        FileNotFoundError
            If no WSI named `case_id` is found in `wsi_dir`.

        ValueError
            If several WSIs match.
        """
        wsi_path, count = findFile(
            str(self.wsi_dir),
            case_id,
            fileExtensions=False,
        )

        if not wsi_path:
            raise FileNotFoundError(f"No WSI named `{case_id}` in `{self.wsi_dir}`.")

        if count > 1:
            raise ValueError(
                f"Several WSIs named `{case_id}` were found in `{self.wsi_dir}`."
            )

        reader = get_slide_reader(wsi_path)

        return reader(img_path=wsi_path)

    def _init_patcher(
        self,
        case_id: str,
    ) -> SlidePatcher:
        """Build a `SlidePatcher` on the tiles of the encoded slide."""
        attrs, coords = self._get_info(case_id)
        slide = self._get_slide(case_id)

        return SlidePatcher(
            slide,
            mag_0=slide.magnification,
            mag_target=attrs["target_magnification"],
            patch_size=attrs["target_patch_size"],
            overlap=attrs["target_overlap"],
            custom_xywh=coords,
            mask_tolerance=attrs["tissu_thr"],
            xywh_only=False,
        )

    def _get_thumbnail(
        self,
        case_id: str,
        size: tuple[int, int] = (1024, 1024),
        numpy: bool = True,
    ):
        """Thumbnail of the WSI, as an RGB array if `numpy`."""
        thumbnail = self._get_slide(case_id).get_thumbnail(size)

        if numpy:
            thumbnail = np.array(thumbnail)[:, :, :3]

        return thumbnail

    def _get_image(
        self,
        case_id: str,
        index: int,
    ):
        """Image of the `index`-th tile of the encoded slide."""
        patcher = self._init_patcher(case_id)

        return patcher.get_tile(*patcher.valid_patches[index])

    def _preprocess(
        self,
        case_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load the network inputs of one slide.

        Returns
        -------
        torch.Tensor
            Tile embeddings, shape [1, N, feature_dim].

        torch.Tensor
            Tile coordinates normalised by `level_size`, shape [1, N, 2].
        """
        sample = prep_wsi(
            self._feature_path(case_id),
            self.args,
        )
        tiles = sample["tiles"].unsqueeze(0).to(self.device)
        coords = sample["coords"].unsqueeze(0).to(self.device)

        return tiles, coords


    @torch.inference_mode()
    def _run_network(
        self,
        case_id: str,
    ) -> dict[str, np.ndarray | None]:
        """
        Run one full forward pass and collect the hooked slide outputs.

        Returns
        -------
        dict
            Batch dimension removed. None when the layer does not exist.

            tiles_encoding
                Network input (foundation model embeddings), shape
                (N, feature_dim).
            tiles_transf
                Output of `InstanceTransform`, shape (N, instance_dim).
            attention_heads
                Attention weights after softmax, shape (N, n_heads).
            attention
                `attention_heads` averaged over heads, shape (N,).
            pred
                Slide raw outputs and probabilities, shape (output_dim,).
        """
        tiles, coords = self._preprocess(case_id)

        self.hooker.reset()
        self.model.network(
            tiles,
            coords=coords,
        )

        attention_heads = self.hooker.tiles_attention

        if attention_heads is not None:
            attention_heads = attention_heads[0]

        return {
            "tiles_encoding": tiles[0].detach().float().cpu().numpy(),
            "tiles_transf": self.hooker.tiles_transf[0],
            "attention_heads": attention_heads,
            "attention": (
                attention_heads.mean(axis=-1) if attention_heads is not None else None
            ),
            "scores": self.hooker.scores[0],
            "proba": (self.hooker.proba[0] if self.hooker.proba is not None else None),
        }

    @abstractmethod
    def forward(
        self,
        case_id: str,
    ):
        """Process one slide."""

    @staticmethod
    def _select_tiles(
        attention: np.ndarray | None,
        n_tiles: int,
        att_thres: str | int | None,
    ) -> np.ndarray:
        """
        Select tiles on their attention weight.

        Parameters
        ----------
        attention : numpy.ndarray or None
            Attention weight of each tile, shape (N,). If None (no
            attention pooling), every tile is selected.

        n_tiles : int
            Number of tiles N.

        att_thres : {"otsu"} or int or None
            `otsu`: tiles above the Otsu threshold. int: the `att_thres`
            tiles with the highest attention (0 selects none). None: all
            tiles.

        Returns
        -------
        numpy.ndarray
            Indices of the selected tiles.

        Raises
        ------
        ValueError
            If `att_thres` is not one of the options above.
        """
        if attention is None or att_thres is None:
            return np.arange(n_tiles)

        if att_thres == "otsu":
            if np.ptp(attention) == 0:
                return np.arange(n_tiles)

            threshold = filters.threshold_otsu(attention)

            return np.flatnonzero(attention >= threshold)

        if isinstance(att_thres, int) and not isinstance(att_thres, bool):
            size_select = min(max(att_thres, 0), n_tiles)

            if size_select == 0:
                return np.array([], dtype=int)

            return np.argsort(attention)[-size_select:]

        raise ValueError(
            f"`att_thres` must be 'otsu', an int or None, received {att_thres!r}."
        )

    def _select_ensemble_members(
        self,
        case_id: str,
        members: Sequence[BaseTileVisualizer],
    ) -> list[BaseTileVisualizer]:
        """
        Keep the ensemble members whose held-out fold contains the case.

        Falls back to every member, with a warning, when the case has no
        fold or when no member was tested on its fold.
        """
        if "fold" not in self.table.columns:
            return list(members)

        folds = self.table.loc[self.table["case_id"] == str(case_id), "fold"]

        if folds.empty:
            LOGGER.warning(
                "Case %s is not in the target table, using all %s models.",
                case_id,
                len(members),
            )
            return list(members)

        fold = folds.iloc[0]
        matching = [
            member
            for member in members
            if getattr(member.args, "test_fold", None) == fold
        ]

        if not matching:
            LOGGER.warning(
                "No model has test fold %s (case %s), using all %s models.",
                fold,
                case_id,
                len(members),
            )
            return list(members)

        return matching

    @staticmethod
    def _average_outputs(
        outputs: Sequence[dict[str, np.ndarray | None]],
    ) -> dict[str, np.ndarray | None]:
        """Mean over members of each output. None if any member lacks it."""
        averaged = {}

        for key in outputs[0]:
            values = [output[key] for output in outputs]

            if any(value is None for value in values):
                averaged[key] = None
            else:
                averaged[key] = np.mean(
                    values,
                    axis=0,
                )

        return averaged


    def _thumbnail_and_coords(
        self,
        slide,
        attrs: dict[str, Any],
        coords: np.ndarray,
        downsample: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Thumbnail of the slide at `level / downsample` and the tile boxes on it.

        Returns
        -------
        numpy.ndarray
            RGB thumbnail, shape (H, W, 3).

        numpy.ndarray
            Tile boxes (x, y, w, h) in thumbnail pixels, shape (N, 4).
        """
        level_width, level_height = slide.level_dimensions[attrs["level"]]
        thumbnail = slide.get_thumbnail(
            (
                int(level_width / downsample),
                int(level_height / downsample),
            )
        )
        thumbnail = np.array(thumbnail)[:, :, :3]
        thumbnail_height, thumbnail_width, _ = thumbnail.shape
        patch_size = max(1, int(attrs["level_patch_size"] / downsample))

        thumbnail_coords = np.zeros(
            (len(coords), 4),
            dtype=int,
        )

        for index, (x, y, _, _) in enumerate(coords):
            x, y = get_x_y_to(
                (x, y),
                (level_width, level_height),
                (thumbnail_width, thumbnail_height),
                integer=True,
            )
            thumbnail_coords[index, :] = [x, y, patch_size, patch_size]

        return thumbnail, thumbnail_coords

    @staticmethod
    def fill_heatmap(
        size: tuple[int, int],
        xywh: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Paint one score per tile box.

        Parameters
        ----------
        size : tuple of int
            (width, height) of the heatmap.

        xywh : numpy.ndarray
            Tile boxes in heatmap pixels, shape (N, 4).

        scores : numpy.ndarray
            Score of each tile, shape (N,).

        Returns
        -------
        numpy.ndarray
            Heatmap, shape (height, width), 0 on the background.

        numpy.ndarray
            Background mask (pixels covered by no tile).
        """
        heatmap = np.full(
            (size[1], size[0]),
            np.nan,
            dtype=np.float32,
        )

        for (x, y, w, h), score in zip(xywh, scores):
            heatmap[y : y + h, x : x + w] = score

        background = np.isnan(heatmap)

        return np.where(background, 0, heatmap), background

    @staticmethod
    def overlay_heatmap_on_thumbnail(
        thumbnail: np.ndarray,
        heatmap: np.ndarray,
        background: np.ndarray,
        smooth: float | None = None,
        alpha: float = 0.5,
        cmap: str = "jet",
    ) -> np.ndarray:
        """
        Colour the heatmap and blend it on the thumbnail.

        Parameters
        ----------
        thumbnail : numpy.ndarray
            RGB thumbnail, shape (H, W, 3).

        heatmap, background : numpy.ndarray
            Output of `fill_heatmap`.

        smooth : float, optional
            Sigma of a Gaussian smoothing restricted to the tissue. None or
            0 disables it.

        alpha : float, default=0.5
            Opacity of the heatmap.

        cmap : str, default="jet"
            Matplotlib colormap.

        Returns
        -------
        numpy.ndarray
            Blended RGB image.
        """
        tissue = ~background

        if not tissue.any():
            return thumbnail.copy()

        # Min-max normalisation on the tissue only.
        values = heatmap[tissue]
        value_range = values.max() - values.min()
        normalized = np.zeros_like(heatmap, dtype=np.float32)
        normalized[tissue] = (
            (values - values.min()) / value_range if value_range > 0 else 0.5
        )

        # Normalised convolution, so the background does not bleed in.
        if smooth:
            weights = gaussian_filter(tissue.astype(np.float32), sigma=smooth)
            normalized = gaussian_filter(normalized, sigma=smooth) / np.maximum(
                weights,
                1e-6,
            )
            normalized = np.where(tissue, np.clip(normalized, 0, 1), 0)

        colormap = plt.get_cmap(cmap)
        colored_heatmap = (colormap(normalized)[..., :3] * 255).astype(np.uint8)
        colored_heatmap[background] = 0

        return blend_images(
            thumbnail,
            colored_heatmap,
            background_color=(0, 0, 0),
            alpha=alpha,
        )
