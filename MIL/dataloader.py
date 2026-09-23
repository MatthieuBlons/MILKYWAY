"""
Data loading utilities for Patched & Encoded WSI with patient level
target.

Supported WSI format
--------------------------
HDF5 containing:
    features: numpy.array of size NxF
        with attributes:
            - "encoder":
            - "name":
            - "dst":

    coords: numpy.array of size Nx4 (left corner: x, y, h, w)
        with attributes:
            - "magnification":
            - "mpp":
            - "target_magnification":
            - "target_patch_size":
            - "target_mpp":
            - "target_overlap":
            - "level":
            - "level_size":
            - "level_patch_size":
            - "level_overlap":
            - "tissue_thr":
            - "mask":
            - "name":
            - "savetodir":

Supported Patch selection strategies
--------------------------
all
random
random_strict
niche
exact_coords

Supported Multiple Instance Learning tasks
--------------------------
survival
    Predict one patient-level risk score from observed survival time and
    event status.

classification
    Predict a binary clinical endpoint such as death within five years.

regression
    Predict one or more continuous targets, such as gene-expression values.
"""

from __future__ import annotations

from pathlib import Path

from typing import Any, Iterator, Sequence

import pandas as pd
import torch

from slide.tile import EncodingSampler
from slide.utils import read_h5_coords, read_h5_features

import logging
from argparse import Namespace
from collections import Counter
from functools import reduce

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
    SubsetRandomSampler,
)

from MIL.io import read_table

SUPPORTED_TASKS = {
    "survival",
    "classification",
    "regression",
}

class WSIEncoded(Dataset):
    """
    Implements a Dataset for encoded WSI.

    Each WSI is represented by a .h5 array of size NxF with N the number of patches
    in the WSI and F the number of features in the embeding space (depending on the
    encoder used) stored in an HDF5 file with coords accessible if needed.

    Note: no transformation is implemented as tiles are already encoded in vector
    form.


    Three target types are supported:

    survival :
        Returns observed survival time and event status.

    classification :
        Returns one or several class labels.

    regression :
        Returns one or several continuous targets.

    The target should have:
        * an ID column with the name of the images (without the extension)
        * $args.target_name column(s)
        * a test columns, stating to which test_fold each image belong.

    Parameters
    ----------
    args : argparse.Namespace
        Runtime configuration.

        Required attributes:

        enc_dir
            Directory containing one HDF5 map per case.

        target_path
            Clinical or target table.

        task
            One of survival, classification, or regression.

        test_fold
            Fold reserved for testing when the target table contains the
            configured fold column.


        Task-specific attributes:

        feature_dim
            number of features to use. Passed that, features are dropped

        sampling_strat
            strategy to use for tile sampling in the WSI

        n_tiles
            number of tiles to sample from each WSI

        For survival:
            time_name and event_name.

        For classification:
            target_name containing exactly one column.

        For regression:
            target_name containing one or more columns.

        Optional attributes:

        id_name
            Case identifier column. Defaults to ID.

        fold_name
            Cross-validation fold column. Defaults to test.

        stratif_name
            Stratification column. Defaults to stratif.

        encode_labels
            Encode non-numeric classification labels with LabelEncoder.
            Defaults to False.

    use_train : bool
        If True, retain cases outside test_fold. If False,
        retain cases assigned to test_fold.

    sampling_mode : str, default="train"
        If data is for training, use the sampling strategy in args

    predict : bool, default=False
        If True, ignore test-fold filtering and retain all cases.

    file_format : str, default="h5"
        Extension of the wsi files.

    verbose : bool, default=False
        If enable, display info

    Raises
    ------
    ValueError
        If `args.task` is not in `SUPPORTED_TASKS`.

    FileNotFoundError
        If `args.enc_dir` does not exist.
    """

    def __init__(
        self,
        args: Namespace,
        use_train: bool,
        sampling_mode: str = "train",
        predict: bool = False,
        file_format: str = "h5",
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self.args = args
        self.input_dir = Path(args.enc_dir)
        self.use_train = bool(use_train)
        self.sampling_mode = str(sampling_mode).lower()
        self.predict = bool(predict)
        self.file_format = file_format.lstrip(".")
        self.logger = logging.getLogger("WSIEncoded")
        self.logger.disabled = not verbose
        self.id_name = getattr(args, "id_name", "ID")
        self.fold_name = getattr(args, "fold_name", "test")
        self.stratif_name = getattr(args, "stratif_name", "stratif")
        self.encode_labels = getattr(args, "encode_labels", False)

        self.task = str(args.task).lower()

        if self.task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task '{self.task}'. "
                f"Expected one of {sorted(SUPPORTED_TASKS)}."
            )

        if not self.input_dir.is_dir():
            raise FileNotFoundError(
                f"Encoded WSI directory does not exist: {self.input_dir}"
            )

        self.target_table = read_table(args.target_path)

        self.target_names: list[str] = []
        self.target_columns: list[str] = []
        self.output_dim: int = 1
        self.label_encoder: LabelEncoder | None = None

        (
            self.files,
            self.target_dict,
            self.stratif_dict,
            self.label_encoder,
        ) = self._make_db()

    def __len__(self):
        """Return the number of cases in the dataset."""
        return len(self.files)

    # Slide level encoding does not make sense in the MIL framework
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Load a sample of tiles from a WSI and its task-specific target.

        Parameters
        ----------
        idx : int
            Dataset index.

        Returns
        -------
        sample : dict
            Always contains:

            tiles: sample of patches encoded
                Float tensor with shape (N, F).

            coords: normalized (x, y) coords of each tile
                Float tensor with shape (N, 2).

            case_id
                Case identifier.

            path
                Source HDF5 path.

            For survival tasks:

            time
                Scalar float tensor.

            event
                Scalar Boolean tensor.

            For classification tasks:

            target
                Scalar float tensor suitable for one-logit binary
                classification.

            For regression tasks:

            target
                Float tensor with shape (n_targets,).
        """
        path = self.files[idx]

        feat_array = self._get_embeddings(path)
        if self.args.feature_dim > feat_array.shape[1]:
            raise ValueError(
                f"Requested feature_dim={self.args.feature_dim}, "
                f"but '{path.name}' contains only "
                f"{feat_array.shape[1]} features."
            )

        coord_array = self._get_coords(path)
        if feat_array.shape[0] != coord_array.shape[0]:
            raise ValueError(
                f"Feature/coordinate mismatch for '{path.name}': "
                f"{feat_array.shape[0]} features vs "
                f"{coord_array.shape[0]} coordinates."
            )

        feats = feat_array[:, : self.args.feature_dim]
        feats = np.ascontiguousarray(feats, dtype=np.float32)
        tiles, indices = self._select_tiles(path, feats)
        ttensor = torch.from_numpy(tiles)

        coords = coord_array[indices, :2]
        coords = np.ascontiguousarray(coords, dtype=np.float32)
        ctensor = torch.from_numpy(coords)

        sample: dict[str, Any] = {
            "tiles": ttensor,
            "coords": ctensor,
            "case_id": path.stem,
            "path": str(path),
        }

        if self.task == "survival":
            time, event = self.target_dict[path]

            sample["time"] = torch.tensor(
                time,
                dtype=torch.float32,
            )
            sample["event"] = torch.tensor(
                event,
                dtype=torch.bool,
            )

        elif self.task == "classification":
            sample["target"] = torch.tensor(
                self.target_dict[path],
                dtype=torch.long,
            )

        else:
            sample["target"] = torch.from_numpy(self.target_dict[path]).to(
                dtype=torch.float32
            )

        return sample

    def _get_embeddings(self, path):
        """
        Read the tile embeddings of one case.

        Parameters
        ----------
        path : pathlib.Path
            HDF5 file of the case.

        Returns
        -------
        numpy.ndarray
            Tile embeddings with shape (N, F).
        """
        _, feats = read_h5_features(path)
        return feats

    def _get_coords(self, path):
        """
        Read the tile coordinates of one case, normalised by the slide size.

        Parameters
        ----------
        path : pathlib.Path
            HDF5 file of the case.

        Returns
        -------
        numpy.ndarray
            Tile (x, y) coordinates divided by the `level_size` attribute,
            with shape (N, 2). If `level_size` is missing, raw coordinates
            are returned.

        Raises
        ------
        ValueError
            If `level_size` does not have shape (2,).
        """
        attrs, coords = read_h5_coords(path)

        coords = coords[:, :2]

        slide_size = attrs.get("level_size")

        if slide_size is None:
            slide_size = np.array(
                [1, 1]
            )  # TMP FIX, Need to add level_size to old coords attrs
            # raise ValueError(f"Missing 'level_size' attribute in '{path.name}'.")

        slide_size = np.asarray(
            slide_size,
            dtype=np.float32,
        )

        if slide_size.shape != (2,):
            raise ValueError(
                f"Invalid WSI size for '{path.name}': "
                f"expected shape (2,), got {slide_size.shape}."
            )

        return coords / slide_size

    def _select_tiles(self, path, mat):
        """_select_tiles.

        Samples tiles from the WSI.

        Parameters
        ----------
        path : pathlib.Path or str
            path to the current WSI

        mat : ndarray
            matrix of the embedded wsi.

        Returns
        -------
        sample : ndarray
            matrix of the stacked selected tiles.

        indices : torch.tensor, size (n_samples,), selected tiles' ID.
        """
        if self.sampling_mode == "train":
            train_sampler = EncodingSampler(
                sampler_name=self.args.sampling_strat,
                feat_path=path,
                n_samples=self.args.n_tiles,
            )
            indices = train_sampler.sampler()
            sample = mat[indices, :]
        else:
            test_sampler = EncodingSampler(
                sampler_name="all",
                feat_path=path,
            )
            indices = test_sampler.sampler()
            sample = mat[indices, :]
        return sample, indices

    def _make_db(
        self,
    ) -> tuple[
        list[Path],
        dict[Path, Any],
        dict[Path, Any],
        LabelEncoder | None,
    ]:
        """
        Match available HDF5 maps with transformed clinical targets.

        Returns
        -------
        files : list[pathlib.Path]
            Files retained in the requested dataset partition.

        target_dict : dict
            Mapping from file path to task-specific target.

        stratif_dict : dict
            Mapping from file path to train-validation stratification label.

        label_encoder : sklearn.preprocessing.LabelEncoder or None
            Classification label encoder, when applicable.

        Raises
        ------
        RuntimeError
            If no cases can be matched.
        """
        table, label_encoder = self._transform_target()

        files: list[Path] = []
        target_dict: dict[Path, Any] = {}
        stratif_dict: dict[Path, Any] = {}

        for row in table.itertuples(index=False):
            row_dict = row._asdict()
            case_id = str(row_dict["case_id"])
            path = self.input_dir / f"{case_id}.{self.file_format}"

            if not path.exists():
                self.logger.debug(
                    "No WSI encoded found for case %s: %s",
                    case_id,
                    path,
                )
                continue

            if not self._is_in_db(row_dict):
                continue

            if self.task == "survival":
                target: Any = (
                    np.float32(row_dict["target_time"]),
                    np.bool_(row_dict["target_event"]),
                )

            elif self.task == "classification":
                target = np.int64(row_dict["target"])

            else:
                target = np.asarray(
                    [row_dict[column] for column in self.target_columns],
                    dtype=np.float32,
                )

            files.append(path)
            target_dict[path] = target

            if "stratif" in row_dict:
                stratif_dict[path] = row_dict["stratif"]
            else:
                stratif_dict[path] = self._default_stratification_label(row_dict)

        if not files:
            raise RuntimeError(
                "No cases were matched between the target table and "
                f"the map directory '{self.input_dir}'."
            )

        self.logger.info(
            "Constructed %s dataset with %d cases.",
            "training" if self.use_train else "evaluation",
            len(files),
        )

        return files, target_dict, stratif_dict, label_encoder

    def _transform_target(
        self,
    ) -> tuple[pd.DataFrame, LabelEncoder | None]:
        """
        Validate and standardize task-specific targets.

        The input table is renamed internally to stable column names:

        case_id
            Case identifier.

        fold
            Cross-validation test-fold assignment, when available.

        stratif
            User-provided stratification label, when available.

        For survival tasks, this method creates:

        target_time
            Observed follow-up or survival time.

        target_event
            Binary event indicator.

        For classification tasks, it creates:

        target
            Binary value encoded as zero or one.

        For regression tasks, it creates:

        target_0, ..., target_n
            Continuous targets.

        Returns
        -------
        table : pandas.DataFrame
            Transformed target table.

        label_encoder : sklearn.preprocessing.LabelEncoder or None
            Encoder fitted for non-numeric classification labels.

        Raises
        ------
        KeyError
            If required columns are missing.

        ValueError
            If target values or task arguments are invalid.
        """
        table = self.target_table.copy()

        if self.id_name not in table.columns:
            raise KeyError(f"Case identifier column '{self.id_name}' was not found.")

        rename_mapping = {
            self.id_name: "case_id",
        }

        if self.fold_name in table.columns:
            rename_mapping[self.fold_name] = "fold"

        if self.stratif_name in table.columns:
            rename_mapping[self.stratif_name] = "stratif"

        table = table.rename(columns=rename_mapping)
        table["case_id"] = table["case_id"].astype(str)

        if table["case_id"].duplicated().any():
            duplicated = (
                table.loc[
                    table["case_id"].duplicated(keep=False),
                    "case_id",
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                "The target table contains duplicated case identifiers: "
                f"{duplicated[:10]}"
            )

        label_encoder: LabelEncoder | None = None

        if self.task == "survival":
            table = self._transform_survival_target(table)

        elif self.task == "classification":
            table, label_encoder = self._transform_classification_target(table)

        else:
            table = self._transform_regression_target(table)

        self.target_table = table

        return table, label_encoder

    def _transform_survival_target(
        self,
        table: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform observed time and event columns for survival analysis.
        """
        time_name = getattr(self.args, "time_name", None)
        event_name = getattr(self.args, "event_name", None)

        if not time_name:
            raise ValueError("`time_name` is required when task='survival'.")

        if not event_name:
            raise ValueError("`event_name` is required when task='survival'.")

        required = [time_name, event_name]
        self._require_columns(table, required)

        table = table.dropna(subset=required).copy()

        table["target_time"] = pd.to_numeric(
            table[time_name],
            errors="raise",
        ).astype(np.float32)

        event = pd.to_numeric(
            table[event_name],
            errors="raise",
        )

        if (table["target_time"] < 0).any():
            invalid_cases = table.loc[
                table["target_time"] < 0,
                "case_id",
            ].tolist()

            raise ValueError(
                "Survival times must be non-negative. Invalid cases: "
                f"{invalid_cases[:10]}"
            )

        event_values = set(event.unique())

        if not event_values.issubset({0, 1}):
            raise ValueError(
                "The event column must contain only 0 and 1. "
                f"Observed values: {sorted(event_values)}"
            )

        table["target_event"] = event.astype(bool)

        self.target_names = [time_name, event_name]
        self.target_columns = [
            "target_time",
            "target_event",
        ]
        self.output_dim = 1

        return table

    def _transform_classification_target(
        self,
        table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, LabelEncoder | None]:
        """
        Transform a single categorical target for binary or multiclass
        classification.

        The target is stored internally in the target column as contiguous
        integer class indices in [0, n_classes - 1].

        Parameters
        ----------
        table : pandas.DataFrame
            Target table containing the classification column.

        Returns
        -------
        table : pandas.DataFrame
            Table containing the integer-encoded target column.

        label_encoder : sklearn.preprocessing.LabelEncoder or None
            Fitted encoder used to map original labels to integer class indices.
            Returns None only when the original target is already encoded as
            contiguous integers starting at zero.

        Raises
        ------
        KeyError
            If the requested target column is absent.

        ValueError
            If multiple target columns are provided, fewer than two classes are
            available, or labels cannot be converted safely.
        """
        target_names = self._get_target_names()

        if len(target_names) != 1:
            raise ValueError(
                "Classification requires exactly one target column. "
                f"Received: {target_names}"
            )

        target_name = target_names[0]
        self._require_columns(table, [target_name])

        table = table.dropna(subset=[target_name]).copy()
        raw_target = table[target_name]

        if raw_target.empty:
            raise ValueError(
                f"No valid values remain for classification target " f"'{target_name}'."
            )

        label_encoder: LabelEncoder | None = None

        numeric_target = pd.to_numeric(
            raw_target,
            errors="coerce",
        )

        is_integer_numeric = numeric_target.notna().all() and np.allclose(
            numeric_target.to_numpy(),
            np.round(numeric_target.to_numpy()),
        )

        if is_integer_numeric:
            integer_target = numeric_target.astype(np.int64)
            observed_classes = np.sort(integer_target.unique())

            expected_classes = np.arange(len(observed_classes))

            if np.array_equal(observed_classes, expected_classes):
                table["target"] = integer_target

            else:
                label_encoder = LabelEncoder()
                table["target"] = label_encoder.fit_transform(integer_target).astype(
                    np.int64
                )

        else:
            label_encoder = LabelEncoder()
            table["target"] = label_encoder.fit_transform(
                raw_target.astype(str)
            ).astype(np.int64)

        n_classes = int(table["target"].nunique())

        if n_classes < 2:
            raise ValueError(
                "Classification requires at least two classes. "
                f"Found {n_classes} class in target '{target_name}'."
            )

        self.target_names = [target_name]
        self.target_columns = ["target"]
        self.output_dim = n_classes
        self.classes_ = (
            label_encoder.classes_.tolist()
            if label_encoder is not None
            else list(range(n_classes))
        )

        self.logger.info(
            "Classification target '%s' | classes=%d | mapping=%s",
            target_name,
            n_classes,
            {str(original): encoded for encoded, original in enumerate(self.classes_)},
        )

        return table, label_encoder

    def _transform_regression_target(
        self,
        table: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform one or multiple continuous regression targets.
        """
        target_names = self._get_target_names()
        self._require_columns(table, target_names)

        table = table.dropna(subset=target_names).copy()
        internal_columns: list[str] = []

        for index, target_name in enumerate(target_names):
            internal_name = f"target_{index}"

            table[internal_name] = pd.to_numeric(
                table[target_name],
                errors="raise",
            ).astype(np.float32)

            values = table[internal_name].to_numpy()

            if not np.isfinite(values).all():
                invalid_cases = table.loc[
                    ~np.isfinite(values),
                    "case_id",
                ].tolist()

                raise ValueError(
                    f"Regression target '{target_name}' contains "
                    f"non-finite values for cases: {invalid_cases[:10]}"
                )

            internal_columns.append(internal_name)

        self.target_names = target_names
        self.target_columns = internal_columns
        self.output_dim = len(target_names)

        return table

    def _get_target_names(self) -> list[str]:
        """
        Return args.target_name as a validated list of column names.
        """
        target_name = getattr(self.args, "target_name", None)

        if target_name is None:
            raise ValueError(f"`target_name` is required when task='{self.task}'.")

        if isinstance(target_name, str):
            target_names = [target_name]
        else:
            target_names = list(target_name)

        if not target_names:
            raise ValueError(f"`target_name` is empty for task='{self.task}'.")

        if len(set(target_names)) != len(target_names):
            raise ValueError(f"Duplicated target columns: {target_names}")

        return target_names

    @staticmethod
    def _require_columns(
        table: pd.DataFrame,
        columns: Sequence[str],
    ) -> None:
        """Raise an error when required columns are absent."""
        missing = set(columns).difference(table.columns)

        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

    def _is_in_db(self, row: dict[str, Any]) -> bool:
        """
        Determine whether a clinical record belongs to this partition.

        In prediction mode, every case is retained. If no fold column is
        present, every matched case is retained. Otherwise, the configured
        test fold is included only in the evaluation dataset.

        Parameters
        ----------
        row : dict
            Transformed clinical-table record.

        Returns
        -------
        bool
            Whether the case should be included.
        """
        if self.predict or "fold" not in row:
            return True

        is_test = row["fold"] == self.args.test_fold

        return not is_test if self.use_train else is_test

    def _default_stratification_label(
        self,
        row: dict[str, Any],
    ) -> str:
        """
        Construct a default stratification label when none is provided.

        Survival data are stratified on event status. Classification data
        are stratified on class. Regression data receive a common label;
        for regression, a precomputed discrete stratification column is
        preferable.
        """
        if self.task == "survival":
            return f"event_{int(row['target_event'])}"

        if self.task == "classification":
            return f"class_{int(row['target'])}"

        return "regression"


from torch.utils.data import default_collate


def collate_variable_size(batch: Any):
    """collate_variable_size.

    Collate a batch of WSI samples with variable numbers of instances.

    Variable-size tensors such as tiles and coordinates are put in a lists
    of tensor.

    WSI-level tensors and scalar targets are collated normally.

    Parameters
    ----------
    batch : dict
        batch of wsi sample.

        Required keys:
            tiles
            coords
            case_id
            path
            target | time + event

    Retruns
    ----------
    collated: dict

    """
    variable_keys = {"tiles", "coords"}

    collated = {}

    for key in batch[0]:
        values = [sample[key] for sample in batch]

        if key in variable_keys:
            collated[key] = values
        else:
            collated[key] = default_collate(values)

    return collated


class DatasetHandler:
    """
    Build datasets, samplers, and data loaders for model training.

    In standard training mode, the cohort is divided into:

    - a training dataset containing all folds except test_fold;
    - a held-out test dataset containing test_fold.

    The training dataset can optionally be split into training and
    validation subsets using stratified shuffle splitting.

    Parameters
    ----------
    args : argparse.Namespace
        Data-loading and splitting configuration.

        Expected attributes include:

        batch_size
            Training batch size.

        num_workers
            Number of DataLoader worker processes.

        use_val
            Whether to reserve a validation subset.

        val_fraction
            Fraction assigned to validation. Defaults to 0.2.

        seed
            Random seed. Defaults to 29.

        pin_memory
            Enable pinned host memory. Defaults to False.

        persistent_workers
            Keep workers alive between epochs. Defaults to False.

        no_strat_sampling
            Disable weighted sampling. Defaults to True.

        sample_wr_whole_label
            Select the legacy weighted-sampling strategy.

        constant_size
            If True, every bag has the same number of tiles and batches are
            stacked. Otherwise `collate_variable_size` is used. Defaults to
            True.

    predict : bool, default=False
        If True, construct one dataset containing all available cases.

    verbose : bool, default=True
        If True, log dataset and split information.

    Raises
    ------
    ValueError
        If `val_fraction` is not strictly between zero and one.
    """

    def __init__(
        self,
        args: Namespace,
        predict: bool = False,
        verbose: bool = True,
    ) -> None:
        self.args = args
        self.predict = bool(predict)
        self.logger = logging.getLogger("DatasetHandler")
        self.logger.disabled = not verbose
        self.num_workers = int(getattr(args, "num_workers", 0))
        self.seed = int(getattr(args, "seed", 29))
        self.constant_size = bool(getattr(args, "constant_size", True))
        self.use_val = bool(getattr(args, "use_val", True))
        self.val_fraction = float(getattr(args, "val_fraction", 0.2))

        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("`val_fraction` must be strictly between zero and one.")

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        if self.predict:
            self.dataset_predict = self._get_dataset(training=False, mode="eval")
            self.dataset_train = None
            self.dataset_test = None
            self.dataset_val = None
            self.train_sampler = None
            self.val_sampler = None

        else:
            self.dataset_predict = None
            self.dataset_train = self._get_dataset(training=True, mode="train")
            self.dataset_val = None
            if self.use_val:
                self.dataset_val = self._get_dataset(training=True, mode="eval")
            self.dataset_test = self._get_dataset(training=False, mode="eval")

            (
                self.train_sampler,
                self.val_sampler,
            ) = self._get_sampler(
                self.dataset_train,
                use_val=self.use_val,
            )

    def _get_dataset(
        self,
        training: bool,
        mode: str,
    ) -> WSIEncoded:
        """
        Construct one dataset partition.

        Parameters
        ----------
        use_train : bool
            Build the development partition when True and the held-out
            test partition when False.

        mode : str

        Returns
        -------
        WSIEncoded
            Configured dataset.
        """
        return WSIEncoded(
            self.args,
            use_train=training,
            sampling_mode=mode,
            predict=self.predict,
            verbose=not self.logger.disabled,
        )

    def get_loader(
        self,
        training: bool,
    ) -> tuple[DataLoader, DataLoader] | DataLoader:
        """
        Construct training, validation, test, or prediction loaders.

        Parameters
        ----------
        training : bool
            Return training and validation loaders when True.
            Otherwise return a deterministic test or prediction loader.

        Returns
        -------
        tuple[DataLoader, DataLoader] or DataLoader
            Requested loaders.
        """
        common_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": bool(getattr(self.args, "pin_memory", False)),
            "persistent_workers": bool(
                getattr(
                    self.args,
                    "persistent_workers",
                    False,
                )
            )
            and self.num_workers > 0,
        }

        if training:
            collate_fn = None if self.args.constant_size else collate_variable_size

            if self.predict:
                raise RuntimeError(
                    "Training loaders are unavailable in prediction mode."
                )

            train_loader = DataLoader(
                dataset=self.dataset_train,
                batch_size=self.args.batch_size,
                sampler=self.train_sampler,
                collate_fn=collate_fn,
                drop_last=False,
                **common_kwargs,
            )

            if self.use_val:
                if self.args.eval_batch_size > 1:
                    val_loader = DataLoader(
                        dataset=self.dataset_val,
                        batch_size=self.args.eval_batch_size,
                        sampler=self.val_sampler,
                        collate_fn=collate_fn,
                        drop_last=False,
                        **common_kwargs,
                    )
                else:
                    val_loader = DataLoader(
                        dataset=self.dataset_val,
                        batch_size=1,
                        sampler=self.val_sampler,
                        drop_last=False,
                        **common_kwargs,
                    )

                return train_loader, val_loader

            return train_loader, None

        dataset = self.dataset_predict if self.predict else self.dataset_test

        return DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        )

    def _get_sampler(
        self,
        dataset: WSIEncoded,
        use_val: bool = True,
    ) -> tuple[Sampler[int], Sampler[int]]:
        """
        Create training and validation index samplers.

        When validation is enabled, a stratified shuffle split is applied
        to the development dataset. Weighted sampling can optionally be
        enabled for classification-style balancing.

        Parameters
        ----------
        dataset : ProteoMultiChannel
            Development dataset.

        use_val : bool, default=True
            Whether to reserve a validation subset.

        Returns
        -------
        train_sampler : torch.utils.data.Sampler
            Training sampler.

        val_sampler : torch.utils.data.Sampler
            Validation sampler.
        """
        n_cases = len(dataset)

        if n_cases == 0:
            raise ValueError("Cannot sample from an empty dataset.")

        all_indices = np.arange(n_cases, dtype=np.int64)

        if not use_val:
            train_sampler = SubsetRandomSampler(
                all_indices.tolist(),
                generator=self.generator,
            )
            val_sampler = None
            return train_sampler, val_sampler

        labels = np.asarray([dataset.stratif_dict[path] for path in dataset.files])

        label_counts = Counter(labels.tolist())
        insufficient = {
            label: count for label, count in label_counts.items() if count < 2
        }

        if insufficient:
            raise ValueError(
                "Every stratification group must contain at least two "
                f"cases. Insufficient groups: {insufficient}"
            )

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.val_fraction,
            random_state=self.seed,
        )

        train_indices, val_indices = next(
            splitter.split(
                X=np.zeros(n_cases),
                y=labels,
            )
        )

        use_weighted_sampling = not bool(
            getattr(
                self.args,
                "no_strat_sampling",
                True,
            )
        )

        if use_weighted_sampling:
            if dataset.task == "survival":
                raise ValueError(
                    "Weighted sampling with replacement is not supported "
                    "for survival analysis."
                )

            train_labels = labels[train_indices]

            weights = self._get_weights_sampling(
                labels=train_labels.tolist(),
                wr_whole_label=bool(
                    getattr(
                        self.args,
                        "sample_wr_whole_label",
                        False,
                    )
                ),
                no_strat_sampling=False,
            )

            train_sampler: Sampler[int] = WeightedRandomSamplerFromList(
                weights=weights,
                indices=train_indices,
                num_samples=len(train_indices),
                replacement=True,
                generator=self.generator,
            )

        else:
            train_sampler = SubsetRandomSampler(
                train_indices.tolist(),
                generator=self.generator,
            )

        val_sampler: Sampler[int] = SequentialSubsetSampler(val_indices.tolist())

        self.logger.info(
            "Development split: train=%d, validation=%d.",
            len(train_indices),
            len(val_indices),
        )

        return train_sampler, val_sampler

    def _get_weights_sampling(
        self,
        labels: Sequence[Any],
        wr_whole_label: bool = False,
        no_strat_sampling: bool = False,
    ) -> list[float]:
        """
        Compute sample weights from discrete stratification labels.

        Parameters
        ----------
        labels : sequence
            Stratification label for each eligible training case.

        wr_whole_label : bool, default=False
            If True, apply inverse-frequency weighting directly to each
            complete stratification label.

            If False, retain the legacy conditional balancing strategy,
            which assumes labels contain underscore-separated components
            and that the penultimate component encodes the prediction
            target.

        no_strat_sampling : bool, default=False
            Return uniform weights.

        Returns
        -------
        list[float]
            One sampling weight per label.

        Notes
        -----
        Weighted sampling with replacement can duplicate patients within a
        batch. This is generally unsuitable for survival analysis
        because it changes the composition of risk sets.
        """
        if no_strat_sampling:
            return [1.0] * len(labels)

        counts = Counter(labels)

        if wr_whole_label:
            return [1.0 / counts[label] for label in labels]

        if self.dataset_train is None:
            raise RuntimeError("Conditional weighting requires a training dataset.")

        target_names = self.dataset_train.target_names

        if len(target_names) != 1:
            self.logger.warning(
                "Conditional weighting is unavailable for multiple "
                "targets; using inverse-frequency weights instead."
            )
            return [1.0 / counts[label] for label in labels]

        raw_target_name = target_names[0]

        if raw_target_name not in self.dataset_train.target_table.columns:
            return [1.0 / counts[label] for label in labels]

        target_values = (
            self.dataset_train.target_table[raw_target_name]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        weights: list[float] = []

        for label in labels:
            label_parts = str(label).split("_")

            if len(label_parts) < 2:
                weights.append(1.0 / counts[label])
                continue

            current_target = label_parts[-2]

            if current_target not in target_values:
                weights.append(1.0 / counts[label])
                continue

            equivalent_count = 0

            for target_value in target_values:
                alternative = label_parts.copy()
                alternative[-2] = target_value
                alternative_label = reduce(
                    lambda x, y: f"{x}_{y}",
                    alternative,
                )
                equivalent_count += counts[alternative_label]

            weights.append(equivalent_count / counts[label])

        return weights


class WeightedRandomSamplerFromList(Sampler[int]):
    """
    Sample weighted entries from a list of dataset indices.

    Parameters
    ----------
    weights : sequence of float
        Non-negative sampling weights.

    indices : sequence of int
        Dataset indices eligible for sampling.

    num_samples : int
        Number of indices yielded per epoch.

    replacement : bool, default=True
        Whether indices can be sampled repeatedly.

    generator : torch.Generator, optional
        Generator used for reproducible sampling.

    Raises
    ------
    ValueError
        If `weights` and `indices` differ in length or are empty, if
        `num_samples` is not positive, if the weights are not finite and
        non-negative with a positive sum, or if `num_samples` exceeds the
        number of indices without replacement.
    """

    def __init__(
        self,
        weights: Sequence[float],
        indices: Sequence[int],
        num_samples: int,
        replacement: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        if len(weights) != len(indices):
            raise ValueError("`weights` and `indices` must have the same length.")

        if len(weights) == 0:
            raise ValueError("`weights` and `indices` cannot be empty.")

        if num_samples <= 0:
            raise ValueError("`num_samples` must be strictly positive.")

        weights_tensor = torch.as_tensor(
            weights,
            dtype=torch.double,
        )

        if not torch.isfinite(weights_tensor).all():
            raise ValueError("Sampling weights must be finite.")

        if (weights_tensor < 0).any():
            raise ValueError("Sampling weights must be non-negative.")

        if weights_tensor.sum() <= 0:
            raise ValueError("At least one sampling weight must be positive.")

        if not replacement and num_samples > len(indices):
            raise ValueError(
                "Without replacement, `num_samples` cannot exceed "
                "the number of eligible indices."
            )

        self.weights = weights_tensor
        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )
        self.num_samples = int(num_samples)
        self.replacement = bool(replacement)
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        """Draw `num_samples` dataset indices according to `weights`."""
        sampled_positions = (
            torch.multinomial(
                self.weights,
                self.num_samples,
                self.replacement,
                generator=self.generator,
            )
            .cpu()
            .numpy()
        )

        sampled_indices = self.indices[sampled_positions]

        return iter(sampled_indices.tolist())

    def __len__(self) -> int:
        """Return the number of samples produced per epoch."""
        return self.num_samples


class SequentialSubsetSampler(Sampler[int]):
    """
    Yield a predefined subset of dataset indices in deterministic order.

    Parameters
    ----------
    indices : sequence of int
        Dataset indices to yield.
    """

    def __init__(
        self,
        indices: Sequence[int],
    ) -> None:
        self.indices = [int(index) for index in indices]

    def __iter__(self) -> Iterator[int]:
        """Yield the stored indices in their original order."""
        return iter(self.indices)

    def __len__(self) -> int:
        """Return the number of stored indices."""
        return len(self.indices)
