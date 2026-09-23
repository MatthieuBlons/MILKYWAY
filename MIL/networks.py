"""
Implements the MIL networks that can be used for MIL training.

"""

from torch.nn import (
    Module,
)
from torchinfo import summary
from torch import Tensor
from MIL.custom_layers import InstanceTransform, PoolingFunction, PredictionHead

from argparse import Namespace


class ABMIL(Module):
    """ABMIL.

    Attention-based multiple-instance learning (embedding-level).

    Tiles are transformed, pooled into one slide embedding, then mapped to
    the task outputs: `InstanceTransform` -> `PoolingFunction` ->
    `PredictionHead`.

    Parameters
    ----------
    args : argparse.Namespace
        Network configuration. Only the attributes listed in `ARG_NAMES`
        are read:

        feature_dim
            Size of the input tile embeddings.

        instance_dim
            Size of the tile embeddings after `InstanceTransform`.

        instance_transf
            Instance transform strategy (see `InstanceTransform`).

        pooling
            Pooling mode (see `PoolingFunction`).

        top_k
            Number of tiles kept by the `top_k` pooling.

        attention_dim
            Hidden size of the attention scoring layer.

        num_heads
            Number of heads of the attention pooling and of the transformer
            instance transforms.

        width_fe
            Hidden layer sizes of the `PredictionHead`.

        dropout
            Dropout probability of the attention scoring layer. The instance
            transform and prediction head use a fixed dropout of 0.1.

    output_dim : int
        Number of outputs (classes, regression targets, or 1 for survival).
    """

    ARG_NAMES = {
        "feature_dim",
        "instance_dim",
        "instance_transf",
        "pooling",
        "top_k",
        "attention_dim",
        "num_heads",
        "width_fe",
        "dropout",
    }

    def __init__(
        self,
        args: Namespace,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.feature_dim = args.feature_dim
        self.instance_dim = args.instance_dim
        self.output_dim = output_dim

        self.pooling = args.pooling
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        self.instance_transform = InstanceTransform(
            strategy=args.instance_transf,
            input_dim=self.feature_dim,
            output_dim=self.instance_dim,
            dropout=0.1,
            num_heads=args.num_heads,
        )

        self.pooling_layer = PoolingFunction(
            mode=args.pooling,
            input_dim=self.instance_dim,
            atn_dim=args.attention_dim,
            n_heads=args.num_heads,
            dropout=args.dropout,
            top_k=args.top_k,
        )

        pooled_dim = self._get_pooled_dim()

        self.prediction_layer = PredictionHead(
            input_dim=pooled_dim,
            hidden_dims=args.width_fe,
            output_dim=self.output_dim,
            dropout=0.1,
        )

    def _get_pooled_dim(self) -> int:
        """
        Size of the slide embedding returned by the pooling layer.

        Returns
        -------
        int
            `instance_dim * num_heads` for the attention-based pooling modes,
            `instance_dim` otherwise.
        """
        if self.pooling in {
            "attention",
            "gated_attention",
            "top_k",
        }:
            return self.instance_dim * self.num_heads

        return self.instance_dim

    def forward(
        self,
        x: Tensor,
        coords: Tensor | None = None,
    ) -> Tensor:
        """
        Predict slide-level outputs from a bag of tiles.

        Parameters
        ----------
        x : torch.Tensor
            Tile embeddings with shape [B, N, feature_dim].

        coords : torch.Tensor, optional
            Normalized tile coordinates with shape [B, N, 2]. Required by
            the `roformer` and `rposbias` instance transforms.

        Returns
        -------
        torch.Tensor
            Raw task predictions with shape [B, output_dim].
        """
        x = self.instance_transform(
            x,
            coords=coords,
        )

        x = self.pooling_layer(x)

        return self.prediction_layer(x)


class IBMIL(Module):
    """IBMIL.

    Instance-based multiple-instance learning.

    Each tile is transformed and mapped to its own task outputs, which are
    then pooled into a slide prediction: `InstanceTransform` ->
    `PredictionHead` -> `PoolingFunction`.

    Parameters
    ----------
    args : argparse.Namespace
        Network configuration. Only the attributes listed in `ARG_NAMES`
        are read:

        feature_dim
            Size of the input tile embeddings.

        instance_dim
            Size of the tile embeddings after `InstanceTransform`.

        instance_transf
            Instance transform strategy (see `InstanceTransform`).

        pooling
            Pooling mode (see `PoolingFunction`).

        top_k
            Number of tiles kept by the `top_k` pooling.

        attention_dim
            Hidden size of the attention scoring layer.

        num_heads
            Number of heads of the attention pooling and of the transformer
            instance transforms.

        width_fe
            Hidden layer sizes of the `PredictionHead`.

        dropout
            Dropout probability of the attention scoring layer. The instance
            transform and prediction head use a fixed dropout of 0.1.

    output_dim : int
        Number of outputs (classes, regression targets, or 1 for survival).
    """

    ARG_NAMES = {
        "feature_dim",
        "instance_dim",
        "instance_transf",
        "pooling",
        "top_k",
        "attention_dim",
        "num_heads",
        "width_fe",
        "dropout",
    }

    def __init__(
        self,
        args: Namespace,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.feature_dim = args.feature_dim
        self.instance_dim = args.instance_dim
        self.output_dim = output_dim
        self.dropout = args.dropout

        self.instance_transform = InstanceTransform(
            strategy=args.instance_transf,
            input_dim=self.feature_dim,
            output_dim=self.instance_dim,
            dropout=0.1,
            num_heads=args.num_heads,
        )

        self.prediction_layer = PredictionHead(
            input_dim=self.instance_dim,
            hidden_dims=args.width_fe,
            output_dim=self.output_dim,
            dropout=0.1,
        )

        self.pooling_layer = PoolingFunction(
            mode=args.pooling,
            input_dim=self.output_dim,
            atn_dim=args.attention_dim,
            n_heads=args.num_heads,
            dropout=args.dropout,
            top_k=args.top_k,
        )

    def forward(
        self,
        x: Tensor,
        coords: Tensor | None = None,
    ) -> Tensor:
        """
        Predict slide-level outputs from a bag of tiles.

        Parameters
        ----------
        x : torch.Tensor
            Tile embeddings with shape [B, N, feature_dim].

        coords : torch.Tensor, optional
            Normalized tile coordinates with shape [B, N, 2]. Required by
            the `roformer` and `rposbias` instance transforms.

        Returns
        -------
        torch.Tensor
            Raw task predictions with shape [B, output_dim].
        """
        x = self.instance_transform(
            x,
            coords=coords,
        )

        x = self.prediction_layer(x)

        return self.pooling_layer(x)


MIL_NETWORKS = {
    "abmil": ABMIL,
    "ibmil": IBMIL,
}


class CustomMIL(Module):
    """
    Unified interface around the available MIL architectures.

    Picks the network class from `MIL_NETWORKS[args.model]` and passes it
    only the arguments listed in its `ARG_NAMES`. Input shapes are checked
    before the forward pass.

    Parameters
    ----------
    args : argparse.Namespace
        Runtime configuration. Must contain `model` and every attribute in
        the `ARG_NAMES` of the selected network.

    output_dim : int
        Number of outputs (classes, regression targets, or 1 for survival).

    Raises
    ------
    ValueError
        If `args.model` is not a key of `MIL_NETWORKS`, or if an argument
        required by the network is missing.
    """

    def __init__(
        self,
        args: Namespace,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.name = args.model

        if self.name not in MIL_NETWORKS:
            raise ValueError(
                f"Unknown MIL architecture '{self.name}'. "
                f"Supported architectures: "
                f"{sorted(MIL_NETWORKS)}."
            )

        self.network_args = self._fetch_network_args(args)
        self.network = MIL_NETWORKS[self.name](
            args=self.network_args,
            output_dim=output_dim,
        )
        self.trainable = [
            *self.network.parameters(),
        ]

    def _fetch_network_args(
        self,
        args: Namespace,
    ) -> Namespace:
        """
        Keep only the arguments required by the selected network.

        Parameters
        ----------
        args : argparse.Namespace
            Full runtime configuration.

        Returns
        -------
        argparse.Namespace
            Namespace restricted to the network `ARG_NAMES`.

        Raises
        ------
        ValueError
            If one of the `ARG_NAMES` is missing from `args`.
        """
        network_cls = MIL_NETWORKS[self.name]

        missing = [name for name in network_cls.ARG_NAMES if not hasattr(args, name)]

        if missing:
            raise ValueError(
                f"Missing arguments for model '{self.name}': " f"{sorted(missing)}."
            )

        return Namespace(
            **{name: getattr(args, name) for name in network_cls.ARG_NAMES}
        )

    def forward(
        self,
        x: Tensor,
        coords: Tensor | None = None,
    ) -> Tensor:
        """
        Check input shapes and run the selected network.

        Parameters
        ----------
        x : torch.Tensor
            Tile embeddings with shape [B, N, F].

        coords : torch.Tensor, optional
            Normalized tile coordinates with shape [B, N, 2].

        Returns
        -------
        torch.Tensor
            Raw task predictions with shape [B, output_dim].

        Raises
        ------
        ValueError
            If `x` is not 3D, or if `coords` does not match [B, N, 2].
        """
        if x.ndim != 3:
            raise ValueError(
                "`x` must have shape [B, N, F], " f"received {tuple(x.shape)}."
            )

        if coords is not None:
            if coords.ndim != 3 or coords.shape[:2] != x.shape[:2]:
                raise ValueError(
                    "`coords` must have shape [B, N, 2] and "
                    "match the batch and tile dimensions of `x`. "
                    f"Received x={tuple(x.shape)}, "
                    f"coords={tuple(coords.shape)}."
                )

            if coords.shape[-1] != 2:
                raise ValueError("The last coordinate dimension must be 2.")

        return self.network(
            x,
            coords=coords,
        )

    def print_summary(
        self,
        depth: int = 4,
        verbose: int = 1,
    ) -> None:
        """
        Print a torchinfo summary of the network.

        Parameters
        ----------
        depth : int, default=4
            Depth of nested modules to display.

        verbose : int, default=1
            torchinfo verbosity level.
        """
        summary(
            self.network,
            depth=depth,
            verbose=verbose,
        )
