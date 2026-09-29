"""
Forward hooks that expose the intermediate outputs of the DeepMIL networks.

Supported networks
------------------
abmil
    `InstanceTransform` -> `PoolingFunction` -> `PredictionHead`. The head
    maps the slide embedding to the slide outputs.
ibmil
    `InstanceTransform` -> `PredictionHead` -> `PoolingFunction`. The head
    maps every tile to its own outputs, which are then pooled.

Captured attributes
-------------------
Every array keeps the batch dimension.

`None` means that the layer does not exist for this configuration
(e.g. no attention with `mean` pooling).

tiles_transf
    Output of `InstanceTransform`, shape (B, N, instance_dim).
self_attention
    Softmaxed self-attention map of the transformer instance transforms,
    shape (B, heads, N, N). Only filled when `store_self_attention=True`.
tiles_weights
    Attention scores before the softmax over tiles, shape (B, N, n_heads).
tiles_attention
    Attention weights after the softmax over tiles, shape (B, N, n_heads).
head_average
    Output of `PoolingFunction`, one row per head: (B, n_heads, D) for the
    attention modes, (B, 1, D) for `mean` / `max`. D is `instance_dim` for
    ABMIL and `output_dim` for IBMIL.
reprewsi
    Input of the last linear layer of `PredictionHead`. ABMIL: slide
    representation, shape (B, hidden). IBMIL: tile representations, shape
    (B, N, hidden).
tiles_scores, tiles_proba
    IBMIL only. Per-tile raw outputs of `PredictionHead`, shape
    (B, N, output_dim), and their softmax (classification only).
scores, proba
    Slide-level raw outputs, shape (B, output_dim), and their softmax
    (classification only).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

LOGGER = logging.getLogger(__name__)

SUPPORTED_NETWORKS = {
    "ABMIL",
    "IBMIL",
}

TRANSFORMER_STRATEGIES = {
    "transformer",
    "roformer",
    "rposbias",
}


def _to_numpy(tensor: Tensor) -> np.ndarray:
    """Detach a tensor and return it as a numpy array on cpu."""
    return tensor.detach().float().cpu().numpy()


def resolve_mil_network(network: Module) -> Module:
    """
    Find the ABMIL / IBMIL module inside a model wrapper.

    Parameters
    ----------
    network : torch.nn.Module
        A `DeepMIL` model, a `CustomMIL` wrapper, or the ABMIL / IBMIL
        network itself.

    Returns
    -------
    torch.nn.Module
        The innermost network exposing `instance_transform`,
        `pooling_layer` and `prediction_layer`.

    Raises
    ------
    ValueError
        If no supported network is found.
    """
    current = network

    while type(current).__name__ not in SUPPORTED_NETWORKS:
        inner = getattr(current, "network", None)

        if not isinstance(inner, Module):
            raise ValueError(
                f"Could not find a supported MIL network in "
                f"`{type(network).__name__}`. "
                f"Expected one of {sorted(SUPPORTED_NETWORKS)}."
            )

        current = inner

    return current


class HookerMIL:
    """
    Manage the forward hooks of an ABMIL or IBMIL network.

    The hooks fill the attributes listed in the module docstring at every
    forward pass of the network (or of one of its sub-layers). Use it in
    eval mode, otherwise the captured values include dropout.

    Parameters
    ----------
    network : torch.nn.Module
        A `DeepMIL` model, a `CustomMIL` wrapper, or the ABMIL / IBMIL
        network itself.

    num_heads : int or None, default=None
        Kept for backward compatibility. The number of heads is now read
        from the attention layer. When given and different from it (with
        attention pooling only), a `ValueError` is raised.

    task : str, default="classification"
        Task of the model. `proba` and `tiles_proba` are only computed for
        `classification`.

    store_self_attention : bool, default=False
        If True, also capture the (B, heads, N, N) self-attention map of
        the transformer instance transforms. Can be large for big bags.

    Raises
    ------
    ValueError
        If `network` does not contain a supported MIL network.
    """

    def __init__(
        self,
        network: Module,
        num_heads: int | None = None,
        task: str = "classification",
        store_self_attention: bool = False,
    ) -> None:
        self.network = resolve_mil_network(network)
        self.architecture = type(self.network).__name__
        self.task = task
        self.store_self_attention = store_self_attention

        attention = self.network.pooling_layer.attention
        self.num_heads = attention[0].n_heads if attention is not None else 1

        # Without attention pooling, `num_heads` only configures the transformer
        # instance transform, so there is nothing to compare.
        if (
            attention is not None
            and num_heads is not None
            and num_heads != self.num_heads
        ):
            raise ValueError(
                f"`num_heads`={num_heads} differs from the network attention "
                f"heads ({self.num_heads})."
            )

        self.handles: list[RemovableHandle] = []
        self.reset()
        self.place_hooks()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear every captured value."""
        self.tiles_transf: np.ndarray | None = None
        self.self_attention: np.ndarray | None = None
        self.tiles_weights: np.ndarray | None = None
        self.tiles_attention: np.ndarray | None = None
        self.head_average: np.ndarray | None = None
        self.reprewsi: np.ndarray | None = None
        self.tiles_scores: np.ndarray | None = None
        self.tiles_proba: np.ndarray | None = None
        self.scores: np.ndarray | None = None
        self.proba: np.ndarray | None = None

    def _probabilities(self, outputs: Tensor) -> np.ndarray | None:
        """Softmax of the raw outputs for classification, None otherwise."""
        if self.task != "classification":
            return None

        return _to_numpy(torch.softmax(outputs, dim=-1))

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _get_instance_transf_hook(self):
        def hook_instance_transf(module, inputs, output):
            """Hooks the tile embeddings after `InstanceTransform`."""
            self.tiles_transf = _to_numpy(output)

        return hook_instance_transf

    def _get_self_attention_hook(self):
        def hook_self_attention(module, inputs, output):
            """Hooks the softmaxed self-attention map (input of `attn_drop`)."""
            self.self_attention = _to_numpy(inputs[0])

        return hook_self_attention

    def _get_attention_scores_hook(self):
        def hook_attention_scores(module, inputs, output):
            """Hooks the attention scores before the softmax over tiles."""
            self.tiles_weights = _to_numpy(output)

        return hook_attention_scores

    def _get_attention_weights_hook(self):
        def hook_attention_weights(module, inputs, output):
            """Hooks the attention weights after the softmax over tiles."""
            self.tiles_attention = _to_numpy(output)

        return hook_attention_weights

    def _get_pooling_hook(self):
        def hook_pooling(module, inputs, output):
            """Hooks the pooled representation, one row per head."""
            batch_size = output.shape[0]
            self.head_average = _to_numpy(
                output.reshape(
                    batch_size,
                    self.num_heads,
                    -1,
                )
            )

            if self.architecture == "IBMIL":
                self.scores = _to_numpy(output)
                self.proba = self._probabilities(output)

        return hook_pooling

    def _get_representation_hook(self):
        def hook_representation(module, inputs, output):
            """Hooks the input of the last linear layer of `PredictionHead`."""
            self.reprewsi = _to_numpy(inputs[0])

        return hook_representation

    def _get_prediction_hook(self):
        def hook_prediction(module, inputs, output):
            """Hooks the raw outputs of `PredictionHead`."""
            if self.architecture == "ABMIL":
                self.scores = _to_numpy(output)
                self.proba = self._probabilities(output)
            else:
                self.tiles_scores = _to_numpy(output)
                self.tiles_proba = self._probabilities(output)

        return hook_prediction

    def _register(
        self,
        module: Module,
        hook,
    ) -> None:
        """Register a forward hook and keep its handle."""
        self.handles.append(module.register_forward_hook(hook))

    def place_hooks(self) -> None:
        """
        Register the forward hooks on the network layers.

        Notes
        -----
        Hooked modules, shared by ABMIL and IBMIL:

        - `instance_transform`: tile embeddings.
        - `instance_transform.transform.attn_drop`: self-attention map
          (transformer strategies, only if `store_self_attention`).
        - `pooling_layer.attention[0]` / `[1]`: attention scores before /
          after the softmax (attention-based pooling only).
        - `pooling_layer`: pooled representation.
        - `prediction_layer.network[-1]`: last linear layer, its input is
          the representation.
        - `prediction_layer`: raw outputs.
        """
        self.remove_hooks()

        instance_transform = self.network.instance_transform
        self._register(
            instance_transform,
            self._get_instance_transf_hook(),
        )

        if (
            self.store_self_attention
            and instance_transform.strategy in TRANSFORMER_STRATEGIES
        ):
            self._register(
                instance_transform.transform.attn_drop,
                self._get_self_attention_hook(),
            )

        pooling_layer = self.network.pooling_layer
        if pooling_layer.attention is not None:
            scorer, softmax = pooling_layer.attention
            self._register(
                scorer,
                self._get_attention_scores_hook(),
            )
            self._register(
                softmax,
                self._get_attention_weights_hook(),
            )

        self._register(
            pooling_layer,
            self._get_pooling_hook(),
        )

        prediction_layer = self.network.prediction_layer
        self._register(
            prediction_layer.network[-1],
            self._get_representation_hook(),
        )
        self._register(
            prediction_layer,
            self._get_prediction_hook(),
        )

    def remove_hooks(self) -> None:
        """Remove every hook placed by this hooker."""
        for handle in self.handles:
            handle.remove()

        self.handles = []


# Backward-compatible name.
HookerAttnMIL = HookerMIL
