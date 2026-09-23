"""
Implements Useful Custom layers that can be used to build MIL
networks.

"""

from torch.nn import (
    Linear,
    Module,
    Sequential,
    Softmax,
    Identity,
    GELU,
    SiLU,
    Dropout,
    LayerNorm,
)
from torch.nn.parameter import Parameter
import torch
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
import math
from torch import Tensor


class LinearBlock(Module):
    """
    Fully connected block: Linear -> LayerNorm (optional) -> GELU -> Dropout.

    Parameters
    ----------
    input_dim : int
        Size of the input features.

    output_dim : int
        Size of the output features.

    dropout : float
        Dropout probability applied after the activation.

    use_norm : bool, default=True
        If True, apply LayerNorm after the linear layer. Otherwise use an
        identity.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        self.block = Sequential(
            Linear(input_dim, output_dim),
            (
                LayerNorm(output_dim) if use_norm else Identity()
            ),  # make sure is the right way to normalize
            GELU(),
            Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Apply the block to the last dimension of the input.

        Parameters
        ----------
        inputs : torch.Tensor
            Input features with shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Output features with shape (..., output_dim).
        """
        return self.block(inputs)


class PredictionHead(Module):
    """
    Multilayer perceptron producing outputs.

    No final activation is applied. Classification logits, regression
    predictions, and survival risk scores are interpreted by the model
    class according to the selected task.

    Parameters
    ----------
    input_dim : int
        Size of the slide-level or tile-level representation.

    hidden_dims : list[int]
        Sizes of the hidden layers, each built as a `LinearBlock`. An empty
        list gives a single linear layer connected to the output.

    output_dim : int
        Size of the output (classes, regression targets, or 1 for survival).

    dropout : float
        Dropout probability used in every hidden block.

    use_norm : bool, default=True
        If True, apply LayerNorm in every hidden block.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(
                LinearBlock(
                    input_dim=current_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    use_norm=use_norm,
                )
            )
            current_dim = hidden_dim

        layers.append(
            Linear(
                current_dim,
                output_dim,
            )
        )

        self.network = Sequential(*layers)

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        """
        Map input features to raw task outputs.

        Parameters
        ----------
        features : torch.Tensor
            Input features with shape (B, input_dim).

        Returns
        -------
        torch.Tensor
            Raw logits or scores with shape (B, output_dim).
        """
        return self.network(features)


class MultiHeadAttention(Module):
    """
    Multi-head attention scoring for MIL (Ilse et al., 2018).

    Each tile embedding is projected to `atn_dim`, passed through tanh, and
    split into `n_heads` chunks of size `atn_dim // n_heads`. Each head
    scores its chunk with its own linear layer, which gives one score per
    tile and head. The scores are not normalised. The softmax over tiles is
    applied in `PoolingFunction`.

    Parameters
    ----------
    input_dim : int
        Size of the tile embeddings.

    atn_dim : int, default=256
        Size of the hidden attention layer.

    n_heads : int, default=1
        Number of attention heads.

    dropout : float, default=0.3
        Dropout probability applied to the hidden attention layer.

    **kwargs
        Ignored. Lets callers pass a shared argument dict.

    """

    def __init__(
        self,
        input_dim: int,
        atn_dim: int = 256,
        n_heads: int = 1,
        dropout: float = 0.3,
        **kwargs,
    ):
        super(MultiHeadAttention, self).__init__()

        self.input_dim = input_dim
        self.atn_dim = atn_dim
        self.n_heads = n_heads

        self.head_dim = self.atn_dim // self.n_heads

        self.atn_layer_1_weights = Parameter(torch.Tensor(self.atn_dim, self.input_dim))
        self.atn_layer_2_weights = Parameter(
            torch.Tensor(1, 1, self.n_heads, self.head_dim, 1)
        )
        self.atn_layer_1_bias = Parameter(torch.empty((self.atn_dim)))
        self.atn_layer_2_bias = Parameter(torch.empty((1, self.n_heads, 1, 1)))

        self._init_weights()

        self.dropout = dropout

    def _init_weights(self):
        """Xavier-uniform init for the weights, zeros for the biases."""
        xavier_uniform_(self.atn_layer_1_weights)
        xavier_uniform_(self.atn_layer_2_weights)
        constant_(self.atn_layer_1_bias, 0)
        constant_(self.atn_layer_2_bias, 0)

    def forward(self, x):
        """
        Compute unnormalised attention scores for every tile.

        Parameters
        ----------
        x : torch.Tensor
            Tile embeddings with shape (B, N, input_dim).

        Returns
        -------
        torch.Tensor
            Attention scores with shape (B, N, n_heads).
        """
        bs, nbt, _ = x.shape

        # Weights extraction
        x = F.linear(x, weight=self.atn_layer_1_weights, bias=self.atn_layer_1_bias)
        x = torch.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view((bs, nbt, self.n_heads, 1, self.head_dim))
        x = torch.matmul(x, self.atn_layer_2_weights) + self.atn_layer_2_bias  # scores.
        x = x.view(bs, nbt, -1)  # shape (bs, nbt, nheads)
        return x


class MultiHeadGatedAttention(Module):
    """
    Gated multi-head attention scoring for MIL (Ilse et al., 2018).

    Same as `MultiHeadAttention`, but the hidden layer is the product of a
    tanh branch (`V`) and a sigmoid gate (`U`). Each head then scores its
    `atn_dim // n_heads` chunk with its own linear layer (`W`). The scores
    are not normalised. The softmax over tiles is applied in
    `PoolingFunction`.

    Parameters
    ----------
    input_dim : int
        Size of the tile embeddings.

    atn_dim : int, default=256
        Size of the hidden attention layer.

    n_heads : int, default=1
        Number of attention heads.

    dropout : float, default=0.3
        Dropout probability applied to the gated hidden layer.

    **kwargs
        Ignored. Lets callers pass a shared argument dict.

    """

    def __init__(
        self,
        input_dim: int,
        atn_dim: int = 256,
        n_heads: int = 1,
        dropout: float = 0.3,
        **kwargs,
    ):
        super(MultiHeadGatedAttention, self).__init__()

        self.input_dim = input_dim
        self.atn_dim = atn_dim
        self.n_heads = n_heads

        self.head_dim = self.atn_dim // self.n_heads

        self.V = Parameter(torch.Tensor(self.atn_dim, self.input_dim))
        self.V_bias = Parameter(torch.empty((self.atn_dim)))
        self.U = Parameter(torch.Tensor(self.atn_dim, self.input_dim))
        self.U_bias = Parameter(torch.empty((self.atn_dim)))
        self.W = Parameter(torch.Tensor(1, 1, self.n_heads, self.head_dim, 1))
        self.W_bias = Parameter(torch.empty((1, self.n_heads, 1, 1)))

        self._init_weights()

        self.dropout = dropout

    def _init_weights(self):
        """Xavier-uniform init for the weights, zeros for the biases."""
        xavier_uniform_(self.U)
        xavier_uniform_(self.V)
        xavier_uniform_(self.W)
        constant_(self.U_bias, 0)
        constant_(self.V_bias, 0)
        constant_(self.W_bias, 0)

    def forward(self, x):
        """
        Compute unnormalised attention scores for every tile.

        Parameters
        ----------
        x : torch.Tensor
            Tile embeddings with shape (B, N, input_dim).

        Returns
        -------
        torch.Tensor
            Attention scores with shape (B, N, n_heads).
        """
        bs, nbt, _ = x.shape
        # Weights extraction
        v = F.linear(x, weight=self.V, bias=self.V_bias)
        v = torch.tanh(v)
        u = F.linear(x, weight=self.U, bias=self.U_bias)
        u = torch.sigmoid(u)
        x = v * u
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view((bs, nbt, self.n_heads, 1, self.head_dim))
        x = torch.matmul(x, self.W) + self.W_bias  # scores.
        x = x.view(bs, nbt, -1)  # shape (bs, nbt, nheads)
        return x


class PoolingFunction(Module):
    """PoolingFunction.

    Aggregate tile representations into a slide-level representation.

    Supported
    ---------
    mean, max
        Parameter-free pooling over all tiles.
    attention, gated_attention
        Attention-weighted sum of all tiles, one per head.
    top_k
        Attention-weighted sum of the `top_k` highest-scoring tiles, one per
        head. Attention weights are renormalized over the selected tiles.

    Parameters
    ----------
    mode : str, default="mean"
        Pooling strategy, one of the options above.

    **kwargs
        `top_k` (int) is read for the `top_k` mode. The remaining keywords
        are forwarded to `MultiHeadAttention` or `MultiHeadGatedAttention`
        (`input_dim`, `atn_dim`, `n_heads`, `dropout`).
    """

    def __init__(self, mode: str = "mean", **kwargs):
        super(PoolingFunction, self).__init__()

        self.pooling = mode
        self.attention = None
        self.top_k = kwargs.pop("top_k", None)
        if self.pooling in ["attention", "top_k"]:
            self.attention = Sequential(MultiHeadAttention(**kwargs), Softmax(dim=-2))

        elif self.pooling in ["gated_attention"]:
            self.attention = Sequential(
                MultiHeadGatedAttention(**kwargs), Softmax(dim=-2)
            )

    def forward(self, x):
        """
        Pool tile embeddings into a slide-level representation.

        Parameters
        ----------
        x : torch.Tensor
            Tile embeddings with shape (B, N, F).

        Returns
        -------
        torch.Tensor
            Slide representation with shape (B, F) for `mean` and `max`, or
            (B, n_heads * F) for `attention`, `gated_attention` and `top_k`.
        """
        if self.pooling == "mean":
            return torch.mean(x, -2)  # (bs, nfeatures)

        if self.pooling == "max":
            slide, _ = torch.max(x, dim=-2)  # (bs, nfeatures)
            return slide

        if self.pooling in ["attention", "gated_attention"]:
            w = self.attention(x)  # (bs, nbt, nheads)
            w = torch.transpose(w, -1, -2)  # (bs, nheads, nbt)
            slide = torch.matmul(w, x)  # (bs, nheads, nfeatures)
            slide = slide.flatten(1, -1)  # (bs, nheads*nfeatures)
            return slide

        if self.pooling == "top_k":
            w = self.attention(x)  # (bs, nbt, nheads)
            k = min(self.top_k, w.shape[-2])
            top_w, top_inds = torch.topk(w, k, dim=-2)  # (bs, k, nheads)
            top_w = top_w / top_w.sum(
                dim=-2, keepdim=True
            )  # normalize selected weights
            top_w = torch.transpose(top_w, -1, -2)  # (bs, nheads, k)
            top_inds = torch.transpose(top_inds, -1, -2)  # (bs, nheads, k)
            top_x = torch.gather(
                x.unsqueeze(1).expand(-1, top_inds.shape[1], -1, -1),
                2,
                top_inds.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]),
            )  # (bs, nheads, k, nfeatures)
            slide = torch.matmul(top_w.unsqueeze(-2), top_x)  # (bs, nheads, 1, nf)
            slide = slide.flatten(1, -1)  # (bs, nheads*nfeatures)
            return slide


class InstanceTransform(Module):
    """InstanceTransform.

    Transform tile embeddings before MIL aggregation.

    Supported
    ---------
    identity
        No transform. Requires `input_dim == output_dim`.
    linear
        One `LinearBlock` applied to each tile independently.
    transformer
        One `TransLayer` (self-attention across tiles, no positional
        information).
    roformer
        `TransLayer` with rotary position embeddings from tile coordinates.
    rposbias
        `TransLayer` with a learned relative position bias from tile
        coordinates.

    Parameters
    ----------
    strategy : str
        Transform to apply, one of the options above.

    input_dim : int
        Size of the input tile embeddings.

    output_dim : int
        Size of the output tile embeddings. Must equal `input_dim` for
        every strategy except `linear`.

    dropout : float, default=0.1
        Dropout probability of the linear block or transformer layer.

    num_heads : int, default=1
        Number of self-attention heads (transformer only).

    mlp_dim : int, default=512
        Hidden size of the transformer gated MLP (transformer only).

    Raises
    ------
    ValueError
        If `strategy` is unknown, or if `input_dim != output_dim` for a
        strategy that requires them to match.
    """

    COORD_TRANSFORMS = {
        "roformer",
        "rposbias",
    }

    def __init__(
        self,
        strategy: str,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        num_heads: int = 1,
        mlp_dim: int = 512,
    ) -> None:
        super().__init__()

        self.strategy = strategy
        self.requires_coords = strategy in self.COORD_TRANSFORMS

        if strategy == "identity":
            if input_dim != output_dim:
                raise ValueError("identity transform assumes input_dim == output_dim.")

            self.transform = Identity()

        elif strategy == "linear":
            self.transform = LinearBlock(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=dropout,
            )

        elif strategy in [
            "transformer",
            "roformer",
            "rposbias",
        ]:
            if input_dim != output_dim:
                raise ValueError(
                    "Transformer instance transforms currently require "
                    "input_dim == output_dim."
                )

            self.transform = TransLayer(
                hidden_dim=output_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                drop=dropout,
                rope=strategy == "roformer",
                rpb=strategy == "rposbias",
            )

        else:
            raise ValueError(f"Unknown instance transform: {strategy}")

    def forward(
        self,
        x: Tensor,
        coords: Tensor | None = None,
    ) -> Tensor:
        """
        Transform tile embedding.

        Parameters
        ----------
        x : torch.Tensor
            Tile embeddings with shape (B, N, input_dim).

        coords : torch.Tensor or None, default=None
            Tile coordinates with shape (B, N, 2). Required for `roformer`
            and `rposbias`, ignored otherwise.

        Returns
        -------
        torch.Tensor
            Transformed embeddings with shape (B, N, output_dim).

        Raises
        ------
        ValueError
            If the strategy needs coordinates and `coords` is None.
        """
        if self.requires_coords and coords is None:
            raise ValueError(
                f"Coordinates are required for "
                f"instance_transform='{self.strategy}'."
            )

        if self.strategy in {
            "identity",
            "linear",
        }:
            return self.transform(x)

        return self.transform(
            x,
            coords=coords,
        )


# transformers related:
class SinCosEncoding(Module):
    """
    Fixed sinusoidal positional encoding added to tile embeddings.

    With `dim=2`, half of the features encode the x coordinate and the
    other half encode the y coordinate. Otherwise, one scalar position per
    tile is encoded over all features.

    Parameters
    ----------
    feature_dim : int
        Size of the tile embeddings. Must be divisible by `dim`.

    dim : int, default=2
        Number of spatial dimensions of the positions.

    freq : float, default=10000
        Base of the geometric progression of wavelengths.

    Raises
    ------
    ValueError
        If `feature_dim` is not divisible by `dim`.
    """

    def __init__(
        self,
        feature_dim,
        dim=2,
        freq=10000,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.dim = dim
        if not self.feature_dim % self.dim == 0:
            raise ValueError(f"SinCosEncoding requires // {self.dim}")
        self.freq = freq

    def sincos(self, pos, feature_dim, freq):
        """
        Sinusoidal encoding of one scalar position per tile.

        Parameters
        ----------
        pos : torch.Tensor
            Positions with shape (B, N).

        feature_dim : int
            Size of the encoding. Must be even.

        freq : float
            Base of the geometric progression of wavelengths.

        Returns
        -------
        torch.Tensor
            Encoding with shape (B, N, feature_dim). Even indices hold sines,
            odd indices hold cosines.
        """
        device = pos.device
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2, device=device)
            * (-math.log(freq) / feature_dim)
        )  # (dim/2,)
        pos = pos.unsqueeze(-1)  # (B, N, 2)
        pe = torch.zeros(*pos.shape[:-1], feature_dim, device=device)  # (B, N, dim)
        pe[..., 0::2] = torch.sin(pos * div_term)
        pe[..., 1::2] = torch.cos(pos * div_term)
        return pe

    def forward(self, x, pos):
        """
        Add the positional encoding to the tile embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Tile embeddings with shape (B, N, feature_dim).

        pos : torch.Tensor
            Tile positions with shape (B, N, 2) when `dim=2`, or (B, N)
            otherwise.

        Returns
        -------
        torch.Tensor
            Encoded embeddings with shape (B, N, feature_dim).
        """
        if self.dim == 2:
            x_pos = pos[..., 0]  # (B, N)
            y_pos = pos[..., 1]  # (B, N)
            pe_x = self.sincos(x_pos, self.feature_dim // 2, self.freq)
            pe_y = self.sincos(y_pos, self.feature_dim // 2, self.freq)
            pe = torch.cat([pe_x, pe_y], dim=-1)  # (B, N, D)
            return x + pe
        else:
            pe = self.sincos(pos, self.feature_dim, self.freq)  # (B, N, D)
            return x + pe


class LearnedPosEncoding(Module):
    """
    Learned positional encoding: a 2-layer MLP maps each (x, y) coordinate
    to a vector that is added to the tile embedding.

    Parameters
    ----------
    feature_dim : int
        Size of the tile embeddings.

    hidden_dim : int, default=128
        Hidden size of the MLP.
    """

    def __init__(
        self,
        feature_dim,
        hidden_dim=128,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = Sequential(
            Linear(2, hidden_dim), GELU(), Linear(hidden_dim, feature_dim)
        )

    def forward(self, x, pos):
        """
        Add the learned positional encoding to the tile embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Tile embeddings with shape (B, N, feature_dim).

        pos : torch.Tensor
            Tile coordinates with shape (B, N, 2).

        Returns
        -------
        torch.Tensor
            Encoded embeddings with shape (B, N, feature_dim).
        """
        pe = self.mlp(pos)  # (B, N, D)
        return x + pe


class PosEncoding(Module):
    """
    Absolute positional encoding, selected by name.

    Supported
    ---------
    None
        No encoding. The input is returned unchanged.
    sincos
        `SinCosEncoding`.
    learned
        `LearnedPosEncoding`.

    Parameters
    ----------
    feature_dim : int
        Size of the tile embeddings. Must be even when `strategy` is set.

    strategy : str or None, default=None
        Encoding strategy, one of the options above (case-insensitive).

    *args, **kwargs
        Forwarded to the encoding class.
    """

    pos_encoding_mapping = {
        "sincos": SinCosEncoding,
        "learned": LearnedPosEncoding,
        # "relative": RelativePosEncoding,
    }

    def __init__(self, feature_dim, strategy=None, *args, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.strategy = strategy

        if self.strategy is not None:
            assert feature_dim % 2 == 0, "PosEncoding requires even dimension"
            self.position = self._get_encoding_strategy(self.strategy, self.feature_dim, *args, **kwargs)

    def _get_encoding_strategy(self, enc_strategy: str, feature_dim: int, *args, **kwargs) -> callable:
        """
        Instantiate the encoding module registered under `enc_strategy`.

        Parameters
        ----------
        enc_strategy : str
            Key of `pos_encoding_mapping` (case-insensitive).

        *args, **kwargs
            Forwarded to the encoding class.

        Returns
        -------
        torch.nn.Module
            Positional encoding module.

        Raises
        ------
        NotImplementedError
            If `enc_strategy` is not a key of `pos_encoding_mapping`.
        """
        if enc_strategy.lower() in PosEncoding.pos_encoding_mapping.keys():
            return PosEncoding.pos_encoding_mapping[enc_strategy.lower()](
                feature_dim,
                *args,
                **kwargs,
            )
        # Otherwise raise an error
        else:
            raise NotImplementedError(
                "pos encoding strategy: {} is not implemented.".format(enc_strategy)
                + f"\nPlease choose a valid strategy from: {' ,'.join(PosEncoding.pos_encoding_mapping.keys())}."
            )

    def forward(self, x, coords=None):
        """
        Add the positional encoding to the tile embeddings, if any.

        Parameters
        ----------
        x : torch.Tensor
            Tile embeddings with shape (B, N, feature_dim).

        coords : torch.Tensor or None, default=None
            Tile coordinates with shape (B, N, 2). Required when `strategy`
            is set.

        Returns
        -------
        torch.Tensor
            Embeddings with shape (B, N, feature_dim).

        Raises
        ------
        ValueError
            If `strategy` is set and `coords` is None.
        """
        if self.strategy is None:
            return x

        if coords is None:
            raise ValueError("Coords are required for positional encoding")

        return self.position(x, coords)


class RoPE(Module):
    """
    Rotary position embedding (RoPE) for queries and keys.

    The 2D tile coordinates are reduced to one scalar position (x + y).
    Consecutive feature pairs of the queries and keys are then rotated by an
    angle proportional to that position. Attention scores then depend on
    relative positions.

    Parameters
    ----------
    feature_dim : int
        Size of each attention head. Must be even.

    freq : float, default=10000
        Base of the geometric progression of rotation frequencies.
    """

    def __init__(self, feature_dim, freq=10000):
        super().__init__()
        assert feature_dim % 2 == 0, "RoPE requires even dimension"

        self.feature_dim = feature_dim
        self.freq = freq

        inv_freq = 1.0 / (
            freq ** (torch.arange(0, feature_dim, 2).float() / feature_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _get_angles(self, coords):
        """
        Compute the rotation angles of every tile.

        Parameters
        ----------
        coords : torch.Tensor
            Tile coordinates with shape (B, N, 2).

        Returns
        -------
        torch.Tensor
            Angles with shape (B, N, feature_dim // 2).
        """
        # simple 2D → 1D projection (sum works well in practice)
        pos = coords[..., 0] + coords[..., 1]  # (B, N)

        freqs = torch.einsum("bn,d->bnd", pos, self.inv_freq)
        return freqs

    def _rotate_half(self, x):
        """Map each feature pair (x1, x2) to (-x2, x1) along the last axis."""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, q, k, coords):
        """
        Rotate queries and keys according to the tile positions.

        Parameters
        ----------
        q, k : torch.Tensor
            Queries and keys with shape (B, H, N, feature_dim).

        coords : torch.Tensor
            Tile coordinates with shape (B, N, 2).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Rotated queries and keys, same shape as the inputs.
        """
        freqs = self._get_angles(coords)

        cos = torch.cos(freqs).unsqueeze(1)
        sin = torch.sin(freqs).unsqueeze(1)

        # expand to full dim
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot


class RelativePositionBias(Module):
    """
    Learned relative position bias for self-attention.

    A 2-layer MLP maps the offset (dx, dy) between every pair of tiles to
    one bias per head. The bias is added to the attention logits.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.

    hidden_dim : int, default=128
        Hidden size of the MLP.

    Notes
    -----
    Memory grows as N^2 with the number of tiles, since the MLP is applied
    to all N x N pairs.
    """

    def __init__(self, num_heads, hidden_dim=128):
        super().__init__()
        self.num_heads = num_heads

        self.mlp = Sequential(
            Linear(2, hidden_dim), GELU(), Linear(hidden_dim, num_heads)
        )

    def forward(self, coords):
        """
        Compute the attention bias of every pair of tiles.

        Parameters
        ----------
        coords : torch.Tensor
            Tile coordinates with shape (B, N, 2).

        Returns
        -------
        torch.Tensor
            Attention bias with shape (B, num_heads, N, N).
        """
        B, N, _ = coords.shape

        # Compute pairwise relative positions
        dist = coords[:, :, None, :] - coords[:, None, :, :]  # (B, N, N, 2)

        # Flatten for MLP
        dist_flat = dist.view(B * N * N, 2)

        bias = self.mlp(dist_flat)  # (B*N*N, num_heads)

        bias = bias.view(B, N, N, self.num_heads)
        bias = bias.permute(0, 3, 1, 2)  # (B, H, N, N)

        return bias


class GatedMLP(Module):
    """
    Gated feed-forward block (SwiGLU): the input is projected to two
    branches of size `hidden_dim`, one branch is gated by SiLU of the other,
    then the result is projected back to `dim`.

    Parameters
    ----------
    dim : int
        Size of the input and output features.

    hidden_dim : int
        Size of each hidden branch.

    drop : float, default=0.0
        Dropout probability applied after the gating and after the output
        projection.
    """

    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()

        # project to 2 * hidden_dim (for gating)
        self.fc1 = Linear(dim, hidden_dim * 2)

        self.act = SiLU()

        self.fc2 = Linear(hidden_dim, dim)
        self.drop = Dropout(drop)

    def forward(self, x):
        """
        Apply the gated MLP to every tile.

        Parameters
        ----------
        x : torch.Tensor
            Tile embeddings with shape (B, N, dim).

        Returns
        -------
        torch.Tensor
            Output embeddings with shape (B, N, dim).
        """
        x_proj = self.fc1(x)  # (B, N, 2 * hidden_dim)

        x, gate = x_proj.chunk(2, dim=-1)  # split

        x = x * self.act(gate)  # gating

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class TransLayer(Module):
    """
    Pre-norm transformer layer over the tiles of a bag.

    Multi-head self-attention followed by a `GatedMLP`, each with a residual
    connection and optional LayerScale.

    Positional information can be injected with
        `RoPE` (`rope=True`) or
        `RelativePositionBias` (`rpb=True`).

    Parameters
    ----------
    hidden_dim : int, default=128
        Size of the tile embeddings. Should be divisible by `num_heads`.

    num_heads : int, default=1
        Number of self-attention heads.

    mlp_dim : int, default=2048
        Hidden size of the gated MLP.

    drop : float, default=0
        Dropout probability after the attention projection and in the MLP.

    attn_drop : float, default=0
        Dropout probability on the attention weights.

    rope : bool, default=False
        If True, apply rotary position embeddings to queries and keys.

    rpb : bool, default=False
        If True, add a learned relative position bias to the attention
        logits.

    scale : bool, default=True
        If True, scale each residual branch by a learned per-feature vector
        (LayerScale).

    ls_init : float, default=1e-5
        Initial value of the LayerScale vectors.
    """

    def __init__(
        self,
        hidden_dim=128,
        num_heads=1,
        mlp_dim=2048,
        drop=0,
        attn_drop=0,
        rope=False,
        rpb=False,
        scale=True,
        ls_init=1e-5,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Self-Attention
        self.norm1 = LayerNorm(hidden_dim)
        self.qkv = Linear(hidden_dim, hidden_dim * 3)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(hidden_dim, hidden_dim)
        self.proj_drop = Dropout(drop)
        self.ls1 = Parameter(ls_init * torch.ones(hidden_dim)) if scale else None

        # Rope if needed
        self.rope = RoPE(self.head_dim) if rope else None

        # Or Relative Position Bias
        self.rpb = RelativePositionBias(num_heads) if rpb else None

        # Gated MLP
        self.norm2 = LayerNorm(hidden_dim)
        self.mlp = GatedMLP(hidden_dim, mlp_dim, drop)
        self.ls2 = Parameter(ls_init * torch.ones(hidden_dim)) if scale else None

    def forward(self, x, coords=None):
        """
        Apply self-attention and the gated MLP to the bag.

        Parameters
        ----------
        x : torch.Tensor
            Tile embeddings with shape (B, N, hidden_dim).

        coords : torch.Tensor or None, default=None
            Tile coordinates with shape (B, N, 2). Required when `rope` or
            `rpb` is True.

        Returns
        -------
        torch.Tensor
            Output embeddings with shape (B, N, hidden_dim).
        """
        B, N, D = x.shape
        # Attention
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)  # (B, H, N, D_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k, coords)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)  # (B, H, N, N)

        if self.rpb is not None:
            bias = self.rpb(coords)  # (B, H, N, N)
            attn = attn + bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)

        out = self.proj(out)
        out = self.proj_drop(out)

        if self.ls1 is not None:
            x = x + self.ls1 * out
        else:
            x = x + out

        # MLP
        mlp_out = self.mlp(self.norm2(x))

        if self.ls2 is not None:
            x = x + self.ls2 * mlp_out
        else:
            x = x + mlp_out

        return x
