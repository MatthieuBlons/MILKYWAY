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
    Fully connected block with optional LayerNorm, GELU, and dropout.
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
        return self.block(inputs)


class PredictionHead(Module):
    """
    Multilayer perceptron producing raw task outputs.

    No final activation is applied. Classification logits, regression
    predictions, and survival risk scores are interpreted by the model
    class according to the selected task.
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
        return self.network(features)


class MultiHeadAttention(Module):
    """MultiHeadedAttentionMIL.

    Implements the multihead attention module.

    Parameters
    ----------
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
        if not self.head_dim * self.n_heads == atn_dim:
            raise ValueError("atn_dim must be divisible by n_heads")

        self.atn_layer_1_weights = Parameter(torch.Tensor(self.atn_dim, self.input_dim))
        self.atn_layer_2_weights = Parameter(
            torch.Tensor(1, 1, self.n_heads, self.head_dim, 1)
        )
        self.atn_layer_1_bias = Parameter(torch.empty((self.atn_dim)))
        self.atn_layer_2_bias = Parameter(torch.empty((1, self.n_heads, 1, 1)))

        self._init_weights()

        self.dropout = dropout

    def _init_weights(self):
        xavier_uniform_(self.atn_layer_1_weights)
        xavier_uniform_(self.atn_layer_2_weights)
        constant_(self.atn_layer_1_bias, 0)
        constant_(self.atn_layer_2_bias, 0)

    def forward(self, x):
        """Extracts a series of attention scores.

        Parameters
        ----------
            x: torch.Tensor size (batch, nb_tiles, features)

        Returns
        ----------
            x: torch.Tensor: size (batch, nb_tiles, nb_heads)
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
    """MultiHeadGatedAttention.

    Implements the gated multihead attention module.

    Parameters
    ----------
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
        if not self.head_dim * self.n_heads == atn_dim:
            raise ValueError("atn_dim must be divisible by n_heads")

        self.V = Parameter(torch.Tensor(self.atn_dim, self.input_dim))
        self.V_bias = Parameter(torch.empty((self.atn_dim)))
        self.U = Parameter(torch.Tensor(self.atn_dim, self.input_dim))
        self.U_bias = Parameter(torch.empty((self.atn_dim)))
        self.W = Parameter(torch.Tensor(1, 1, self.n_heads, self.head_dim, 1))
        self.W_bias = Parameter(torch.empty((1, self.n_heads, 1, 1)))

        self._init_weights()

        self.dropout = dropout

    def _init_weights(self):
        xavier_uniform_(self.U)
        xavier_uniform_(self.V)
        xavier_uniform_(self.W)
        constant_(self.U_bias, 0)
        constant_(self.V_bias, 0)
        constant_(self.W_bias, 0)

    def forward(self, x):
        """Extracts a series of attention scores.

        Parameters
        ----------
            x: torch.Tensor size (batch, nb_tiles, features)

        Returns
        ----------
            x: torch.Tensor: size (batch, nb_tiles, nb_heads)
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
    """PoolingFunction."""

    def __init__(self, mode: str = "mean", **kwargs):
        super(PoolingFunction, self).__init__()

        self.pooling = mode
        self.attention = None

        if self.pooling in ["attention", "max_attention", "top_k"]:
            self.attention = Sequential(MultiHeadAttention(**kwargs), Softmax(dim=-2))

        elif self.pooling in ["gated_attention"]:
            self.attention = Sequential(
                MultiHeadGatedAttention(**kwargs), Softmax(dim=-2)
            )

    def forward(self, x):
        """Pool slide-level representation.

        Parameters
        ----------
            x: torch.Tensor size (batch, nb_tiles, features)

        Returns
        ----------
            slide: torch.Tensor: size (batch, features)
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

        if self.pooling == "max_attention":
            w = self.attention(x)  # (bs, nbt, nheads)
            _, inds = torch.max(w, dim=-2)
            slide = torch.gather(
                x, 1, torch.cat([inds.unsqueeze(-1)] * self.instance_dim, axis=-1)
            )
            slide = slide.squeeze(-2)
            return slide

        if self.pooling == "top_k":
            return None


class InstanceTransform(Module):
    """InstanceTransform.

    Transform tile embeddings before MIL aggregation.
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
        pos: (B, N)
        returns: (B, N, dim)
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
        x: (B, N, D)
        pos: (B, N, dim)  (pixel positions)
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
    def __init__(
        self,
        feature_dim,
        hidden_dim=128,
    ):
        """
        dim: embedding dimension
        hidden_dim: size of hidden layer in MLP
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = Sequential(
            Linear(2, hidden_dim), GELU(), Linear(hidden_dim, feature_dim)
        )

    def forward(self, x, pos):
        """
        x: (B, N, D)
        pos: (B, N, 2)  (pixel coordinates)
        """
        pe = self.mlp(pos)  # (B, N, D)
        return x + pe


class PosEncoding(Module):
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
            self.position = self._get_encoding_strategy(self.strategy, *args, **kwargs)(
                self.feature_dim
            )

    def _get_encoding_strategy(self, enc_strategy: str, *args, **kwargs) -> callable:
        if enc_strategy.lower() in PosEncoding.pos_encoding_mapping.keys():
            return PosEncoding.pos_encoding_mapping[enc_strategy.lower()](
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
        if self.strategy is None:
            return x

        if coords is None:
            raise ValueError("Coords are required for positional encoding")

        return self.position(x, coords)


class RoPE(Module):
    def __init__(self, feature_dim, freq=10000):
        """ """
        super().__init__()
        assert feature_dim % 2 == 0, "RoPE requires even dimension"

        self.feature_dim = feature_dim
        self.freq = freq

        inv_freq = 1.0 / (
            freq ** (torch.arange(0, feature_dim, 2).float() / feature_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _get_angles(self, coords):
        """ """
        # simple 2D → 1D projection (sum works well in practice)
        pos = coords[..., 0] + coords[..., 1]  # (B, N)

        freqs = torch.einsum("bn,d->bnd", pos, self.inv_freq)
        return freqs

    def _rotate_half(self, x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, q, k, coords):
        """ """
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
    def __init__(self, num_heads, hidden_dim=128):
        super().__init__()
        self.num_heads = num_heads

        self.mlp = Sequential(
            Linear(2, hidden_dim), GELU(), Linear(hidden_dim, num_heads)
        )

    def forward(self, coords):
        """
        coords: (B, N, 2)

        returns:
            bias: (B, num_heads, N, N)
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
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()

        # project to 2 * hidden_dim (for gating)
        self.fc1 = Linear(dim, hidden_dim * 2)

        self.act = SiLU()

        self.fc2 = Linear(hidden_dim, dim)
        self.drop = Dropout(drop)

    def forward(self, x):
        x_proj = self.fc1(x)  # (B, N, 2 * hidden_dim)

        x, gate = x_proj.chunk(2, dim=-1)  # split

        x = x * self.act(gate)  # gating

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class TransLayer(Module):
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
