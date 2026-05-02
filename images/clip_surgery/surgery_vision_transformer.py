"""Surgery Vision Transformer (ViT) for SatCLIP.

This file is adapted from `CLIP_Surgery/satclip/satclip/surgery_vision_transformer.py`
and slightly modified to live in the local `clip_surgery` package. The implementation
is otherwise kept identical so that pretrained weights and checkpoints from the
original SatCLIP models remain compatible while changing only the inference behavior.
"""

import copy
import logging
import math
import os
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)

from .modified_attention import ConsistentAttention
from timm.layers import (
    Attention,
    DiffAttention,
    AttentionPoolLatent,
    PatchEmbed,
    Mlp,
    SwiGLUPacked,
    SwiGLU,
    LayerNorm,
    RmsNorm,
    DropPath,
    calculate_drop_path_rates,
    PatchDropout,
    trunc_normal_,
    lecun_normal_,
    resample_patch_embed,
    resample_abs_pos_embed,
    use_fused_attn,
    get_act_layer,
    get_norm_layer,
    maybe_add_mask,
    LayerType,
    LayerScale,
)
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import named_apply, checkpoint, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ["VisionTransformer"]  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)


ATTN_LAYERS = {
    "": Attention,
    "attn": Attention,
    "consistent_attn": ConsistentAttention,
    # "diff": DiffAttention,
}


def _create_attn(
    attn_layer: LayerType,
    dim: int,
    num_heads: int,
    qkv_bias: bool = False,
    qk_norm: bool = False,
    scale_norm: bool = False,
    proj_bias: bool = True,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    norm_layer: Optional[Type[nn.Module]] = None,
    depth: int = 0,
    **kwargs,
) -> nn.Module:
    if isinstance(attn_layer, str):
        attn_layer = ATTN_LAYERS.get(attn_layer, None)
        assert attn_layer is not None, f"Unknown attn_layer: {attn_layer}"

    # Only pass depth to attention layers that use it
    if issubclass(attn_layer, DiffAttention):
        kwargs["depth"] = depth

    return attn_layer(
        dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        scale_norm=scale_norm,
        proj_bias=proj_bias,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
        norm_layer=norm_layer,
        **kwargs,
    )


class Block(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
        attn_layer: LayerType = Attention,
        depth: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}

        self.norm1 = norm_layer(dim, **dd)
        self.attn = _create_attn(
            attn_layer,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            depth=depth,
            **dd,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, **dd) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, **dd)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
            **dd,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, **dd) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(self.attn, ConsistentAttention):
            # custom dual path behavior
            # dual paths for blocks deeper than "d"
            if isinstance(x, list):
                x, x_ori = x
                # LS1 and DROP 1 are not needed for inference and dont know
                x_attn, x_ori_attn = self.drop_path1(self.ls1(self.attn(self.norm1(x_ori), attn_mask=attn_mask)))
                x_ori += x_ori_attn
                x_ori_ffn = self.drop_path2(self.ls2(self.mlp(self.norm2(x_ori))))
                x_ori += x_ori_ffn

                x += x_attn  # skip ffn for the new path
                return [x, x_ori]

            # start of dual path
            else:
                x_attn, x_ori_attn = self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
                x_ori = x + x_ori_attn
                x_ori_ffn = self.drop_path2(self.ls2(self.mlp(self.norm2(x_ori))))
                x_ori += x_ori_ffn
                x += x_attn  # skip ffn for the new path
                return [x, x_ori]
        else:
            # default behavior
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResPostBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
        attn_layer: LayerType = Attention,
        depth: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.init_values = init_values

        self.attn = _create_attn(
            attn_layer,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            depth=depth,
            **dd,
        )
        self.norm1 = norm_layer(dim, **dd)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
            **dd,
        )
        self.norm2 = norm_layer(dim, **dd)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x, attn_mask=attn_mask)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelScalingBlock(nn.Module):
    """Parallel ViT block (MLP & Attention in parallel).

    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        mlp_layer: Optional[Type[nn.Module]] = None,  # not used
        attn_layer: Optional[LayerType] = None,  # not used
        depth: int = 0,  # not used
        fuse_out_proj: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert not scale_attn_norm and not scale_mlp_norm, "Scale norms not supported"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim, **dd)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias, **dd)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            # mlp_bias is combined with qkv_bias in in_proj.bias
            self.register_parameter("mlp_bias", None)
        else:
            self.mlp_bias = nn.Parameter(torch.empty(mlp_hidden_dim, **dd))

        self.q_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()

        if fuse_out_proj:
            # Fused output projection for both attention and MLP
            self.out_proj = nn.Linear(dim + mlp_hidden_dim, dim, bias=proj_bias, **dd)
            self.attn_out_proj = None
            self.mlp_out_proj = None
        else:
            # Separate output projections
            self.out_proj = None
            self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias, **dd)
            self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim, bias=proj_bias, **dd)

        self.ls = LayerScale(dim, init_values=init_values, **dd) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # TODO: skip init when on meta device when safe to do so
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters and buffers."""
        if getattr(self, "mlp_bias", None) is not None:
            nn.init.zeros_(self.mlp_bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)
        if getattr(self, "mlp_bias", None) is not None:
            x_mlp = x_mlp + self.mlp_bias

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.fused_attn:
            x_attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v

        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)

        # MLP activation & dropout
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)

        # Output projection (fused or separate)
        if self.out_proj is not None:
            y = self.out_proj(torch.cat((x_attn, x_mlp), dim=-1))
        else:
            y = self.attn_out_proj(x_attn) + self.mlp_out_proj(x_mlp)

        # Add residual w/ drop path & layer scale applied
        x = x + self.drop_path(self.ls(y))
        return x


class DiffParallelScalingBlock(nn.Module):
    """Parallel ViT block with Differential Attention (MLP & Attention in parallel).

    Combines the parallel MLP+Attention structure from 'Scaling Vision Transformers to
    22 Billion Parameters' (https://arxiv.org/abs/2302.05442) with differential attention
    from 'Differential Transformer' (https://arxiv.org/abs/2410.05258).
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        mlp_layer: Optional[Type[nn.Module]] = None,
        attn_layer: Optional[LayerType] = None,
        depth: int = 0,
        dual_lambda: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert not scale_attn_norm and not scale_mlp_norm, "Scale norms not supported"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2  # Half head_dim for diff attention
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim, **dd)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias, **dd)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            # mlp_bias is combined with qkv_bias in in_proj.bias
            self.register_parameter("mlp_bias", None)
        else:
            self.mlp_bias = nn.Parameter(torch.empty(mlp_hidden_dim, **dd))

        self.q_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_p = attn_drop

        # Differential attention specific
        self.sub_norm = RmsNorm(2 * self.head_dim, eps=1e-5, **dd)
        self.dual_lambda = dual_lambda
        if dual_lambda:
            self.lambda_a = nn.Parameter(torch.empty((), dtype=torch.float32, device=device))
            self.lambda_b = nn.Parameter(torch.empty((), dtype=torch.float32, device=device))
            self.lambda_q1 = self.lambda_k1 = self.lambda_q2 = self.lambda_k2 = None
        else:
            self.lambda_a = self.lambda_b = None
            self.lambda_q1 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32, device=device))
            self.lambda_k1 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32, device=device))
            self.lambda_q2 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32, device=device))
            self.lambda_k2 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32, device=device))

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()

        # Fused output projection for both attention and MLP
        self.out_proj = nn.Linear(dim + mlp_hidden_dim, dim, bias=proj_bias, **dd)

        self.ls = LayerScale(dim, init_values=init_values, **dd) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.lambda_init = 0.8
        self.set_lambda_init(depth)

        # TODO: skip init when on meta device when safe to do so
        self.reset_parameters()

    def set_lambda_init(self, depth: int):
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)

    def reset_parameters(self) -> None:
        """Initialize parameters and buffers."""
        if getattr(self, "mlp_bias", None) is not None:
            nn.init.zeros_(self.mlp_bias)
        if self.dual_lambda:
            nn.init.zeros_(self.lambda_a)
            nn.init.zeros_(self.lambda_b)
        else:
            nn.init.normal_(self.lambda_q1, mean=0, std=0.1)
            nn.init.normal_(self.lambda_k1, mean=0, std=0.1)
            nn.init.normal_(self.lambda_q2, mean=0, std=0.1)
            nn.init.normal_(self.lambda_k2, mean=0, std=0.1)

    def _compute_lambda(self) -> torch.Tensor:
        if self.lambda_a is not None:
            lambda_1 = torch.exp(self.lambda_a)
            lambda_2 = torch.exp(self.lambda_b)
        else:
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        return lambda_1 - lambda_2 + self.lambda_init

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)
        if getattr(self, "mlp_bias", None) is not None:
            x_mlp = x_mlp + self.mlp_bias

        # Reshape for differential attention (2x heads with half head_dim for q/k)
        q = q.reshape(B, N, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, 2 * self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        lambda_full = self._compute_lambda().type_as(q)

        if self.fused_attn:
            q = q.reshape(B, self.num_heads, 2, N, self.head_dim)
            k = k.reshape(B, self.num_heads, 2, N, self.head_dim)
            q1, q2 = q.unbind(2)
            k1, k2 = k.unbind(2)

            dropout_p = self.attn_drop_p if self.training else 0.0
            attn1 = F.scaled_dot_product_attention(q1, k1, v, attn_mask=attn_mask, dropout_p=dropout_p)
            attn2 = F.scaled_dot_product_attention(q2, k2, v, attn_mask=attn_mask, dropout_p=dropout_p)

            x_attn = attn1 - lambda_full * attn2
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            attn = attn.view(B, self.num_heads, 2, N, N)
            attn = attn[:, :, 0] - lambda_full * attn[:, :, 1]
            x_attn = attn @ v

        x_attn = self.sub_norm(x_attn)
        x_attn = x_attn * (1 - self.lambda_init)
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)

        # MLP activation & dropout
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)

        # Fused output projection
        y = self.out_proj(torch.cat((x_attn, x_mlp), dim=-1))

        # Add residual w/ drop path & layer scale applied
        x = x + self.drop_path(self.ls(y))
        return x


class ParallelThingsBlock(nn.Module):
    """Parallel ViT block (N parallel attention followed by N parallel MLP).

    Based on:
      `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_parallel: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
        attn_layer: LayerType = Attention,
        depth: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("norm", norm_layer(dim, **dd)),
                            (
                                "attn",
                                _create_attn(
                                    attn_layer,
                                    dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_norm=qk_norm,
                                    scale_norm=scale_attn_norm,
                                    proj_bias=proj_bias,
                                    attn_drop=attn_drop,
                                    proj_drop=proj_drop,
                                    norm_layer=norm_layer,
                                    depth=depth,
                                    **dd,
                                ),
                            ),
                            ("ls", LayerScale(dim, init_values=init_values, **dd) if init_values else nn.Identity()),
                            ("drop_path", DropPath(drop_path) if drop_path > 0.0 else nn.Identity()),
                        ]
                    )
                )
            )
            self.ffns.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("norm", norm_layer(dim, **dd)),
                            (
                                "mlp",
                                mlp_layer(
                                    dim,
                                    hidden_features=int(dim * mlp_ratio),
                                    act_layer=act_layer,
                                    norm_layer=norm_layer if scale_mlp_norm else None,
                                    bias=proj_bias,
                                    drop=proj_drop,
                                    **dd,
                                ),
                            ),
                            ("ls", LayerScale(dim, init_values=init_values, **dd) if init_values else nn.Identity()),
                            ("drop_path", DropPath(drop_path) if drop_path > 0.0 else nn.Identity()),
                        ]
                    )
                )
            )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attn_mask is not None:
            attn_out = []
            for attn in self.attns:
                x_attn = attn.norm(x)
                x_attn = attn.attn(x_attn, attn_mask=attn_mask)
                x_attn = attn.ls(x_attn)
                x_attn = attn.drop_path(x_attn)
                attn_out.append(x_attn)
            x = x + torch.stack(attn_out).sum(dim=0)
        else:
            x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x


def global_pool_nlc(
    x: torch.Tensor,
    pool_type: str = "token",
    num_prefix_tokens: int = 1,
    reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == "token":
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == "avg":
            x = x.mean(dim=1)
        elif pool_type == "avgmax":
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == "max":
            x = x.amax(dim=1)
        else:
            assert not pool_type, f"Unknown pool type {pool_type}"

    return x


def _merge_satclip_dual_path_nlc(
    x: Union[torch.Tensor, List[torch.Tensor]],
) -> torch.Tensor:
    """Merge ``[x_new, x_ori]`` from ConsistentAttention blocks into one NLC tensor.

    Matches the end-of-backbone merge in :meth:`VisionTransformer.forward_features`.
    """
    if isinstance(x, list) and len(x) == 2:
        x_new, x_ori = x
        out = x_new.clone()
        out[:, 0, :] = x_ori[:, 0, :]
        return out
    assert isinstance(x, torch.Tensor)
    return x


class VisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = "learn",
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        pool_include_prefix: bool = False,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "reset", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        embed_layer: Callable = PatchEmbed,
        embed_norm_layer: Optional[LayerType] = None,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        attn_layer: LayerType = Attention,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
        assert class_token or global_pool != "token"
        assert pos_embed in ("", "none", "learn")
        use_fc_norm = global_pool in ("avg", "avgmax", "max") if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or LayerNorm
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.in_chans = in_chans
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class
        self.pool_include_prefix = pool_include_prefix
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        if embed_norm_layer is not None:
            embed_args["norm_layer"] = embed_norm_layer
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
            **dd,
        )
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, "feat_ratio") else patch_size

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, **dd)) if class_token else None
        self.reg_token = nn.Parameter(torch.empty(1, reg_tokens, embed_dim, **dd)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == "none":
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.empty(1, embed_len, embed_dim, **dd))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim, **dd) if pre_norm else nn.Identity()

        dpr = calculate_drop_path_rates(drop_path_rate, depth)  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # apply architecture surgery on the last 6 blocks
            start_index = max(0, depth - 6)
            apply_surgery = i >= start_index
            attn_layer_i = "consistent_attn" if apply_surgery else attn_layer
            self.blocks.append(
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    scale_attn_norm=scale_attn_norm,
                    scale_mlp_norm=scale_mlp_norm,
                    proj_bias=proj_bias,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    attn_layer=attn_layer_i,
                    depth=i,
                    **dd,
                )
            )
        self.blocks = nn.Sequential(*self.blocks)
        self.feature_info = [dict(module=f"blocks.{i}", num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim, **dd) if final_norm and not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == "map":
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                **dd,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim, **dd) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()

        self.weight_init_mode = "reset" if weight_init == "skip" else weight_init
        self.fix_init = fix_init
        # TODO: skip init when on meta device when safe to do so
        if weight_init != "skip":
            self.init_weights(needs_reset=False)

    def fix_init_weight(self) -> None:
        """Apply weight initialization fix (scaling w/ layer index)."""
        with torch.no_grad():
            for layer_id, layer in enumerate(self.blocks):
                scale = math.sqrt(2.0 * (layer_id + 1))
                layer.attn.proj.weight.div_(scale)
                layer.mlp.fc2.weight.div_(scale)

    def init_weights(self, mode: str = "", needs_reset: bool = True) -> None:
        """Initialize model weights."""
        mode = mode or self.weight_init_mode
        assert mode in ("jax", "jax_nlhb", "moco", "reset", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)

        named_apply(get_init_weights_vit(mode, head_bias, needs_reset=needs_reset), self)

        if self.fix_init:
            self.fix_init_weight()

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for a single module (compatibility method)."""
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = "") -> None:
        """Load pretrained weights."""
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Set of parameters that should not use weight decay."""
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Union[str, List]]:
        """Create regex patterns for parameter grouping."""
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing."""
        self.grad_checkpointing = enable
        if hasattr(self.patch_embed, "set_grad_checkpointing"):
            self.patch_embed.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier head."""
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classifier head."""
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map" and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_input_size(
        self,
        img_size: Optional[Tuple[int, int]] = None,
        patch_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update the input image resolution and patch size."""
        prev_grid_size = self.patch_embed.grid_size
        self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)
        if self.pos_embed is not None:
            num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
            num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
            if num_new_tokens != self.pos_embed.shape[1]:
                self.pos_embed = nn.Parameter(
                    resample_abs_pos_embed(
                        self.pos_embed,
                        new_size=self.patch_embed.grid_size,
                        old_size=prev_grid_size,
                        num_prefix_tokens=num_prefix_tokens,
                        verbose=True,
                    )
                )

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to input."""
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
        output_dict: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """Forward features that returns intermediates."""
        assert output_fmt in ("NCHW", "NLC"), "Output format must be one of NCHW or NLC."
        reshape = output_fmt == "NCHW"
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[: max_index + 1]
        for i, blk in enumerate(blocks):
            if attn_mask is not None:
                x = blk(x, attn_mask=attn_mask)
            elif self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
            if i in take_indices:
                # Dual-path blocks return [x_new, x_ori]; merge before norm / storage (see forward_features).
                xm = _merge_satclip_dual_path_nlc(x)
                # Must clone: later ConsistentAttention blocks mutate `x` / list paths in-place; shared
                # tensor refs would make every saved "layer" converge to the final activations.
                to_store = self.norm(xm) if norm else xm
                intermediates.append(to_store.detach().clone())

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0 : self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens :] for y in intermediates]
        else:
            prefix_tokens = None

        if reshape:
            # reshape to BCHW output format
            H, W = self.patch_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        # For dictionary output, handle prefix tokens separately
        if output_dict:
            result_dict = {}
            # Intermediates are always included
            result_dict["image_intermediates"] = intermediates
            if prefix_tokens is not None and return_prefix_tokens:
                result_dict["image_intermediates_prefix"] = prefix_tokens

            # Only include features if not intermediates_only
            if not intermediates_only:
                x_final = self.norm(_merge_satclip_dual_path_nlc(x))
                result_dict["image_features"] = x_final

            return result_dict

        # For non-dictionary output, maintain the original behavior
        if not torch.jit.is_scripting() and return_prefix_tokens and prefix_tokens is not None:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(_merge_satclip_dual_path_nlc(x))

        return x, intermediates

    def prune_intermediate_layers(
        self,
        indices: Union[int, List[int]] = 1,
        prune_norm: bool = False,
        prune_head: bool = True,
    ) -> List[int]:
        """Prune layers not required for specified intermediates."""
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[: max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, "")
        return take_indices

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, List[int], Tuple[int]] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Get intermediate layer outputs (DINO interface compatibility)."""
        return self.forward_intermediates(
            x,
            n,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            output_fmt="NCHW" if reshape else "NLC",
            intermediates_only=True,
            attn_mask=attn_mask,
        )

    def forward_features(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if attn_mask is not None:
            # If mask provided, we need to apply blocks one by one
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
        elif self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        # x is a list [x_new, x_ori] because last blocks use ConsistentAttention
        if isinstance(x, list) and len(x) == 2:
            x_new, x_ori = x
            x_new[:, 0, :] = x_ori[:, 0, :]  # cls token from the original path, img tokens from the new path
            x = x_new

        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        """Apply pooling to feature tokens."""
        if self.attn_pool is not None:
            if not self.pool_include_prefix:
                x = x[:, self.num_prefix_tokens :]
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(
            x,
            pool_type=pool_type,
            num_prefix_tokens=self.num_prefix_tokens,
            reduce_include_prefix=self.pool_include_prefix,
        )
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classifier head."""
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.forward_features(x, attn_mask=attn_mask)
        x = self.forward_head(x)
        return x


def init_weights_vit_timm(module: nn.Module, name: str = "", needs_reset: bool = True) -> None:
    """ViT weight initialization, original timm impl (for reproducibility)."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()
    elif needs_reset and hasattr(module, "reset_parameters"):
        module.reset_parameters()


def init_weights_vit_jax(
    module: nn.Module,
    name: str = "",
    head_bias: float = 0.0,
    needs_reset: bool = True,
) -> None:
    """ViT weight initialization, matching JAX (Flax) impl."""
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if "mlp" in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()
    elif needs_reset and hasattr(module, "reset_parameters"):
        module.reset_parameters()


def init_weights_vit_moco(module: nn.Module, name: str = "", needs_reset: bool = True) -> None:
    """ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed."""
    if isinstance(module, nn.Linear):
        if "qkv" in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()
    elif needs_reset and hasattr(module, "reset_parameters"):
        module.reset_parameters()


def init_weights_reset_parameters(module: nn.Module, name: str = "", needs_reset: bool = True) -> None:
    if needs_reset and hasattr(module, "reset_parameters"):
        module.reset_parameters()


def get_init_weights_vit(mode: str = "jax", head_bias: float = 0.0, needs_reset: bool = True) -> Callable:
    if mode.startswith("jax"):
        return partial(init_weights_vit_jax, head_bias=head_bias, needs_reset=needs_reset)
    elif mode.startswith("moco"):
        return partial(init_weights_vit_moco, needs_reset=needs_reset)
    elif mode == "reset":
        # "reset" means only call reset_parameters() on modules
        return partial(init_weights_reset_parameters, needs_reset=needs_reset)
    else:
        # timm init is default
        return partial(init_weights_vit_timm, needs_reset=needs_reset)


def resize_pos_embed(
    posemb: torch.Tensor,
    posemb_new: torch.Tensor,
    num_prefix_tokens: int = 1,
    gs_new: Tuple[int, int] = (),
    interpolation: str = "bicubic",
    antialias: bool = False,
) -> torch.Tensor:
    """Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of using resample_abs_pos_embed
    """
    ntok_new = posemb_new.shape[1] - num_prefix_tokens
    ntok_old = posemb.shape[1] - num_prefix_tokens
    gs_old = [int(math.sqrt(ntok_old))] * 2
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    return resample_abs_pos_embed(
        posemb,
        gs_new,
        gs_old,
        num_prefix_tokens=num_prefix_tokens,
        interpolation=interpolation,
        antialias=antialias,
        verbose=True,
    )


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = "", load_bfloat16: bool = False) -> None:
    """Load weights from .npz checkpoints for official Google Brain Flax implementation."""
    import numpy as np

    if load_bfloat16:
        import jax.numpy as jnp
        import ml_dtypes

    def _n2p(_w, t=True, idx=None):
        if idx is not None:
            _w = _w[idx]

        if load_bfloat16:
            _w = _w.view(ml_dtypes.bfloat16).astype(jnp.float32)
            _w = np.array(_w)

        if _w.ndim == 4 and _w.shape[0] == _w.shape[1] == _w.shape[2] == 1:
            _w = _w.flatten()
        if t:
            if _w.ndim == 4:
                _w = _w.transpose([3, 2, 0, 1])
            elif _w.ndim == 3:
                _w = _w.transpose([2, 0, 1])
            elif _w.ndim == 2:
                _w = _w.transpose([1, 0])

        _w = torch.from_numpy(_w)
        return _w

    if load_bfloat16:
        w = jnp.load(checkpoint_path)
    else:
        w = np.load(checkpoint_path)

    interpolation = "bilinear"
    antialias = False
    big_vision = False
    if not prefix:
        if "opt/target/embedding/kernel" in w:
            prefix = "opt/target/"
        elif "params/embedding/kernel" in w:
            prefix = "params/"
            big_vision = True
        elif "params/img/embedding/kernel" in w:
            prefix = "params/img/"
            big_vision = True

    if hasattr(model.patch_embed, "backbone"):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, "stem")
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f"{prefix}conv_root/kernel"])))
        stem.norm.weight.copy_(_n2p(w[f"{prefix}gn_root/scale"]))
        stem.norm.bias.copy_(_n2p(w[f"{prefix}gn_root/bias"]))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f"{prefix}block{i + 1}/unit{j + 1}/"
                    for r in range(3):
                        getattr(block, f"conv{r + 1}").weight.copy_(_n2p(w[f"{bp}conv{r + 1}/kernel"]))
                        getattr(block, f"norm{r + 1}").weight.copy_(_n2p(w[f"{bp}gn{r + 1}/scale"]))
                        getattr(block, f"norm{r + 1}").bias.copy_(_n2p(w[f"{bp}gn{r + 1}/bias"]))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f"{bp}conv_proj/kernel"]))
                        block.downsample.norm.weight.copy_(_n2p(w[f"{bp}gn_proj/scale"]))
                        block.downsample.norm.bias.copy_(_n2p(w[f"{bp}gn_proj/bias"]))
        embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f"{prefix}embedding/kernel"])
        )
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f"{prefix}pos_embedding"], t=False)
    else:
        pos_embed_w = _n2p(w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        num_prefix_tokens = 0 if getattr(model, "no_embed_class", False) else getattr(model, "num_prefix_tokens", 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
    model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))
    if (
        isinstance(model.head, nn.Linear)
        and f"{prefix}head/bias" in w
        and model.head.bias.shape[0] == w[f"{prefix}head/bias"].shape[-1]
    ):
        model.head.weight.copy_(_n2p(w[f"{prefix}head/kernel"]))
        model.head.bias.copy_(_n2p(w[f"{prefix}head/bias"]))
    if model.attn_pool is not None:
        block_prefix = f"{prefix}MAPHead_0/"
        mha_prefix = block_prefix + "MultiHeadDotProductAttention_0/"
        model.attn_pool.latent.copy_(_n2p(w[f"{block_prefix}probe"], t=False))
        model.attn_pool.kv.weight.copy_(
            torch.cat(
                [_n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T for n in ("key", "value")]
            )
        )
        model.attn_pool.kv.bias.copy_(
            torch.cat([_n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1) for n in ("key", "value")])
        )
        model.attn_pool.q.weight.copy_(_n2p(w[f"{mha_prefix}query/kernel"], t=False).flatten(1).T)
        model.attn_pool.q.bias.copy_(_n2p(w[f"{mha_prefix}query/bias"], t=False).reshape(-1))
        model.attn_pool.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"]).flatten(1))
        model.attn_pool.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"]))
        model.attn_pool.norm.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
        model.attn_pool.norm.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
        for r in range(2):
            getattr(model.attn_pool.mlp, f"fc{r + 1}").weight.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_0/Dense_{r}/kernel"])
            )
            getattr(model.attn_pool.mlp, f"fc{r + 1}").bias.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_0/Dense_{r}/bias"])
            )

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        if f"{prefix}Transformer/encoderblock/LayerNorm_0/scale" in w:
            block_prefix = f"{prefix}Transformer/encoderblock/"
            idx = i
        else:
            block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
            idx = None
        mha_prefix = block_prefix + f"MultiHeadDotProductAttention_{mha_sub}/"
        block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"], idx=idx))
        block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"], idx=idx))
        block.attn.qkv.weight.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/kernel"], t=False, idx=idx).flatten(1).T
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.qkv.bias.copy_(
            torch.cat(
                [_n2p(w[f"{mha_prefix}{n}/bias"], t=False, idx=idx).reshape(-1) for n in ("query", "key", "value")]
            )
        )
        block.attn.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"], idx=idx).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"], idx=idx))
        block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_{ln1_sub}/scale"], idx=idx))
        block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_{ln1_sub}/bias"], idx=idx))
        for r in range(2):
            getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel"], idx=idx)
            )
            getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias"], idx=idx)
            )


def _convert_openai_clip(
    state_dict: Dict[str, torch.Tensor],
    model: VisionTransformer,
    prefix: str = "visual.",
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ("conv1", "patch_embed.proj"),
        ("positional_embedding", "pos_embed"),
        ("transformer.resblocks.", "blocks."),
        ("ln_pre", "norm_pre"),
        ("ln_post", "norm"),
        ("ln_", "norm"),
        ("in_proj_", "qkv."),
        ("out_proj", "proj"),
        ("mlp.c_fc", "mlp.fc1"),
        ("mlp.c_proj", "mlp.fc2"),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, "")
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == "proj":
            k = "head.weight"
            v = v.transpose(0, 1)
            out_dict["head.bias"] = torch.zeros(v.shape[0])
        elif k == "class_embedding":
            k = "cls_token"
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == "pos_embed":
            v = v.unsqueeze(0)
        out_dict[k] = v
    return out_dict


def _convert_dinov2(
    state_dict: Dict[str, torch.Tensor],
    model: VisionTransformer,
) -> Dict[str, torch.Tensor]:
    import re

    out_dict = {}
    state_dict.pop("mask_token", None)
    if "register_tokens" in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict["reg_token"] = state_dict.pop("register_tokens")
        out_dict["cls_token"] = state_dict.pop("cls_token") + state_dict["pos_embed"][:, 0]
        out_dict["pos_embed"] = state_dict.pop("pos_embed")[:, 1:]
    for k, v in state_dict.items():
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict


def _convert_aimv2(
    state_dict: Dict[str, torch.Tensor],
    model: VisionTransformer,
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    for k, v in state_dict.items():
        k = k.replace("norm_1", "norm1")
        k = k.replace("norm_2", "norm2")
        k = k.replace("preprocessor.patchifier.", "patch_embed.")
        k = k.replace("preprocessor.pos_embed", "pos_embed")
        k = k.replace("trunk.", "")
        k = k.replace("post_trunk_norm.", "norm.")
        k = k.replace("mlp.fc1", "mlp.fc1_g")
        k = k.replace("mlp.fc3", "mlp.fc1_x")
        out_dict[k] = v
    return out_dict


def _convert_beit3(state_dict: dict, model):
    """
    Turn a BEiT-3 checkpoint into a standard VisionTransformer state-dict.
    """
    import re

    state_dict = state_dict.get("model", state_dict)  # unwrap if needed

    # Prune unused
    for k in ("beit3.text_embed.weight", "beit3.vision_embed.mask_token"):
        state_dict.pop(k, None)

    # Key renaming rules
    rules = [
        ("beit3.", ""),
        ("vision_embed.cls_token", "cls_token"),
        ("vision_embed.", "patch_embed."),
        ("embed_positions.", "pos_embed."),
        ("encoder.", ""),
        ("layers.", "blocks."),
        ("ffn_layernorm.", "norm."),
        ("ffn.", "mlp."),
        ("self_attn_layer_norm.", "norm1."),
        ("self_attn.", "attn."),
        ("final_layer_norm.", "norm2."),
        ("inner_attn_ln", "norm"),
        ("out_proj", "proj"),
        (r"\.A\.", "."),
    ]

    # First pass, rename keys
    tmp = {}
    for k, v in state_dict.items():
        if ".B." in k:
            continue  # use branch-A only
        for old, new in rules:
            k = re.sub(old, new, k)
        if k == "pos_embed.weight":
            # strip first two positions, [1, N+1, D]
            tmp["pos_embed"] = v[2:].unsqueeze(0)
        else:
            tmp[k] = v

    # Second pass, fuse q, k, v
    out, buf = {}, {}
    pat = re.compile(r"blocks\.(\d+)\.attn\.(q|k|v)_proj\.(weight|bias)$")
    for k, v in tmp.items():
        m = pat.fullmatch(k)
        if not m:  # anything not q/k/v -> copy through
            out[k] = v
            continue

        blk, which, kind = m.groups()  # block idx, "q"/"k"/"v", "weight"/"bias"
        stash = buf.setdefault((blk, kind), {})  # Gather by block & param type
        stash[which] = v
        if len(stash) == 3:  # Have q, k, v -> concatenate
            out[f"blocks.{blk}.attn.qkv.{kind}"] = torch.cat(
                [stash["q"], stash["k"], stash["v"]], dim=0
            )

    return out


def checkpoint_filter_fn(
    state_dict: Dict[str, torch.Tensor],
    model: VisionTransformer,
    adapt_layer_scale: bool = False,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """Convert patch embedding weight from manual patchify + linear proj to conv."""
    import re

    out_dict = {}
    state_dict = state_dict.get("model", state_dict)
    state_dict = state_dict.get("state_dict", state_dict)
    prefix = ""

    if "visual.class_embedding" in state_dict:
        state_dict = _convert_openai_clip(state_dict, model)
    elif "module.visual.class_embedding" in state_dict:
        state_dict = _convert_openai_clip(state_dict, model, prefix="module.visual.")
    elif "mask_token" in state_dict:
        state_dict = _convert_dinov2(state_dict, model)
    elif any("beit3." in k for k in state_dict.keys()):
        # BEiT3 model - multimodal checkpoint with beit3.* prefix
        state_dict = _convert_beit3(state_dict, model)
    elif "encoder" in state_dict:
        # IJEPA, vit in an "encoder" submodule
        state_dict = state_dict["encoder"]
        prefix = "module."
    elif "visual.trunk.pos_embed" in state_dict or "visual.trunk.blocks.0.norm1.weight" in state_dict:
        # OpenCLIP model with timm vision encoder
        prefix = "visual.trunk."
        if "visual.head.proj.weight" in state_dict and isinstance(model.head, nn.Linear):
            # remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
            out_dict["head.weight"] = state_dict["visual.head.proj.weight"]
            out_dict["head.bias"] = torch.zeros(state_dict["visual.head.proj.weight"].shape[0])
    elif "module.visual.trunk.pos_embed" in state_dict:
        prefix = "module.visual.trunk."
    elif "preprocessor.patchifier.proj.weight" in state_dict:
        state_dict = _convert_aimv2(state_dict, model)

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == "pos_embed" and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, "no_embed_class", False) else getattr(
                model, "num_prefix_tokens", 1
            )
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and "gamma_" in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r"gamma_([0-9])", r"ls\1.gamma", k)
        elif "pre_logits" in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict

