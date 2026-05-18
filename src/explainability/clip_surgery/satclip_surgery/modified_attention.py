from typing import Final, Optional, Type

import torch
from torch import nn as nn
from torch.nn import functional as F

from timm.layers._fx import register_notrace_function
from timm.layers.config import use_fused_attn
from timm.layers.pos_embed_sincos import apply_rot_embed_cat  # noqa: F401 (kept for parity with source)


@torch.fx.wrap
@register_notrace_function
def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask


class ConsistentAttention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_head_dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[nn.Module]] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        dim_out = dim_out or dim
        head_dim = attn_head_dim
        if head_dim is None:
            assert dim % num_heads == 0, "dim should be divisible by num_heads"
            head_dim = dim // num_heads
        if qk_norm or scale_norm:
            assert norm_layer is not None, "norm_layer must be provided if qk_norm or scale_norm is True"

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, self.attn_dim * 3, bias=qkv_bias, **dd)
        self.q_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(self.attn_dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(self.attn_dim, dim_out, bias=proj_bias, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x_ori = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn_ori = maybe_add_mask(attn, attn_mask)
            attn_ori = attn_ori.softmax(dim=-1)
            attn_ori = self.attn_drop(attn_ori)
            x_ori = attn_ori @ v

        # Dual path, replace q & k by v
        k = v
        q = k
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
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
            x = attn @ v

        x_ori = x_ori.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)

        x_ori = self.norm(x_ori)
        x_ori = self.proj(x_ori)
        x_ori = self.proj_drop(x_ori)

        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return [x, x_ori]

