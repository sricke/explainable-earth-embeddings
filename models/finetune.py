import torch.nn as nn
import torch.nn.functional as F

import copy


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)

        d_in, d_out = linear.in_features, linear.out_features
        device = next(linear.parameters()).device
        self.A = nn.Linear(d_in, r, bias=False, device=device)
        self.B = nn.Linear(r, d_out, bias=False, device=device)
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.linear(x) + self.B(self.A(x)) * self.scale


class LoRAMultiheadAttention(nn.Module):
    """LoRA on Q and V projections, using PyTorch's scaled_dot_product_attention.
    Keeps the fused in_proj_weight to preserve memory-efficient attention kernels."""

    def __init__(self, attn: nn.Module, r: int, alpha: float, open_clip: bool = False):
        super().__init__()
        self.seq_first = open_clip or not getattr(attn, 'batch_first', False)
        self.num_heads = attn.num_heads
        self.head_dim = attn.embed_dim // attn.num_heads
        d = attn.embed_dim

        self.register_buffer("in_proj_weight", attn.in_proj_weight.detach())
        if attn.in_proj_bias is not None:
            self.register_buffer("in_proj_bias", attn.in_proj_bias.detach())
        else:
            self.register_buffer("in_proj_bias", None)

        device = attn.in_proj_weight.device
        self.q_A = nn.Linear(d, r, bias=False, device=device)
        self.q_B = nn.Linear(r, d, bias=False, device=device)
        self.v_A = nn.Linear(d, r, bias=False, device=device)
        self.v_B = nn.Linear(r, d, bias=False, device=device)
        nn.init.normal_(self.q_A.weight, std=0.01)
        nn.init.zeros_(self.q_B.weight)
        nn.init.normal_(self.v_A.weight, std=0.01)
        nn.init.zeros_(self.v_B.weight)
        self.scale = alpha / r

        self.out_proj = copy.deepcopy(attn.out_proj)
        for p in self.out_proj.parameters():
            p.requires_grad_(False)

    def forward(self, q_x, k_x=None, v_x=None, attn_mask=None, **_):
        is_self_attn = k_x is None
        if k_x is None:
            k_x = q_x
        if v_x is None:
            v_x = q_x

        if self.seq_first:
            q_x = q_x.transpose(0, 1)
            k_x = k_x.transpose(0, 1)
            v_x = v_x.transpose(0, 1)

        B, T, d = q_x.shape
        S = k_x.shape[1]

        W_q = self.in_proj_weight[:d]
        W_k = self.in_proj_weight[d:2 * d]
        W_v = self.in_proj_weight[2 * d:]
        b_q = b_k = b_v = None
        if self.in_proj_bias is not None:
            b_q = self.in_proj_bias[:d]
            b_k = self.in_proj_bias[d:2 * d]
            b_v = self.in_proj_bias[2 * d:]

        Q = F.linear(q_x, W_q, b_q) + self.q_B(self.q_A(q_x)) * self.scale
        K = F.linear(k_x, W_k, b_k)
        V = F.linear(v_x, W_v, b_v) + self.v_B(self.v_A(v_x)) * self.scale

        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Drop attn_mask for cross-attention (open_clip never passes one there)
        if not is_self_attn:
            attn_mask = None

        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, d)
        out = self.out_proj(out)

        if self.seq_first:
            out = out.transpose(0, 1)
        return out, None


def apply_lora(model: nn.Module, r: int = 4, alpha: float = 1.0, open_clip: bool = False):
    """Freeze the base model and replace attention modules with LoRA-wrapped versions."""
    for p in model.parameters():
        p.requires_grad_(False)
    for name, module in list(model.named_children()):
        if hasattr(module, 'in_proj_weight'):
            setattr(model, name, LoRAMultiheadAttention(module, r, alpha, open_clip=open_clip))
        else:
            apply_lora(module, r, alpha, open_clip=open_clip)
