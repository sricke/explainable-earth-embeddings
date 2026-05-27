"""
Helpers for GeoCLIP + CLIP surgery: single vision forward with per-layer hiddens.

Use ``CLIPSurgeryVisionTransformer.forward_intermediates`` then project with
``post_layernorm`` + ``visual_projection`` (and GeoCLIP ``mlp`` if needed).
"""

from __future__ import annotations

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling


def geoclip_project_to_image_embed(
    vision_model,
    hidden_states: torch.Tensor,
    visual_projection: torch.nn.Module,
) -> torch.Tensor:
    """Encoder NLD (pre/post_layernorm) → clip embedding dim. Pass *post* layernorm states."""
    return visual_projection(hidden_states)


def geoclip_encode_image_and_layer_hiddens(
    vision_model,
    pixel_values: torch.Tensor,
    interpolate_pos_encoding: bool = False,
    indices: int | list[int] | None = None,
    **kwargs,
) -> tuple[BaseModelOutputWithPooling, list[torch.Tensor]]:
    """Single ``forward_intermediates`` call: same first return as :meth:`CLIPSurgeryVisionTransformer.forward`
    plus per-layer merged hiddens (pre ``post_layernorm``).

    For per-patch surgery features, apply ``post_layernorm`` to ``last_hidden_state`` then
    ``visual_projection`` and GeoCLIP ``mlp`` (CLS-only image embedding uses ``pooler_output``).
    """
    return vision_model.forward_intermediates(
        pixel_values=pixel_values,
        interpolate_pos_encoding=interpolate_pos_encoding,
        indices=indices,
        **kwargs,
    )


def geoclip_layer_tokens_to_embed(
    vision_model,
    layer_hidden_pre_ln: torch.Tensor,
    visual_projection: torch.nn.Module,
) -> torch.Tensor:
    """Per-layer encoder output (pre ``post_layernorm``) → same space as projected image tokens."""
    x = vision_model.post_layernorm(layer_hidden_pre_ln)
    return visual_projection(x)
