from typing import Literal
import torch.nn as nn

FINETUNE_MODE_OPTIONS = Literal[
    "none",
    "mlp",
    "linear_only",
    "all",
    "geoclip_text_mlp_only"
]

def set_finetune_mode(text_model, finetune_mode: str):
    """Set which parts of the TextEncoder are trainable.

    Note: Call this with the `TextEncoder` instance (not the wrapped
    backend model), so we can also handle optional projection heads.
    """
    for param in text_model.parameters():
        param.requires_grad = False

    if finetune_mode == "none":
        return

    elif finetune_mode in {"linear_only", "mlp"}:
        if getattr(text_model, "embed_project", None) is None:
            raise ValueError("mlp/linear_only requires text_model.embed_project to exist")
        for param in text_model.embed_project.parameters():
            param.requires_grad = True

    elif finetune_mode == "all":
        for param in text_model.parameters():
            param.requires_grad = True

    elif finetune_mode == "geoclip_text_mlp_only":
        # Unfreeze only GeoCLIP MLP (and, if present, the projection head).
        if not hasattr(text_model, "model") or not hasattr(text_model.model, "mlp"):
            raise ValueError(
                "geoclip_text_mlp_only requires a GeoCLIP-backed text model with `.model.mlp`"
            )
        for p in text_model.model.mlp.parameters():
            p.requires_grad = True
        # If we added a projection layer to match target_dim, it must be trainable
        # or it will bottleneck alignment.
        if getattr(text_model, "embed_project", None) is not None:
            for p in text_model.embed_project.parameters():
                p.requires_grad = True

    else:
        raise ValueError(f"Unsupported finetune mode: {finetune_mode}")