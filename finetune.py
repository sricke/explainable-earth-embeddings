from typing import Literal
import torch.nn as nn

FINETUNE_MODE_OPTIONS = Literal[
    "none",
    "linear_only",
    "all",
    "geoclip_text_mlp_only"
]

def set_finetune_mode(model, finetune_mode: str):
    for param in model.parameters():
        param.requires_grad = False

    if finetune_mode == "none":
        return

    elif finetune_mode == "linear_only":
        for param in model.text_model.embed_project.parameters():
            param.requires_grad = True

    elif finetune_mode == "all":
        for param in model.text_model.parameters():
            param.requires_grad = True

    elif finetune_mode == "geoclip_text_mlp_only":
        # freeze all text model params
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze only GeoCLIP MLP
        for p in model.mlp.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unsupported finetune mode: {finetune_mode}")