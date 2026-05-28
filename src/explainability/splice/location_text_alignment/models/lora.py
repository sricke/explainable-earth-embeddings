import torch.nn as nn
from peft import LoraConfig, get_peft_model


def apply_lora(
    model: nn.Module, r: int = 4, alpha: float = 1.0, last_n_layers: int | None = None
) -> nn.Module:
    """
    Apply LoRA (Low-Rank Adaptation) to a model's attention projections (q_proj, v_proj).
    See https://arxiv.org/abs/2106.09685.

    Uses peft LoraConfig. If last_n_layers is specified, LoRA is applied only to the
    final n layers of the text encoder; otherwise it is applied to all layers.

    Parameters:
        model: model to apply LoRA to (expected nn.Module with text_model.encoder.layers)
        r: LoRA rank
        alpha: LoRA scaling factor
        last_n_layers: if set, restrict LoRA to the last n layers

    Returns:
        model wrapped with PEFT LoRA adapters
    """
    layers_to_transform = None

    if last_n_layers is not None:
        num_layers = len(model.text_model.encoder.layers)
        layers_to_transform = list(range(num_layers - last_n_layers, num_layers))

    config = LoraConfig(
        r=int(r),
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        layers_to_transform=layers_to_transform,
    )
    return get_peft_model(model, config)
