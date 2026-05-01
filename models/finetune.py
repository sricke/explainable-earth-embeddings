import torch.nn as nn
from peft import get_peft_model, LoraConfig


def apply_lora(model: nn.Module, r: int = 4, alpha: float = 1.0, last_n_layers: int | None = None) -> nn.Module:
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
