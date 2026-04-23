import torch.nn as nn
from peft import get_peft_model, LoraConfig


def apply_lora(model: nn.Module, r: int = 4, alpha: float = 1.0) -> nn.Module:
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(model, config)
