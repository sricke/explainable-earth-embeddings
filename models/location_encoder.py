import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download

from external.satclip.satclip.load import get_satclip

from models.layers import EmbeddingProjection

LOCATION_EMBEDDING_DIMENSIONS = {
    "geoclip": 512,
    "satclip": 256,
    "aef": 64,
}

LOCATION_MODEL_IDS = {
    "satclip": "microsoft/SatCLIP-ViT16-L40",
    # might need to include L10 as well
}

LOCATION_MODEL_CHECKPOINTS = {
    "satclip": "satclip-vit16-l40.ckpt",
}


def load_model(location_model: str, device: str = "cuda:0"):
    """Load a pretrained location encoder."""
    if location_model == "satclip":
        return get_satclip(
            hf_hub_download(LOCATION_MODEL_IDS["satclip"], LOCATION_MODEL_CHECKPOINTS["satclip"]),
            device=device,
        )
    elif location_model == "geoclip":
        from geoclip import GeoCLIP
        return GeoCLIP().location_encoder
    else:
        raise ValueError(f"Location model '{location_model}' is not supported")


class LocationEncoder(nn.Module):

    def __init__(
        self,
        location_model: str = None,
        embed_project: EmbeddingProjection = None,
        precomputed: bool = True,
    ):
        super().__init__()
        assert location_model is not None, "Must specify location model"
        self.location_model = location_model
        self.precomputed = precomputed
        self.location_embedding_dim = LOCATION_EMBEDDING_DIMENSIONS[location_model]
        self.embed_project = embed_project

        if not precomputed:
            self.location_encoder = load_model(location_model)
            self.location_encoder.requires_grad_(False)
            self.location_encoder.eval()

        if embed_project is not None:
            embed_project.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.precomputed:
            target_device = None
            if self.embed_project is not None:
                target_device = next(self.embed_project.parameters()).device
            if target_device is not None and x.device != target_device:
                x = x.to(target_device, non_blocking=True)
            embedding = F.normalize(x.to(dtype=torch.float32), dim=-1)
        else:
            assert x.shape[1] == 2, "Forward expects (lat, lon) pairs"
            x = x[:, [1, 0]].double() if self.location_model == "satclip" else x.float()
            # Ensure coords live on the same device as the encoder.
            enc_device = next(self.location_encoder.parameters()).device
            if x.device != enc_device:
                x = x.to(enc_device, non_blocking=True)
            embedding = F.normalize(self.location_encoder(x).float(), dim=-1)

        if self.embed_project is not None:
            embedding = F.normalize(self.embed_project(embedding), dim=-1)
        return embedding
