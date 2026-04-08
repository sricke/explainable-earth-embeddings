import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download

from external.satclip.satclip.load import get_satclip

from modeling.layers import EmbeddingProjection

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


def load_model(location_model: str):
    """Load a pretrained location encoder."""
    if location_model == "satclip":
        return get_satclip(
            hf_hub_download(LOCATION_MODEL_IDS["satclip"], LOCATION_MODEL_CHECKPOINTS["satclip"]),
            device="cpu",
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
    ):
        super().__init__()
        assert location_model is not None, "Must specify location model"
        self.location_model = location_model

        self.location_encoder = load_model(location_model)
        self.location_encoder.requires_grad_(False)
        self.location_encoder.eval()

        self.embed_project = embed_project
        if self.embed_project is not None:
            self.embed_project.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 2, "Forward expects (lat, lon) pairs"

        if self.location_model == "satclip":
            x = x[:, [1, 0]]  # SatCLIP expects (lon, lat)

        location_embedding = self.location_encoder(x)
        location_embedding = F.normalize(location_embedding, dim=-1)

        if self.embed_project is not None:
            location_embedding = F.normalize(self.embed_project(location_embedding), dim=-1)

        return location_embedding
