import logging

import torch
import torch.nn as nn

from models.utils import make_text_encoder, make_location_encoder

logger = logging.getLogger(__name__)

class TextLocationModel(nn.Module):
    """
    Contains both text and location encoder
    """

    def __init__(self, text_encoder: nn.Module = None, location_encoder: nn.Module = None):
        super().__init__()
        
        assert text_encoder is not None, "Must include text encoder"
        assert location_encoder is not None, "Must include location encoder"

        self.text_encoder = text_encoder
        self.location_encoder = location_encoder

    @property
    def output_dim(self) -> int:
        text_dim = self.text_encoder.embed_project.net[-1].out_features if self.text_encoder.embed_project else self.text_encoder.output_dim
        loc_dim = self.location_encoder.embed_project.net[-1].out_features if self.location_encoder.embed_project else self.location_encoder.location_embedding_dim
        assert text_dim == loc_dim, f"Text encoder output dim ({text_dim}) != location encoder output dim ({loc_dim})"
        return text_dim

    def location_model_predict(self, locations: torch.Tensor) -> torch.Tensor:
        return self.location_encoder(locations)

    def text_model_predict(self, texts, normalize: bool = True) -> torch.Tensor:
        emb = self.text_encoder(texts)
        if normalize:
            emb = nn.functional.normalize(emb, dim=-1)
        return emb

    def forward(self, texts, locations, out_dict=False):
        if out_dict:
            features_dict = {
                'text_features': self.text_encoder(texts),
                'location_features': self.location_encoder(locations)
            }
            return features_dict

        return self.text_encoder(texts), self.location_encoder(locations)


def build_model(
    text_encoder,
    location_encoder,
    text_projection,
    location_projection,
    shared_dim,
    text_finetune_mode,
    loc_finetune_mode,
    text_proj_hidden_layers,
    text_proj_hidden_features,
    loc_proj_hidden_layers,
    loc_proj_hidden_features,
    text_nonlinearity,
    loc_nonlinearity,
    precomputed_text_embeddings=True,
    precomputed_location_embeddings=True,
    *,
    device,
):
    text_enc = make_text_encoder(
        text_encoder, text_projection, shared_dim, text_finetune_mode,
        num_hidden_layers=text_proj_hidden_layers, num_hidden_features=text_proj_hidden_features,
        nonlinearity=text_nonlinearity,
        precomputed=precomputed_text_embeddings,
    )
    loc_enc = make_location_encoder(
        location_encoder, location_projection, shared_dim, loc_finetune_mode,
        num_hidden_layers=loc_proj_hidden_layers, num_hidden_features=loc_proj_hidden_features,
        nonlinearity=loc_nonlinearity,
        precomputed=precomputed_location_embeddings,
    )
    logger.info(f"Using text encoder={text_encoder}, location encoder={location_encoder}")
    return TextLocationModel(text_encoder=text_enc, location_encoder=loc_enc).to(device)