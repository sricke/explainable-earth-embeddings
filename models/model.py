import torch
import torch.nn as nn

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

    def location_model(self, locations: torch.Tensor) -> torch.Tensor:
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