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

    def forward(self, texts, locations, out_dict=False):
        if out_dict:
            features_dict = {
                'text_features': self.text_encoder(texts),
                'location_features': self.location_encoder(locations)
            }
            return features_dict
        
        return self.text_encoder(texts), self.location_encoder(locations)