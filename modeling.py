from typing import Literal, Optional, Union
import sys
from pathlib import Path

# Add project root to path so we can import satclip
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import open_clip
import torch.nn as nn
from huggingface_hub import hf_hub_download
from satclip.satclip.load import get_satclip

class Encoder(nn.Module):
    def __init__(self, model, train_encoder: bool, target_dim: int = None):
        super().__init__()
        self.model = model
        self.model.requires_grad_(train_encoder)
        for param in self.model.parameters():
            param.requires_grad = train_encoder
        self.embed_project = None
        self.location_dim= self.model.nnet.last_layer.dim_out
        if target_dim is not None:
            self.embed_project = nn.Linear(self.location_dim, target_dim).double()
        self.target_dim = target_dim
        self.train_encoder = train_encoder
        
    def encode_features(self, x):
        pass
        
    def forward(self, x, normalize=False):
        last_hidden_state = self.encode_features(x, normalize)
        if self.embed_project is not None:
            embedding_output = self.embed_project(last_hidden_state)
        else:
            raise ValueError("No projection layer added to train")
            embedding_output = last_hidden_state
 
        return embedding_output

class LocationEmbeddingModel(Encoder):
    def __init__(
        self,
        location_model: str,
        location_model_filename: str,
        train_location_model: bool,
        target_dim: int = None,
    ):
        model = get_satclip(hf_hub_download(location_model, location_model_filename), device="cpu")
        super().__init__(model, train_location_model, target_dim)

    
    def encode_features(self, x, normalize=False):
        ## Encode location
        if self.train_encoder_model:
            embedding = self.model(x.double())
        else:
            with torch.no_grad():
                embedding = self.model(x.double())
        if normalize:
            embedding = embedding / embedding.norm(dim=1, keepdim=True)
        return embedding
            
            
class TextEmbeddingModel(Encoder):
    def __init__(self, text_model: str, text_model_filename: str, train_text_model: bool, target_dim: int = None):
        model = open_clip.create_model(text_model, pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer(text_model)
        super().__init__(model, train_text_model, target_dim)
        
    def encode_features(self, x, normalize=False):
        if isinstance(x, str) or isinstance(x, list):
            x = self.tokenizer(x).to(self.device)
        if self.train_encoder:
            embedding = self.model.encode_text(x)
        else:
            with torch.no_grad():
                embedding = self.model.encode_text(x)
        if normalize:
            embedding = embedding / embedding.norm(dim=1, keepdim=True)
        return embedding
