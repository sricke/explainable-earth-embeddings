from typing import Literal, Optional, Union
import sys
from pathlib import Path
from utils import get_location_model_output_dim
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
        if target_dim is not None:
            self.text_output_dim = self.model.text_projection.shape[1]
            self.embed_project = nn.Linear(self.text_output_dim, target_dim)
        self.target_dim = target_dim
        self.train_encoder = train_encoder
        if not train_encoder:
            self.model.eval()
        
    def encode_features(self, x):
        pass
        
    def forward(self, x):
        last_hidden_state = self.encode_features(x)
        if self.embed_project is not None:
            last_hidden_state = self.embed_project(last_hidden_state)
 
        return last_hidden_state

class LocationEmbeddingModel(Encoder):
    def __init__(
        self,
        location_model: str,
        location_model_filename: str,
        train_location_model: bool,
        target_dim: int = None,
        dtype: torch.dtype = torch.float32,
    ):
        model = get_satclip(hf_hub_download(location_model, location_model_filename), device="cpu")
        self.dtype = dtype
        super().__init__(model, train_location_model, target_dim)
    
    def encode_features(self, x):
        ## Encode location
        if self.train_encoder:
            embedding = self.model(x.double())
        else:
            with torch.no_grad():
                embedding = self.model(x.double())
        embedding = embedding.type(self.dtype)
        return embedding
            
            
class TextEmbeddingModel(Encoder):
    def __init__(self, text_model: str, text_vocabulary: str, train_text_model: bool, target_dim: int = None, dtype: torch.dtype = torch.float32):
        model = open_clip.create_model(text_model, pretrained=text_vocabulary)
        
        # don't need vision encoder
        del model.visual
        
        self.tokenizer = open_clip.get_tokenizer(text_model)
        self.dtype = dtype
        super().__init__(model, train_text_model, target_dim)
        
    def encode_features(self, x):
        if isinstance(x, str) or isinstance(x, list):
            x = self.tokenizer(x).to(self.device)
        if self.train_encoder:
            embedding = self.model.encode_text(x)
        else:
            with torch.no_grad():
                embedding = self.model.encode_text(x)
        embedding = embedding.type(self.dtype)
        return embedding
