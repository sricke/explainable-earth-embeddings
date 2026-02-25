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
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch.nn.functional as F
import torch.nn as nn

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
        location_model_type: str, #satclip, or geoclip
        location_model: str,
        location_model_filename: str,
        train_location_model: bool,
        target_dim: int = None,
        dtype: torch.dtype = torch.float32,
    ):
        if location_model_type == "satclip":
            model = get_satclip(hf_hub_download(location_model, location_model_filename), device="cpu")
        elif location_model_type == "geoclip":
            from geoclip import GeoCLIP
            model = GeoCLIP().location_encoder
        else:
            raise ValueError(f"Unsupported location model type: {location_model_type}")
        
        self.dtype = dtype
        super().__init__(model, train_location_model, target_dim)
    
    def encode_features(self, x):
        ## Encode location
        if self.train_encoder:
            embedding = self.model(x.float())
        else:
            with torch.no_grad():
                embedding = self.model(x.float())
        embedding = F.normalize(embedding, dim=-1)
        return embedding.type(self.dtype)
            
            
class TextEmbeddingModel(Encoder):
    def __init__(self, text_model_type: str, text_model: str, text_vocabulary: str, train_text_model: bool, target_dim: int = None, dtype: torch.dtype = torch.float32):
        self.text_model_type = text_model_type
        self.dtype = dtype

        if text_model == 'ViT-B-32':
            model = open_clip.create_model(text_model, pretrained=text_vocabulary)
            del model.visual
            self.tokenizer = open_clip.get_tokenizer(text_model)
            output_dim = model.text_projection.shape[1]

        elif text_model == 'geoclip':
            from geoclip import GeoCLIP
            from transformers import AutoTokenizer

            loc_image_clip_model = GeoCLIP()
            image_encoder = loc_image_clip_model.image_encoder
            clip_model = image_encoder.CLIP          # full CLIPModel, not clip_model.text_model
            mlp = image_encoder.mlp

            self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

            class CLIPTextWithMLP(nn.Module):
                def __init__(self, clip_model, mlp):
                    super().__init__()
                    self.clip_model = clip_model
                    self.mlp = mlp

                def forward(self, **kwargs):
                    text_features = self.clip_model.get_text_features(**kwargs)

                    # Handle HuggingFace-style outputs
                    if not isinstance(text_features, torch.Tensor):
                        # Typically you want the pooled embedding
                        text_features = text_features.pooler_output

                    return F.normalize(self.mlp(text_features), dim=-1)

            model = CLIPTextWithMLP(clip_model, mlp)
            output_dim = 512  # GeoCLIP MLP output dim

        else:
            raise ValueError(f"Unsupported text model: {text_model}")

        # Bypass Encoder's text_projection logic by initialising directly
        nn.Module.__init__(self)
        self.model = model
        self.model.requires_grad_(train_text_model)
        self.train_encoder = train_text_model
        if not train_text_model:
            self.model.eval()
        # self.embed_project = nn.Linear(output_dim, target_dim) if target_dim is not None else None
        self.embed_project = None
        self.target_dim = target_dim

    def encode_features(self, x):
        if not isinstance(x, (str, list)):
            raise ValueError(f"Expected str or list, got {type(x)}")

        if self.text_model_type == 'geoclip':
            tokens = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
        else:
            tokens = self.tokenizer(x).to(self.device)

        with torch.set_grad_enabled(self.train_encoder):
            if self.text_model_type == 'geoclip':
                embedding = self.model(**tokens)
            else:
                embedding = self.model.encode_text(tokens)

        return embedding.type(self.dtype)
    
    @property
    def device(self):
        return next(self.model.parameters()).device   

