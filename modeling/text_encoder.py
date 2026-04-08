from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def _build_openclip(model_id: str, pretrained: str):
    import open_clip
    model = open_clip.create_model(model_id, pretrained=pretrained)
    del model.visual
    tokenizer = open_clip.get_tokenizer(model_id)
    output_dim = model.text_projection.shape[1]
    return model, tokenizer, output_dim


def _build_geoclip(tokenizer):
    from geoclip import GeoCLIP
    from transformers import CLIPTokenizer

    # This is the same pretrained CLIP that GeoCLIP uses
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    geoclip = GeoCLIP()
    output_dim = 512
    clip_model = geoclip.image_encoder.CLIP

    # Isolate the mlp, then will create a model with text encoder and this MLP on top
    mlp = geoclip.image_encoder.mlp

    class CLIPTextWithMLP(nn.Module):
        def __init__(self, clip_model, mlp):
            super().__init__()
            self.clip_model = clip_model
            self.mlp = mlp

        def forward(self, input_ids):
            """
            input_ids: already tokenized input, shape [B, L]
            """
            text_features = self.clip_model.encode_text(input_ids)

            text_proj = self.mlp(text_features)
            
            # Normalize (location embeddings are normalized)
            text_proj = nn.functional.normalize(text_proj, dim=-1)

            return text_proj

    model = CLIPTextWithMLP(clip_model, mlp)
    return model, tokenizer, output_dim

class EmbeddingProjection(nn.Module):
    """
    Either linear layer or small MLP placed on top of the location embedding
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int = 0,
        num_hidden_features: int = None,
        nonlinearity: str = None
    ):
        super().__init__()

        # Linear case
        if num_hidden_layers == 0:
            self.net = self._create_linear_project(in_features, out_features)

        # MLP case
        else:
            self.net = self._create_mlp_project(in_features, out_features, num_hidden_layers, num_hidden_features, nonlinearity)

    def _create_linear_project(self, in_features, out_features):
        layers = []
        linear_layer = nn.Linear(in_features, out_features)
        layers.append(linear_layer)
        return nn.Sequential(*layers)
    
    def _create_mlp_project(self, in_features, out_features, num_hidden_layers, num_hidden_features, nonlinearity):
        assert num_hidden_features is not None, "Need hidden size for MLP"
        assert nonlinearity is not None, "Need to specify nonlinearity"

        layers = []
        activation = self._create_activation(nonlinearity)
    
        # Input layer
        input_layer = nn.Linear(in_features, num_hidden_features)
        input_activation = activation()
        layers.append([input_layer, input_activation])

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            hidden_layer = nn.Linear(num_hidden_features, num_hidden_features)
            hidden_activation = activation()
            layers.append([hidden_layer, hidden_activation])

        # Output layer
        layers.append(nn.Linear(num_hidden_features, out_features))

        return nn.Sequential(*layers)
    

    def _create_activation(self, nonlinearity: str = "sine"):
        if nonlinearity == "relu":
            return nn.ReLU
        else:
            raise NotImplementedError(f"Nonlinearity {nonlinearity} is not implemented yet")

    def _init_weights(self, nonlinearity: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):

                if nonlinearity == "relu":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class TextEncoder(nn.Module):
    """Encodes text into fixed-size embeddings.
    """

    def __init__(
        self,
        text_model: str = None,
        embed_project: nn.Module = None,
        finetune_mode: str = None,
    ):
        super().__init__()
        assert text_model is not None, "Need to specify text model"
        text_encoder, tokenizer, output_dim = self._build_model(text_model)

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.output_dim = output_dim

        # Train embedding projection if there is one
        self.embed_project = embed_project
        self._set_finetune_mode(finetune_mode)


    def _build_model(self, text_model):
        if text_model == "open_clip":
            return _build_openclip()
        elif text_model == "geoclip":
            return _build_geoclip()
        else:
            raise NotImplementedError("Other text models are not implemented yet")
        
    def _set_finetune_mode(self, finetune_mode):
        # We always want to finetune the embed project layer
        # Sometimes, we may want to also finetune the whole text encoder
        if finetune_mode == "all":
            self.text_encoder._requires_grad(True)
            self.text_encoder.eval()

        elif finetune_mode == "only_proj":
            self.text_encoder._requires_grad(False)

        if self.embed_project is not None:
            self.embed_project._requires_grad(True)

    def encode_texts(self, texts) -> torch.Tensor:
        tokens = self.tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
        input_ids = tokens.input_ids # [B, L]

        text_embeddings = self.text_encoder(input_ids)

        if isinstance(text_embeddings, dict) and 'text_embeds' in text_embeddings:
            text_embeddings = text_embeddings['text_embeds']

        text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)

        return text_embeddings

    def project_embeddings(self, embeddings) -> torch.Tensor:
        # Sometimes, we may want to directly use this if we've already precomputed embeddings
        if self.embed_project is not None:
            embeddings = self.embed_project(embeddings)
            
            # Normalize output again
            embeddings = nn.functional.normalize(embeddings, dim=-1)

        return embeddings

    def forward(self, texts) -> torch.Tensor:
        return self.project_embeddings(self.encode_texts(texts))
