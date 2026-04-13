from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import EmbeddingProjection

TEXT_MODEL_IDS = {
    "open_clip": ("ViT-L-14", "openai"),
}

TEXT_EMBEDDING_DIMENSIONS = {
    "open_clip": 768, # Double check this
    "geoclip": 512,
}


def _build_openclip():
    import open_clip
    import types

    model_id, pretrained = TEXT_MODEL_IDS["open_clip"]
    clip = open_clip.create_model(model_id, pretrained=pretrained)
    del clip.visual
    output_dim = clip.text_projection.shape[1]

    # Wrap model so forward() calls encode_text
    class _OpenCLIPText(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids): return self.m.encode_text(input_ids)

    # Wrap tokenizer so it returns an object with .input_ids 
    raw_tok = open_clip.get_tokenizer(model_id)
    def tokenizer(texts, **_):
        return types.SimpleNamespace(input_ids=raw_tok(texts))

    return _OpenCLIPText(clip), tokenizer, output_dim


def _build_geoclip():
    from geoclip import GeoCLIP
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    geoclip = GeoCLIP()
    clip_model = geoclip.image_encoder.CLIP
    mlp = geoclip.image_encoder.mlp
    output_dim = 512

    class CLIPTextWithMLP(nn.Module):
        def __init__(self, clip_model, mlp):
            super().__init__()
            self.clip_model = clip_model
            self.mlp = mlp

        def forward(self, input_ids):
            """input_ids: already tokenized, shape [B, L]"""
            text_features = self.clip_model.encode_text(input_ids)
            text_proj = self.mlp(text_features)
            return F.normalize(text_proj, dim=-1)

    model = CLIPTextWithMLP(clip_model, mlp)
    return model, tokenizer, output_dim


class TextEncoder(nn.Module):
    """Encodes text into fixed-size embeddings."""

    def __init__(
        self,
        text_model: str = None,
        embed_project: EmbeddingProjection = None,
        finetune_mode: str = None,
    ):
        super().__init__()
        assert text_model is not None, "Need to specify text model"
        self.text_model = text_model
        self.text_encoder, self.tokenizer, self.output_dim = self._build_model(text_model)
        self.embed_project = embed_project
        self._set_finetune_mode(finetune_mode)

    def _build_model(self, text_model: str):
        if text_model == "open_clip":
            return _build_openclip()
        elif text_model == "geoclip":
            return _build_geoclip()
        else:
            raise NotImplementedError(f"Text model '{text_model}' is not implemented")

    def _set_finetune_mode(self, finetune_mode: str):
        assert finetune_mode in ['all', 'only_proj']
        if finetune_mode == "all":
            self.text_encoder.requires_grad_(True)
            self.text_encoder.train()
        elif finetune_mode == "only_proj":
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

        if self.embed_project is not None:
            self.embed_project.requires_grad_(True)

    def encode_texts(self, texts) -> torch.Tensor:
        tokens = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        input_ids = tokens.input_ids  # [B, L]
        text_embeddings = self.text_encoder(input_ids)

        if isinstance(text_embeddings, dict):
            text_embeddings = text_embeddings["text_embeds"]

        return F.normalize(text_embeddings, dim=-1)

    def project_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply projection head. Use this when embeddings are precomputed."""
        if self.embed_project is not None:
            embeddings = F.normalize(self.embed_project(embeddings), dim=-1)
        return embeddings

    def forward(self, texts) -> torch.Tensor:
        return self.project_embeddings(self.encode_texts(texts))
