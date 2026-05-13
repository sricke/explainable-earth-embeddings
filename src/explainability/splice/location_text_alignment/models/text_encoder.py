from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import EmbeddingProjection

TEXT_MODEL_IDS = {
    "open_clip": "openai/clip-vit-large-patch14",
    "open_clip_vit_l": "openai/clip-vit-large-patch14",
    "open_clip_vit_h": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}

TEXT_EMBEDDING_DIMENSIONS = {
    "open_clip": 768,
    "open_clip_vit_l": 768,
    "open_clip_vit_h": 1024,
    "geoclip": 512,
}


def _build_openclip(variant: str = "open_clip"):
    """
    Builds an OpenCLIP model.
    """
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    model_id = TEXT_MODEL_IDS[variant]
    clip = CLIPTextModelWithProjection.from_pretrained(model_id)
    output_dim = clip.config.projection_dim

    # Simplified class with just the text encoder
    class CLIPText(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask=None):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).text_embeds

    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    return CLIPText(clip), tokenizer, output_dim


def _build_geoclip():
    """
    Builds GeoCLIP text encoder (which is also OpenCLIP model + MLP with trained weights from GeoCLIP)
    """
    from geoclip import GeoCLIP
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    geoclip = GeoCLIP()
    clip_model = geoclip.image_encoder.CLIP
    mlp = geoclip.image_encoder.mlp
    output_dim = 512

    # Class for OpenCLIP text encoder (used by GeoCLIP) + trained MLP from GeoCLIP
    class CLIPTextWithMLP(nn.Module):
        def __init__(self, clip_model, mlp):
            super().__init__()
            self.clip_model = clip_model
            self.mlp = mlp

        def forward(self, input_ids, attention_mask=None):
            """input_ids: already tokenized, shape [B, L]"""
            # GeoCLIP's MLP was trained on pooler_output, not text_embeds (no projection head)
            text_features = self.clip_model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).pooler_output
            text_proj = self.mlp(text_features)
            return F.normalize(text_proj, dim=-1)

    model = CLIPTextWithMLP(clip_model, mlp)
    return model, tokenizer, output_dim


class TextEncoder(nn.Module):
    """
    Text Encoder Class.

    Parameters:
        text_model: name of text model (open_clip_vit_l, open_clip_vit_h, geoclip)
        embed_project: linear or MLP layer (often needed to match the location embedding dimension)
        finetune_mode: all, lora, or only_proj
        precomputed: option to precompute embeddings if not finetuning the base model
    """

    def __init__(
        self,
        text_model: str = None,
        embed_project: EmbeddingProjection = None,
        finetune_mode: str = None,
        precomputed: bool = True,
        **lora_kwargs,
    ):
        super().__init__()
        assert text_model is not None, "Need to specify text model"
        self.text_model = text_model
        self.precomputed = precomputed
        self.embed_project = embed_project

        # Precomputed mode is incompatible with finetuning the base model
        if precomputed:
            assert finetune_mode not in ['all', 'lora'], "Cannot finetune all with finetune_mode with precomputed=True"
        else:
            self.text_encoder, self.tokenizer, self.output_dim = self._build_model(text_model)
            self._set_finetune_mode(finetune_mode, **lora_kwargs)

        # Always need to train the embedding projection since it's not initialized with pretrained weights
        if embed_project is not None:
            embed_project.requires_grad_(True)

    def _target_device(self):
        if self.embed_project is not None:
            return next(self.embed_project.parameters()).device
        if hasattr(self, "text_encoder"):
            return next(self.text_encoder.parameters()).device
        return None

    def _build_model(self, text_model: str):
        """
        Build model from specified text_model
        """
        if text_model in ("open_clip", "open_clip_vit_l", "open_clip_vit_h"):
            return _build_openclip(text_model)
        elif text_model == "geoclip":
            return _build_geoclip()
        else:
            raise NotImplementedError(f"Text model '{text_model}' is not implemented")

    def _set_finetune_mode(self, finetune_mode: str, lora_r: int = 4, lora_alpha: float = 1.0, lora_last_n_layers: int = None):
        """
        Set finetune mode
        """
        assert finetune_mode in ['all', 'lora', 'only_proj'], f"Finetune mode {finetune_mode} not accepted"
        if finetune_mode == "all":
            self.text_encoder.requires_grad_(True)
            self.text_encoder.train()
            self.text_encoder.m.gradient_checkpointing_enable()
        elif finetune_mode == "lora":
            from models.lora import apply_lora
            self.text_encoder.requires_grad_(False)
            self.text_encoder = apply_lora(self.text_encoder.m, r=lora_r, alpha=lora_alpha, last_n_layers=lora_last_n_layers)
            self.text_encoder.gradient_checkpointing_enable()  # recompute activations to save memory
            self.text_encoder.train()
        elif finetune_mode == "only_proj":
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

    def encode_texts(self, texts) -> torch.Tensor:
        """
        Tokenize and encode raw text into normalized embeddings. 
        Does not apply embed_project.
        """
        assert not self.precomputed, "Cannot call encode_texts in precomputed mode"

        device = next(self.text_encoder.parameters()).device

        # Tokenize if raw strings
        # otherwise expect a pre-tokenized dict/BatchEncoding from collate_fn
        if isinstance(texts, (str, list, tuple)):
            if isinstance(texts, tuple):
                texts = list(texts)
            tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
            input_ids = tokens.input_ids.to(device)
            attention_mask = tokens.attention_mask.to(device) if hasattr(tokens, "attention_mask") else None
        else:
            input_ids = texts["input_ids"].to(device)
            attention_mask = texts["attention_mask"].to(device) if "attention_mask" in texts else None

        text_embeddings = self.text_encoder(input_ids, attention_mask)
        if isinstance(text_embeddings, dict):
            text_embeddings = text_embeddings["text_embeds"]

        return F.normalize(text_embeddings, dim=-1)

    def forward(self, x) -> torch.Tensor:
        """
        Full encoding. Accepts either a precomputed embedding tensor or raw text.
        Applies embed_project on top if set.
        """

        embed_dim = TEXT_EMBEDDING_DIMENSIONS[self.text_model]

        if isinstance(x, torch.Tensor) and x.shape[-1] == embed_dim:
            target_device = self._target_device()
            if target_device is not None and x.device != target_device:
                x = x.to(target_device, non_blocking=True)
            embedding = F.normalize(x.to(dtype=torch.float32), dim=-1)
        else:
            assert not self.precomputed, "Precomputed mode expects a float tensor with the correct embedding dimension"
            embedding = self.encode_texts(x)

        # Project into the shared embedding space (to match location encoder dimension)
        if self.embed_project is not None:
            embedding = F.normalize(self.embed_project(embedding), dim=-1)
            
        return embedding
