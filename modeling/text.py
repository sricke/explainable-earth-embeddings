from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.chunking import (
    chunk_text_strict,
    max_text_tokens,
    pool_chunk_embeddings,
    token_length,
)
from modeling.model_specs import DEFAULT_TEXT_CONFIGS, resolve_text_config

# ---------------------------------------------------------------------------
# Backend builders – each returns (model, tokenizer, output_dim)
# ---------------------------------------------------------------------------

def _build_openclip(model_id: str, pretrained: str):
    import open_clip
    model = open_clip.create_model(model_id, pretrained=pretrained)
    del model.visual
    tokenizer = open_clip.get_tokenizer(model_id)
    output_dim = model.text_projection.shape[1]
    return model, tokenizer, output_dim


def _build_geoclip():
    from geoclip import GeoCLIP
    from transformers import AutoTokenizer

    geoclip = GeoCLIP()
    clip_model = geoclip.image_encoder.CLIP
    mlp = geoclip.image_encoder.mlp
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    class CLIPTextWithMLP(nn.Module):
        def __init__(self, clip_model, mlp):
            super().__init__()
            self.clip_model = clip_model
            self.mlp = mlp

        def forward(self, **kwargs):
            feats = self.clip_model.get_text_features(**kwargs)
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output
            return F.normalize(self.mlp(feats), dim=-1)

    model = CLIPTextWithMLP(clip_model, mlp)
    return model, tokenizer, 512


def _build_gritlm(model_id: str):
    from gritlm import GritLM
    model = GritLM(model_id, mode="embedding", torch_dtype="auto")
    tokenizer = model.tokenizer
    backbone = model.model.module if hasattr(model.model, "module") else model.model
    output_dim = int(backbone.config.hidden_size)
    return model, tokenizer, output_dim


# ---------------------------------------------------------------------------
# Uses HF-style tokenization (kwargs dict) vs OpenCLIP-style (single tensor).
# ---------------------------------------------------------------------------
_HF_STYLE_BACKENDS = {"geoclip"}

# ---------------------------------------------------------------------------
# TextEncoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Encodes text into fixed-size embeddings.

    Supported backends (selected via *backend*):
      - ``"geoclip"``  – GeoCLIP CLIP + MLP  (77-token CLIP limit)
      - ``"gritlm"``   – GritLM / any HF causal model  (long context)
      - ``"ViT-B-32"`` – OpenCLIP text encoder  (77-token limit)
    """

    DEFAULT_CONFIGS = DEFAULT_TEXT_CONFIGS

    def __init__(
        self,
        backend: str = "geoclip",
        model_id: str | None = None,
        pretrained: str | None = None,
        trainable: bool | None = None,
        projection_head: str = "linear", # linear, mlp2, two_layer_mlp
        projection_hidden_dim: int | None = None,
        projection_dropout: float = 0.0,
        *,
        # legacy aliases (for old configs/scripts)
        text_model_type: str | None = None,
        text_model: str | None = None,
        text_vocabulary: str | None = None,
        train_text_model: bool | None = None,
        target_dim: int | None = None,
        dtype: torch.dtype = torch.float32,
        text_chunk_granularity: str | None = None,
        text_chunk_pooling: str = "mean",
    ):
        super().__init__()
        self.config = resolve_text_config(
            backend=backend,
            model_id=model_id,
            pretrained=pretrained,
            legacy_type=text_model_type,
            legacy_model=text_model,
            legacy_vocabulary=text_vocabulary,
        )
        self.text_model_type = self.config["backend"]
        self.dtype = dtype
        self.text_chunk_granularity = text_chunk_granularity
        self.text_chunk_pooling = text_chunk_pooling
        self.projection_head = projection_head
        self.projection_hidden_dim = projection_hidden_dim
        self.projection_dropout = float(projection_dropout)
        if self.projection_dropout < 0.0 or self.projection_dropout >= 1.0:
            raise ValueError("projection_dropout must be in [0.0, 1.0).")
        if trainable is None:
            trainable = bool(train_text_model)

        # ---------- build backend ----------
        if self.text_model_type == "openclip":
            model, tokenizer, output_dim = _build_openclip(
                self.config["model_id"], self.config["pretrained"]
            )
        elif self.text_model_type == "geoclip":
            model, tokenizer, output_dim = _build_geoclip()
        elif self.text_model_type == "gritlm":
            model, tokenizer, output_dim = _build_gritlm(self.config["model_id"])
        else:
            raise ValueError(f"Unsupported text backend: {self.text_model_type}")

        self.model = model
        self.tokenizer = tokenizer
        self.text_output_dim = output_dim

        self.model.requires_grad_(trainable)
        self.train_encoder = trainable
        if not trainable:
            self.model.eval()

        # ---------- optional projection ----------
        self.embed_project = None
        if target_dim is not None:
            needs_projection = (target_dim != output_dim) or (projection_head != "none")
            if needs_projection:
                if projection_head in {"linear", "none"}:
                    self.embed_project = nn.Linear(output_dim, target_dim)
                    nn.init.normal_(self.embed_project.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(self.embed_project.bias)
                elif projection_head in {"mlp2", "two_layer_mlp"}:
                    hidden = projection_hidden_dim or max(output_dim, target_dim)
                    layers: list[nn.Module] = [nn.Linear(output_dim, hidden), nn.GELU()]
                    if self.projection_dropout > 0.0:
                        layers.append(nn.Dropout(self.projection_dropout))
                    layers.append(nn.Linear(hidden, target_dim))
                    self.embed_project = nn.Sequential(*layers)
                    for module in self.embed_project:
                        if isinstance(module, nn.Linear):
                            nn.init.normal_(module.weight, mean=0.0, std=0.02)
                            nn.init.zeros_(module.bias)
                else:
                    raise ValueError(
                        f"Unsupported projection_head={projection_head!r}. "
                        "Use 'none', 'linear', or 'mlp2'."
                    )
        self.target_dim = target_dim

    # ---- forwarding helpers ------------------------------------------------

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _tokenize(self, texts: list[str]) -> dict | torch.Tensor:
        if self.text_model_type in _HF_STYLE_BACKENDS:
            tokens = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            return {k: v.to(self.device) for k, v in tokens.items()}
        return self.tokenizer(texts).to(self.device)

    def _encode_tokens(self, tokens) -> torch.Tensor:
        with torch.set_grad_enabled(self.train_encoder):
            if isinstance(tokens, dict):
                return self.model(**tokens)
            return self.model.encode_text(tokens)

    def _validate_unchunked_token_limits(self, texts: list[str]) -> None:
        if self.text_model_type not in _HF_STYLE_BACKENDS:
            return
        limit = max_text_tokens(self.tokenizer, self.text_model_type)
        for i, txt in enumerate(texts):
            n = token_length(txt, self.tokenizer, self.text_model_type)
            if n > limit:
                raise ValueError(
                    f"Text at index {i} has {n} tokens > limit {limit}. "
                    "Set text_chunk_granularity to avoid truncation."
                )

    def _prepare_text_batches(
        self, texts: list[str], chunking: bool
    ) -> tuple[list[str], list[int], list[int]]:
        if not chunking:
            self._validate_unchunked_token_limits(texts)
            sample_map = list(range(len(texts)))
            chunk_lens = [1] * len(texts)
            return texts, sample_map, chunk_lens

        flat_texts: list[str] = []
        sample_map: list[int] = []
        chunk_lens: list[int] = []
        for idx, txt in enumerate(texts):
            chunks = chunk_text_strict(
                txt,
                self.text_chunk_granularity,
                self.tokenizer,
                self.text_model_type,
            )
            for ch in chunks:
                flat_texts.append(ch)
                sample_map.append(idx)
                chunk_lens.append(token_length(ch, self.tokenizer, self.text_model_type))
        return flat_texts, sample_map, chunk_lens

    def _pool_embeddings_by_sample(
        self,
        embedding: torch.Tensor,
        sample_map: list[int],
        chunk_lens: list[int],
        n_samples: int,
    ) -> torch.Tensor:
        pooled: list[torch.Tensor] = []
        for sid in range(n_samples):
            idxs = [i for i, m in enumerate(sample_map) if m == sid]
            pooled.append(
                pool_chunk_embeddings(
                    embedding[idxs],
                    [chunk_lens[i] for i in idxs],
                    method=self.text_chunk_pooling,
                )
            )
        return torch.stack(pooled, dim=0)

    # ---- public interface --------------------------------------------------

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            embedding = x.to(device=self.device, dtype=self.dtype)
            if self.embed_project is not None:
                if embedding.shape[-1] == self.text_output_dim:
                    embedding = self.embed_project(embedding)
                elif self.target_dim is not None and embedding.shape[-1] != self.target_dim:
                    raise ValueError(
                        "Cached embedding dimension does not match either "
                        f"text_output_dim={self.text_output_dim} or target_dim={self.target_dim}. "
                        f"Got shape {tuple(embedding.shape)}."
                    )
            return embedding

        if isinstance(x, tuple):
            x = list(x)
        if isinstance(x, str):
            x = [x]

        # GritLM supports long contexts, so skip chunking entirely.
        chunking = (
            self.text_chunk_granularity is not None
            and self.text_model_type != "gritlm"
        )

        texts, sample_map, chunk_lens = self._prepare_text_batches(x, chunking)

        # --- encode ---
        if self.text_model_type == "gritlm":
            embedding = self.model.encode(
                texts, convert_to_tensor=True,
            ).to(device=self.device, dtype=self.dtype)
        else:
            tokens = self._tokenize(texts)
            embedding = self._encode_tokens(tokens).to(device=self.device, dtype=self.dtype)

        if self.embed_project is not None:
            embedding = self.embed_project(embedding)

        if not chunking:
            return embedding

        # --- pool chunks back to one embedding per sample ---
        return self._pool_embeddings_by_sample(
            embedding, sample_map, chunk_lens, n_samples=len(x)
        )
