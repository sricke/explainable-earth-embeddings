"""Text chunking and chunk-embedding pooling utilities.

These are used by TextEncoder to split long texts into tokenizer-safe
chunks and then pool chunk embeddings back into one vector per sample.
"""

from __future__ import annotations

import re

import torch


def max_text_tokens(tokenizer, text_model_type: str) -> int:
    """Return the effective max token count for the given backend."""
    if text_model_type == "geoclip":
        tok_max = int(getattr(tokenizer, "model_max_length", 77))
        if tok_max > 100_000:
            tok_max = 77
        return min(tok_max, 77)

    # OpenCLIP default
    return 77


def token_length(text: str, tokenizer, text_model_type: str) -> int:
    """Count tokens for *text* using a HuggingFace-style tokenizer."""
    if text_model_type in {"geoclip"}:
        encoded = tokenizer(
            [text],
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        return int(encoded["input_ids"].shape[1])

    raise NotImplementedError(
        "Strict no-truncation token-length checks are only implemented "
        "for geoclip text_model_type."
    )


def split_text_units(text: str, granularity: str) -> list[str]:
    """Split *text* into sentence or phrase units, preserving all characters."""
    # if granularity is None, return the text as a single unit
    # if text is empty, return an empty list
    if not text or granularity is None:
        return [text] #return the text as a single unit
    if granularity == "sentence":
        units = re.findall(r"[^.!?]+[.!?]*\s*", text, flags=re.MULTILINE)
    elif granularity == "phrase":
        units = re.findall(r"[^,;:.!?]+[,;:.!?]*\s*", text, flags=re.MULTILINE)
    else:
        raise ValueError(
            f"Unsupported granularity={granularity!r}. Use 'sentence' or 'phrase'."
        )
    units = [u for u in units if u]
    return units or [text]


def chunk_text_strict(
    text: str,
    granularity: str | None,
    tokenizer,
    text_model_type: str,
) -> list[str]:
    """Split *text* into chunks that each fit within the tokenizer limit.

    Raises if any single linguistic unit exceeds the limit or if the
    reconstruction doesn't match the original text exactly.
    """
    if granularity is None:
        return [text]

    limit = max_text_tokens(tokenizer, text_model_type)
    units = split_text_units(text, granularity)

    chunks: list[str] = []
    current = ""
    for unit in units:
        unit_len = token_length(unit, tokenizer, text_model_type)
        if unit_len > limit:
            raise ValueError(
                f"Single {granularity} exceeds tokenizer limit "
                f"({unit_len}>{limit}) and cannot be represented without truncation."
            )
        candidate = current + unit
        if token_length(candidate, tokenizer, text_model_type) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = unit

    if current:
        chunks.append(current)

    if "".join(chunks) != text:
        raise RuntimeError("Chunking altered text content; refusing to continue.")
    return chunks


def pool_chunk_embeddings(
    chunk_emb: torch.Tensor,
    chunk_lengths: list[int],
    method: str = "mean",
) -> torch.Tensor:
    """Aggregate *chunk_emb* ``[n_chunks, dim]`` into a single vector."""
    if chunk_emb.ndim != 2:
        raise ValueError(
            f"Expected chunk_emb [n_chunks, dim], got shape {tuple(chunk_emb.shape)}"
        )
    if method == "mean":
        return chunk_emb.mean(dim=0)
    if method == "length_weighted_mean":
        w = torch.tensor(
            chunk_lengths, dtype=chunk_emb.dtype, device=chunk_emb.device
        ).clamp_min(1.0)
        w = w / w.sum()
        return (chunk_emb * w.unsqueeze(1)).sum(dim=0)
    if method == "max":
        return chunk_emb.max(dim=0).values
    raise ValueError(
        f"Unsupported pooling method={method!r}. "
        "Use 'mean', 'length_weighted_mean', or 'max'."
    )
