#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from modeling.text import TextEncoder


def _resolve_text_column(df: pd.DataFrame, csv_path: Path) -> str:
    if "description" in df.columns:
        return "description"
    if "text" in df.columns:
        return "text"
    raise ValueError(
        f"{csv_path} must contain either 'description' or 'text' column."
    )


def _sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def _encode_texts(
    encoder: TextEncoder,
    texts: list[str],
    desc: str,
    batch_size: int,
) -> torch.Tensor:
    all_embeddings: list[torch.Tensor] = []

    encoder.eval()
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc=desc,
            unit="batch",
        ):
            batch_texts = texts[start:start + batch_size]
            emb = encoder(batch_texts)
            all_embeddings.append(emb.detach().cpu())

    if not all_embeddings:
        raise ValueError("No texts found to encode.")
    return torch.cat(all_embeddings, dim=0)
