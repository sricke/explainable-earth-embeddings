import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


def embed_vocab(model, device: torch.device, vocab_path: Path) -> Tuple[List[str], torch.Tensor]:
    """Embed all concepts from a vocab JSON file using the text model."""
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    reserved = {"lat", "lon", "fn", "mse", "row_idx"}
    if isinstance(vocab, dict):
        words_raw = list(vocab.keys())
    else:
        words_raw = list(vocab)
    vocab = [w for w in words_raw if w not in reserved]
    print(f"[SpLiCE] Loaded vocab '{vocab_path}': {len(vocab)} usable concepts (raw size={len(words_raw)})")

    words: list[str] = []
    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for word in tqdm(vocab, desc=f"Embedding vocab: {vocab_path.name}"):
            emb = model.text_model_predict(word, normalize=True)
            if emb.ndim == 2:
                emb = emb.squeeze(0)
            embeddings.append(emb.cpu())
            words.append(word)

    torch.cuda.empty_cache()

    emb_tensor = torch.stack(embeddings).to(device)
    print(f"[SpLiCE] Vocab embedding tensor shape before norm: {emb_tensor.shape}")
    emb_tensor = F.normalize(emb_tensor, dim=1)
    emb_tensor = F.normalize(
        emb_tensor - emb_tensor.mean(dim=0, keepdim=True),
        dim=1,
    )
    return words, emb_tensor


def vocab_size(vocab_path: Path) -> int:
    """Return the number of concepts in a vocab JSON file."""
    with vocab_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return len(data)
    if isinstance(data, list):
        return len(data)
    raise ValueError(f"Unexpected vocab format in {vocab_path}: {type(data)}")

