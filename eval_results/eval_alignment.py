import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import GeoTextDataset
from main import build_model


def _recall_at_k(
    query_emb: torch.Tensor,
    key_emb: torch.Tensor,
    ks=(1, 5, 10),
    chunk_size: int = 512,
) -> dict[str, float]:
    """
    Chunked recall computation — avoids materialising the full (N, N) matrix.
    query_emb / key_emb: (N, D), L2-normalised.
    The ground-truth match for query i is key i.
    """
    n = query_emb.shape[0]
    ranks = torch.empty(n, dtype=torch.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # (chunk, N) similarity for this block of queries
        sim_chunk = query_emb[start:end] @ key_emb.T
        # rank of correct key (diagonal of the full matrix = index start..end)
        sorted_idx = sim_chunk.argsort(dim=1, descending=True)   # (chunk, N)
        correct = torch.arange(start, end, device=query_emb.device)
        chunk_ranks = (sorted_idx == correct.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
        ranks[start:end] = chunk_ranks.float() + 1  # 1-indexed

    results = {}
    for k in ks:
        results[f"R@{k}"] = (ranks <= k).float().mean().item()
    results["MedR"] = ranks.median().item()
    results["MRR"] = (1.0 / ranks).mean().item()
    return results


def _alignment(text_emb: torch.Tensor, loc_emb: torch.Tensor) -> float:
    """Mean cosine similarity of matched (text_i, loc_i) pairs."""
    return F.cosine_similarity(text_emb, loc_emb, dim=-1).mean().item()


def _uniformity(emb: torch.Tensor, t: float = 2.0, max_samples: int = 8192) -> float:
    """
    Uniformity loss from Wang & Isola 2020.
    uniformity = log( mean exp(-t * ||u - v||^2) )  over all pairs u != v.
    Embeddings are assumed to already be L2-normalised.

    Subsamples to max_samples when N is large — the expectation is well
    approximated by a random subset, avoiding the O(N²) full cdist.
    """
    n = emb.shape[0]
    if n > max_samples:
        idx = torch.randperm(n, device=emb.device)[:max_samples]
        emb = emb[idx]
        n = max_samples
    # ||u - v||^2 = 2 - 2*<u,v> for L2-normalised vectors (avoids cdist)
    sq_dist = 2 - 2 * (emb @ emb.T)
    mask = ~torch.eye(n, dtype=torch.bool, device=emb.device)
    return sq_dist[mask].mul(-t).exp().mean().log().item()


# ── main ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(checkpoint_path: str, dataset_path: str | None, split: str, device: str, batch_size: int, num_samples: int | None = None):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_args = argparse.Namespace(**ckpt["args"])

    if dataset_path is None:
        dataset_path = saved_args.dataset_path
    print(f"Dataset : {dataset_path}  |  split={split}")
    print(f"Checkpoint : {checkpoint_path}  (epoch {ckpt['epoch']}, best_val_loss={ckpt['best_val_loss']:.4f})")

    # Reconstruct model from saved args
    model = build_model(saved_args, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = GeoTextDataset(
        root=dataset_path,
        split=split,
        precomputed_text_embeddings=getattr(saved_args, "precomputed_text_embeddings", True),
        precomputed_location_embeddings=getattr(saved_args, "precomputed_location_embeddings", True),
    )
    if num_samples is not None and num_samples < len(dataset):
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_text, all_loc = [], []
    for locs, texts in loader:
        if isinstance(texts, torch.Tensor):
            texts = texts.to(device)
        locs = locs.to(device)
        text_emb, loc_emb = model(texts, locs)
        all_text.append(text_emb.cpu())
        all_loc.append(loc_emb.cpu())

    text_emb = torch.cat(all_text)   # (N, D)
    loc_emb  = torch.cat(all_loc)    # (N, D)
    # Both are already L2-normalised by the encoders
    n = text_emb.shape[0]
    print(f"\nEvaluating on {n} samples  (embedding dim = {text_emb.shape[1]})\n")

    t2l = _recall_at_k(text_emb, loc_emb,  ks=(1, 5, 10), chunk_size=batch_size)
    l2t = _recall_at_k(loc_emb,  text_emb, ks=(1, 5, 10), chunk_size=batch_size)
    align    = _alignment(text_emb, loc_emb)
    uni_text = _uniformity(text_emb)
    uni_loc  = _uniformity(loc_emb)

    # ── print results ────────────────────────────────────────────────────────
    width = 40
    print("─" * width)
    print(f"{'Text → Location retrieval':^{width}}")
    print("─" * width)
    for k in (1, 5, 10):
        print(f"  R@{k:<3}  {t2l[f'R@{k}']:.3f}")
    print(f"  MedR   {t2l['MedR']:.1f}  (of {n})")
    print(f"  MRR    {t2l['MRR']:.4f}")

    print()
    print("─" * width)
    print(f"{'Location → Text retrieval':^{width}}")
    print("─" * width)
    for k in (1, 5, 10):
        print(f"  R@{k:<3}  {l2t[f'R@{k}']:.3f}")
    print(f"  MedR   {l2t['MedR']:.1f}  (of {n})")
    print(f"  MRR    {l2t['MRR']:.4f}")

    print()
    print("─" * width)
    print(f"{'Alignment & Uniformity':^{width}}")
    print("─" * width)
    print(f"  Alignment     {align:.4f}   (↑ higher = matched pairs closer)")
    print(f"  Uniformity (text) {uni_text:.4f}  (↓ lower = more spread out)")
    print(f"  Uniformity (loc)  {uni_loc:.4f}  (↓ lower = more spread out)")
    print("─" * width)

    return {
        "t2l": t2l, "l2t": l2t,
        "alignment": align, "uniformity_text": uni_text, "uniformity_loc": uni_loc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True, help="Path to best.pt")
    parser.add_argument("--dataset_path",  default=None,  help="Override dataset path from checkpoint args")
    parser.add_argument("--split",         default="test", choices=["train", "val", "test"])
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--batch_size",    type=int, default=512)
    parser.add_argument("--num_samples",   type=int, default=None, help="Randomly sample this many examples before eval (seed=42)")
    parser.add_argument("--output",        default=None,  help="Save printed results to this .txt file")
    args = parser.parse_args()

    if args.output:
        import io, sys as _sys
        buf = io.StringIO()
        _tee = type("Tee", (), {
            "write":  lambda s, x: (buf.write(x), _sys.__stdout__.write(x)),
            "flush":  lambda s:    _sys.__stdout__.flush(),
        })()
        _sys.stdout = _tee

    evaluate(args.checkpoint, args.dataset_path, args.split, args.device, args.batch_size, args.num_samples)

    if args.output:
        _sys.stdout = _sys.__stdout__
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(buf.getvalue())
