from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F

from external.splice.splice import SPLICE


def cosine_to_vocab(location_embeddings: torch.Tensor, vocab_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Returns cosine similarities.
    - location_embeddings: [D] or [B, D]
    - vocab_embeddings:    [C, D]
    - returns:             [C] or [B, C]
    """
    loc = F.normalize(location_embeddings, dim=-1)
    vocab = F.normalize(vocab_embeddings, dim=-1)

    if loc.ndim == 1:
        return vocab @ loc  # [C]
    return loc @ vocab.T   # [B, C]


def run_splice_for_vocab(
    device: torch.device,
    batch_size: int,
    base_df: pd.DataFrame,
    latlon_embeddings: torch.Tensor,
    mean_embedding_locs: torch.Tensor,
    vocab_words: List[str],
    vocab_embs: torch.Tensor,
    l1_penalty: float,
    out_path: Path,
    top_k: int,
):
    """Run SpLiCE decomposition for a single vocab and write a parquet file."""
    splicemodel = SPLICE(
        mean_embedding_locs,
        vocab_embs,
        clip=None,
        device=str(device),
        solver="skl",
        return_weights=True,
        return_cosine=True,
        l1_penalty=l1_penalty,
    )

    n = len(latlon_embeddings)
    print(
        f"[SpLiCE] Running decomposition for {n} locations, "
        f"vocab size={vocab_embs.shape[0]}, "
        f"embedding_dim={vocab_embs.shape[1]}, "
        f"l1_penalty={l1_penalty}"
    )
    results = []
    all_active_counts: list[int] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = latlon_embeddings[start:end]

        sparse_chunk = splicemodel.decompose(chunk)
        recon_chunk = splicemodel.recompose_image(sparse_chunk)
        mse_chunk = F.mse_loss(recon_chunk, chunk, reduction="none").mean(dim=1)

        print_decomposition_batch_summary(
            start_idx=start,
            end_idx=end,
            chunk=chunk,
            vocab_embs=vocab_embs,
            base_df=base_df,
            mse_chunk=mse_chunk,
            sparse_chunk=sparse_chunk,
            vocab_words=vocab_words,
            top_k=top_k,
        )
        batch_active = (sparse_chunk > 0).sum(dim=1).tolist()
        all_active_counts.extend(int(x) for x in batch_active)

        for i in range(end - start):
            idx = start + i
            row = base_df.iloc[idx]
            record = {
                "lat": row["lat"],
                "lon": row["lon"],
                "fn": row["fn"],
                "mse": mse_chunk[i].item(),
                "row_idx": idx,
                "active_concepts": batch_active[i],
            }
            for c_idx, word in enumerate(vocab_words):
                record[word] = sparse_chunk[i, c_idx].item()
            results.append(record)

    if all_active_counts:
        active_tensor = torch.tensor(all_active_counts, dtype=torch.float32)
        print(
            "Overall active concepts per location: "
            f"mean={active_tensor.mean().item():.2f}, "
            f"min={active_tensor.min().item():.0f}, "
            f"max={active_tensor.max().item():.0f}"
        )

    df_out = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    return df_out


def print_decomposition_batch_summary(
    start_idx: int,
    end_idx: int,
    chunk: torch.Tensor,
    vocab_embs: torch.Tensor,
    base_df: pd.DataFrame,
    mse_chunk: torch.Tensor,        # shape [B]
    sparse_chunk: torch.Tensor,     # shape [B, C]
    vocab_words: List[str],
    top_k: int = 5,
) -> None:
    """Print summary stats and a few example decompositions for one batch."""
    batch_size = end_idx - start_idx
    print(f"\nBatch {start_idx}–{end_idx} (size={batch_size})")
    print(
        f"  MSE stats: mean={mse_chunk.mean().item():.4f}, "
        f"min={mse_chunk.min().item():.4f}, "
        f"max={mse_chunk.max().item():.4f}"
    )

    # active concept counts for this batch
    active_counts = (sparse_chunk > 0).sum(dim=1)  # [B]
    if batch_size > 0:
        print(
            "  Active concepts per location: "
            f"mean={active_counts.float().mean().item():.2f}, "
            f"min={active_counts.min().item():.0f}, "
            f"max={active_counts.max().item():.0f}"
        )

    # pick up to 3 example indices within this batch: first, middle, last
    if batch_size == 0:
        return
    example_local_indices = [0]
    if batch_size > 2:
        example_local_indices.append(batch_size // 2)
    if batch_size > 1:
        example_local_indices.append(batch_size - 1)
    example_local_indices = sorted(set(example_local_indices))

    for local_i in example_local_indices:
        global_idx = start_idx + local_i
        row = base_df.iloc[global_idx]
        mse_val = mse_chunk[local_i].item()
        weights = sparse_chunk[local_i]  # [C]

        num_active = (weights > 0).sum().item()
        print(
            f"  Location idx={global_idx} "
            f"(lat={row['lat']:.4f}, lon={row['lon']:.4f}, fn={row['fn']}, mse={mse_val:.4f})"
        )
        print(f"    active concepts: {int(num_active)}")

        top_vals, top_idx = torch.topk(weights, k=min(top_k, weights.shape[0]))
        concepts = [vocab_words[j] for j in top_idx.tolist()]
        for concept, wt in zip(concepts, top_vals.tolist()):
            print(f"    {concept}: {wt:.4f}")

        cos = cosine_to_vocab(chunk[local_i], vocab_embs)  # [C]
        cos_vals, cos_idx = torch.topk(cos, k=min(top_k, cos.shape[0]))
        print(f"    Top-{min(top_k, cos.shape[0])} cosine-similar vocab terms:")
        for j, s in zip(cos_idx.tolist(), cos_vals.tolist()):
            print(f"      {vocab_words[j]}: {s:.4f}")


def summarize_to_text(
    df: pd.DataFrame,
    vocab_name: str,
    model_label: str,
    ckpt_path: Path | None,
    l1_penalty: float,
    out_txt: Path,
    vocab_words: List[str],
    vocab_embs: torch.Tensor,
    embeddings: torch.Tensor | None,
    top_k: int,
):
    """Generate a human-readable text report from a SpLiCE decomposition parquet."""
    buf = StringIO()
    buf.write(f"=== SpLiCE report for vocab: {vocab_name} ===\n")
    buf.write(f"Model: {model_label}\n")
    buf.write(f"Checkpoint: {ckpt_path if ckpt_path is not None else '(GeoCLIP init only)'}\n")
    buf.write(f"L1 penalty: {l1_penalty}\n\n")

    # Reconstruction stats
    buf.write("Reconstruction quality stats:\n")
    buf.write(f"Mean MSE: {df['mse'].mean():.3f}\n")
    buf.write(f"Min MSE:  {df['mse'].min():.3f}\n")
    buf.write(f"Max MSE:  {df['mse'].max():.3f}\n\n")

    drop_cols = ["lat", "lon", "fn", "mse", "row_idx"]
    concept_cols = df.columns.drop([c for c in drop_cols if c in df.columns])

    # Most common concepts
    sparsity = (df[concept_cols] > 0).sum()
    most_common = sparsity.nlargest(20)
    buf.write("Top 20 most frequently used concepts:\n")
    buf.write(most_common.to_string())
    buf.write("\n\n")

    # Average active concepts per location
    avg_nonzero = (df[concept_cols] > 0).sum(axis=1).mean()
    buf.write(f"Average number of active concepts per location: {avg_nonzero:.1f}\n\n")

    # Example decompositions: best / median / worst MSE
    if len(df) > 0:
        buf.write(f"Example decompositions (top {top_k} concepts per location):\n")
        df_sorted = df.sort_values("mse").reset_index(drop=True)

        example_indices: list[int] = [0]
        if len(df_sorted) > 2:
            example_indices.append(len(df_sorted) // 2)
        if len(df_sorted) > 1:
            example_indices.append(len(df_sorted) - 1)
        example_indices = sorted(set(example_indices))

        for idx in example_indices:
            loc = df_sorted.iloc[idx]
            weights = pd.to_numeric(loc[concept_cols], errors="coerce")
            top_concepts = weights.nlargest(top_k)

            buf.write(
                f"\nLocation idx={idx} "
                f"(lat={loc['lat']:.4f}, lon={loc['lon']:.4f}, mse={loc['mse']:.4f}, fn={loc.get('fn', 'N/A')}):\n"
            )
            buf.write(top_concepts.to_string())
            buf.write("\n")

            if "row_idx" in loc and embeddings is not None:
                row_i = int(loc["row_idx"])
                emb = embeddings[row_i]            # [D]
                cos = cosine_to_vocab(emb, vocab_embs)  # [C]
                cos_vals, cos_idx = torch.topk(cos, k=min(top_k, cos.shape[0]))

                buf.write(f"Top-{min(top_k, cos.shape[0])} cosine-similar vocab terms:\n")
                for j, s in zip(cos_idx.tolist(), cos_vals.tolist()):
                    buf.write(f"  {vocab_words[j]}: {s:.4f}\n")
                buf.write("\n")

        # All named-location decompositions: one block per known location
        buf.write(
            f"\nAll named-location decompositions (top {top_k} concepts per location):\n"
        )
        df_reset = df.reset_index(drop=True)
        for idx, loc in df_reset.iterrows():
            weights = pd.to_numeric(loc[concept_cols], errors="coerce")
            top_concepts = weights.nlargest(top_k)

            buf.write(
                f"\nLocation '{loc.get('fn', 'N/A')}' "
                f"(lat={loc['lat']:.4f}, lon={loc['lon']:.4f}, mse={loc['mse']:.4f}):\n"
            )
            buf.write(top_concepts.to_string())
            buf.write("\n")

            if "row_idx" in loc and embeddings is not None:
                row_i = int(loc["row_idx"])
                emb = embeddings[row_i]
                cos = cosine_to_vocab(emb, vocab_embs)
                cos_vals, cos_idx = torch.topk(cos, k=min(top_k, cos.shape[0]))

                buf.write(f"Top-{min(top_k, cos.shape[0])} cosine-similar vocab terms:\n")
                for j, s in zip(cos_idx.tolist(), cos_vals.tolist()):
                    buf.write(f"  {vocab_words[j]}: {s:.4f}\n")
                buf.write("\n")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(buf.getvalue())

