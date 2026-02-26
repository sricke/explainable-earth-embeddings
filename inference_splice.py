#!/usr/bin/env python
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import yaml

from main import Location2TextLightningModule
from splice import SPLICE
from interpret_splice import (
    check_reconstruction_quality,
    find_most_common_concepts,
    analyze_sparsity,
)


@dataclass
class SpliceConfig:
    device: str
    num_samples: int | None
    batch_size: int
    max_vocab_size: int
    dataset_csv: Path
    models: list[dict[str, Any]]
    vocab_dirs: list[Path]
    l1_penalties: list[float]
    known_locations: dict[str, tuple[float, float]]
    parquet_dir: Path
    report_dir: Path


def load_config(path: Path) -> SpliceConfig:
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    root = path.parent.parent  # assume configs/splice.yaml -> project root one level up

    return SpliceConfig(
        device=raw["device"],
        num_samples=raw.get("num_samples"),
        batch_size=raw["batch_size"],
        max_vocab_size=raw["max_vocab_size"],
        dataset_csv=(root / raw["dataset_csv"]).resolve(),
        models=raw["models"],
        vocab_dirs=[(root / d).resolve() for d in raw["vocab_dirs"]],
        l1_penalties=raw["l1_penalties"],
        known_locations={k: tuple(v) for k, v in raw["known_locations"].items()},
        parquet_dir=(root / raw["output"]["parquet_dir"]).resolve(),
        report_dir=(root / raw["output"]["report_dir"]).resolve(),
    )


def load_model(device: torch.device, ckpt_path: Path | None) -> Location2TextLightningModule:
    model = Location2TextLightningModule(
        location_model_type="geoclip",
        location_model=None,
        location_model_filename=None,
        text_model_type="geoclip",
        text_model="geoclip",
        text_vocabulary="openai",
        train_text_model=False,
        learning_rate=1e-4,
        weight_decay=1e-2,
        logit_scale_temperature=0.07,
        lambda_alignment=1.0,
        sigma=1.0,
    )

    if ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def embed_vocab(model, device: torch.device, vocab_path: Path):
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    words: list[str] = []
    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for word in tqdm(vocab, desc=f"Embedding vocab: {vocab_path.name}"):
            emb = model.text_model_predict(word, normalize=True)
            if emb.ndim == 2: # if embedding has dim [1, d], make it [d]
                emb = emb.squeeze(0)
            embeddings.append(emb.cpu())
            words.append(word)

    torch.cuda.empty_cache()

    emb_tensor = torch.stack(embeddings).to(device)
    emb_tensor = F.normalize(emb_tensor, dim=1) # should already be normalized!!!
    emb_tensor = F.normalize(
        emb_tensor - emb_tensor.mean(dim=0, keepdim=True),
        dim=1,
    ) # mean normalize as in splice; need to run on a large enough dataset to get a good mean
    return words, emb_tensor


def vocab_size(vocab_path: Path) -> int:
    with vocab_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return len(data)
    if isinstance(data, list):
        return len(data)
    raise ValueError(f"Unexpected vocab format in {vocab_path}: {type(data)}")


def embed_locations(model, device: torch.device, dataset_csv: Path, num_samples: int | None):
    df = pd.read_csv(dataset_csv)
    if num_samples is not None:
        df = df.sample(n=num_samples, random_state=42)

    output_dim = model.output_dim
    latlon_embeddings = torch.zeros(len(df), output_dim, device=device)

    locs = df[["lon", "lat"]].values.astype("float64")
    locs = torch.tensor(locs, device=device)

    from torch.utils.data import DataLoader

    loader = DataLoader(locs, batch_size=512, shuffle=False)
    i = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding locations"):
            latlon_embeddings[i : i + batch.shape[0]] = model.location_model(batch)
            i += batch.shape[0]
            if i % 5120 == 0:
                torch.cuda.empty_cache()

    latlon_embeddings = F.normalize(latlon_embeddings, dim=1) #should already be normalized
    latlon_embeddings = F.normalize(
        latlon_embeddings - latlon_embeddings.mean(dim=0, keepdim=True),
        dim=1,
    ) # mean normalize as in splice; need to run on a large enough dataset to get a good mean
    return df.reset_index(drop=True), latlon_embeddings


def embed_known_locations(model, device: torch.device, known_locations: dict[str, tuple[float, float]]):
    rows = []
    coords = []
    for name, (lat, lon) in known_locations.items():
        rows.append({"fn": name, "lat": lat, "lon": lon})
        coords.append([lat, lon])  # CHANGE FOR SATCLIP

    df = pd.DataFrame(rows).reset_index(drop=True)
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)

    with torch.no_grad():
        embeddings = model.location_model(coords_tensor)
        embeddings = F.normalize(embeddings, dim=1)

    return df, embeddings

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
        return (vocab @ loc)  # [C]
    return (loc @ vocab.T)    # [B, C]

def run_splice_for_vocab(
    device: torch.device,
    batch_size: int,
    base_df: pd.DataFrame,
    latlon_embeddings: torch.Tensor,
    vocab_words: list[str],
    vocab_embs: torch.Tensor,
    l1_penalty: float,
    out_path: Path,
):

    mean_embedding_locs = latlon_embeddings.mean(dim=0) 

    splicemodel = SPLICE(
        mean_embedding_locs,
        vocab_embs,
        clip=None,
        device=str(device),
        solver='admm',
        return_weights=True,
        return_cosine=True,
        l1_penalty=l1_penalty,
    )

    n = len(latlon_embeddings)
    results = []

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
            top_k=5,
        )

        for i in range(end - start):
            idx = start + i
            row = base_df.iloc[idx]
            record = {
                "lat": row["lat"],
                "lon": row["lon"],
                "fn": row["fn"],
                "mse": mse_chunk[i].item(),
                "row_idx": idx
            }
            for c_idx, word in enumerate(vocab_words):
                record[word] = sparse_chunk[i, c_idx].item()
            results.append(record)

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
    vocab_words: list[str],
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

        top_vals, top_idx = torch.topk(weights, k=min(top_k, weights.shape[0]))
        concepts = [vocab_words[j] for j in top_idx.tolist()]

        print(
            f"  Location idx={global_idx} "
            f"(lat={row['lat']:.4f}, lon={row['lon']:.4f}, fn={row['fn']}, mse={mse_val:.4f})"
        )
        for concept, wt in zip(concepts, top_vals.tolist()):
            print(f"    {concept}: {wt:.4f}")

        cos = cosine_to_vocab(chunk[local_i], vocab_embs)  # [C]
        cos_vals, cos_idx = torch.topk(cos, k=min(5, cos.shape[0]))
        print("    Top-5 cosine-similar vocab terms:")
        for j, s in zip(cos_idx.tolist(), cos_vals.tolist()):
            print(f"      {vocab_words[j]}: {s:.4f}")


def summarize_to_text(
    df: pd.DataFrame,
    df_known: pd.DataFrame | None,
    vocab_name: str,
    model_label: str,
    ckpt_path: Path | None,
    l1_penalty: float,
    out_txt: Path,
    vocab_words: list[str],
    vocab_embs: torch.Tensor,
    base_embeddings: torch.Tensor,
    known_embeddings: torch.Tensor | None,
):
    from io import StringIO

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

    # Example decompositions (base locations): best / median / worst MSE
    if len(df) > 0:
        buf.write("Example decompositions (top 10 concepts per location):\n")
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
            top_concepts = weights.nlargest(10)

            buf.write(
                f"\nLocation idx={idx} "
                f"(lat={loc['lat']:.4f}, lon={loc['lon']:.4f}, mse={loc['mse']:.4f}):\n"
            )
            buf.write(top_concepts.to_string())
            buf.write("\n")

            if "row_idx" in loc:
                row_i = int(loc["row_idx"])
                emb = base_embeddings[row_i]            # [D]
                cos = cosine_to_vocab(emb, vocab_embs)  # [C]
                cos_vals, cos_idx = torch.topk(cos, k=min(5, cos.shape[0]))

                buf.write("Top-5 cosine-similar vocab terms:\n")
                for j, s in zip(cos_idx.tolist(), cos_vals.tolist()):
                    buf.write(f"  {vocab_words[j]}: {s:.4f}\n")
                buf.write("\n")

    # Named-location decompositions (df_known)
    if df_known is not None and len(df_known) > 0:
        buf.write(
            "\nNamed-location decompositions (top 10 concepts per specified city):\n"
        )
        for _, loc in df_known.iterrows():
            weights = pd.to_numeric(loc[concept_cols], errors="coerce")
            top_concepts = weights.nlargest(10)

            buf.write(
                f"\nLocation '{loc['fn']}' "
                f"(lat={loc['lat']:.4f}, lon={loc['lon']:.4f}, mse={loc['mse']:.4f}):\n"
            )
            buf.write(top_concepts.to_string())
            buf.write("\n")

            if "row_idx" in loc and known_embeddings is not None:
                row_i = int(loc["row_idx"])
                emb = known_embeddings[row_i]
                cos = cosine_to_vocab(emb, vocab_embs)
                cos_vals, cos_idx = torch.topk(cos, k=min(5, cos.shape[0]))

                buf.write("Top-5 cosine-similar vocab terms:\n")
                for j, s in zip(cos_idx.tolist(), cos_vals.tolist()):
                    buf.write(f"  {vocab_words[j]}: {s:.4f}\n")
                buf.write("\n")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(buf.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/splice.yaml"),
        help="Path to splice config YAML.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab_files: list[Path] = []
    for d in cfg.vocab_dirs:
        vocab_files.extend(sorted(d.glob("*.json")))

    for model_spec in cfg.models:
        model_label = model_spec["label"]
        ckpt_path = Path(model_spec["ckpt_path"]) if model_spec["ckpt_path"] is not None else None

        print(f"\n\n######## Model: {model_label} ########")
        print(f"Checkpoint path: {ckpt_path}")

        model = load_model(device, ckpt_path)

        base_df, latlon_embeddings = embed_locations(
            model,
            device,
            cfg.dataset_csv,
            cfg.num_samples,
        )

        known_df, known_embeddings = embed_known_locations(
            model,
            device,
            cfg.known_locations,
        )

        for l1_penalty in cfg.l1_penalties:
            print(f"\n---- L1 penalty: {l1_penalty} ----")

            for vocab_path in vocab_files:
                size = vocab_size(vocab_path)
                if size >= cfg.max_vocab_size:
                    print(f"Skipping vocab {vocab_path.name} (size {size} >= {cfg.max_vocab_size})")
                    continue

                vocab_name = vocab_path.stem
                print(
                    f"\n=== Running SpLiCE for model={model_label}, "
                    f"vocab={vocab_name} (size={size}), l1={l1_penalty} ==="
                )

                words, vocab_embs = embed_vocab(model, device, vocab_path)

                out_parquet = (
                    cfg.parquet_dir
                    / f"{model_label}_location_weights_{vocab_name}_l1_{l1_penalty}.parquet"
                )
                df_out = run_splice_for_vocab(
                    device,
                    cfg.batch_size,
                    base_df,
                    latlon_embeddings,
                    words,
                    vocab_embs,
                    l1_penalty=l1_penalty,
                    out_path=out_parquet,
                )

                known_parquet = (
                    cfg.parquet_dir
                    / f"{model_label}_knownlocs_{vocab_name}_l1_{l1_penalty}.parquet"
                )
                df_known = run_splice_for_vocab(
                    device,
                    cfg.batch_size,
                    known_df,
                    known_embeddings,
                    words,
                    vocab_embs,
                    l1_penalty=l1_penalty,
                    out_path=known_parquet,
                )

                report_txt = (
                    cfg.report_dir
                    / f"{model_label}_report_{vocab_name}_l1_{l1_penalty}.txt"
                )
                summarize_to_text(
                    df_out,
                    df_known,
                    vocab_name,
                    model_label,
                    ckpt_path,
                    l1_penalty,
                    report_txt,
                    words,             # vocab_words
                    vocab_embs,        # vocab_embs
                    latlon_embeddings, # base_embeddings
                    known_embeddings,  # known_embeddings
                )

                print(
                    f"\nQuick stats for model={model_label}, "
                    f"vocab={vocab_name}, l1={l1_penalty}:"
                )
                check_reconstruction_quality(df_out)
                find_most_common_concepts(df_out)
                analyze_sparsity(df_out)


if __name__ == "__main__":
    main()