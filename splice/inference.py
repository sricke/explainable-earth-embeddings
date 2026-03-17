from pathlib import Path
from typing import Any, Dict

import torch

from splice.config import load_config
from splice.model import load_model, compute_mean_location_embedding, embed_known_locations
from splice.vocab import embed_vocab, vocab_size
from splice.decomposition import run_splice_for_vocab, summarize_to_text
from splice.interpret import (
    check_reconstruction_quality,
    find_most_common_concepts,
    analyze_sparsity,
)


def main(cfg: Dict[str, Any]) -> None:
    """Run SpLiCE inference pipeline for all models / vocabs in the config."""
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for model_spec in cfg["models"]:
        model_label = model_spec["label"]
        ckpt_path = model_spec.get("ckpt_path")

        print(f"\n\n######## Model: {model_label} ########")
        print(f"Checkpoint path: {ckpt_path}")

        model = load_model(device, ckpt_path)

        # Use the dataset only to estimate the global mean embedding for SpLiCE.
        mean_embedding_locs = compute_mean_location_embedding(
            model,
            device,
            cfg["dataset_csv"],
            cfg["num_samples"],
        )

        known_df, known_embeddings = embed_known_locations(
            model,
            device,
            cfg["known_locations"],
        )

        for l1_penalty in cfg["l1_penalties"]:
            print(f"\n---- L1 penalty: {l1_penalty} ----")

            for vocab_path in cfg["vocab_files"]:
                size = vocab_size(vocab_path)
                if size >= cfg["max_vocab_size"]:
                    print(f"Skipping vocab {vocab_path.name} (size {size} >= {cfg['max_vocab_size']})")
                    continue

                vocab_name = vocab_path.stem
                print(
                    f"\n=== Running SpLiCE for model={model_label}, "
                    f"vocab={vocab_name} (size={size}), l1={l1_penalty} ==="
                )

                words, vocab_embs = embed_vocab(model, device, vocab_path)

                known_parquet = cfg["parquet_dir"] / f"{model_label}_knownlocs_{vocab_name}_l1_{l1_penalty}.parquet"
                df_known = run_splice_for_vocab(
                    device,
                    cfg["batch_size"],
                    known_df,
                    known_embeddings,
                    mean_embedding_locs,
                    words,
                    vocab_embs,
                    l1_penalty=l1_penalty,
                    out_path=known_parquet,
                    top_k=cfg["top_k"],
                )

                report_txt = cfg["report_dir"] / f"{model_label}_report_{vocab_name}_l1_{l1_penalty}.txt"
                summarize_to_text(
                    df_known,
                    vocab_name,
                    model_label,
                    ckpt_path,
                    l1_penalty,
                    report_txt,
                    words,             # vocab_words
                    vocab_embs,        # vocab_embs
                    known_embeddings,  # embeddings for df_known
                    cfg["top_k"],
                )

                print(
                    f"\nQuick stats for model={model_label}, "
                    f"vocab={vocab_name}, l1={l1_penalty}:"
                )
                check_reconstruction_quality(df_known)
                find_most_common_concepts(df_known)
                analyze_sparsity(df_known)

