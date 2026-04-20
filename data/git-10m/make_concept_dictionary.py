#!/usr/bin/env python3

import argparse, json, numpy as np, pandas as pd, spacy
from collections import Counter
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path.home() / "data" / "modified-git-10M"
OUT_DIR  = Path(__file__).parent
SPLITS   = ["train", "val", "test"]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_texts(num_samples=None):
    texts = pd.concat(
        [pd.read_parquet(DATA_DIR / f"{s}.parquet")["text"] for s in SPLITS],
        ignore_index=True,
    ).dropna()
    if num_samples is not None:
        texts = texts.sample(n=min(num_samples, len(texts)), random_state=42)
    return texts.tolist()


def valid_chunk(chunk):
    """Return lemmatised concept string or None. Keeps single NOUNs only."""
    words = [t for t in chunk if t.is_alpha and not t.is_stop]
    if len(words) == 1 and words[0].pos_ == "NOUN":
        return words[0].lemma_.lower()
    return None


def extract_concepts(texts, n_process=4, batch_size=4096):
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    counts = Counter()
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), total=len(texts), desc="extracting"):
        for chunk in doc.noun_chunks:
            c = valid_chunk(chunk)
            if c:
                counts[c] += 1
    return counts.most_common()  # list of (concept, count)


# ── semantic grouping ─────────────────────────────────────────────────────────

def semantic_cluster(ranked, top_n, n_clusters):
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans

    vocab  = [c for c, _ in ranked[:top_n]]
    counts = {c: n for c, n in ranked[:top_n]}
    embs   = SentenceTransformer("all-MiniLM-L6-v2").encode(
        vocab, batch_size=512, show_progress_bar=True, convert_to_numpy=True
    )
    labels  = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit_predict(embs)
    centres = np.array([embs[labels == k].mean(0) for k in range(n_clusters)])

    reps = []
    for k in range(n_clusters):
        idx  = np.where(labels == k)[0]
        best = idx[np.linalg.norm(embs[idx] - centres[k], axis=1).argmin()]
        c    = vocab[best]
        reps.append({"concept": c, "count": counts[c]})
    return sorted(reps, key=lambda x: -x["count"])


# ── main ──────────────────────────────────────────────────────────────────────

def load_or_extract(out, n_process, num_samples=None):
    """Return ranked list of (concept, count). Load existing JSON if present, else extract."""
    if out.exists():
        print(f"Loading existing {out} ...")
        raw = json.loads(out.read_text())
        if raw and isinstance(raw[0], dict):
            ranked = [(item["concept"], item["count"]) for item in raw]
        else:
            ranked = [(c, 0) for c in raw]
        ranked = [(c, n) for c, n in ranked if len(c.split()) == 1]
        print(f"Loaded {len(ranked):,} concepts")
        return ranked
    texts = load_texts(num_samples=num_samples)
    return extract_concepts(texts, n_process=n_process)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters",  type=int, default=-1,
                        help="number of semantic clusters (-1 = skip clustering)")
    parser.add_argument("--top_n",      type=int, default=50_000,
                        help="top-N concepts to cluster")
    parser.add_argument("--n_process",  type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="randomly sample this many texts before extracting concepts")
    args = parser.parse_args()

    suffix = f"_n{args.num_samples}" if args.num_samples is not None else ""
    out = OUT_DIR / f"concept_dictionary{suffix}.json"
    ranked = load_or_extract(out, n_process=args.n_process, num_samples=args.num_samples)
    print(f"Unique concepts: {len(ranked):,}")

    with open(out, "w") as f:
        json.dump([c for c, _ in ranked], f, indent=2)
    print(f"Saved → {out}")

    if args.n_clusters > 0:
        reps = semantic_cluster(ranked, top_n=args.top_n, n_clusters=args.n_clusters)
        out2 = OUT_DIR / f"concept_dictionary_n{args.num_samples}_{args.n_clusters}_clustered.json"
        with open(out2, "w") as f:
            json.dump([r["concept"] for r in reps], f, indent=2)
        print(f"Clustered → {out2}  ({len(reps):,} representatives)")


if __name__ == "__main__":
    main()
