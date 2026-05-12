"""
Processes git-10M parquet splits into an expanded, concept-level dataset.

Approach
--------
Uses spaCy's dependency parse to extract substantive noun phrases — the actual
geographic / landscape things described — without hardcoded sentence templates.

For each caption sentence we:
  1. Strip the `{num}_GOOGLE_LEVEL_` prefix.
  2. Split into sentences and skip ones with hedging / variation language.
  3. Parse with spaCy.
  4. For every noun chunk, extend it to include its immediate prepositional
     complement ("patchwork" → "patchwork of rectangular fields").
  5. Drop chunks that refer to the image/camera, spatial positions, or are too
     vague/generic to encode a scene concept.
  6. Strip leading determiners, lowercase, deduplicate (by lemma).

Output: ~/data/expanded-git-10M/{split}.parquet  (columns: lat, lon, caption)
"""

import re
import spacy
import pandas as pd
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path.home() / "data" / "git-10M"
OUTPUT_DIR = Path.home() / "data" / "expanded-git-10M"

nlp = spacy.load("en_core_web_sm", disable=["ner"])

_LEVEL_PREFIX  = re.compile(r"^\d+_GOOGLE_LEVEL_\s*")
_SENTENCE_END  = re.compile(r"(?<=[.!?])\s+")
_LEADING_DET   = re.compile(r"^(?:a|an|the|this|these|that|those)\s+", re.IGNORECASE)
_POSSESSIVE    = re.compile(r"^(?:its|their|his|her|our|your)\s+", re.IGNORECASE)

# Sentences to skip — they describe uncertainty, change, or connectivity
# rather than naming what's in the scene.
# Use explicit suffix groups so inflections are caught ("connecting", not just "connect").
_SKIP_SENTENCE = re.compile(
    r"\b(?:indicat(?:es?|ed|ing)"
    r"|suggest(?:s|ed|ing)?"
    r"|appear(?:s|ed|ing)?"
    r"|seem(?:s|ed|ing)?"
    r"|likely|possibly|probably"
    r"|var(?:ies?|ied|ying)"
    r"|differ(?:s|ed|ing|ent)?"
    r"|connect(?:s|ed|ing)?"
    r"|travers(?:e|es|ed|ing)?"
    r"|runs?\s+along|winds?\s+through)\b",
    re.IGNORECASE,
)

# Root lemmas/texts that mean "the image" rather than "a place".
_IMAGE_ROOTS = {
    "image", "photo", "photograph", "picture", "satellite",
    "view", "scene", "shot", "camera", "resolution",
}

# Root lemmas that are pure spatial references within the frame, not concepts.
_POSITIONAL_ROOTS = {
    "top", "bottom", "left", "right", "center", "middle",
    "front", "back", "side", "corner", "edge",
    "north", "south", "east", "west",
}

# Root lemmas that are too vague to be useful unless paired with a descriptive
# modifier (amod or compound child).  Chunks whose root lemma is in this set
# and have no such modifier will be dropped.
_VAGUE_HEADS = {
    # spatial / relational
    "area", "region", "part", "section", "portion", "spot",
    "place", "location", "site", "point", "vicinity",
    # generic infrastructure
    "road", "path", "building", "structure", "boundary", "lot",
    # partitive / collective
    "mix", "mixture", "combination", "variety", "array", "number", "set",
    "piece",
    # pronouns / determiners
    "it", "they", "this", "these", "that", "those",
    # abstract meta-terms
    "feature", "thing", "element", "aspect", "characteristic",
    # generic landscape/scene nouns — meaningful with a modifier, not alone
    "landscape", "terrain", "field", "pathway", "surface", "ground",
    "environment", "scenery",
}


def _extend_with_prep(chunk):
    """
    Extend a noun chunk to include its root's immediate prep+pobj subtree.

    Rules:
    - Skip if the pobj is a positional / image-frame reference.
    - Skip if the pobj itself has prep children — the child noun chunk will
      carry that content, avoiding double-counting.
      e.g. "landscape [with a patchwork [of fields]]" → don't extend the
      parent; the separate "patchwork" chunk extends to "patchwork of fields".
    - "body [of water]"  → extend (water has no further prep children)
    """
    root = chunk.root
    end = chunk.end

    for child in root.children:
        if child.dep_ != "prep":
            continue
        for grandchild in child.children:
            if grandchild.dep_ != "pobj":
                continue
            if grandchild.lemma_.lower() in _POSITIONAL_ROOTS | _IMAGE_ROOTS:
                continue
            # If the pobj itself has a prep child, its own chunk will be richer
            pobj_has_prep = any(t.dep_ == "prep" for t in grandchild.children)
            if pobj_has_prep:
                continue
            subtree_end = grandchild.i + 1
            for tok in grandchild.children:
                if tok.dep_ in {"amod", "compound", "nummod", "det", "nmod"}:
                    subtree_end = max(subtree_end, tok.i + 1)
            end = max(end, child.i + 1, subtree_end)

    return chunk.doc[chunk.start:end]


def _keep_chunk(chunk) -> bool:
    root_lemma = chunk.root.lemma_.lower()
    root_text  = chunk.root.text.lower()

    if root_lemma in _IMAGE_ROOTS or root_text in _IMAGE_ROOTS:
        return False
    if root_lemma in _POSITIONAL_ROOTS or root_text in _POSITIONAL_ROOTS:
        return False

    # Possessive references ("its roof", "their borders")
    if _POSSESSIVE.match(chunk.text):
        return False

    # Single-token DET/PRON
    if len(chunk) == 1 and chunk[0].pos_ in {"DET", "PRON"}:
        return False

    # Vague root: only keep if there is at least one *descriptive* modifier
    # (amod or compound that is not itself positional or ordinal).
    # "paved road" → keep; "few roads" / "top part" → drop.
    if root_lemma in _VAGUE_HEADS:
        has_qualifier = any(
            t.dep_ in {"amod", "compound"}
            and t.lemma_.lower() not in _POSITIONAL_ROOTS
            and t.pos_ != "JJR"   # comparative ("upper", "lower")
            for t in chunk.root.children
        )
        if not has_qualifier:
            return False

    return True


def _normalise(text: str) -> str:
    text = _LEADING_DET.sub("", text).strip().rstrip(".,;:")
    text = _POSSESSIVE.sub("", text).strip()
    return text


def _lemma_key(text: str) -> str:
    """Rough dedup key: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_concepts(raw: str) -> list[str]:
    text = _LEVEL_PREFIX.sub("", raw).strip()
    sentences = _SENTENCE_END.split(text)

    candidates: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or _SKIP_SENTENCE.search(sentence):
            continue

        doc = nlp(sentence)

        for base_chunk in doc.noun_chunks:
            if not _keep_chunk(base_chunk):
                continue

            extended = _extend_with_prep(base_chunk)
            norm = _normalise(extended.text)
            if not norm:
                continue

            # Require at least 3 characters
            if len(norm) < 3:
                continue

            # Drop if the head (before extension) was vague and extension added nothing
            head_stripped = _LEADING_DET.sub("", base_chunk.text).strip().lower()
            if head_stripped in _VAGUE_HEADS and _lemma_key(norm) == head_stripped:
                continue

            candidates.append(norm)

    # Longest-match dedup: drop any candidate whose text is a substring of a
    # longer candidate (case-insensitive).  Sort longest first so we keep the
    # most informative form of overlapping phrases.
    candidates.sort(key=lambda s: len(s), reverse=True)
    kept: list[str] = []
    seen_keys: set[str] = set()
    for cand in candidates:
        key = _lemma_key(cand)
        if key in seen_keys:
            continue
        # Drop if this candidate is fully contained inside an already-kept one
        if any(key in _lemma_key(k) for k in kept):
            continue
        seen_keys.add(key)
        kept.append(cand)

    return kept


BATCH_SIZE = 6_000


def process_split(split_path: Path, output_dir: Path, batch_size: int = BATCH_SIZE) -> None:
    df = pd.read_parquet(split_path, columns=["text", "lat", "lon"])
    batches = [df.iloc[i : i + batch_size] for i in range(0, len(df), batch_size)]

    processed = []
    for batch in tqdm(batches, desc=split_path.name, unit="batch"):
        batch = batch.copy()
        batch["caption"] = batch["text"].map(extract_concepts)
        out = (
            batch[["lat", "lon", "caption"]]
            .explode("caption")
            .dropna(subset=["caption"])
        )
        out = out[out["caption"].str.strip().str.len() > 0]
        processed.append(out)

    out_df = pd.concat(processed, ignore_index=True)
    out_path = output_dir / split_path.name
    out_df.to_parquet(out_path, index=False)
    print(f"  {len(df):,} locations → {len(out_df):,} concept rows → {out_path}")


def extract_split_from_data(split_path: Path, output_dir: Path) -> None:
    data_path = output_dir / "data.parquet"
    print(f"  Reading {split_path.name} for lat/lon keys...")
    split_df = pd.read_parquet(split_path, columns=["lat", "lon"])
    print(f"  {len(split_df):,} locations in {split_path.name}")
    print(f"  Reading data.parquet ({data_path})...")
    data_df = pd.read_parquet(data_path)
    print(f"  {len(data_df):,} rows in data.parquet")
    keys = set(zip(split_df["lat"], split_df["lon"]))
    print(f"  {len(keys):,} unique (lat, lon) keys")
    mask = [t in keys for t in zip(data_df["lat"], data_df["lon"])]
    subset = data_df[mask]
    print(f"  Matched {len(subset):,} rows ({len(subset)/len(data_df)*100:.1f}% of data.parquet)")
    out_path = output_dir / split_path.name
    subset.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = sorted(INPUT_DIR.glob("*.parquet"))
    if not splits:
        raise FileNotFoundError(f"No parquet files found in {INPUT_DIR}")
    data_parquet = OUTPUT_DIR / "data.parquet"
    print(f"INPUT_DIR:  {INPUT_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"data.parquet exists: {data_parquet.exists()}")
    print(f"Found {len(splits)} split(s): {[s.name for s in splits]}")
    for split_path in splits:
        out_path = OUTPUT_DIR / split_path.name
        print(f"\n[{split_path.name}] output exists: {out_path.exists()}")
        if not out_path.exists():
            if data_parquet.exists():
                print(f"  → Extracting from data.parquet")
                extract_split_from_data(split_path, OUTPUT_DIR)
            else:
                print(f"  → Processing with NLP pipeline")
                process_split(split_path, OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
