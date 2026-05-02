"""
Processes git-10M parquet splits and writes modified versions to ~/data/modified-git-10M/.

Transformations applied to the `text` column:
1. Strip leading `{num}_GOOGLE_LEVEL_` prefix.
2. Keep only the first sentence.
3. Rewrite the sentence to start with "a satellite image of ...".
"""

import os
import re
import pandas as pd
from pathlib import Path

INPUT_DIR = Path.home() / "data" / "git-10M"
OUTPUT_DIR = Path.home() / "data" / "modified-git-10M"

# Verbs used in the standard openers
_VERBS = r"shows?|depicts?|displays?|presents?|reveals?|captures?|features?|contains?|includes?"

# Ordered list of (pattern, replacement_prefix) pairs.
# Each pattern matches from the start of the sentence; the rest becomes the
# content after "a satellite image of ".
_OPENERS = [
    # "The/This (satellite )?(image|view) <verb> ..."
    re.compile(
        rf"^(?:The|This|An?)\s+(?:satellite\s+)?(?:image|view)\s+(?:{_VERBS})\s+",
        re.IGNORECASE,
    ),
    # "This is a/an satellite image (of|showing) ..."
    re.compile(
        r"^This\s+is\s+(?:a|an)\s+satellite\s+image\s+(?:of\s+|showing\s+)?",
        re.IGNORECASE,
    ),
    # "This is a/an aerial (view|image) of ..."
    re.compile(
        r"^This\s+is\s+(?:a|an)\s+aerial\s+(?:view|image)\s+of\s+",
        re.IGNORECASE,
    ),
    # "Aerial (view|image) of ..."
    re.compile(r"^Aerial\s+(?:view|image)\s+of\s+", re.IGNORECASE),
    # "A satellite image <verb> ..."
    re.compile(
        rf"^A\s+satellite\s+image\s+(?:{_VERBS})\s+",
        re.IGNORECASE,
    ),
]

_LEVEL_PREFIX = re.compile(r"^\d+_GOOGLE_LEVEL_")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def clean_text(raw: str) -> str:
    # Step 1: remove leading {num}_GOOGLE_LEVEL_
    text = _LEVEL_PREFIX.sub("", raw)

    # Step 2: keep only the first sentence
    first = _SENTENCE_SPLIT.split(text.strip())[0].rstrip(".")

    # Step 3: rewrite opener to "a satellite image of ..."
    for pattern in _OPENERS:
        m = pattern.match(first)
        if m:
            rest = first[m.end():]
            return f"a satellite image of {rest}"

    # Fallback: return the cleaned first sentence unchanged
    return first


def process_split(split_path: Path, output_dir: Path) -> None:
    print(f"Processing {split_path.name} ...")
    df = pd.read_parquet(split_path)
    df["text"] = df["text"].map(clean_text)
    out_path = output_dir / split_path.name
    df.to_parquet(out_path, index=False)
    print(f"  Saved {len(df):,} rows -> {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = sorted(INPUT_DIR.glob("*.parquet"))
    if not splits:
        raise FileNotFoundError(f"No parquet files found in {INPUT_DIR}")
    for split_path in splits:
        process_split(split_path, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
