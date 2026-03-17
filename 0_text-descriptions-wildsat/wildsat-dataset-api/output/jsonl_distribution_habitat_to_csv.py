"""
jsonl_distribution_habitat_to_csv.py

Utility to convert wiki_species_text_raw.jsonl produced by fetch_wiki_species_text.py
into a flat CSV containing only the "Distribution and habitat" section.

Input JSONL format (one object per species, as written by fetch_wiki_species_text.py):
{
  "taxon_id": int,
  "species_name": str,
  "lat": float,
  "lon": float,
  "sections": [
    {
      "section_index": int,
      "section_name": str,
      "section_type": str,
      "text": str,
    },
    ...
  ]
}

Output CSV columns:
  taxon_id, species_name, section_name, section_type, text, lat, lon
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def jsonl_distribution_habitat_to_csv(
    jsonl_path: str,
    csv_path: str,
) -> pd.DataFrame:
    """
    Read wiki_species_text_raw.jsonl and write a CSV with only the
    "Distribution and habitat" section.
    """
    rows: list[dict] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            sections = obj.get("sections", [])
            for sec in sections:
                # Match the exact composite section commonly used on Wikipedia
                if sec.get("section_name") == "Distribution and habitat":
                    rows.append(
                        {
                            "taxon_id": obj.get("taxon_id"),
                            "species_name": obj.get("species_name", ""),
                            "section_name": sec.get("section_name", ""),
                            "section_type": sec.get("section_type", ""),
                            "text": sec.get("text", ""),
                            "lat": obj.get("lat"),
                            "lon": obj.get("lon"),
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    size_mb = os.path.getsize(csv_path) / 1e6
    print(f"Saved CSV → {csv_path}  ({size_mb:.1f} MB, {len(df):,} rows)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert wiki_species_text_raw.jsonl to a CSV containing only the "
            '"Distribution and habitat" section.'
        )
    )
    parser.add_argument(
        "--jsonl",
        default="wiki_species_text_raw.jsonl",
        help=(
            "Path to input JSONL file (default: wiki_species_text_raw.jsonl "
            "in the current directory)."
        ),
    )
    parser.add_argument(
        "--out_csv",
        default="geolocated_text_distribution_habitat_only.csv",
        help=(
            "Path to output CSV file "
            "(default: geolocated_text_distribution_habitat_only.csv "
            "in the current directory)."
        ),
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {jsonl_path}")

    csv_path = Path(args.out_csv)

    print(f"Reading JSONL from {jsonl_path}")
    jsonl_distribution_habitat_to_csv(str(jsonl_path), str(csv_path))


if __name__ == "__main__":
    main()

