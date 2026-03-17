# concatenate_vocab.py
from pathlib import Path
import json

mscoco_txt = Path("../SpLiCE/data/vocab/mscoco.txt")
geospatial_json = Path("geospatial_concepts.json")
out_json = Path("mscoco_plus_geospatial.json")

# block forbidden tokens
FORBIDDEN = {"lat", "lon", "latitude", "longitude"}

def load_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_geo(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return list(data.keys())
    else:
        raise ValueError(f"unexpected type for {path}: {type(data)}")

if __name__ == "__main__":
    vocab = []
    vocab.extend(load_lines(mscoco_txt))
    vocab.extend(load_geo(geospatial_json))

    # normalize + remove forbidden tokens
    cleaned = []
    for w in vocab:
        w_clean = w.strip()
        if w_clean.lower() not in FORBIDDEN:
            cleaned.append(w_clean)

    # preserve order, drop duplicates
    seen = set()
    merged = []
    for w in cleaned:
        if w not in seen:
            seen.add(w)
            merged.append(w)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(merged)} entries to {out_json}")