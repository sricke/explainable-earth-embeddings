import sys, pickle, json, glob
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from paths import SKYSCRIPT_DIR

meta2_dir = SKYSCRIPT_DIR / "meta2"
out_path  = SKYSCRIPT_DIR / "concepts.json"

metas = [pickle.load(open(f, "rb")) for f in glob.glob(str(meta2_dir / "*.pickle"))]

concepts = set()
for m in tqdm(metas, desc="Meta files"):
    concepts.update(f"{k}" for k, _ in m["center_tags"].items())
    concepts.update(f"{v}" for _, v in m["center_tags"].items())
    for t in m["surrounding_tags"]:
        concepts.update(f"{k}" for k, _ in t.items())
        concepts.update(f"{v}" for _, v in t.items())

json.dump(sorted(concepts), open(out_path, "w"), indent=2)
print(f"Saved {len(concepts)} concepts")
