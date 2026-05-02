import pickle, json, glob
from tqdm import tqdm

meta2_dir = "/home/libe2152/data/skyscript/meta2"
out_path = "/home/libe2152/data/skyscript/concepts.json"

metas = [pickle.load(open(f, "rb")) for f in glob.glob(f"{meta2_dir}/*.pickle")]

concepts = set()
for m in tqdm(metas, desc="Meta files"):
    concepts.update(f"{k}" for k, _ in m["center_tags"].items())
    concepts.update(f"{v}" for _, v in m["center_tags"].items())
    for t in m["surrounding_tags"]:
        concepts.update(f"{k}" for k, _ in t.items())
        concepts.update(f"{v}" for _, v in t.items())

json.dump(sorted(concepts), open(out_path, "w"), indent=2)
print(f"Saved {len(concepts)} concepts")
