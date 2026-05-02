import sys, requests
from pathlib import Path
from tqdm import tqdm

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

print("Fetching file list...")
files = requests.get(
    "https://api.figshare.com/v2/articles/26026798"
).json()["files"]

print(f"Found {len(files)} files")

for i, f in enumerate(files, 1):
    name = f["name"]
    url = f["download_url"]
    path = out / name

    print(f"[{i}/{len(files)}] Downloading {name}")

    r = requests.get(url, stream=True)
    total = int(r.headers.get("content-length", 0))

    with open(path, "wb") as fp, tqdm(
        total=total, unit="B", unit_scale=True, desc=name, leave=False
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            fp.write(chunk)
            bar.update(len(chunk))

print("Done.")
