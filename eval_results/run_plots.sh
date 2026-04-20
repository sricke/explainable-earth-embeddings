#!/usr/bin/env bash
# Usage: ./run_plots.sh [config.yaml]
set -euo pipefail

CONFIG="${1:-$(dirname "$0")/config.yaml}"
SCRIPT="$(dirname "$0")/plot.py"

python3 - "$CONFIG" "$SCRIPT" <<'EOF'
import sys, subprocess
from pathlib import Path
import yaml

cfg  = yaml.safe_load(open(sys.argv[1]))
plot = Path(sys.argv[2])

ckpt_p = Path(cfg["checkpoints"])
ckpts  = sorted(ckpt_p.glob("**/best.pt")) if ckpt_p.is_dir() else [ckpt_p]

cs_p     = Path(cfg["concepts"])
concepts = sorted(cs_p.glob("*.json")) if cs_p.is_dir() else [cs_p]

extra = []
if cfg.get("num_samples"):           extra += ["--num_samples", str(cfg["num_samples"])]
if cfg.get("stratify_by_continent"): extra += ["--stratify_by_continent"]
if cfg.get("lat_col", "lat") != "lat": extra += ["--lat_col", cfg["lat_col"]]
if cfg.get("lon_col", "lon") != "lon": extra += ["--lon_col", cfg["lon_col"]]

total = len(ckpts) * len(concepts)
for i, (ckpt, cs) in enumerate([(c, s) for c in ckpts for s in concepts], 1):
    out = Path(cfg.get("output_dir", "plots")) / ckpt.parent.name / cs.stem
    print(f"[{i}/{total}] {ckpt.parent.name} x {cs.stem}")
    cmd = ["python", str(plot), "--model_path", str(ckpt), "--concept_set", str(cs),
           "--locations_path", cfg["locations"], "--output_dir", str(out), *extra]
    subprocess.run(cmd, check=True)
EOF
