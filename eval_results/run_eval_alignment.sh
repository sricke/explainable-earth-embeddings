#!/usr/bin/env bash
# Usage: ./run_eval_alignment.sh [config.yaml]
set -euo pipefail

CONFIG="${1:-$(dirname "$0")/config.yaml}"
SCRIPT="$(dirname "$0")/eval_alignment.py"
ROOT="$(dirname "$0")/.."

python3 - "$CONFIG" "$SCRIPT" "$ROOT" <<'EOF'
import sys, subprocess
from pathlib import Path
import yaml

cfg    = yaml.safe_load(open(sys.argv[1]))
script = Path(sys.argv[2])
root   = Path(sys.argv[3]).resolve()

ckpt_p = Path(cfg["checkpoints"])
ckpts  = sorted(ckpt_p.glob("**/best.pt")) if ckpt_p.is_dir() else [ckpt_p]

extra = []
if cfg.get("num_samples"): extra += ["--num_samples",  str(cfg["num_samples"])]

import os
env = {**os.environ, "PYTHONPATH": str(root)}

out_dir = Path(cfg.get("output_dir", str(root / "eval_alignment")))

total = len(ckpts)
for i, ckpt in enumerate(ckpts, 1):
    out_file = out_dir / f"{ckpt.parent.name}.txt"
    print(f"[{i}/{total}] {ckpt.parent.name}  ->  {out_file}")
    cmd = ["python", str(script), "--checkpoint", str(ckpt), "--output", str(out_file), *extra]
    subprocess.run(cmd, check=True, env=env)
EOF
