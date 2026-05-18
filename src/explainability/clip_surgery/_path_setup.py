import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # explainable-earth-embeddings


for _p in (_ROOT, _ROOT / "src"):
    _s = str(_p)
    if _p.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)
