import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from paths import _p


def load_paths() -> dict:
    splice = {k: v for k, v in _p["splice"].items() if isinstance(v, dict)}
    return {
        **_p["prediction"],
        "sparse_models": {"splice": splice},
    }
