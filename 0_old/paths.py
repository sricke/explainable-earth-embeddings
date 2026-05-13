import re
import yaml
from pathlib import Path

_CONFIGS = Path(__file__).parent / "configs"


def _subst(s, ctx):
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r'\$\{(\w+)\}', lambda m: str(ctx.get(m.group(1), m.group(0))), s)
    return s


def _resolve(node, ctx=None):
    assert isinstance(node, dict)
    merged = {**(ctx or {}), **{k: v for k, v in node.items() if not isinstance(v, dict)}}
    result = {}
    for k, v in node.items():
        if isinstance(v, dict):
            result[k] = _resolve(v, merged)
        elif isinstance(v, str):
            result[k] = _subst(v, merged)
        else:
            result[k] = v
    return result


def _load(name, ctx=None):
    raw = yaml.safe_load((_CONFIGS / name).read_text())
    assert isinstance(raw, dict)
    return _resolve(raw, ctx)


def load_paths() -> dict:
    base = _load("base.yaml")
    ctx = {k: v for k, v in base.items() if isinstance(v, str)}
    return {
        "base":              base,
        "data":              _load("data.yaml", ctx),
        "location_encoders": _load("location_encoders.yaml", ctx),
        "models":            _load("models.yaml", ctx),
        "concept_sets":      _load("concept_sets.yaml", ctx),
        "splice":            _load("splice.yaml", ctx),
        "prediction":        _load("prediction.yaml", ctx),
    }


_p = load_paths()
DATA_ROOT            = Path(_p["data"]["root"])
SKYSCRIPT_DIR        = Path(_p["data"]["skyscript"])
GIT10M_DIR           = Path(_p["data"]["git10m"])
SHAPEFILE            = Path(_p["data"]["shapefile"])
CSP_FMOW_CHECKPOINT  = Path(_p["location_encoders"]["csp_fmow"]["checkpoint"]).expanduser()
CSP_INAT_CHECKPOINT  = Path(_p["location_encoders"]["csp_inat"]["checkpoint"]).expanduser()
SINR_CHECKPOINT      = Path(_p["location_encoders"]["sinr"]["checkpoint"])
