import re
import yaml
from pathlib import Path

_YAML = Path(__file__).parent / "paths.yaml"


def load_paths() -> dict:
    raw = yaml.safe_load(_YAML.read_text())
    assert isinstance(raw, dict)
    return _resolve(raw)


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


_PROJECT_ROOT = Path(__file__).parent

_p = load_paths()
DATA_ROOT     = Path(_p["data"]["root"])
SKYSCRIPT_DIR = Path(_p["data"]["skyscript"])
GIT10M_DIR    = Path(_p["data"]["git10m"])
SHAPEFILE     = Path(_p["data"]["shapefile"])

CSP_FMOW_CHECKPOINT = Path(_p["location_encoders"]["csp_fmow"]).expanduser()
CSP_INAT_CHECKPOINT = Path(_p["location_encoders"]["csp_inat"]).expanduser()
SINR_CHECKPOINT     = _PROJECT_ROOT / "external/sinr/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt"
