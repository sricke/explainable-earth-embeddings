import re
import yaml
from pathlib import Path

_YAML = Path(__file__).parent / "paths.yaml"


def load_paths() -> dict:
    raw = yaml.safe_load(_YAML.read_text())
    assert isinstance(raw, dict)
    return _resolve(raw)


def _resolve(node, ctx=None):
    assert isinstance(node, dict)
    merged = {**(ctx or {}), **{k: v for k, v in node.items() if not isinstance(v, dict)}}
    result = {}
    for k, v in node.items():
        if isinstance(v, dict):
            result[k] = _resolve(v, merged)
        elif isinstance(v, str):
            result[k] = re.sub(r'\$\{(\w+)\}', lambda m: str(merged.get(m.group(1), m.group(0))), v)
        else:
            result[k] = v
    return result
