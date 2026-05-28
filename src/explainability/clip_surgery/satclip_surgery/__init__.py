import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
_inner = _root / "satclip" / "satclip"

# Bare imports inside satclip/satclip/*.py (e.g. ``from loss import ...``) must resolve
# to satclip/satclip/loss.py, not repo-root loss.py. Insert at 0 in this order so
# ``_inner`` ends up ahead of ``_root`` after both prepend operations.
for p in (_root, _inner):
    s = str(p)
    if p.is_dir():
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)

from .load import get_satclip
from .main_surgery import SatCLIPSurgeryLightningModule
from .model_surgery import SatCLIPSurgery
