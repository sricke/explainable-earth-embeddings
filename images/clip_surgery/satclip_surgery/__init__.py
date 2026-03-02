
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[3]
_inner = _root / "satclip" / "satclip"
for p in (_inner, _root):
    s = str(p)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)

from .load import get_satclip
from .main_surgery import SatCLIPSurgeryLightningModule
from .model_surgery import SatCLIPSurgery

