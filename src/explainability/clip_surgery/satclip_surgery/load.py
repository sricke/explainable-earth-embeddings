import sys
from pathlib import Path

import torch

_SATCLIP_PY_ROOT = Path(__file__).resolve().parents[3] / "satclip" / "satclip"
if _SATCLIP_PY_ROOT.is_dir() and str(_SATCLIP_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SATCLIP_PY_ROOT))

from location_encoders.satclip.main import SatCLIPLightningModule
from .main_surgery import SatCLIPSurgeryLightningModule


def get_satclip(ckpt_path, device, surgery: bool = False, return_all: bool = False):
    """Load SatCLIP model with optional CLIP-Surgery vision encoder."""
    ckpt = torch.load(ckpt_path, map_location=device)
    for key in ["eval_downstream", "air_temp_data_path", "election_data_path"]:
        if key in ckpt.get("hyper_parameters", {}):
            ckpt["hyper_parameters"].pop(key)

    Lightning = SatCLIPSurgeryLightningModule if surgery else SatCLIPLightningModule
    lightning_model = Lightning(**ckpt["hyper_parameters"]).to(device)

    lightning_model.load_state_dict(ckpt["state_dict"])
    lightning_model.eval()
    geo_model = lightning_model.model

    if return_all:
        return geo_model
    else:
        return geo_model.location

