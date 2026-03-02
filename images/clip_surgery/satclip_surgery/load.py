import torch
from satclip.satclip.main import SatCLIPLightningModule
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

