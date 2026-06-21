import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_SRC_DIR = Path(__file__).resolve().parents[1]
_CSP_MODEL_DIR = _SRC_DIR / "location_encoders" / "csp" / "model_dir"
_LTA_DIR = _SRC_DIR / "explainability" / "splice" / "location_text_alignment"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_LTA_DIR) not in sys.path:
    sys.path.insert(0, str(_LTA_DIR))

LOCATION_EMBEDDING_DIMENSIONS = {
    "geoclip": 512,
    "satclip": 256,
    "climplicit": 1024,
    "csp_fmow": 256,
    "sinr": 256,
}

LOCATION_MODEL_IDS = {
    "satclip": "microsoft/SatCLIP-ViT16-L40",
    "climplicit": "Jobedo/climplicit",
}

LOCATION_MODEL_CHECKPOINTS = {
    "satclip": "satclip-vit16-l40.ckpt",
    "sinr": str(
        _SRC_DIR
        / "location_encoders/sinr/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt"
    ),
}

# Models that expect (lon, lat) rather than (lat, lon).
_LON_FIRST_MODELS = {"satclip", "climplicit", "sinr"}


def _load_sinr(checkpoint_path, device):
    """
    Load SINR and wrap it so raw (lon, lat) can be passed at inference time.
    """
    from location_encoders.sinr.models import get_model as sinr_get_model
    from location_encoders.sinr.utils import CoordEncoder

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = sinr_get_model(ckpt["params"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device).eval()
    coord_enc = CoordEncoder(ckpt["params"]["input_enc"])

    # SINR expects sin/cos-encoded coordinates, not raw lat/lon — wrap to apply encoding inline.
    class _SINRWrapper(nn.Module):
        def __init__(self, model, coord_encoder):
            super().__init__()
            self.model, self.coord_encoder = model, coord_encoder

        def forward(self, x, return_feats=False):
            return self.model(
                self.coord_encoder.encode(x.clone()), return_feats=return_feats
            )

    return _SINRWrapper(model, coord_enc)


def _load_csp(variant: str, device: str):
    """
    Load a CSP location encoder from model_dir/model_{variant}/*.pth.tar.
    """
    model_dir = _CSP_MODEL_DIR / f"model_{variant}"
    checkpoints = sorted(model_dir.glob("*.pth.tar"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    checkpoint_path = checkpoints[0]

    from location_encoders.csp.main.models import LocationImageEncoder
    from location_encoders.csp.main.utils import get_model as csp_get_model

    logger.info(f"Loading CSP from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    params = ckpt["params"]
    params["coord_dim"] = 2

    loc_enc = csp_get_model(
        train_locs=None,
        params=params,
        spa_enc_type=params["spa_enc_type"],
        num_inputs=params.get("num_loc_feats", 2),
        num_classes=params["num_classes"],
        num_filts=params["num_filts"],
        num_users=params.get("num_users", 1),
        device=device,
    )
    wrapper = LocationImageEncoder(
        loc_enc=loc_enc,
        train_loss=params["train_loss"],
        unsuper_loss=params["unsuper_loss"],
        cnn_feat_dim=params.get("cnn_feat_dim", 2048),
        spa_enc_type=params["spa_enc_type"],
    ).to(device)
    wrapper.load_state_dict(ckpt["state_dict"])
    return wrapper.loc_enc.eval()


def _get_satclip_visual_encoder(device):
    from huggingface_hub import hf_hub_download
    from location_encoders.satclip.load import get_satclip

    satclip_model = get_satclip(
        hf_hub_download(
            LOCATION_MODEL_IDS["satclip"], LOCATION_MODEL_CHECKPOINTS["satclip"]
        ),
        device=device,
        return_all=True)
    
    image_encoder = satclip_model.visual.eval()
    return image_encoder

def get_visual_encoder(dataset, device):
    if dataset == "s2-100k":
        return _get_satclip_visual_encoder(device)
    else:
        from geoclip import GeoCLIP
        geoclip_model = GeoCLIP().to(device).double()
        image_encoder = geoclip_model.image_encoder.eval()
        return image_encoder

def _load_model(location_model: str, device: str = "cuda:0"):
    """
    Load a pretrained location encoder by name.
    """
    if location_model == "satclip":
        from huggingface_hub import hf_hub_download

        from location_encoders.satclip.load import get_satclip

        satclip_model = get_satclip(
            hf_hub_download(
                LOCATION_MODEL_IDS["satclip"], LOCATION_MODEL_CHECKPOINTS["satclip"]
            ),
            device=device,
            return_all=True)
        
        location_encoder = satclip_model.location

    elif location_model == "geoclip":
        from geoclip import GeoCLIP
        geoclip_model = GeoCLIP().to(device).double()
        location_encoder = geoclip_model.location_encoder.eval()

    elif location_model == "climplicit":
        from rshf.climplicit import Climplicit

        location_encoder = Climplicit.from_pretrained(
            LOCATION_MODEL_IDS["climplicit"], config={"return_chelsa": False}
        ).to(device)

    elif location_model.startswith("csp"):
        variant = location_model[len("csp_") :]
        location_encoder = _load_csp(variant, device)

    elif location_model == "sinr":
        location_encoder = _load_sinr(LOCATION_MODEL_CHECKPOINTS["sinr"], device)

    else:
        raise ValueError(f"Location model '{location_model}' is not supported")
    
    location_encoder.eval()
    return location_encoder


class LocationEncoder(nn.Module):
    """
    Wraps a pretrained location encoder with an optional projection head.
    """

    def __init__(
        self,
        location_model: str = None,
        finetune_mode: str = None,
        precomputed: bool = True,
    ):
        super().__init__()
        assert location_model is not None, "Must specify location model"
        self.location_model = location_model
        self.precomputed = precomputed
        self.location_embedding_dim = LOCATION_EMBEDDING_DIMENSIONS[location_model]

        if not precomputed:
            self.location_encoder = _load_model(location_model)
            self._set_finetune_mode(finetune_mode)
        elif finetune_mode in ("all", "lora"):
            raise ValueError(
                f"Cannot use finetune_mode='{finetune_mode}' with precomputed=True"
            )

    def _set_finetune_mode(self, finetune_mode: str):
        """
        Freeze or unfreeze the base encoder.
        """
        assert finetune_mode in ["all", "lora", "only_proj"], (
            f"Unknown finetune_mode: {finetune_mode}"
        )

        if finetune_mode == "all":
            self.location_encoder.requires_grad_(True)
            self.location_encoder.train()

        elif finetune_mode in ("lora", "only_proj"):
            self.location_encoder.requires_grad_(False)
            self.location_encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2) tensor of (lat, lon) coordinates, or (B, D) precomputed embeddings.
        Returns:
            L2-normalised embeddings of shape (B, embed_dim).
        """
        if self.precomputed:
            embedding = F.normalize(x.to(dtype=torch.float32), dim=-1)

        else:
            assert x.shape[1] == 2, "Forward expects (lat, lon) pairs"

            if (
                self.location_model in _LON_FIRST_MODELS
                or self.location_model.startswith("csp")
            ):
                x = x[:, [1, 0]]

            x = x.double() if self.location_model == "satclip" else x.float()

            enc_device = next(self.location_encoder.parameters()).device
            if x.device != enc_device:
                x = x.to(enc_device, non_blocking=True)

            if self.location_model.startswith("csp") or self.location_model == "sinr":
                raw = self.location_encoder(x, return_feats=True)
            else:
                raw = self.location_encoder(x)

            embedding = F.normalize(raw.float(), dim=-1)

        return embedding
