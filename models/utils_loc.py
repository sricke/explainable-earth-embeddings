import torch
import torch.nn as nn
from pathlib import Path
from paths import _p

_loc = _p["location_encoders"]

LOCATION_EMBEDDING_DIMENSIONS = {
    k: v["dim"] for k, v in _loc.items() if isinstance(v, dict) and "dim" in v
}

LOCATION_MODEL_IDS = {
    k: str(v["model_id"]) for k, v in _loc.items() if isinstance(v, dict) and "model_id" in v
}

LOCATION_MODEL_CHECKPOINTS = {
    k: v["checkpoint"]
    for k, v in _loc.items() if isinstance(v, dict) and "checkpoint" in v
}


def load_sinr(checkpoint_path, device):
    from external.sinr.models import get_model as sinr_get_model
    from external.sinr.utils import CoordEncoder

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = sinr_get_model(ckpt['params'])
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.to(device).eval()
    coord_enc = CoordEncoder(ckpt['params']['input_enc'])

    class _SINRWrapper(nn.Module):
        def __init__(self, model, coord_encoder):
            super().__init__()
            self.model, self.coord_encoder = model, coord_encoder
        def forward(self, x, return_feats=False):
            return self.model(self.coord_encoder.encode(x.clone()), return_feats=return_feats)

    return _SINRWrapper(model, coord_enc)


def load_csp(checkpoint_path, device):
    """Load CSP location encoder."""
    from external.csp.main.utils import get_model as csp_get_model
    from external.csp.main.models import LocationImageEncoder
    import os
    checkpoint_path = os.path.expanduser(checkpoint_path)
    print(f"Loading CSP from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    params = ckpt['params']
    params['coord_dim'] = 2

    loc_enc = csp_get_model(
        train_locs=None,
        params=params,
        spa_enc_type=params['spa_enc_type'],
        num_inputs=params.get('num_loc_feats', 2),
        num_classes=params['num_classes'],
        num_filts=params['num_filts'],
        num_users=params.get('num_users', 1),
        device=device,
    )
    wrapper = LocationImageEncoder(
        loc_enc=loc_enc,
        train_loss=params['train_loss'],
        unsuper_loss=params['unsuper_loss'],
        cnn_feat_dim=params.get('cnn_feat_dim', 2048),
        spa_enc_type=params['spa_enc_type']
    ).to(device)
    wrapper.load_state_dict(ckpt['state_dict'])
    return wrapper.loc_enc.eval()
