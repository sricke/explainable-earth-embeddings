import torch
import torch.nn as nn

LOCATION_EMBEDDING_DIMENSIONS = {
    "geoclip": 512,
    "satclip": 256,
    "gair": 768,
    "climplicit": 1024, # this defaults to 256 dim embeddings times 4 months!
    "csp_fmow": 512,
    "csp_inat": 512,
    "sinr": 256
}

LOCATION_MODEL_IDS = {
    "satclip": "microsoft/SatCLIP-ViT16-L40",
    # might need to include L10 as well, or can also look at ResNet-based models
    "climplicit": "Jobedo/climplicit",
    "gair": "PingL/GAIR",
    "taxabind": "MVRL/taxabind-config"
}

LOCATION_MODEL_CHECKPOINTS = {
    "satclip": "satclip-vit16-l40.ckpt",
    "gair": "checkpoint.pth",
    "csp_fmow": "~/data/csp/model_dir/model_fmow/model_fmow_gridcell_0.0010_32_0.1000000_1_512_gelu_UNSUPER-contsoftmax_0.000050_1.000_1_0.100_TMP1.0000_1.0000_1.0000.pth.tar",
    "csp_inat": "~/data/csp/model_dir/model_inat/model_inat_2018_gridcell_0.0010_32_0.1000000_1_512_leakyrelu_UNSUPER-contsoftmax_0.000500_1.000_1_1.000_TMP20.0000_1.0000_1.0000.pth.tar",
    "sinr": "external/sinr/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt",
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