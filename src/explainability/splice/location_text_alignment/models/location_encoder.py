import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC_DIR = Path(__file__).resolve().parents[4]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from location_encoders.location_encoder import (  # noqa: E402, F401
    LOCATION_EMBEDDING_DIMENSIONS,
    LOCATION_MODEL_CHECKPOINTS,
    LOCATION_MODEL_IDS,
    LocationEncoder as _LocationEncoder,
    _LON_FIRST_MODELS,
    _load_csp,
    _load_model,
    _load_sinr,
)


class LocationEncoder(_LocationEncoder):
    """
    LocationEncoder with an optional learned projection head.
    """

    def __init__(self, embed_project: nn.Module = None, **kwargs):
        super().__init__(**kwargs)
        self.embed_project = embed_project
        if embed_project is not None:
            embed_project.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = super().forward(x)
        if self.embed_project is not None:
            proj_device = next(self.embed_project.parameters()).device
            if embedding.device != proj_device:
                embedding = embedding.to(proj_device, non_blocking=True)
            embedding = F.normalize(self.embed_project(embedding), dim=-1)
        return embedding
