import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from external.satclip.satclip.load import get_satclip
from modeling.model_specs import DEFAULT_LOCATION_CONFIGS, resolve_location_config


class LocationEncoder(nn.Module):
    """Encodes (lat, lon) pairs into fixed-size embeddings.

    Supports SatCLIP and GeoCLIP location-encoder backends.
    Convention: datasets yield [lat, lon]; SatCLIP expects [lon, lat] internally.
    """

    DEFAULT_CONFIGS = DEFAULT_LOCATION_CONFIGS

    def __init__(
        self,
        backend: str = "satclip",
        model_id: str | None = None,
        checkpoint_filename: str | None = None,
        *,
        # legacy aliases (for old configs/scripts)
        location_model_type: str | None = None,
        location_model: str | None = None,
        location_model_filename: str | None = None,
        train_location_model: bool = False,
        target_dim: int | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.config = resolve_location_config(
            backend=backend,
            model_id=model_id,
            checkpoint_filename=checkpoint_filename,
            legacy_type=location_model_type,
            legacy_model=location_model,
            legacy_filename=location_model_filename,
        )
        self.location_model_type = self.config["backend"]
        self.dtype = dtype

        if self.location_model_type == "satclip":
            model = get_satclip(
                hf_hub_download(
                    self.config["model_id"], self.config["checkpoint_filename"]
                ),
                device="cpu",
            )
        elif self.location_model_type == "geoclip":
            from geoclip import GeoCLIP
            model = GeoCLIP().location_encoder
        else:
            raise ValueError(
                f"Unsupported location model type: {self.location_model_type}"
            )

        self.model = model
        self.model.requires_grad_(train_location_model)
        self.train_encoder = train_location_model
        if not train_location_model:
            self.model.eval()

        self.embed_project = None
        if target_dim is not None:
            output_dim = self._probe_output_dim()
            if output_dim != target_dim:
                self.embed_project = nn.Linear(output_dim, target_dim)

    def _probe_output_dim(self) -> int:
        param = next(self.model.parameters())
        dummy = torch.zeros(1, 2, device=param.device, dtype=param.dtype)
        with torch.no_grad():
            return self.model(dummy).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != 2:
            raise ValueError(f"Expected last dim == 2 for (lat, lon), got shape {x.shape}")

        if self.location_model_type == "satclip":
            x_model = torch.stack([x[..., 1], x[..., 0]], dim=-1)
        else:
            x_model = x

        param = next(self.model.parameters())
        x_model = x_model.to(device=param.device, dtype=param.dtype)

        if self.train_encoder:
            embedding = self.model(x_model)
        else:
            with torch.no_grad():
                embedding = self.model(x_model)

        embedding = F.normalize(embedding, dim=-1)

        if self.embed_project is not None:
            embedding = self.embed_project(embedding)

        return embedding.to(dtype=self.dtype)
