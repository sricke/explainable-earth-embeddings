import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from external.satclip.satclip.load import get_satclip

LOCATION_EMBEDDING_DIMENSIONS = {
    "geoclip": 512,
    "satclip": 256,
    "aef": 64
}

LOCATION_MODEL_IDS = {
    "satclip": "microsoft/SatCLIP-ViT16-L40",
    #"satclip_l10": "microsoft/SatCLIP-ViT16-L10"
}

LOCATION_MODEL_CHECKPOINTS = {
    "satclip": "satclip-vit16-l40.ckpt",
    #"satclip_l10": "satclip-vit16-l10.ckpt"
}

def load_model(location_model: str):
    """
    Load location model for satclip or geoclip
    """
    if location_model == "satclip":
        model = get_satclip(
            hf_hub_download(
                LOCATION_MODEL_IDS["satclip"], LOCATION_MODEL_CHECKPOINTS["satclip"]
            ),
            device="cpu",
        )
    elif location_model == "geoclip":
        from geoclip import GeoCLIP
        model = GeoCLIP().location_encoder
    else:
        raise ValueError(f"Location Model {location_model} is not supported")
    
    return model

class Sine(nn.Module):
    """
    Sine activation (to mimic SIREN style location encoder in SatCLIP)
    """
    def __init__(self, w0: float = 1.0):
        super.__init__()
        self.w0 = w0 # frequency scaling for SIREN style

    def forward(self, x):
        return torch.sin(self.w0 * x)

class EmbeddingProjection(nn.Module):
    """
    Either linear layer or small MLP placed on top of the location embedding
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int = 0,
        num_hidden_features: int = None,
        nonlinearity: str = None
    ):
        super().__init__()

        # Linear case
        if num_hidden_layers == 0:
            self.net = self._create_linear_project(in_features, out_features)

        # MLP case
        else:
            self.net = self._create_mlp_project(in_features, out_features, num_hidden_layers, num_hidden_features, nonlinearity)

    def _create_linear_project(self, in_features, out_features):
        layers = []
        linear_layer = nn.Linear(in_features, out_features)
        layers.append(linear_layer)
        return nn.Sequential(*layers)
    
    def _create_mlp_project(self, in_features, out_features, num_hidden_layers, num_hidden_features, nonlinearity):
        assert num_hidden_features is not None, "Need hidden size for MLP"
        assert nonlinearity is not None, "Need to specify nonlinearity"

        layers = []
        activation = self._create_activation(nonlinearity)
    
        # Input layer
        input_layer = nn.Linear(in_features, num_hidden_features)
        input_activation = activation()
        layers.append([input_layer, input_activation])

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            hidden_layer = nn.Linear(num_hidden_features, num_hidden_features)
            hidden_activation = activation()
            layers.append([hidden_layer, hidden_activation])

        # Output layer
        layers.append(nn.Linear(num_hidden_features, out_features))

        return nn.Sequential(*layers)

    def _create_activation(self, nonlinearity: str = "sine"):
        if nonlinearity == "relu":
            return nn.ReLU
        elif nonlinearity == "sine":
            return Sine
        else:
            raise NotImplementedError(f"Nonlinearity {nonlinearity} is not implemented yet")

    def _init_weights(self, nonlinearity: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):

                if nonlinearity == "relu":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                elif nonlinearity == "sine":
                    self._sine_init(m)

                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _sine_init(self, m):
        """
        Special init for sine
        """
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                in_features = m.weight.size(-1)
                m.weight.uniform_(-1 / in_features, 1 / in_features)

    def forward(self, x):
        return self.net(x)


class LocationEncoder(nn.Module):

    def __init__(
        self,
        location_model: str = None,
        use_model_weights: bool = True,
        embed_project: nn.Module = None
    ):
        super().__init__()
        assert location_model is not None, "Must specify location model"
        self.location_model = location_model

        # Either use model weights or directly obtain embeddings (like AEF)
        if use_model_weights:
            location_encoder = load_model(self.location_model)
        else:
            raise NotImplementedError("Can only run on loaded model weights")

        self.location_encoder = location_encoder

        # Don't train location encoder
        self.location_encoder.requires_grad_(False)
        self.location_encoder.eval()

        # Train embedding projection if there is one
        self.embed_project = embed_project
        if self.embed_project is not None:
            self.embed_project._requires_grad(True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 2, "Forward expects lat, lon"

        if self.location_model == "satclip":
            x = x[: [1, 0]] # SatCLIP expects lon, lat

        location_embedding = self.location_encoder(x)

        # First normalize after location encoder (this is standard at least for GeoCLIP and SatCLIP)
        location_embedding = nn.functional.normalize(location_embedding, dim=-1)

        if self.embed_project is not None:
            location_embedding = self.embed_project(location_embedding)
            # Normalize final location embedding
            location_embedding = nn.functional.normalize(location_embedding, dim=-1)
        
        return location_embedding
        

        
