import torch
import torch.nn as nn


class Sine(nn.Module):
    """Sine activation for SIREN-style networks."""

    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "sine": Sine,
}


class EmbeddingProjection(nn.Module):
    """Linear or MLP projection head placed on top of an embedding."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int = 0,
        num_hidden_features: int = None,
        nonlinearity: str = None,
    ):
        super().__init__()

        if num_hidden_layers == 0:
            self.net = nn.Sequential(nn.Linear(in_features, out_features))
        else:
            assert num_hidden_features is not None, "Need hidden size for MLP"
            assert nonlinearity is not None, "Need to specify nonlinearity"

            activation_cls = _ACTIVATIONS.get(nonlinearity)
            if activation_cls is None:
                raise NotImplementedError(f"Nonlinearity '{nonlinearity}' is not implemented")

            layers = [nn.Linear(in_features, num_hidden_features), activation_cls()]
            for _ in range(num_hidden_layers - 1):
                layers += [nn.Linear(num_hidden_features, num_hidden_features), activation_cls()]
            layers.append(nn.Linear(num_hidden_features, out_features))

            self.net = nn.Sequential(*layers)

        self._init_weights(nonlinearity)

    def _init_weights(self, nonlinearity: str):
        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue
            if nonlinearity in ("relu", "gelu"):
                # kaiming with relu is the standard approximation for gelu
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif nonlinearity == "sine":
                with torch.no_grad():
                    m.weight.uniform_(-1 / m.weight.size(1), 1 / m.weight.size(1))
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        w = self.net[0].weight
        assert x.dtype == w.dtype and x.device == w.device, (
            f"embedding projection expects {w.dtype} on {w.device}, got {x.dtype} on {x.device}"
        )
        return self.net(x)
