import torch.nn as nn

NONLINEARITIES = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int = 1,
        dropout: float = 0.0,
        nonlinearity: str = "relu",
    ):
        super().__init__()
        act_cls = NONLINEARITIES[nonlinearity]
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
