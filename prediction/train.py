import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import r2_score

from mlp import MLP

_LOSSES = {"mse": nn.MSELoss}


def train_ridge(X_train, y_train, alpha: float, cv: bool = False):
    model = RidgeCV() if cv else Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_logistic(X_train, y_train, C: float = 1.0, max_iter: int = 1000):
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train.astype(int))
    return model


def train_nn(
    model: MLP,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer,
    epochs: int,
    device: str,
):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        if val_loader:
            model.eval()
            with torch.no_grad():
                preds = torch.cat([model(x.to(device)).cpu() for x, _ in val_loader])
                targets = torch.cat([y for _, y in val_loader])
            print(f"Epoch {epoch + 1}/{epochs}  val R²={r2_score(targets, preds):.4f}")
    return model


def train(mode: str, X_train, y_train, X_val, y_val, **kwargs):
    if mode == "ridge":
        return train_ridge(X_train, y_train, alpha=kwargs["alpha"], cv=kwargs.get("CV", False))
    if mode == "logistic":
        return train_logistic(X_train, y_train, C=kwargs.get("C", 1.0), max_iter=kwargs.get("max_iter", 1000))

    device = kwargs["device"]
    batch_size = kwargs["batch_size"]

    def _loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    model = MLP(
        in_dim=X_train.shape[1],
        hidden_dims=kwargs["hidden_dims"],
        dropout=kwargs["dropout"],
        nonlinearity=kwargs["nonlinearity"],
    )
    val_loader = _loader(X_val, y_val, shuffle=False) if X_val is not None else None
    loss_fn = _LOSSES[kwargs["loss"]]()
    opt = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])
    return train_nn(model, _loader(X_train, y_train, shuffle=True), val_loader,
                    loss_fn, opt, epochs=kwargs["epochs"], device=device)
