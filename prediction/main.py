import argparse
import sys
import numpy as np
from pathlib import Path

import torch
from sklearn.metrics import r2_score, accuracy_score

sys.path.append("..")

from prediction import load_paths
from train import train
from eval import top_concepts


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_dir", required=True, help="Directory containing <prefix>_{train,val,test}.pt")
    p.add_argument("--prefix", required=True, help="Filename prefix (e.g. satclip, geoclip, geoclip_geoyfcc)")
    p.add_argument("--mode", choices=["ridge", "logistic", "nn"], default="ridge")
    p.add_argument("--alpha", type=float, default=1.0, help="[ridge]")
    p.add_argument("--CV", action="store_true", help="[ridge] use RidgeCV instead of Ridge(alpha)")
    p.add_argument("--C", type=float, default=1.0, help="[logistic]")
    p.add_argument("--max_iter", type=int, default=1000, help="[logistic]")
    p.add_argument("--hidden_dims", nargs="*", type=int, default=[256, 256], help="[nn]")
    p.add_argument("--dropout", type=float, default=0.0, help="[nn]")
    p.add_argument("--nonlinearity", default="relu", help="[nn]")
    p.add_argument("--loss", default="mse", choices=["mse"], help="[nn]")
    p.add_argument("--lr", type=float, default=1e-3, help="[nn]")
    p.add_argument("--epochs", type=int, default=20, help="[nn]")
    p.add_argument("--batch_size", type=int, default=1024, help="[nn]")
    p.add_argument("--device", default="cuda", help="[nn]")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--model", default=None, help="Named sparse model from paths.yaml to resolve concepts (e.g. geoclip_geoyfcc)")
    return p.parse_args()


def _load(pt: Path):
    d = torch.load(pt, map_location="cpu", weights_only=False)
    X = d["embeddings"].numpy() if d.get("codes") is None else d["codes"].numpy()
    y = d["values"].numpy() if d.get("vals") is None else d["vals"].numpy()
    nan_mask = np.isnan(y)
    return X[~nan_mask], y[~nan_mask]


def main():
    a = get_args()
    emb_dir = Path(a.emb_dir)
    X_train, y_train = _load(emb_dir / f"{a.prefix}_train.pt")
    X_val, y_val = _load(emb_dir / f"{a.prefix}_val.pt") if (emb_dir / f"{a.prefix}_val.pt").exists() else (None, None)
    X_test, y_test = _load(emb_dir / f"{a.prefix}_test.pt")

    m = train(a.mode, X_train, y_train, X_val, y_val,
              alpha=a.alpha, CV=a.CV, C=a.C, max_iter=a.max_iter,
              hidden_dims=a.hidden_dims, dropout=a.dropout, nonlinearity=a.nonlinearity,
              loss=a.loss, lr=a.lr, epochs=a.epochs, batch_size=a.batch_size, device=a.device)
    preds = m.predict(X_test) if hasattr(m, "predict") else m(torch.tensor(X_test)).detach().cpu().numpy()
    metric = accuracy_score(y_test.astype(int), preds.astype(int)) if a.mode == "logistic" else r2_score(y_test, preds)
    print(f"{a.mode} test metric: {metric:.4f}")

    if a.mode == "ridge":
        idxs = top_concepts(m, k=a.topk)
        if getattr(m.coef_, "ndim", 1) == 1:
            concepts = None
            if a.model:
                paths = load_paths()
                concepts_pt = paths["sparse_models"]["splice"][a.model]["concepts"]
                concepts = torch.load(concepts_pt, weights_only=False)
            names = [concepts[i] for i in idxs] if concepts else idxs.tolist()
            print("top concepts:", names)
        else:
            print("top concepts:", idxs)


if __name__ == "__main__":
    main()
