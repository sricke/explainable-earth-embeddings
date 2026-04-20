import argparse, json
from pathlib import Path
import pandas as pd
import torch, torch.nn.functional as F
from external.splice.splice.model import SPLICE
from main import build_model

import random
# Make class so this works with the original SpLiCE implementation
class _LocWrapper(torch.nn.Module):
    def __init__(self, enc): super().__init__(); self.enc = enc
    def encode_image(self, x): return self.enc(x)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",    required=True, help="Model path to location-text aligned model")
    p.add_argument("--lat_lon_csv",   required=True, help="CSV with 'lat','lon' columns (used to compute mean embedding)")
    p.add_argument("--concepts_json", required=True, help="JSON list of concept strings")
    p.add_argument("--l1_penalty",    type=float, default=0.1)
    p.add_argument("--prompt",        default=None, help='Prompt template with {concept} placeholder, e.g. "A satellite image of a {concept}"')
    p.add_argument("--device",        default="cuda")
    return p.parse_args()

def main():
    args = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.model_path, map_location=device)
    model_args = argparse.Namespace(**ckpt['args'])
    model_args.precomputed_location_embeddings = False  # need SatCLIP for raw coord inference
    model = build_model(model_args, device)
    model.load_state_dict(ckpt['model'], strict=False)  # SatCLIP weights not in ckpt, just projection
    model.eval()

    df = pd.read_csv(args.lat_lon_csv)
    latlon = torch.tensor(df[["lat", "lon"]].values, dtype=torch.float32).to(device)
    with torch.no_grad():
        mean_emb = model.location_model_predict(latlon).mean(0)

    with open(args.concepts_json) as f:
        concepts = json.load(f)
    with torch.no_grad():
        prompted = [args.prompt.format(concept=c) for c in concepts] if args.prompt else concepts
        emb_concepts = model.text_model_predict(prompted)
    emb_concepts = F.normalize(emb_concepts - emb_concepts.mean(0), dim=1)

    splice_model = SPLICE(mean_emb, emb_concepts,
                          clip=_LocWrapper(model.location_encoder).to(device),
                          solver='admm', device=device, return_weights=True, return_cosine=True,
                          l1_penalty=args.l1_penalty)

    out = Path("splice_results"); out.mkdir(exist_ok=True)
    torch.save(splice_model, out / "splice_model.pt")
    torch.save(concepts, out / "concepts.pt")

if __name__ == "__main__":
    main()
