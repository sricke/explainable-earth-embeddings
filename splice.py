import argparse, json
from pathlib import Path
import pandas as pd
import torch, torch.nn.functional as F
from external.splice.splice.model import SPLICE
from models.model import build_model

BATCH_SIZE = 512

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
    model = build_model(
        text_encoder=model_args.text_encoder,
        location_encoder=model_args.location_encoder,
        text_projection=model_args.text_projection,
        location_projection=model_args.location_projection,
        shared_dim=model_args.shared_dim,
        text_finetune_mode=model_args.text_finetune_mode,
        loc_finetune_mode=model_args.loc_finetune_mode,
        text_proj_hidden_layers=model_args.text_proj_hidden_layers,
        text_proj_hidden_features=model_args.text_proj_hidden_features,
        loc_proj_hidden_layers=model_args.loc_proj_hidden_layers,
        loc_proj_hidden_features=model_args.loc_proj_hidden_features,
        text_nonlinearity=model_args.text_nonlinearity,
        loc_nonlinearity=model_args.loc_nonlinearity,
        precomputed_text_embeddings=model_args.precomputed_text_embeddings,
        precomputed_location_embeddings=model_args.precomputed_location_embeddings,
        device=device,
    )
    model.load_state_dict(ckpt['model'], strict=False)  # SatCLIP weights not in ckpt, just projection
    model.eval()

    df = pd.read_csv(args.lat_lon_csv)
    latlon = torch.tensor(df[["lat", "lon"]].values, dtype=torch.float32).to(device)
    with torch.no_grad():
        mean_emb = model.location_model_predict(latlon).mean(0)

    with open(args.concepts_json) as f:
        concepts = json.load(f)
    prompted = [args.prompt.format(concept=c) for c in concepts] if args.prompt else concepts
    with torch.no_grad():
        emb_concepts = torch.cat([
            model.text_model_predict(prompted[i:i + BATCH_SIZE])
            for i in range(0, len(prompted), BATCH_SIZE)
        ])
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
