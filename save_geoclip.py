"""
save_geoclip_pretrained_ckpt.py

Instantiates Location2TextLightningModule with pretrained GeoCLIP weights
(no fine-tuning) and saves the checkpoint.
"""

import torch
from main import Location2TextLightningModule


if __name__ == "__main__":
    CKPT_DIR = "/home/libe2152/outputs/explainable-earth-embeddings/geoclip/pretrained/checkpoints"
    CKPT_OUT = f"{CKPT_DIR}/location2text_pretrained.ckpt"

    # 1. build — pretrained weights loaded automatically in each encoder's __init__
    model = Location2TextLightningModule(
        location_model={"backend": "geoclip"},
        text_model={"backend": "geoclip", "train_text_model": False},
        finetune_mode="none",  # freeze all text model weights
    )
    model.eval()

    # 2. sanity check
    with torch.no_grad():
        dummy_loc  = torch.tensor([[48.8566, 2.3522]])          # lat/lon Paris
        dummy_text = ["France", "a photo of a mountain"]
        loc_emb  = model.location_model(dummy_loc)
        text_emb = model.text_model(dummy_text)
        print(f"Location embedding shape: {loc_emb.shape}")
        print(f"Text embedding shape:     {text_emb.shape}")


    # 3. save
    torch.save({"state_dict": model.state_dict()}, CKPT_OUT)
    print(f"Saved checkpoint → {CKPT_OUT}")