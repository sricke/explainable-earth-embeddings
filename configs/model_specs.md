# Model Config Defaults

Use these clear keys in `model:`:

- `location_backend`, `location_model_id`, `location_checkpoint`
- `text_backend`, `text_model_id`, `text_pretrained`

## Defaults (auto-filled when omitted)

- Location
  - `satclip`: `model_id=microsoft/SatCLIP-ViT16-L40`, `checkpoint=satclip-vit16-l40.ckpt`
  - `geoclip`: `model_id=geoclip`
- Text
  - `geoclip`: `model_id=geoclip`, `pretrained=openai`
  - `gritlm`: `model_id=GritLM/GritLM-7B`
  - `openclip`: `model_id=ViT-B-32`, `pretrained=openai`

## Backend Rules

- `location_backend=satclip` requires a checkpoint filename.
- `location_backend=geoclip` ignores checkpoint/model id.
- `text_backend=geoclip` always uses `model_id=geoclip` and `pretrained=openai`.
- `text_backend=openclip` requires both `text_model_id` and `text_pretrained`.
- `text_backend=gritlm` defaults to `GritLM/GritLM-7B` if model id is omitted.

## Where defaults are shown

At fit start, `main.py` prints:
- the default preset table
- the fully resolved location/text configs actually used for the run
