# Explainable Earth Embeddings

PyTorch Lightning framework for training location-to-text embedding models using satellite imagery (SatCLIP) and text descriptions.

## Installation

1. Clone the SatCLIP repository in the project root directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Weights & Biases:
   ```bash
   wandb login
   ```

## Usage

Start training:
```bash
python main.py
```

Configuration can be modified in `configs/train.yaml`.
