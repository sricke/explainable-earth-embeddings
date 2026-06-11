# SAE

Code for training a sparse autoencoder (SAE) on location embeddings and analyzing what the learned features represent.

## 1. Train an SAE

Run `main.py` to train a BatchTopK SAE on top of a frozen location encoder's embeddings.

```bash
python main.py
```

This will:
- load the location encoder (e.g. `satclip`)
- load the dataset (e.g. `urban_geoyfcc` or `S2100k`)
- train the SAE and log to wandb
- save checkpoints to `/home/results/sae_geo_embeddings/<dataset>/<encoder>/<trainer>/dict_size=...,k=.../<timestamp>/`

The dataset, location encoder, and SAE hyperparameters (`dict_size`, `k`, `steps`, `lr`, etc.) are set directly at the bottom of `main.py`, so edit them there before running. Specifically, in the `if __name__ == "__main__":` block:

- `location_encoder_name` (line 32) — which location encoder to use, e.g. `"satclip"`.
- `dataset_name` (line 39) — which dataset to train on, e.g. `"urban_geoyfcc"` or `"S2100k"`.
- the data path on line 40, `"/home/datasets/earth_embeddings/{}/".format(dataset_name)` — change `/home/datasets/earth_embeddings/` if your data lives somewhere else.
- `steps` (line 43) and the `trainer_cfg` dict (lines 44-54) — SAE hyperparameters like `dict_size`, `k`, `lr`.
- `save_dir` (line 59) — where checkpoints get saved. Change `/home/results/sae_geo_embeddings` if you want results written elsewhere.

## 2. Export sparse activations

Once you have a trained SAE checkpoint (`best_ae.pt`), use `monosemanticity_analysis.py` to run the SAE over a dataset and save the sparse activations + visual embeddings for each point:

```bash
python monosemanticity_analysis.py \
    --data-dir /home/datasets/earth_embeddings/s2-100k \
    --sae-path /home/results/sae_geo_embeddings/S2100k/satclip/BatchTopKTrainer/dict_size=1024,k=20/<timestamp>
```

This writes `xAI/sparse_activations.csv` and `xAI/visual_embeddings.pt` inside the SAE folder, and also runs the monosemanticity scoring (see below).

## 3. Monosemanticity analysis

`monosemanticity.py` computes, for each SAE neuron, how well its activations align with the visual embeddings (i.e. how "monosemantic" each neuron is). This runs automatically at the end of step 2 and saves the scores under `xAI/visual_monosemanticity/`.

## 4. Visualize neurons

To save example satellite image crops for the most (or least) monosemantic neurons:

```bash
python visualize_monosemantic_neurons.py
```

Edit the paths at the bottom of the script before running:

- `base_dir` — should point to the `xAI/` folder produced in step 2 (the SAE run dir from step 1, plus `/xAI/`).
- `index_csv_path` — path to the dataset's `index.csv` (e.g. `/home/datasets/earth_embeddings/s2-100k/index.csv`).
- `largest=False` — set to `True` to visualize the most monosemantic neurons instead of the least.

To plot where neurons activate on a world map:

```bash
# single neuron, full world map
python plot_activation_maps.py --input-csv /path/to/sparse_activations.csv

# multiple neurons on one wide map
python joint_activation_map.py --input-csv /path/to/sparse_activations.csv
```

Both of these scripts have the neuron column names hardcoded near the top of the file — edit these before running:

- `plot_activation_maps.py` line 73: `activation_columns = ["act406"]`
- `joint_activation_map.py` line 38: `activation_columns = ["act872", "act395", "act294"]`

Set these to whichever `actN` columns (neuron indices) from `sparse_activations.csv` you want to plot.
