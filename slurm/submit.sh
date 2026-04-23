#!/bin/bash
# Usage: bash slurm/submit.sh

PROJECT_DIR=$(realpath "$(dirname "$0")/..")
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fai
cd "$PROJECT_DIR"

NUM_RUNS=$(python -c "
import yaml, math
cfg = yaml.safe_load(open('sweep_git10m.yaml'))
n = math.prod(len(p['values']) for p in cfg['parameters'].values() if isinstance(p.get('values'), list))
print(n)
")

# Create the sweep and extract its ID from the output
SWEEP_ID=$(wandb sweep sweep_git10m.yaml 2>&1 | grep -oP "(?<=Run sweep agent with: wandb agent )\\S+")
echo "Sweep ID: $SWEEP_ID"

mkdir -p slurm/logs

for i in $(seq 1 $NUM_RUNS); do
    sbatch --export=SWEEP_ID="$SWEEP_ID" slurm/worker.sh
done
