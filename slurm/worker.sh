#!/bin/bash
#SBATCH --job-name=xemb
#SBATCH --output=slurm/logs/%j.out
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

PROJECT_DIR=$(realpath "$(dirname "$0")/..")
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fai
cd "$PROJECT_DIR"

wandb agent --count 1 "$SWEEP_ID"
