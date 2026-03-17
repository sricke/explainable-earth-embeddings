#!/bin/bash 
#SBATCH -o slurm/match-sinr-goodsentinel.%j.out 
#SBATCH --mail-type=ALL 
#SBATCH --partition=cpu-long
#SBATCH --constraint="sapphirerapids|zen3"
#SBATCH --nodes=1 
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=30G 
#SBATCH --time=96:00:00 
#SBATCH --job-name=match-sinr-goodsentinel

python 00d_match_h3_cells.py