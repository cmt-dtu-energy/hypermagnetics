#!/usr/bin/bash
#SBATCH --job-name HyperMagnetics
#SBATCH --account DD-23-133
#SBATCH --partition qgpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 128
#SBATCH --time 18:00:00
#SBATCH --gpus 1

ml texlive
conda init
conda activate hypermagnetics
wandb login
python src/hypermagnetics/sweeper.py
