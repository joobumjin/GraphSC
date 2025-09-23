#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:1  --output=optuna_both.out
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH --mem=16g

# Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh

conda activate qbam

# Run program
cd /users/bjoo2/code/qbam/qbam_gnn

echo "Optuna Searching on Both"
python3 grid_search_optuna.py --data /users/bjoo2/data/bjoo2/qbam/data --pred Both