#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:1  --output=optuna_vegf_multi.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 60:00:00
#SBATCH --mem=32g

# Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh

conda activate qbam

# Run program
cd /users/bjoo2/code/qbam/qbam_gnn

echo "Optuna Searching on VEGF"
python3 optuna_search.py --data /users/bjoo2/data/bjoo2/qbam/data --pred VEGF --dataset AMD --multi_opt