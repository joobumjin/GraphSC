#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:1  --output=bmclip.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mem=100g

# Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh

conda activate biomedclip

# Run program
cd /users/bjoo2/code/qbam/qbam_gnn

echo "Embedding Images with BiomedCLIP"
python3 test_biomedclip.py
