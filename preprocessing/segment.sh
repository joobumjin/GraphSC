#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 60:00:00
#SBATCH --mem=100g

# Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh

conda activate cellpose4

# Run program
cd /users/bjoo2/code/qbam/qbam_gnn/preprocessing
python cellpose4_segment.py

mkdir tiffs npys

mv *.tif tiffs
mv *.npy npys