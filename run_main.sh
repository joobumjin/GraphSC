#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:2

#SBATCH -n 4
#SBATCH -t 48:00:00
#SBATCH --mem=150g

# Load a CUDA module
module load cuda
module load miniconda/4.12.0
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh

conda activate qbam

# Run program
cd /users/bjoo2/data/bjoo2/QBAM/qbam_fork
python3 main.py