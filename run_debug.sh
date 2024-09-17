#!/bin/bash

#SBATCH --partition=gpu-debug --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH --mem=10g

# Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate qbam

# Run program
cd /users/bjoo2/code/qbam/qbam_gnn
python3 grid_search.py --data /users/bjoo2/data/bjoo2/qbam/data --pred TER --chkpt_path /users/bjoo2/data/bjoo2/qbam/checkpoints --img_path /users/bjoo2/data/bjoo2/qbam/graphs --results_path /users/bjoo2/data/bjoo2/qbam/text_out