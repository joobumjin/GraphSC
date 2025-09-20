#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:1 --output=end_singular_donor.out
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 06:00:00
#SBATCH --mem=32g

# Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh

conda activate qbam

# Run program
cd /users/bjoo2/code/qbam/qbam_gnn

echo "Training Singular Model on Donor"
python3 singular_donor.py --data /users/bjoo2/data/bjoo2/qbam/data/DonorSingular --pred Donor 