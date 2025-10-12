#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:1 --output=dnn-f_amd_ter.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 60:00:00
#SBATCH --mem=50g

# Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh

conda activate qbam

# Run program
cd /users/bjoo2/code/qbam/qbam_gnn

echo "Training DNN F on TER"
python3 train_dnnf.py --data /users/bjoo2/data/bjoo2/qbam/data --pred TER --dataset AMD --batch_size 32 
# python3 train_dnnf.py --data /users/bjoo2/data/bjoo2/qbam/data --graph_path /users/bjoo2/data/bjoo2/qbam/dnn_f_results/train_graph --pred "$i" --batch_size 64 --extra_data /users/bjoo2/data/bjoo2/qbam/data/healthy