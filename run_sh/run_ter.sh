#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:1  --output=optuna_ter.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mem=16g

# Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh

conda activate qbam

# Run program
cd /users/bjoo2/code/qbam/qbam_gnn

declare -a arr=("TER")
for i in "${arr[@]}"
do
    echo "Optuna Searching on $i"
    python3 optuna_search.py --data /users/bjoo2/data/bjoo2/qbam/data --dataset AMD --pred "$i"
    # python3 optuna_search.py --data /users/bjoo2/data/bjoo2/qbam/data/combined_data/graphs --pred "$i" --log_path /users/bjoo2/code/qbam/qbam_gnn/optuna_logs
done

# python3 optuna_search.py --data /users/bjoo2/scratch/csam_data/csam15 --pred TER --log_path /users/bjoo2/code/qbam/qbam_gnn/optuna_logs/ --study_name csam_data15
