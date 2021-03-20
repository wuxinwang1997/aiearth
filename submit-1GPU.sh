#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

module load anaconda/3.7
module load nvidia/cuda/10.1
source activate aiearth
python tools/train_net.py
python tools/train_net.py DATASETS.SODA '("False")' SOLVER.BASE_LR '(1e-4)' OUTPUT_DIR '("./usr_data/model_data/resnet18_lstm-epoch30-soda/")'
