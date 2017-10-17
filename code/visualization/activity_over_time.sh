#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=02:00:00
#SBATCH --mem=20000
#SBATCH --account=rkozma

conv_size=${1:-28}
conv_stride=${2:-0}
conv_features=${3:-400}
num_train=${4:-30000}
random_seed=${5:-1}
normalize_inputs=${6:-False}

python csnn_growing_inhibition_activity_over_time.py --conv_size $conv_size --conv_stride $conv_stride --conv_features $conv_features --num_train $num_train --random_seed $random_seed --normalize_inputs $normalize_inputs

exit
