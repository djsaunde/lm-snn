#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=30000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/stdp-mnist/code/bash/job_reports/%j.out

conv_size=${1:-28}
conv_stride=${2:-0}
conv_features=${3:-100}
num_train=${4:-10000}
random_seed=${5:-42}
proportion_grow=${6:-1.0}
max_inhib=${7:-17.5}

cd ../train/

echo "csnn_growing_inhibition_vary_max_inhib_job.sh"

python csnn_growing_inhibition_vary_max_inhib.py --mode=train --num_train=$num_train --conv_size=$conv_size \
						--conv_stride=$conv_stride --conv_features=$conv_features --random_seed=$random_seed \
													--proportion_grow=$proportion_grow --max_inhib=$max_inhib

python csnn_growing_inhibition_vary_max_inhib.py --mode=test --num_test=10000 --num_train=$num_train \
					--conv_size=$conv_size --conv_stride=$conv_stride --conv_features=$conv_features \
					--random_seed=$random_seed  --proportion_grow=$proportion_grow --max_inhib=$max_inhib

exit
