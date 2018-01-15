#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=10000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/stdp-mnist/code/bash/job_reports/%j.out

conv_size=${1:-28}
conv_stride=${2:-0}
conv_features=${3:-100}
num_train=${4:-10000}
random_seed=${5:-42}
proportion_grow=${6:-1.0}

cd ../train/

echo "csnn_growing_inhibition_job.sh"

python csnn_growing_inhibition.py --mode=train --num_train=$num_train --conv_size=$conv_size --conv_stride=$conv_stride \
			--conv_features=$conv_features --random_seed=$random_seed --proportion_grow=$proportion_grow --dt 0.1

python csnn_growing_inhibition.py --mode=test --num_test=10000 --conv_size=$conv_size --conv_stride=$conv_stride --proportion_grow=$proportion_grow \
								--conv_features=$conv_features --num_train=$num_train --random_seed=$random_seed --dt 0.1
exit
