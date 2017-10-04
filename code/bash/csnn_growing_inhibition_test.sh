#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=60000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/stdp-mnist/code/bash/job_reports/%j.out

conv_size=${1:-28}
conv_stride=${2:-0}
conv_features=${3:-100}
num_train=${4:-10000}
random_seed=${5:-42}
test_no_inhibition=${6:-False}
test_max_inhibition=${7:-False}

cd ../train/

echo "csnn_growing_inhibition_test.sh"

python csnn_growing_inhibition.py --mode=test --num_test=10000 --conv_size=$conv_size --conv_stride=$conv_stride --conv_features=$conv_features --num_train=$num_train --random_seed=$random_seed --test_no_inhibition=$test_no_inhibition --test_max_inhibition=$test_max_inhibition

exit
