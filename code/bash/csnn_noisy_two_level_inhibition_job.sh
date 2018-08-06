#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=05-00:00:00
#SBATCH --mem=60000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/stdp-mnist/code/bash/noisy_two_level_job_reports/%j.out

conv_size=${1:-28}
conv_stride=${2:-0}
conv_features=${3:-100}
num_train=${4:-10000}
random_seed=${5:-42}
proportion_low=${6:-1.0}
start_inhib=${7:-0.1}
max_inhib=${8:-17.4}
p_flip=${9:-0.0}

cd ../train/

echo "csnn_noisy_two_level_inhibition_job.sh"

python csnn_two_level_inhibition.py --mode=train --num_train=$num_train --conv_size=$conv_size --conv_stride=$conv_stride \
                                    --conv_features=$conv_features --random_seed=$random_seed --proportion_low=$proportion_low \
                                    --start_inhib=$start_inhib --max_inhib=$max_inhib --dt=0.1 --p_flip=$p_flip

python csnn_two_level_inhibition.py --mode=test --num_test=10000 --conv_size=$conv_size --conv_stride=$conv_stride \
                                    --proportion_low=$proportion_low --conv_features=$conv_features --num_train=$num_train \
                                    --random_seed=$random_seed --start_inhib=$start_inhib --max_inhib=$max_inhib --dt=0.1 --p_flip=p_flip

exit
