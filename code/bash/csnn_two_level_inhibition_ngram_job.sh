#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=05-00:00:00
#SBATCH --mem=60000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/dsanghavi/stdp-mnist/code/bash/ngram_job_reports/%j.out


conv_size=${1:-28}
conv_stride=${2:-0}
conv_features=${3:-100}
num_train=${4:-60000}
random_seed=${5:-0}
proportion_low=${6:-0.1}
start_inhib=${7:-1.0}
max_inhib=${8:-20.0}

cd ../train/

echo "csnn_two_level_inhibition_job.sh"

echo "Learning n-grams ------------------------------------------------------------------"
python csnn_two_level_inhibition.py --mode=test --num_train=$num_train --conv_size=$conv_size \
			--conv_stride=$conv_stride --conv_features=$conv_features --random_seed=$random_seed \
			--proportion_low=$proportion_low --start_inhib=$start_inhib --max_inhib=$max_inhib --dt=0.1 \
			--use_ngram=True --learn_ngram=True --num_examples_ngram=12000

echo "Testing using n-grams  ------------------------------------------------------------"
python csnn_two_level_inhibition.py --mode=test --num_test=10000 --conv_size=$conv_size \
	--conv_stride=$conv_stride --proportion_low=$proportion_low --conv_features=$conv_features \
	--num_train=$num_train --random_seed=$random_seed --start_inhib=$start_inhib --max_inhib=$max_inhib --dt=0.1 \
	--use_ngram=True --learn_ngram=False --num_examples_ngram=12000

exit
