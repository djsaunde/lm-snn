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
proportion_low=${6:-1.0}
start_low_inhib=${7:-0.5}
end_low_inhib=${8:-10.0}
max_inhib=${9:-20.0}

cd ../train/

echo "csnn_two_level_inhibition_no_eth_job.sh"

python csnn_two_level_inhibition_no_eth.py --mode=train --num_train=$num_train --conv_size=$conv_size \
			--conv_stride=$conv_stride --conv_features=$conv_features --random_seed=$random_seed \
					--proportion_low=$proportion_low --start_low_inhib=$start_low_inhib \
							--end_low_inhib=$end_low_inhib --max_inhib=$max_inhib

python csnn_two_level_inhibition_no_eth.py --mode=test --num_test=10000 --conv_size=$conv_size \
	--conv_stride=$conv_stride --proportion_low=$proportion_low --conv_features=$conv_features \
	--num_train=$num_train --random_seed=$random_seed --start_low_inhib=$start_low_inhib \
				--end_low_inhib=$end_low_inhib --max_inhib=$max_inhib

exit
