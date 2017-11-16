#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=04-00:00:00
#SBATCH --mem=50000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/stdp-mnist/code/bash/snn_job_reports/%j.out

conv_features=${1:-100}
num_train=${2:-10000}
random_seed=${3:-42}

cd ../train/

echo "snn_job.sh"

python snn_mnist.py --mode=train --num_train=$num_train --conv_features=$conv_features --random_seed=$random_seed

python snn_mnist.py --mode=test --num_train=$num_train --conv_features=$conv_features --random_seed=$random_seed --num_test=10000

exit
