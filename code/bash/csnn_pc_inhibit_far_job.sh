#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=01-00:00:00
#SBATCH --mem=30000
#SBATCH --account=rkozma
#SBATCH --output=/mnt/nfs/work1/rkozma/djsaunde/stdp-mnist/code/bash/job_reports/%j.out

connectivity=${1:-none}
conv_size=${2:-28}
conv_stride=${3:-0}
conv_features=${4:-100}
lattice_structure=${5:-4}

top_percent=${6:-10}
num_train=${7:-10000}
reduced_dataset=${8:-True}
examples_per_class=${9:-500}
neighborhood=${10:-8}
inhib_scheme=${11:-far}
inhib_const=${12:-5.0}
strengthen_const=${13:-0.5}
noise=${14:-True}
noise_const=${15:-0.1}
random_seed=${16:-42}

cd ../train/

echo 1 $connectivity 2 $conv_size 3 $conv_stride 4 $conv_features 5 $lattice_structure 6 $top_percent 7 $num_train 8 $reduced_dataset 9 $examples_per_class 10 \
	$neighborhood 11 $inhib_scheme 12 $inhib_const 13 $strengthen_const 14 $noise 15 $noise_const 16 $random_seed

python csnn_pc_inhibit_far_mnist.py --mode=train --connectivity=$connectivity --conv_size=$conv_size \
	--conv_stride=$conv_stride --conv_features=$conv_features --lattice_structure=$lattice_structure --top_percent=$top_percent \
	--num_train=$num_train --reduced_dataset=$reduced_dataset --examples_per_class=$examples_per_class --neighborhood=$neighborhood \
	--inhib_scheme=$inhib_scheme --inhib_const=$inhib_const --strengthen_const=$strengthen_const --noise=$noise --noise_const=$noise_const --random_seed=$random_seed

python csnn_pc_inhibit_far_mnist.py --mode=test --connectivity=$connectivity --conv_size=$conv_size \
	--conv_stride=$conv_stride --conv_features=$conv_features --lattice_structure=$lattice_structure --top_percent=$top_percent \
	--num_train=$num_train --num_test=10000 --reduced_dataset=$reduced_dataset --examples_per_class=$examples_per_class --neighborhood=$neighborhood \
        --inhib_scheme=$inhib_scheme --inhib_const=$inhib_const --strengthen_const=$strengthen_const --noise=$noise --noise_const=$noise_const --random_seed=$random_seed

exit
