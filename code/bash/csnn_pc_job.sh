#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=30000
#SBATCH --account=rkozma

connectivity=${1:-all}
conv_size=${2:-16}
conv_stride=${3:-4}
conv_features=${4:-50}
lattice_structure=${5:-4}
weight_sharing=${6:-weight_sharing}
top_percent=${7:-10}

cd ../train/

python csnn_pc_mnist.py --mode=train --connectivity=$connectivity --weight_dependence=no_weight_dependence --post_pre=postpre --conv_size=$conv_size \
	--conv_stride=$conv_stride --conv_features=$conv_features --weight_sharing=$weight_sharing --lattice_structure=$lattice_structure --top_percent=$top_percent
python csnn_pc_mnist.py --mode=test --connectivity=$connectivity --weight_dependence=no_weight_dependence --post_pre=postpre --conv_size=$conv_size \
	--conv_stride=$conv_stride --conv_features=$conv_features --weight_sharing=$weight_sharing --lattice_structure=$lattice_structure --top_percent=$top_percent

exit
