#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=00-03:00:00
#SBATCH --mem=30000
#SBATCH --account=rkozma
connectivity=${1:-none}
conv_size=${2:-28}
conv_stride=${3:-0}
conv_features=${4:-10}
lattice_structure=${5:-4}
weight_sharing=${6:-no_weight_sharing}
top_percent=${7:-10}
num_examples=${8:-100}
inhib_const=${12:-0.100000}
inhib_scheme=${13:-increasing}
do_plot=${14:-False}
save_weights=${15:-True}

cd ../train/

python csnn_pc_inhibit_far_mnist.py --mode=train --connectivity=$connectivity --weight_dependence=no_weight_dependence --post_pre=postpre --conv_size=$conv_size \
	--conv_stride=$conv_stride --conv_features=$conv_features --weight_sharing=$weight_sharing --lattice_structure=$lattice_structure --top_percent=$top_percent\
 --num_examples=$num_examples  --inhib_const=$inhib_const --do_plot=$do_plot --save_weights=$save_weights\

python csnn_pc_inhibit_far_mnist.py --mode=test --connectivity=$connectivity --weight_dependence=no_weight_dependence --post_pre=postpre --conv_size=$conv_size \
--conv_stride=$conv_stride --conv_features=$conv_features --weight_sharing=$weight_sharing --lattice_structure=$lattice_structure --top_percent=$top_percent \
	 --num_examples=$num_examples
exit