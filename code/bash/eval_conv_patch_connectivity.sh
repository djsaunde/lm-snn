#!/bin/bash
#
#SBATCH --partition=defq
#SBATCH --mem=8000
#SBATCH --account=rkozma

connectivity=none
conv_size=27
conv_stride=1
conv_features=50
lattice_structure=8
weight_sharing=no_weight_sharing

python MNIST_conv_patch_connectivity_evaluation.py --mode=train --connectivity=$connectivity --weight_dependence=weight_dependence --post_pre=postpre --conv_size=$conv_size \
        --conv_stride=$conv_stride --conv_features=$conv_features --weight_sharing=$weight_sharing --lattice_structure=$lattice_structure --random_inhibition_prob=0.0 --top_percent=10
exit

