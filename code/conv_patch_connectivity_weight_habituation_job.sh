#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=02-00:00:00
#SBATCH --mem=30000
#SBATCH --account=rkozma

connectivity=all
conv_size=16
conv_stride=4
conv_features=100
lattice_structure=8
weight_sharing=no_weight_sharing

python spiking_conv_patch_connectivity_weight_habituation_MNIST.py --mode=train --connectivity=$connectivity --weight_dependence=no_weight_dependence --post_pre=postpre --conv_size=$conv_size \
        --conv_stride=$conv_stride --conv_features=$conv_features --weight_sharing=$weight_sharing --lattice_structure=$lattice_structure --random_inhibition_prob=0.0 --top_percent=10
python spiking_conv_patch_connectivity_weight_habituation_MNIST.py --mode=test --connectivity=$connectivity --weight_dependence=no_weight_dependence --post_pre=postpre --conv_size=$conv_size \
        --conv_stride=$conv_stride --conv_features=$conv_features --weight_sharing=$weight_sharing --lattice_structure=$lattice_structure --random_inhibition_prob=0.0 --top_percent=10
exit

