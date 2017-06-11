#!/bin/bash
#
#SBATCH --job-name=test_14_2_50_none
#SBATCH --output=test_14_2_50_none.out
#SBATCH --partition=longq
#SBATCH --time=01-00:00:00
#SBATCH --mem=35000
#SBATCH --account=rkozma

connectivity=none
conv_size=14
conv_stride=2
conv_features=50
lattice_structure=4
weight_sharing=no_weight_sharing

python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=$connectivity --conv_size=$conv_size --conv_stride=$conv_stride --conv_features=$conv_features \
	--lattice_structure=$lattice_structure --weight_sharing=$weight_sharing
exit
