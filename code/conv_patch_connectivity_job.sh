#!/bin/bash
#
# SBATCH --partition=longq
# SBATCH --time=03-12:00:00
# SBATCH --mem=20000
# SBATCH --account=rkozma

python spiking_conv_patch_connectivity_MNIST.py --mode=train --connectivity=all --weight_dependence=no_weight_dependence --post_pre=postpre --conv_size=16 \
	--conv_stride=4 --conv_features=50 --weight_sharing=no_weight_sharing --lattice_structure=4 --random_inhibition_prob=0.0 --top_percent=10
python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=all --weight_dependence=no_weight_dependence --post_pre=postpre --conv_size=16 \
	--conv_stride=4 --conv_features=50 --weight_sharing=no_weight_sharing --lattice_structure=4 --random_inhibition_prob=0.0 --top_percent=10