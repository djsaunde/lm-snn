#!/bin/bash
#
#SBATCH --partition=longq
#SBATCH --time=04-00:00:00
#SBATCH --mem=40000
#SBATCH --ntasks-per-node=8
#SBATCH --account=rkozma


connectivity=none
conv_size=14
conv_stride=2
conv_features=25
weight_sharing=no_weight_sharing
lattice_structure=4

python spiking_conv_patch_connectivity_patch_voting_MNIST.py --mode=train --connectivity=$connectivity --conv_size=$conv_size \
        --conv_stride=$conv_stride --conv_features=$conv_features --weight_sharing=$weight_sharing --lattice_structure=$lattice_structure
python spiking_conv_patch_connectivity_patch_voting_MNIST.py --mode=test --connectivity=$connectivity --conv_size=$conv_size \
        --conv_stride=$conv_stride --conv_features=$conv_features --weight_sharing=$weight_sharing --lattice_structure=$lattice_structure
exit

