#!/bin/bash
#
#SBATCH --job-name=nws_none_csnn_contrast
#SBATCH --output=nws_none_csnn_contrast.out
#SBATCH --partition=longq
#SBATCH --time=2-00:00:00
#SBATCH --mem=40000
#SBATCH --account=rkozma

python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=none --conv_size=27 --conv_stride=1 --conv_features=50 --lattice_structure=8 --weight_sharing=no_weight_sharing
python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=none --conv_size=26 --conv_stride=1 --conv_features=50 --lattice_structure=8 --weight_sharing=no_weight_sharing
python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=none --conv_size=24 --conv_stride=2 --conv_features=50 --lattice_structure=8 --weight_sharing=no_weight_sharing
python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=none --conv_size=22 --conv_stride=2 --conv_features=50 --lattice_structure=8 --weight_sharing=no_weight_sharing
python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=none --conv_size=20 --conv_stride=2 --conv_features=50 --lattice_structure=8 --weight_sharing=no_weight_sharing
python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=none --conv_size=18 --conv_stride=2 --conv_features=50 --lattice_structure=8 --weight_sharing=no_weight_sharing
python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=none --conv_size=16 --conv_stride=2 --conv_features=50 --lattice_structure=8 --weight_sharing=no_weight_sharing
python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=none --conv_size=14 --conv_stride=2 --conv_features=50 --lattice_structure=8 --weight_sharing=no_weight_sharing
python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=none --conv_size=12 --conv_stride=2 --conv_features=50 --lattice_structure=8 --weight_sharing=no_weight_sharing
python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=none --conv_size=10 --conv_stride=2 --conv_features=50 --lattice_structure=8 --weight_sharing=no_weight_sharing
python spiking_conv_patch_connectivity_MNIST.py --mode=test --connectivity=none --conv_size=8 --conv_stride=2 --conv_features=50 --lattice_structure=8 --weight_sharing=no_weight_sharing
exit
