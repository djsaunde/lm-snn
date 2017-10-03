# Grid search over hyperparameters for experiments with new inhibition schemes.

num_train=30000
conv_features=400

for random_seed in 1 2 3
do
	sbatch csnn_pc_growing_inhibition_test.sh 28 0 $conv_features $num_train $random_seed
done

exit
