# Grid search over hyperparameters for experiments with new inhibition schemes.

num_train=60000
reduced_dataset="False"
inhib_scheme="strengthen"
conv_features=400

for random_seed in 1 2 3
do
	sbatch csnn_pc_growing_inhibition_job.sh 28 0 $conv_features $num_train $random_seed
done

exit
