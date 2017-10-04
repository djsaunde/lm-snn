# Grid search over hyperparameters for experiments with new inhibition schemes.

num_train=30000
reduced_dataset="False"
inhib_scheme="strengthen"

for random_seed in 1 2 3 
do
	for conv_features in 100 225 400 625 900 1225 1600 2025 2500
	do
		sbatch csnn_growing_inhibition_job.sh 28 0 $conv_features $num_train $random_seed
	done
done

exit
