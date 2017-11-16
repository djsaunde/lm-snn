# Grid search over hyperparameters for experiments with new inhibition schemes.

for num_train in 30000 60000 120000 180000
do
	for random_seed in 1 2 3 4 5
	do
		for conv_features in 1600 6400
		do
			sbatch csnn_growing_inhibition_job.sh 28 0 $conv_features $num_train $random_seed 
		done
	done
done

exit
