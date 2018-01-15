# Grid search over hyperparameters for experiments with new inhibition schemes.

for num_train in 60000
do
	for random_seed in 0 1 2 3 4
	do
		for conv_features in 400 625 900
		do
			for proportion_grow in 0.25 0.5 0.75 1.0
			do
				sbatch csnn_growing_inhibition_job.sh 28 0 $conv_features $num_train $random_seed $proportion_grow
			done
		done
	done
done

exit
