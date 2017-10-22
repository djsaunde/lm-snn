# Grid search over hyperparameters for experiments with new inhibition schemes.

for num_train in 15000 30000 45000 60000
do
	for random_seed in 1 2 3 4 5
	do
		for conv_features in 400
		do
			for normalized_inputs in False
			do
				for proportion_grow in 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0
				do
					sbatch csnn_growing_inhibition_job.sh 28 0 $conv_features $num_train $random_seed $normalized_inputs $proportion_grow
				done
			done
		done
	done
done

exit
