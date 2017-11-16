# Grid search over hyperparameters for experiments with new inhibition schemes.

for num_train in 60000
do
	for random_seed in 1 2 3
	do
		for conv_features in 625
		do
			for proportion_grow in 0.25 0.5 0.75 1.0
			do
				for max_inhib in 5.0 7.5 10.0 12.5 15.0 17.5 20.0 25.0
				do
					sbatch csnn_growing_inhibition_vary_max_inhib_job.sh 28 0 $conv_features \
											$num_train $random_seed $proportion_grow $max_inhib
				done
			done
		done
	done
done

exit
