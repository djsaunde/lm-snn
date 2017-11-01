# Grid search over hyperparameters for experiments with new inhibition schemes.

noise_const=0.0
normalize_inputs=False

for num_train in 45000 60000
do
	for random_seed in 1 2 3
	do
		for conv_features in 400 625
		do
			for proportion_grow in 0.25 0.5 0.75 1.0
			do
				for linear_train_input in True False
				do
					for linear_test_input in True False
					do
						sbatch csnn_growing_inhibition_linear_inputs_job.sh 28 0 $conv_features $num_train $random_seed \
								$normalized_inputs $proportion_grow $noise_const $linear_train_input $linear_test_input
					done
				done
			done
		done
	done
done

exit
