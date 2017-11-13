# Grid search over hyperparameters for experiments with new inhibition schemes.

normalize_inputs='False'
noise_const=0.0

for num_train in 60000
do
	for random_seed in 1 2 3 4 5
	do
		for conv_features in 625
		do
			for proportion_grow in 0.25 0.5 0.75 1.0
			do
				for input_sparsity in 0.1 0.25 0.5 0.75 0.9 1.0
				do
					sbatch csnn_growing_inhibition_job.sh 28 0 $conv_features $num_train $random_seed \
							$normalize_inputs $proportion_grow $noise_const $input_sparsity
				done
			done
		done
	done
done

exit
