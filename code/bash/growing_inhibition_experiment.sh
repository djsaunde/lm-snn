# Grid search over hyperparameters for experiments with new inhibition schemes.

for num_train in 30000 45000 60000
do
	for random_seed in 1 2 3
	do
		for conv_features in 400
		do
			for normalized_inputs in False
			do
				for proportion_grow in 0.25 0.5 0.75 1.0
				do
					for noise_const in 0.1 0.25 0.5 0.75 1.0
					do
						sbatch csnn_growing_inhibition_job.sh 28 0 $conv_features $num_train $random_seed $normalized_inputs $proportion_grow $noise_const
					done
				done
			done
		done
	done
done

exit
