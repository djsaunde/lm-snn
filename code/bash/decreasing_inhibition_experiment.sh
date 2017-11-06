# Grid search over hyperparameters for experiments with new inhibition schemes.

normalize_inputs='False'
noise_const=0.0

for num_train in 45000 60000
do
	for random_seed in 1 2 3 4 5
	do
		for conv_features in 400 625
		do
			for proportion_shrink in 0.25 0.5 0.75 1.0
			do
				for start_inhib in 25.0 20.0 17.5 15.0
				do
					for min_inhib in 0.0 0.5 1.0 2.5
					do
						sbatch csnn_decreasing_inhibition_job.sh 28 0 $conv_features $num_train $random_seed $normalize_inputs $proportion_shrink $noise_const
					done
				done
			done
		done
	done
done

exit
