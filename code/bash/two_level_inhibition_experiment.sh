# Grid search over hyperparameters for experiments with new inhibition schemes.

for num_train in 60000
do
	for random_seed in 1 2 3
	do
		for conv_features in 625
		do
			for proportion_low in 0.25 0.5 0.75
			do
				for start_inhib in 0.1 1.0 2.5
				do
					for max_inhib in 10.0 15.0 17.4 20.0
					do
						sbatch csnn_two_level_inhibition_job.sh 28 0 $conv_features $num_train \
											$random_seed $proportion_low $start_inhib $max_inhib
					done
				done
			done
		done
	done
done

exit
