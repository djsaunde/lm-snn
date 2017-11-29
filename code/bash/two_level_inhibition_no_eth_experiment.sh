# Grid search over hyperparameters for experiments with new inhibition schemes.

for num_train in 60000
do
	for random_seed in 1 2 3
	do
		for conv_features in 625
		do
			for proportion_low in 0.05 0.1 0.25 5.0
			do
				for start_low_inhib in 0.5 1.0 2.5
				do
					for end_low_inhib in 10.0 12.5 15.0 
					do
						for max_inhib in 17.5 18.75 20 21.25 22.5
						do
							sbatch csnn_two_level_inhibition_no_eth_job.sh 28 0 $conv_features $num_train $random_seed \
										$proportion_low $start_low_inhib $end_low_inhib $max_inhib
						done
					done
				done
			done
		done
	done
done

exit
