# Grid search over hyperparameters for experiments with new inhibition schemes.

normalize_inputs='False'
test_rest=0.15

for num_train in 45000 60000
do
	for random_seed in 1 2 3 4 5
	do
		for conv_features in 400 625
		do
			for proportion_grow in 0.25 0.5 0.75 1.0
			do
				for noise_const in 0.0 0.1
				do
					for test_time in 0.05
					do
						for num_tests in 15 30
						do
							sbatch csnn_growing_inhibition_multiple_test.sh 28 0 $conv_features $num_train $random_seed \
									$normalize_inputs $proportion_grow $noise_const $test_time $test_rest $num_tests
						done
					done
				done
			done
		done
	done
done

exit
