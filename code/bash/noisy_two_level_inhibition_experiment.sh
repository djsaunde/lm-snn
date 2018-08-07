for num_train in 60000
do
	for random_seed in 0
	do
		for conv_features in 400
		do
			for proportion_low in 0.1
			do
				for start_inhib in 1.0 20.0
				do
					for max_inhib in 20.0
					do
						for p_flip in $(seq 0.0 0.01 0.5)
						do
							sbatch csnn_noisy_two_level_inhibition_job.sh 28 0 $conv_features $num_train \
								                                          $random_seed $proportion_low \
								                                          $start_inhib $max_inhib $p_flip
						done
					done
				done
			done
		done
	done
done

exit
