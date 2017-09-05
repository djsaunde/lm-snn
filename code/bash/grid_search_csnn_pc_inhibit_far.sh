# Grid search over hyperparameters for experiments with new inhibition schemes.

num_examples=10000
reduced_dataset="True"
examples_per_class=500

for conv_features in 49 81 100 144 196 225 324 400
do
	for inhib_scheme in "far" "increasing" "strengthen"
	do
		for noise in "True" "False"
		do
			if [[ "$noise" -eq "True" ]]
			then
				for noise_const in 0.05 0.1 0.15 0.25 0.5
				do
					if [[ "$inhib_scheme" -ne "far" ]]
					then
						for inhib_const in 1.0 2.5 5.0 10.0
						do
							sbatch csnn_pc_inhibit_far_job.sh none 28 0 $conv_features 4 10 $num_examples $reduced_dataset \
								$examples_per_class 8 $inhib_scheme $inhib_const 0.5 $noise $noise_const
						done
					fi
					if [[ "$inhib_scheme" -eq "strengthen" ]]
					then
						for strengthen_const in 0.05 0.1 0.225 0.375 0.5 0.625
						do
							sbatch csnn_pc_inhibit_far_job.sh none 28 0 $conv_features 4 10 $num_examples $reduced_dataset \
                                                                $examples_per_class 8 $inhib_scheme $inhib_const $strengthen_const $noise $noise_const
						done    
					fi
				done
			else
				if [[ "$inhib_scheme" -ne "far" ]]
				then
					for inhib_const in 1.0 2.5 5.0 10.0
					do
						sbatch csnn_pc_inhibit_far_job.sh none 28 0 $conv_features 4 10 $num_examples $reduced_dataset \
                                                	$examples_per_class 8 $inhib_scheme $inhib_const 0.5 $noise 0.0
					done
				fi
				if [ "$inhib_scheme" -eq "strengthen" ]
				then
					for strengthen_const in 0.05 0.1 0.225 0.375 0.5 0.625
					do
						sbatch csnn_pc_inhibit_far_job.sh none 28 0 $conv_features 4 10 $num_examples $reduced_dataset \
                                                        $examples_per_class 8 $inhib_scheme $inhib_const $strengthen_const $noise 0.0
					done
				fi
			fi
		done	
	done
done
exit
