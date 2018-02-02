# Grid search over hyperparameters for experiments with new inhibition schemes.

source /home/dsanghavi/anaconda2/bin/activate

for num_train in 120000 180000
do
	for random_seed in 0 1 2 3 4
	do
		for conv_features in 1225 1600
		do
			for proportion_low in 0.1
			do
				for start_inhib in 1.0
				do
					for max_inhib in 20.0
					do
						sbatch csnn_two_level_inhibition_ngram_job.sh 28 0 $conv_features $num_train \
								$random_seed $proportion_low $start_inhib $max_inhib
					done
				done
			done
		done
	done
done
source deactivate

exit
