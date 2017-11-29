# Grid search over hyperparameters for experiments with new inhibition schemes.

mkdir snn_job_reports

for num_train in 60000
do
	for random_seed in 1 2 3 4 5 6 7 8 9 10
	do
		for conv_features in 400 625 900 1225 1600 2025
		do
			sbatch snn_job.sh $conv_features $num_train $random_seed
		done
	done
done

exit
