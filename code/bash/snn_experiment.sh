# Grid search over hyperparameters for experiments with new inhibition schemes.

mkdir snn_job_reports

for num_train in 30000 60000 120000 180000
do
	for random_seed in 1 2 3 4 5 6 7 8 9 10
	do
		for conv_features in 400 900 1600 2500 3600 4900 6400
		do
			sbatch snn_mnist_job.sh $conv_features $num_train $random_seed
		done
	done
done

exit
