# Grid search over hyperparameters for experiments with new inhibition schemes.

num_train=30000

for random_seed in 1 2 3 4 5
do
	for conv_features in 225 400 625
	do
		sh save_spikes_csnn_growing_inhibition_job.sh 28 0 $conv_features $num_train $random_seed &
	done
done

exit
