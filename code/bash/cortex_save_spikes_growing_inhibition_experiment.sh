# Grid search over hyperparameters for experiments with new inhibition schemes.

num_train=30000

cd ../train

for random_seed in 1 2 3 4 5
do
	for conv_features in 225 400 625
	do
		python csnn_growing_inhibition.py --conv_size 28 --conv_stride 0 --conv_features $conv_features --num_train $num_train --random_seed $random_seed --save_spikes True &
	done
done

exit
