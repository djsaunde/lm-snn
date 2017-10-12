# Grid search over hyperparameters for experiments with new inhibition schemes.

cd ../train

num_train=30000

for random_seed in 1 2 3 4 5
do
	for conv_features in 225 400 625
	do
		for normalize_inputs in True False
		do
			/home/djsaunde/anaconda2/bin/python csnn_growing_inhibition.py --conv_features 28 --conv_stride 0 --conv_features $conv_features \
					--num_train $num_train --random_seed $random_seed --normalize_inputs $normalize_inputs &
		done
	done
done

exit
