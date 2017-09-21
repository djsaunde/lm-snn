# Grid search over hyperparameters for experiments with new inhibition schemes.

num_train=15000
reduced_dataset="False"
inhib_scheme="strengthen"
conv_features=625

for random_seed in 1
do
	for inhib_const in 0.1 0.5 1.0
	do
		for strengthen_const in 0.01 0.1 0.175 0.25
		do
			sh csnn_pc_inhibit_far_job.sh none 28 0 $conv_features 4 10 $num_train $reduced_dataset 1000 8 $inhib_scheme $inhib_const $strengthen_const False 0.0 $random_seed &
		done
	done
done

exit
