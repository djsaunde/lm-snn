# Grid search over hyperparameters for experiments with new inhibition schemes.

num_train=15000
reduced_dataset="True"
examples_per_class=500
inhib_scheme="increasing"
conv_features=225

for random_seed in 1 2 3
do
	for noise_const in 0.01 0.05 0.1 0.15 0.2 0.25 0.375 0.5
	do
		sbatch csnn_pc_inhibit_far_job.sh none 28 0 $conv_features 4 10 $num_train $reduced_dataset $examples_per_class 8 $inhib_scheme 0.25 0.5 True $noise_const $random_seed
	done
done

exit
