# Grid search over hyperparameters for experiments with new inhibition schemes.

num_train=15000
reduced_dataset="True"
examples_per_class=500
inhib_scheme="increasing"
conv_features=400

for random_seed in 1 2 3
do
	for inhib_const in 0.1 0.5 1.0 2.5 5.0
	do
		sbatch csnn_pc_inhibit_far_job.sh none 28 0 $conv_features 4 10 $num_train $reduced_dataset $examples_per_class 8 $inhib_scheme $inhib_const 0.5 False 0.0 $random_seed
	done
done

exit
