# Grid search over hyperparameters for experiments with new inhibition schemes.

num_train=30000
reduced_dataset="False"
examples_per_class=1000
inhib_scheme="strengthen"
conv_features=400
reset_state_vars="False"

for random_seed in 1
do
	for inhib_const in 0.5 1.0 2.5 5.0 10.0
	do
		for strengthen_const in 0.0 0.05 0.1 0.25 0.5
		do
			for n_clusters in 10 20 30 40 50 60 70 80 90 100
			do
				sbatch csnn_pc_inhibit_far_cluster_filters_job.sh none 28 0 $conv_features 4 10 $num_train $reduced_dataset $examples_per_class \
							8 $inhib_scheme $inhib_const $strengthen_const False 0.0 $random_seed $reset_state_vars $n_clusters
			done
		done
	done
done

exit
