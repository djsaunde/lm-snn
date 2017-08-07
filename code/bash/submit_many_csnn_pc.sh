# Use this file to loop through parameters you wish to use for batch jobs on the swarm2 cluster.

for conv_size in {6..16..2}
do
	for conv_features in {5..25..5}
	do
		sbatch csnn_pc_job.sh 'none' $conv_size 2 $conv_features 'none' 'no_weight_sharing' 10
	done
done

exit
