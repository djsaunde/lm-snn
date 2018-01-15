model_name=${1:-csnn_two_level_inhibition}
weights_type=${2:-best}
regex=${3:-*}
destination=${4:-../../weights/${model_name}/${weights_type}/}

if [ ! -d $destination  ]; then
	mkdir $destination
fi

scp djsaunde@swarm2.cs.umass.edu:/mnt/nfs/work1/rkozma/djsaunde/stdp-mnist/weights/${model_name}/${weights_type}/${regex} ${destination}
