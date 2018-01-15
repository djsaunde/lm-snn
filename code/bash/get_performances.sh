model_name=${1:-csnn_two_level_inhibition}
regex=${2:-*}
destination=${3:-../../performance/${model_name}/}

if [ ! -d $destination  ]; then
	mkdir $destination
fi

scp djsaunde@swarm2.cs.umass.edu:/mnt/nfs/work1/rkozma/djsaunde/stdp-mnist/performance/${model_name}/${regex} ${destination}
