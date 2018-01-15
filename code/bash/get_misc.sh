model_name=${1:-csnn_two_level_inhibition}
misc_type=${2:-best}
regex=${3:-*}
destination=${4:-../../misc/${model_name}/${misc_type}/}

if [ ! -d $destination  ]; then
	mkdir -p $destination
fi

scp djsaunde@swarm2.cs.umass.edu:/mnt/nfs/work1/rkozma/djsaunde/stdp-mnist/misc/${model_name}/${misc_type}/${regex} ${destination}
