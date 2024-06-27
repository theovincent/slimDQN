#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/CarOnHill/out/$EXPERIMENT_NAME/FQI ] || mkdir -p experiments/CarOnHill/out/$EXPERIMENT_NAME/FQI

if [[ $GPU = true ]]
then
    source env_gpu/bin/activate
else
    source env_cpu/bin/activate
fi

echo "launch train fqi local"
for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do  
    car_on_hill_fqi -e $EXPERIMENT_NAME -s $seed $BASE_ARGS $FQI_ARGS &> experiments/CarOnHill/out/$EXPERIMENT_NAME/FQI/seed=$seed.out &
done
