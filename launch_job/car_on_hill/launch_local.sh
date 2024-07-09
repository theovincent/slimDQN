#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/car_on_hill/logs/$EXPERIMENT_NAME/FQI ] || mkdir -p experiments/car_on_hill/logs/$EXPERIMENT_NAME/FQI

if [[ $GPU = true ]]
then
    source env_gpu/bin/activate
else
    source env_cpu/bin/activate
fi

echo "launch train fqi local"
for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    tmux new-session -d -s car_on_hill_${EXPERIMENT_NAME}_${seed}\
    "car_on_hill_fqi -e $EXPERIMENT_NAME -s $seed $BASE_ARGS $FQI_ARGS >> experiments/car_on_hill/logs/$EXPERIMENT_NAME/FQI/seed_$seed.out 2>&1;\
    tmux kill-session -t car_on_hill_${EXPERIMENT_NAME}_${seed}"
done
