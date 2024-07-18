#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/car_on_hill/logs/$EXPERIMENT_NAME/FQI ] || mkdir -p experiments/car_on_hill/logs/$EXPERIMENT_NAME/FQI

if ! tmux has-session -t slimRL; then
    tmux new-session -d -s slimRL
    echo "Created new tmux session - slimRL"
fi

if [[ $GPU = true ]]
then
    tmux send-keys -t slimRL "source env_gpu/bin/activate" ENTER
else
    tmux send-keys -t slimRL "source env_cpu/bin/activate" ENTER
fi

echo "launch train fqi local"
for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    tmux send-keys -t slimRL\
    "car_on_hill_fqi -e $EXPERIMENT_NAME -s $seed $BASE_ARGS $FQI_ARGS >> experiments/car_on_hill/logs/$EXPERIMENT_NAME/FQI/seed_$seed.out 2>&1 &" ENTER
done
tmux send-keys -t slimRL "wait" ENTER
