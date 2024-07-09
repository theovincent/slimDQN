#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/lunar_lander/logs/$EXPERIMENT_NAME/DQN ] || mkdir -p experiments/lunar_lander/logs/$EXPERIMENT_NAME/DQN

if [[ $GPU = true ]]
then
    source env_gpu/bin/activate
else
    source env_cpu/bin/activate
fi

echo "launch train dqn local"
for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    tmux new-session -d -s lunar_lander_${EXPERIMENT_NAME}_${seed}\
    "lunar_lander_dqn -e $EXPERIMENT_NAME -s $seed $BASE_ARGS $DQN_ARGS >> experiments/lunar_lander/logs/$EXPERIMENT_NAME/DQN/seed_$seed.out 2>&1;\
    tmux kill-session -t lunar_lander_${EXPERIMENT_NAME}_${seed}"
done
