#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

if ! tmux has-session -t slimDQN; then
    tmux new-session -d --seed slimDQN
    echo "Created new tmux session - slimDQN"
fi

if [[ $GPU = true ]]
then
    tmux send-keys -t slimDQN "source env_gpu/bin/activate" ENTER
else
    tmux send-keys -t slimDQN "source env_cpu/bin/activate" ENTER
fi

echo "launch train $ALGO_NAME local"
for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    tmux send-keys -t slimDQN\
    "python3 experiments/$ENV_NAME/$ALGO_NAME.py --experiment_name $EXPERIMENT_NAME --seed $seed $ARGS >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &" ENTER
done
tmux send-keys -t slimDQN "wait" ENTER
