#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

if ! tmux has-session -t slimdqn; then
    tmux new-session -d -s slimdqn
    echo "Created new tmux session - slimdqn"
fi

tmux send-keys -t slimdqn "cd $(pwd)" ENTER
if [[ $GPU = true ]]
then
    tmux send-keys -t slimdqn "source env_gpu/bin/activate" ENTER
    FRACTION_GPU=$(echo "scale=2 ; 1 / ($LAST_SEED - $FIRST_SEED + 1)" | bc)
    tmux send-keys -t slimdqn "export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION_GPU" ENTER
else
    tmux send-keys -t slimdqn "source env_cpu/bin/activate" ENTER
fi

echo "launch train $ALGO_NAME local"
for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    tmux send-keys -t slimdqn\
    "python3 experiments/$ENV_NAME/$ALGO_NAME.py --experiment_name $EXPERIMENT_NAME --seed $seed $ARGS >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &" ENTER
done
tmux send-keys -t slimdqn "wait" ENTER
