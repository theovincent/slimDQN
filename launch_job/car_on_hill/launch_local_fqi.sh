#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

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

echo "launch train $ALGO_NAME local"
for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    tmux send-keys -t slimRL\
    "$ENV_NAME\_$ALGO_NAME -e $EXPERIMENT_NAME -s $seed $ARGS >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &" ENTER
done
tmux send-keys -t slimRL "wait" ENTER
