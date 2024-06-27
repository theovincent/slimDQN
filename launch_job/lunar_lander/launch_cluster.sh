#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/LunarLander/out/$EXPERIMENT_NAME/DQN ] || mkdir -p experiments/LunarLander/out/$EXPERIMENT_NAME/DQN

echo "launch train dqn"

if [[ $GPU = true ]]
then
    submission_train_dqn="sbatch -J L_train_dqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=750M --time=2:00:00 --gres=gpu:1 -p stud,stud3080 \
    --output=experiments/LunarLander/out/$EXPERIMENT_NAME/DQN/train_dqn_%a.out \
    launch_job/lunar_lander/train_dqn.sh -e $EXPERIMENT_NAME $BASE_ARGS $DQN_ARGS -g"
else
    submission_train_dqn="sbatch -J L_train_dqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=8 --mem-per-cpu=750M --time=2:00:00 -p stud,stud3080 \
    --output=experiments/LunarLander/out/$EXPERIMENT_NAME/DQN/train_dqn_%a.out \
    launch_job/lunar_lander/train_dqn.sh -e $EXPERIMENT_NAME $BASE_ARGS $DQN_ARGS"
fi
    
$submission_train_dqn
