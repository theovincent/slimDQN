#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

echo "launch train $ALGO_NAME"

if [[ $GPU = true ]]
then
    sbatch -J $EXPERIMENT_NAME-$ALGO_NAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=750M --time=00:10:00 --gres=gpu:1 -p stud,stud3080 \
    --output=experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_%a.out \
    launch_job/$ENV_NAME/train_$ALGO_NAME.sh --algo_name $ALGO_NAME --env_name $ENV_NAME -e $EXPERIMENT_NAME $ARGS -g
else
    sbatch -J $EXPERIMENT_NAME-$ALGO_NAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=750M --time=00:15:00 -p stud,stud3080 \
    --output=experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_%a.out \
    launch_job/$ENV_NAME/train_$ALGO_NAME.sh --algo_name $ALGO_NAME --env_name $ENV_NAME -e $EXPERIMENT_NAME $ARGS
fi
