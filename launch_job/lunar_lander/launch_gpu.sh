#!/bin/bash

source launch_job/lunar_lander/parse_arguments.sh
parse_arguments $@

[ -d experiments/LunarLander/logs/$EXPERIMENT_NAME/DQN/out ] || mkdir -p experiments/LunarLander/logs/$EXPERIMENT_NAME/DQN/out
[ -d experiments/LunarLander/logs/$EXPERIMENT_NAME/DQN/error ] || mkdir -p experiments/LunarLander/logs/$EXPERIMENT_NAME/DQN/error

echo "launch train dqn"
submission_train_dqn="sbatch -J L_train_dqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750M --time=30:00 -p stud"
submission_train_dqn="$submission_train_dqn --output=experiments/LunarLander/logs/$EXPERIMENT_NAME/DQN/out/train_dqn_%a.out --error=experiments/LunarLander/logs/$EXPERIMENT_NAME/DQN/error/train_dqn_%a.out)"
submission_train_dqn="$submission_train_dqn launch_job/lunar_lander/train_dqn.sh -e $EXPERIMENT_NAME -hl $HIDDEN_LAYER -gamma $GAMMA"

if [[ ! -z $RB_CAPACITY ]]; then
    submission_train_dqn="$submission_train_dqn -rb $RB_CAPACITY"
fi

if [[ ! -z $BATCH_SIZE ]]; then
    submission_train_dqn="$submission_train_dqn -B $BATCH_SIZE"
fi

if [[ ! -z $UPDATE_HORIZON ]]; then
    submission_train_dqn="$submission_train_dqn -n $UPDATE_HORIZON"
fi

if [[ ! -z $LEARNING_RATE ]]; then
    submission_train_dqn="$submission_train_dqn -lr $LEARNING_RATE"
fi

if [[ ! -z $LR_EPSILON ]]; then
    submission_train_dqn="$submission_train_dqn -lr_eps $LR_EPSILON"
fi

if [[ ! -z $HORIZON ]]; then
    submission_train_dqn="$submission_train_dqn -H $HORIZON"
fi

if [[ ! -z $UPDATE_TO_DATA ]]; then
    submission_train_dqn="$submission_train_dqn -utd $UPDATE_TO_DATA"
fi

if [[ ! -z $TARGET_UPDATE_PERIOD ]]; then
    submission_train_dqn="$submission_train_dqn -T $TARGET_UPDATE_PERIOD"
fi

if [[ ! -z $N_INITIAL_SAMPLES ]]; then
    submission_train_dqn="$submission_train_dqn -n_init $N_INITIAL_SAMPLES"
fi

if [[ ! -z $END_EPSILON ]]; then
    submission_train_dqn="$submission_train_dqn -eps_e $END_EPSILON"
fi

if [[ ! -z $DURATION_EPSILON ]]; then
    submission_train_dqn="$submission_train_dqn -eps_dur $DURATION_EPSILON"
fi

if [[ ! -z $N_EPOCHS ]]; then
    submission_train_dqn="$submission_train_dqn -E $N_EPOCHS"
fi

if [[ ! -z $N_TRAINING_STEPS_PER_EPOCH ]]; then
    submission_train_dqn="$submission_train_dqn -spe $N_TRAINING_STEPS_PER_EPOCH"
fi

$submission_train_dqn