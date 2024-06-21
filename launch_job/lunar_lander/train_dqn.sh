#!/bin/bash

source launch_job/lunar_lander/parse_arguments.sh
parse_arguments $@

if [[ $GPU = true ]]
then
    source env_gpu/bin/activate
else
    source env_cpu/bin/activate
fi


command="lunar_lander_dqn -e $EXPERIMENT_NAME -s $SLURM_ARRAY_TASK_ID -hl $HIDDEN_LAYER -gamma $GAMMA"

if [[ ! -z $RB_CAPACITY ]]; then
    command="$command -rb $RB_CAPACITY"
fi

if [[ ! -z $BATCH_SIZE ]]; then
    command="$command -B $BATCH_SIZE"
fi

if [[ ! -z $UPDATE_HORIZON ]]; then
    command="$command -n $UPDATE_HORIZON"
fi

if [[ ! -z $LEARNING_RATE ]]; then
    command="$command -lr $LEARNING_RATE"
fi

if [[ ! -z $LR_EPSILON ]]; then
    command="$command -lr_eps $LR_EPSILON"
fi

if [[ ! -z $HORIZON ]]; then
    command="$command -H $HORIZON"
fi

if [[ ! -z $UPDATE_TO_DATA ]]; then
    command="$command -utd $UPDATE_TO_DATA"
fi

if [[ ! -z $TARGET_UPDATE_PERIOD ]]; then
    command="$command -T $TARGET_UPDATE_PERIOD"
fi

if [[ ! -z $N_INITIAL_SAMPLES ]]; then
    command="$command -n_init $N_INITIAL_SAMPLES"
fi

if [[ ! -z $END_EPSILON ]]; then
    command="$command -eps_e $END_EPSILON"
fi

if [[ ! -z $DURATION_EPSILON ]]; then
    command="$command -eps_dur $DURATION_EPSILON"
fi

if [[ ! -z $N_EPOCHS ]]; then
    command="$command -E $N_EPOCHS"
fi

if [[ ! -z $N_TRAINING_STEPS_PER_EPOCH ]]; then
    command="$command -spe $N_TRAINING_STEPS_PER_EPOCH"
fi

eval $command