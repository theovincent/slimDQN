#!/bin/bash

source launch_job/car_on_hill/parse_arguments.sh
parse_arguments $@

[ -d experiments/CarOnHill/logs/$EXPERIMENT_NAME ] || mkdir -p experiments/CarOnHill/logs/$EXPERIMENT_NAME
[ -d experiments/CarOnHill/logs/$EXPERIMENT_NAME/FQI ] || mkdir experiments/CarOnHill/logs/$EXPERIMENT_NAME/FQI


for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    echo "launch train fqi"
    command="car_on_hill_fqi -e $EXPERIMENT_NAME -bi $N_BELLMAN_ITERATIONS -hl $HIDDEN_LAYER -s $seed -gamma $GAMMA"

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

    if [[ ! -z $N_FITTING_STEPS ]]; then
        command="$command -lr $N_FITTING_STEPS"
    fi

    eval $command &> experiments/CarOnHill/logs/$EXPERIMENT_NAME/FQI/seed=$seed.out &
done