#!/bin/bash

function parse_arguments() {
    BASE_ARGS=""
    FQI_ARGS=""
    DQN_ARGS=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e | --experiment_name)
                EXPERIMENT_NAME=$2
                shift
                shift
                ;;
            -frs | --first_seed)
                FIRST_SEED=$2
                shift
                shift
                ;;
            -lrs | --last_seed)
                LAST_SEED=$2
                shift
                shift
                ;;
            -hl | --hidden_layers)
                shift
                HIDDEN_LAYER=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    HIDDEN_LAYER="$HIDDEN_LAYER $1"
                    shift
                done
                BASE_ARGS="$BASE_ARGS -hl $HIDDEN_LAYER"
                ;;
            -rb | --replay_capacity)
                BASE_ARGS="$BASE_ARGS -rb $2"
                shift
                shift
                ;;
            -bs | --batch_size)
                BASE_ARGS="$BASE_ARGS -bs $2"
                shift
                shift
                ;;
            -n | --update_horizon)
                BASE_ARGS="$BASE_ARGS -n $2"
                shift
                shift
                ;;
            -gamma | --gamma)
                BASE_ARGS="$BASE_ARGS -gamma $2"
                shift
                shift
                ;;
            -lr | --lr)
                BASE_ARGS="$BASE_ARGS -lr $2"
                shift
                shift
                ;;
            -hor | --horizon)
                BASE_ARGS="$BASE_ARGS -hor $2"
                shift
                shift
                ;;
            -g | --gpu)
                GPU=true
                shift
                ;;
            -nbi | --n_bellman_iterations)
                FQI_ARGS="$FQI_ARGS -nbi $2"
                shift
                shift
                ;;
            -fs | --n_fitting_steps)
                FQI_ARGS="$FQI_ARGS -fs $2"
                shift
                shift
                ;;
            -ne | --n_epochs)
                DQN_ARGS="$DQN_ARGS -ne $2"
                shift
                shift
                ;;
            -spe | --n_training_steps_per_epoch)
                DQN_ARGS="$DQN_ARGS -spe $2"
                shift
                shift
                ;;
            -utd | --update_to_data)
                DQN_ARGS="$DQN_ARGS -utd $2"
                shift
                shift
                ;;
            -tuf | --target_update_frequency)
                DQN_ARGS="$DQN_ARGS -tuf $2"
                shift
                shift
                ;;
            -n_init | --n_initial_samples)
                DQN_ARGS="$DQN_ARGS -n_init $2"
                shift
                shift
                ;;
            -eps_e | --end_epsilon)
                DQN_ARGS="$DQN_ARGS -eps_e $2"
                shift
                shift
                ;;
            -eps_dur | --duration_epsilon)
                DQN_ARGS="$DQN_ARGS -eps_dur $2"
                shift
                shift
                ;;
            -?*)
                printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
                shift
                shift
                ;;
            ?*)
                printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
                shift
                ;;
        esac
    done

    if [[ $EXPERIMENT_NAME == "" ]]
    then
        echo "experiment name is missing, use -e" >&2
        exit
    elif ( [[ $FIRST_SEED = "" ]] || [[ $LAST_SEED = "" ]] )
    then
        echo "you need to specify -frs and -lrs" >&2
        exit
    fi
    if [[ $GPU == "" ]]
    then
        GPU=false
    fi
}