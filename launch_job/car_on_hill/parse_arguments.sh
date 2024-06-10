#!/bin/bash

function parse_arguments() {
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
                while [[ $1 != -* && $# -gt 0 ]]; do
                    HIDDEN_LAYER="$HIDDEN_LAYER $1"
                    shift
                done
                ;;
            -rb | --replay_capacity)
                RB_CAPACITY=$2
                shift
                shift
                ;;
            -B | --batch_size)
                BATCH_SIZE=$2
                shift
                shift
                ;;
            -n | --update_horizon)
                UPDATE_HORIZON=$2
                shift
                shift
                ;;
            -gamma | --gamma)
                GAMMA=$2
                shift
                shift
                ;;
            -lr)
                LEARNING_RATE=$2
                shift
                shift
                ;;
            -lr_eps | --lr_epsilon)
                LR_EPSILON=$2
                shift
                shift
                ;;
            -H | --horizon)
                HORIZON=$2
                shift
                shift
                ;;
            -bi | --n_bellman_iterations)
                N_BELLMAN_ITERATIONS=$2
                shift
                shift
                ;;
            -fs | --n_fitting_steps)
                N_FITTING_STEPS=$2
                shift
                shift
                ;;
            -g | --gpu)
                GPU=true
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
    elif ( [[ $FIRST_SEED != "" ]] && [[ $LAST_SEED = "" ]] ) || ( [[ $FIRST_SEED == "" ]] && [[ $LAST_SEED != "" ]] )
    then
        echo "you need to specify -frs and -lrs, not only one" >&2
        exit
    elif [[ $N_BELLMAN_ITERATIONS == "" ]]
    then
        echo "n_bellman_iterations is missing, use -bi" >&2
        exit
    elif [[ $HIDDEN_LAYER == "" ]]
    then
        echo "hidden_layers is missing, use -hl" >&2
        exit
    elif [[ $GAMMA == "" ]]
    then
        echo "gamma is missing, use -gamma" >&2
        exit
    fi
    if [[ $GPU == "" ]]
    then
        GPU=false
    fi
}