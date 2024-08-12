#!/bin/bash

function parse_arguments() {
    IFS='_' read -ra splitted_file_name <<< $(basename $0)
    ALGO_NAME=${splitted_file_name[-1]::-3}
    ENV_NAME=$(basename $(dirname ${0}))
    ARGS=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --algo_name)
                ALGO_NAME=$2
                shift
                shift
                ;;
            --env_name)
                ENV_NAME=$2
                shift
                shift
                ;;
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
            -fs | --features)
                shift
                FEATURES=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    FEATURES="$FEATURES $1"
                    shift
                done
                ARGS="$ARGS --features $FEATURES"
                ;;
            -rbc | --replay_buffer_capacity)
                ARGS="$ARGS -replay_buffer_capacity $2"
                shift
                shift
                ;;
            -bs | --batch_size)
                ARGS="$ARGS -batch_size $2"
                shift
                shift
                ;;
            -n | --update_horizon)
                ARGS="$ARGS -update_horizon $2"
                shift
                shift
                ;;
            -gamma | --gamma)
                ARGS="$ARGS -gamma $2"
                shift
                shift
                ;;
            -lr | --learning_rate)
                ARGS="$ARGS -learning_rate $2"
                shift
                shift
                ;;
            -h | --horizon)
                ARGS="$ARGS -horizon $2"
                shift
                shift
                ;;
            # ---- fqi parameters ----
            -g | --gpu)
                GPU=true
                shift
                ;;
            -nbi | --n_bellman_iterations)
                ARGS="$ARGS -n_bellman_iterations $2"
                shift
                shift
                ;;
            -nfs | --n_fitting_steps)
                ARGS="$ARGS -n_fitting_steps $2"
                shift
                shift
                ;;
            # ---- dqn parameters ----
            -ne | --n_epochs)
                ARGS="$ARGS -n_epochs $2"
                shift
                shift
                ;;
            -ntspe | --n_training_steps_per_epoch)
                ARGS="$ARGS -n_training_steps_per_epoch $2"
                shift
                shift
                ;;
            -utd | --update_to_data)
                ARGS="$ARGS -update_to_data $2"
                shift
                shift
                ;;
            -tuf | --target_update_frequency)
                ARGS="$ARGS -target_update_frequency $2"
                shift
                shift
                ;;
            --nis | ---n_initial_samples)
                ARGS="$ARGS --n_initial_samples $2"
                shift
                shift
                ;;
            --ee | --epsilon_end)
                ARGS="$ARGS --epsilon_end $2"
                shift
                shift
                ;;
            --ed | --epsilon_duration)
                ARGS="$ARGS --epsilon_duration $2"
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

    [ -d experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME ] || mkdir -p experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME

}