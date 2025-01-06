#!/bin/bash

EXPERIMENT_NAME=dqn_Breakout
CPUS_PER_TASK=3
MEM_PER_CPU=6
ARGS="--experiment_name $EXPERIMENT_NAME  --first_seed 0 --last_seed 0 --n_parallel_seeds 1 -gpu --disable_wandb \
--features 32 64 64 512 --replay_buffer_capacity 1_000_000 --batch_size 32 --update_horizon 1 --gamma 0.99 --learning_rate 5e-5 \
--horizon 27_000 --n_epochs 6 --n_training_steps_per_epoch 250_000 --update_to_data 4 --target_update_frequency 8000 \
--n_initial_samples 999_998 --epsilon_end 0.01 --epsilon_duration 250_000"


source launch_job/parse_arguments.sh
parse_arguments $ARGS

echo "launch train $ALGO_NAME"

sbatch --job-name $EXPERIMENT_NAME-$ALGO_NAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=$CPUS_PER_TASK --mem-per-cpu=$((N_PARALLEL_SEEDS * MEM_PER_CPU))G --time=02:00:00 --gres=gpu:1 \
--constraint="rtx2080" --partition stud --nodelist="cn04" \
--output=experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_$((N_PARALLEL_SEEDS * (FIRST_SEED - 1) + 1))-$((N_PARALLEL_SEEDS * LAST_SEED)).out \
launch_job/$ENV_NAME/train.sh --algo_name $ALGO_NAME --env_name $ENV_NAME --experiment_name $EXPERIMENT_NAME $ARGS --n_parallel_seeds $N_PARALLEL_SEEDS -gpu
