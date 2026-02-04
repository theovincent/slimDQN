#!/bin/bash
ENV_NAME="mountain_car"

FEATURES="5"
BATCH_SIZE=32
RB=10000
NT=10000
TUP=200

FIRST_SEED=1
LAST_SEED=5
N_PARALLEL_SEEDS=1

ARGS="--architecture_type fc \
    --replay_buffer_capacity $RB \
    --n_training_steps_per_epoch $NT \
    --features $FEATURES $FEATURES \
    --batch_size $BATCH_SIZE \
    --update_horizon 1 \
    --horizon 200 \
    --gamma 0.99 \
    --learning_rate 3e-3 \
    --n_epochs 20 \
    --target_update_period $TUP \
    "
FULL_EXPERIMENT_NAME="DQN_F${FEATURES}_RB${RB}_NT${NT}_T${TUP}_${ENV_NAME}"

####################################################################################


LAUNCHER_SCRIPT="launch_job/${ENV_NAME}/cluster_dqn.sh"

echo "----------------------------------------------------------------"
echo "Launching Job"
echo "Experiment: $EXPERIMENT_TAG"
echo "----------------------------------------------------------------"

bash $LAUNCHER_SCRIPT \
    --experiment_name $FULL_EXPERIMENT_NAME \
    --first_seed $FIRST_SEED \
    --last_seed $LAST_SEED \
    --n_parallel_seeds $N_PARALLEL_SEEDS \
    $ARGS 