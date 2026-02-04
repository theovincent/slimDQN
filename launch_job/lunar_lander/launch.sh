  GNU nano 6.2                                                                                                launch_job/lunar_lander/launch.sh                                                                                                         
#!/bin/bash

#policy expertise
EXPERTISE="ablation"
FEATURES="200 200"
BATCH_SIZE=64

ENV_NAME="lunar_lander"
ALGO="dqn"

EXPERIMENT_TAG="${EXPERTISE}_${ALGO}_${ENV_NAME}"


FIRST_SEED=41
LAST_SEED=50
N_PARALLEL_SEEDS=1


BASE_ARGS="--architecture_type fc \
    --n_epochs 50 \
    --n_training_steps_per_epoch 10000 \
    --replay_buffer_capacity 10000 \
    --update_horizon 1 \
    --gamma 0.99 \
    --learning_rate 1e-3 \
    --target_update_period 100 \
    --horizon 500 \
    --n_initial_samples 1000
    "


POLICY_EXPERTISE_ARGS="--features $FEATURES --batch_size $BATCH_SIZE"


LAUNCHER_SCRIPT="launch_job/${ENV_NAME}/cluster_${ALGO}.sh"
FULL_EXPERIMENT_NAME="${EXPERIMENT_TAG}"

echo "----------------------------------------------------------------"
echo "Launching Job"
echo "Experiment: $FULL_EXPERIMENT_NAME"
echo "Algorithm:  $ALGO"
echo "Policy Expertise Args: $EXPERTISE"
echo "Data Args: $DATA_ARGS"
echo "----------------------------------------------------------------"

#sleep 2000s

bash $LAUNCHER_SCRIPT \
    --experiment_name $FULL_EXPERIMENT_NAME \
    --first_seed $FIRST_SEED \
    --last_seed $LAST_SEED \
    --n_parallel_seeds $N_PARALLEL_SEEDS \
    $BASE_ARGS $POLICY_EXPERTISE_ARGS $DATA_ARGS
