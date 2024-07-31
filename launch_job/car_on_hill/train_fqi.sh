#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@ -frs dummy -lrs dummy

if [[ $GPU = true ]]
then
    source env_gpu/bin/activate
else
    source env_cpu/bin/activate
fi

$ENV_NAME\_$ALGO_NAME -e $EXPERIMENT_NAME -s $SLURM_ARRAY_TASK_ID $ARGS