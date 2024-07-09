#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/car_on_hill/logs/$EXPERIMENT_NAME/FQI ] || mkdir -p experiments/car_on_hill/logs/$EXPERIMENT_NAME/FQI

echo "launch train fqi"

if [[ $GPU = true ]]
then
    submission_train_fqi="sbatch -J $EXPERIMENT_NAME\_fqi --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=750M --time=00:10:00 --gres=gpu:1 -p stud,stud3080 \
    --output=experiments/car_on_hill/logs/$EXPERIMENT_NAME/FQI/train_fqi_%a.out \
    launch_job/car_on_hill/train_fqi.sh -e $EXPERIMENT_NAME $BASE_ARGS $FQI_ARGS -g"
else
    submission_train_fqi="sbatch -J $EXPERIMENT_NAME\_fqi --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=750M --time=00:15:00 -p stud,stud3080 \
    --output=experiments/car_on_hill/logs/$EXPERIMENT_NAME/FQI/train_fqi_%a.out \
    launch_job/car_on_hill/train_fqi.sh -e $EXPERIMENT_NAME $BASE_ARGS $FQI_ARGS"
fi
    
$submission_train_fqi
