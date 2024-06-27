#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/CarOnHill/out/$EXPERIMENT_NAME/FQI ] || mkdir -p experiments/CarOnHill/out/$EXPERIMENT_NAME/FQI

echo "launch train fqi"

if [[ $GPU = true ]]
then
    submission_train_fqi="sbatch -J C_train_fqi --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=750M --time=2:00:00 --gres=gpu:1 -p stud,stud3080 \
    --output=experiments/CarOnHill/out/$EXPERIMENT_NAME/FQI/train_fqi_%a.out \
    launch_job/car_on_hill/train_fqi.sh -e $EXPERIMENT_NAME $BASE_ARGS $FQI_ARGS -g"
else
    submission_train_fqi="sbatch -J C_train_fqi --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=8 --mem-per-cpu=750M --time=2:00:00 -p stud,stud3080 \
    --output=experiments/CarOnHill/out/$EXPERIMENT_NAME/FQI/train_fqi_%a.out \
    launch_job/car_on_hill/train_fqi.sh -e $EXPERIMENT_NAME $BASE_ARGS $FQI_ARGS"
fi
    
$submission_train_fqi
