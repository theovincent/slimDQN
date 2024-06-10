# slimRL
Simple and concise implementation of DQN on toy environments.

## User installation
In the folder where the code is, create a Python virtual environment, activate it, update pip and install the package and its dependencies in editable mode:
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```

## Run the experiments
All the experiments can be ran the same way by simply replacing the name of the environment, here is an example for CarOnHill.

The following command line runs the training and the evaluation of all the algorithms, one after the other:
```Bash
launch_job/car_on_hill/launch_local.sh -e "test" -rb 50000 -bi 30 -hl 40 20 -frs 10 -lrs 29 -gamma 0.95

```

Once all the trainings are done, you can generate the figures shown in the paper by running the jupyter notebook file located at *experiments/CarOnHill/plots.ipynb*. In the first cell of the notebook, please make sure to change the *experiment_name* as needed. 

## Run the tests
Run all tests with
```Bash
pytest
```