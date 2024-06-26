# slimRL

![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to **slimRL** - a gateway to mastering Deep Q-Network (DQN) and Fitted Q Iteration (FQI) algorithms in Reinforcement Learning!🎉 Whether you're a researcher, student, or just curious about RL, slimRL provides a clear, concise, and customizable path to understanding and implementing these powerful algorithms. The simplicity of the implementation allows you to tailor the experimental setup to your requirements. 

### 🚀 Key advantages 
✅ Learn the essentials without the clutter 🧹\
✅ Easily modifiable to implement new research ideas in Online and Offline RL 💬\
✅ Allows quick tailoring for conference reviews and rebuttals ✂️\
✅ Smooth transfer from theory to practice for RL learners ➡️\
✅ Easy to use with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and custom environments 🏋️‍♂️\



Let's dive in!

## User installation
In the folder where the code is, create a Python virtual environment, activate it, update pip and install the package and its dependencies in editable mode:
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

If you are using GPU, update jax with CUDA dependencies:
```bash
pip install -U "jax[cuda12]"
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