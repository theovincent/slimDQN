import unittest
import os
import json
import shutil
import numpy as np
import jax

from experiments.base.utils import prepare_logs


class TestPrepareLogs(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

    def test_prepare_logs(self):
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../experiments/lunar_lander/exp_output/_test_prepare_logs"
        )
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        # Create folders and parameters.json with seed = 1 -> should not throw an error
        try:
            prepare_logs("lunar_lander", "dqn", ["-e", "_test_prepare_logs", "-s", "1"])
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised when running 'prepare_logs'."

        # Create folders and parameters.json with seed = 2 -> should not throw an error
        try:
            prepare_logs("lunar_lander", "dqn", ["-e", "_test_prepare_logs", "-s", "2"])
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised when running 'prepare_logs'."

        # Create again folders and parameters.json with seed = 1 -> should throw an error
        try:
            prepare_logs("lunar_lander", "dqn", ["-e", "_test_prepare_logs", "-s", "1"])
        except Exception as e:
            if type(e) != AssertionError:
                assert 0, f"The exception {type(e).__name__} is raised when running 'prepare_logs'."

        # Create again folders and parameters.json with different first parameter for dqn -> should throw an error
        parameters = json.load(open(os.path.join(save_path, "parameters.json"), "rb"))
        first_dqn_param = list(parameters["dqn"].keys())[1]
        first_dqn_param_value = parameters["dqn"][first_dqn_param]
        try:
            prepare_logs(
                "lunar_lander",
                "dqn",
                ["-e", "_test_prepare_logs", "-s", "1", f"--{first_dqn_param}", f"{first_dqn_param_value}"],
            )
        except Exception as e:
            if type(e) != AssertionError:
                assert 0, f"The exception {type(e).__name__} is raised when running 'prepare_logs'."

        assert os.path.exists(save_path)
        assert os.path.exists(os.path.join(save_path, "parameters.json"))
        shutil.rmtree(save_path)
