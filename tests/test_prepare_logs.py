import unittest
import numpy as np
import jax

from experiments.base.utils import prepare_logs


class TestPrepareLogs(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

    def test_prepare_logs(self):
        # Create folders and parameters.json with seed = 1 -> should not throw an error
        try:
            prepare_logs("lunar_lander", "dqn", ["-e", "test_prepare_logs", "-s", "1"])
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised when running 'prepare_logs'."

        # Create folders and parameters.json with seed = 2 -> should not throw an error
        try:
            prepare_logs("lunar_lander", "dqn", ["-e", "test_prepare_logs", "-s", "2"])
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised when running 'prepare_logs'."

        # Create again folders and parameters.json with seed = 1 -> should throw an error
        try:
            prepare_logs("lunar_lander", "dqn", ["-e", "test_prepare_logs", "-s", "1"])
        except Exception as e:
            if type(e) != AssertionError:
                assert 0, f"The exception {type(e).__name__} is raised when running 'prepare_logs'."
