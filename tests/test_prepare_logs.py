import json
import os
import shutil

from experiments.base.utils import prepare_logs


def test_prepare_logs():
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../experiments/lunar_lander/exp_output/_test_prepare_logs"
    )
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    # Create folders and parameters.json with seed = 1 -> should not throw an error
    try:
        prepare_logs(
            "lunar_lander", "dqn", ["--experiment_name", "_test_prepare_logs", "--seed", "1", "--disable_wandb"]
        )
    except Exception as e:
        assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    # Fake that the returns for seed 1 are stored.
    os.mkdir(os.path.join(save_path, "dqn/episode_returns_and_lengths"))
    json.dump({}, open(os.path.join(save_path, "dqn/episode_returns_and_lengths/1.json"), "w"))

    # Create folders and parameters.json with seed = 2 -> should not throw an error
    try:
        prepare_logs(
            "lunar_lander", "dqn", ["--experiment_name", "_test_prepare_logs", "--seed", "2", "--disable_wandb"]
        )
    except Exception as e:
        assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    # Create again folders and parameters.json with seed = 1 -> should throw an error
    try:
        prepare_logs(
            "lunar_lander", "dqn", ["--experiment_name", "_test_prepare_logs", "--seed", "1", "--disable_wandb"]
        )
        assert 0, "An error saying that this experiment has been run with the same seed should have been thrown."
    except Exception as e:
        if type(e) != AssertionError:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    # Create again folders and parameters.json with different first parameter for dqn -> should throw an error
    parameters = json.load(open(os.path.join(save_path, "parameters.json"), "rb"))
    first_dqn_param = list(parameters["dqn"].keys())[1]
    first_dqn_param_value = parameters["dqn"][first_dqn_param] + 1
    try:
        prepare_logs(
            "lunar_lander",
            "dqn",
            [
                "--experiment_name",
                "_test_prepare_logs",
                "--seed",
                "3",
                f"--{first_dqn_param}",
                f"{first_dqn_param_value}",
                "--disable_wandb",
            ],
        )
        assert (
            0
        ), "An error saying that this experiment has been run with a different parameter should have been thrown."
    except Exception as e:
        if type(e) != AssertionError:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    assert os.path.exists(save_path)
    assert os.path.exists(os.path.join(save_path, "parameters.json"))
    shutil.rmtree(save_path)
