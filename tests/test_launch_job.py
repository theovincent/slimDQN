import subprocess


def test_launch_local():
    returncode = subprocess.run(["launch_job/lunar_lander/local_dqn.sh"]).returncode
    assert returncode > 0, "The command should have raised an error telling that the experiment name is not specified."

    returncode = subprocess.run(
        ["launch_job/lunar_lander/local_dqn.sh", "--experiment_name", "_test_launch_local"]
    ).returncode
    assert returncode > 0, "The command should have raised an error telling that the first seed is not specified."

    returncode = subprocess.run(
        [
            "launch_job/lunar_lander/local_dqn.sh",
            "--experiment_name",
            "_test_launch_local",
            "--first_seed",
            "1",
        ]
    ).returncode
    assert returncode > 0, "The command should have raised an error telling that the last seed is not specified."

    returncode = subprocess.run(
        [
            "launch_job/lunar_lander/local_dqn.sh",
            "--experiment_name",
            "_test_launch_local",
            "--first_seed",
            "10",
            "--last_seed",
            "1",
        ]
    ).returncode
    assert (
        returncode > 0
    ), "The command should have raised an error telling that the last seed should be greater than the first seed."
