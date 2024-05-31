import sys
import subprocess
import random
import concurrent.futures
import argparse


def run_command(command, seed):
    command = command.split() + ["-s", str(seed)]
    result = subprocess.run(command, capture_output=True, text=True)
    return f"Executed: {command}, Return code: {result.returncode}, Output: {result.stdout}, Error: {result.stderr}"


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Find optimal values for CarOnHill.")
    parser.add_argument(
        "-ns",
        "--n_seeds",
        help="Number of seeds.",
        type=int,
        default=20,
    )

    parser.add_argument(
        "-cmd",
        "--command",
        help="Command to run with different seed values.",
        type=str,
        required=True,
    )
    args = parser.parse_args(argvs)
    p = vars(args)

    seed_values = random.sample(range(1000), p["n_seeds"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(run_command, *(p["command"], seed)) for seed in seed_values
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    print("All commands executed.")
