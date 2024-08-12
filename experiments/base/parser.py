import os
import time
import argparse
from experiments.base import DISPLAY_NAME


def base_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-e",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-s",
        "--seed",
        help="Seed of the experiment.",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-fs",
        "--features",
        nargs="*",
        help="List of features for the Q-networks.",
        type=int,
        default=[50, 50],
    )

    parser.add_argument(
        "-rbc",
        "--replay_buffer_capacity",
        help="For DQN: Replay Buffer capacity, For FQI: Dataset size to sample.",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Batch size for training.",
        type=int,
        default=32,
    )

    parser.add_argument(
        "-n",
        "--update_horizon",
        help="Value of n in n-step TD update.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "-gamma",
        "--gamma",
        help="Discounting factor.",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate.",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "-h",
        "--horizon",
        help="Horizon for truncation.",
        type=int,
        default=1000,
    )


def fqi_parser(env_name, argvs):
    algo_name = "fqi"
    print(f"--- Train {DISPLAY_NAME[algo_name]} on {DISPLAY_NAME[env_name]} {time.strftime('%d-%m-%Y %H:%M:%S')}---")
    parser = argparse.ArgumentParser(f"Train {DISPLAY_NAME[algo_name]} on {DISPLAY_NAME[env_name]}.")

    base_parser(parser)
    parser.add_argument(
        "-nbi",
        "--n_bellman_iterations",
        help="Number of Bellman iterations to perform.",
        type=int,
        default=30,
    )

    parser.add_argument(
        "-nfs",
        "--n_fitting_steps",
        help="Number of gradient update steps per Bellman iteration.",
        type=int,
        default=5,
    )
    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = env_name
    p["algo"] = algo_name
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../{env_name}/exp_output/{p['experiment_name']}/{p['algo']}/seed_{p['seed']}",
    )

    return p


def dqn_parser(env_name, argvs):
    algo_name = "dqn"
    print(f"--- Train {DISPLAY_NAME[algo_name]} on {DISPLAY_NAME[env_name]} {time.strftime('%d-%m-%Y %H:%M:%S')}---")
    parser = argparse.ArgumentParser(f"Train {DISPLAY_NAME[algo_name]} on {DISPLAY_NAME[env_name]}.")

    base_parser(parser)
    parser.add_argument(
        "-ne",
        "--n_epochs",
        help="Number of epochs to perform.",
        type=int,
        default=80,
    )

    parser.add_argument(
        "-ntspe",
        "--n_training_steps_per_epoch",
        help="Number of training steps per epoch.",
        type=int,
        default=6000,
    )

    parser.add_argument(
        "-utd",
        "--update_to_data",
        help="Number of data points to collect per online Q-network update.",
        type=float,
        default=1,
    )

    parser.add_argument(
        "-tuf",
        "--target_update_frequency",
        help="Number of training steps before updating the target Q-network.",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--nis",
        "---n_initial_samples",
        help="Number of initial samples before the training starts.",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "-ee",
        "--epsilon_end",
        help="Ending value for the linear decaying epsilon used for exploration.",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--ed",
        "--epsilon_duration",
        help="Duration of epsilon's linear decay used for exploration.",
        type=float,
        default=1000,
    )

    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = env_name
    p["algo"] = algo_name
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../{env_name}/exp_output/{p['experiment_name']}/{p['algo']}",
    )

    return p
