import os
import time
import argparse
from slimRL.environments import DISPLAY_NAME


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
        help="Seed for the experiment.",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-hl",
        "--hidden_layers",
        nargs="*",
        help="Hidden layer sizes.",
        type=int,
        default=[50, 50],
    )

    parser.add_argument(
        "-rb",
        "--replay_capacity",
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
        help="Discounting factor gamma.",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "-lr",
        "--lr",
        help="Learning rate for Adam optimizer.",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "-hor",
        "--horizon",
        help="Horizon for truncation.",
        type=int,
        default=1000,
    )


def fqi_parser(env_name, argvs):
    algo = "FQI"
    print(f"---{DISPLAY_NAME[env_name]}__{algo}__{time.strftime('%d-%m-%Y %H:%M:%S')}---")
    parser = argparse.ArgumentParser(f"Train {algo} on {DISPLAY_NAME[env_name]}.")

    base_parser(parser)
    parser.add_argument(
        "-nbi",
        "--n_bellman_iterations",
        help="No. of Bellman iterations to perform.",
        type=int,
        default=30,
    )

    parser.add_argument(
        "-fs",
        "--n_fitting_steps",
        help="No. of gradient update steps to perform per Bellman iteration.",
        type=int,
        default=5,
    )
    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = DISPLAY_NAME[env_name]
    p["algo"] = algo
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../{env_name}/exp_output/{p['experiment_name']}/{p['algo']}/seed_{p['seed']}",
    )

    return p


def dqn_parser(env_name, argvs):
    algo = "DQN"
    print(f"---{DISPLAY_NAME[env_name]}__{algo}__{time.strftime('%d-%m-%Y %H:%M:%S')}---")
    parser = argparse.ArgumentParser(f"Train {algo} on {DISPLAY_NAME[env_name]}.")

    base_parser(parser)
    parser.add_argument(
        "-ne",
        "--n_epochs",
        help="No. of epochs to train the DQN for.",
        type=int,
        default=80,
    )

    parser.add_argument(
        "-spe",
        "--n_training_steps_per_epoch",
        help="Max. no. of training steps per epoch.",
        type=int,
        default=6000,
    )

    parser.add_argument(
        "-utd",
        "--update_to_data",
        help="No. of data points to collect per online Q-network update.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "-tuf",
        "--target_update_frequency",
        help="Update period for target Q-network.",
        type=int,
        default=200,
    )

    parser.add_argument(
        "-n_init",
        "--n_initial_samples",
        help="No. of initial samples before training begins.",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "-eps_e",
        "--end_epsilon",
        help="Ending value of epsilon for linear schedule.",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "-eps_dur",
        "--duration_epsilon",
        help="Duration(number of steps) over which epsilon decays.",
        type=float,
        default=1000,
    )

    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = DISPLAY_NAME[env_name]
    p["algo"] = algo
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../{env_name}/exp_output/{p['experiment_name']}/{p['algo']}/seed_{p['seed']}",
    )

    return p
