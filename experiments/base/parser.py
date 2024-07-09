import argparse


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
        required=False,
        default=[100],
    )

    parser.add_argument(
        "-rb",
        "--replay_capacity",
        help="For DQN: Replay Buffer capacity, For FQI: Dataset size to sample.",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "-B",
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
        required=False,
        default=0.99,
    )

    parser.add_argument(
        "-lr",
        "--lr",
        help="Starting learning rate for Adam optimizer.",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "-H",
        "--horizon",
        help="Horizon for truncation.",
        type=int,
        default=200,
    )


def fqi_parser(parser: argparse.ArgumentParser):
    base_parser(parser=parser)
    parser.add_argument(
        "-bi",
        "--n_bellman_iterations",
        help="No. of Bellman iterations to perform.",
        type=int,
        required=False,
        default=30,
    )

    parser.add_argument(
        "-fs",
        "--n_fitting_steps",
        help="No. of gradient update steps to perform per Bellman iteration.",
        type=int,
        default=5,
    )


def dqn_parser(parser: argparse.ArgumentParser):
    base_parser(parser=parser)
    parser.add_argument(
        "-utd",
        "--update_to_data",
        help="No. of data points to collect per online Q-network update.",
        type=int,
        default=4,
    )

    parser.add_argument(
        "-T",
        "--target_update_period",
        help="Update period for target Q-network.",
        type=int,
        default=40,
    )

    parser.add_argument(
        "-n_init",
        "--n_initial_samples",
        help="No. of initial samples before training begins.",
        type=int,
        default=2000,
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
        default=25000,
    )

    parser.add_argument(
        "-E",
        "--n_epochs",
        help="No. of epochs to train the DQN for.",
        type=int,
        default=100,
    )

    parser.add_argument(
        "-spe",
        "--n_training_steps_per_epoch",
        help="Max. no. of training steps per epoch.",
        type=int,
        default=2000,
    )
