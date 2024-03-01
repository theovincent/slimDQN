import argparse


def addparse(parser: argparse.ArgumentParser) -> None:
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
        "-log",
        "--log",
        help="Log the parameters, model and returns.",
        type=bool,
        required=False,
        default=True,
    )

    parser.add_argument(
        "-gpu",
        "--use_gpu",
        help="Specify if GPU is to be used.",
        type=bool,
        required=False,
        default=False,
    )

    parser.add_argument(
        "-rb",
        "--replay_capacity",
        help="Size of replay buffer to use.",
        type=int,
        required=False,
        default=50000,
    )

    parser.add_argument(
        "-B",
        "--batch_size",
        help="Batch size for training.",
        type=int,
        required=False,
        default=32,
    )

    parser.add_argument(
        "-n",
        "--update_horizon",
        help="Value of n in n-step TD update.",
        type=int,
        required=False,
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
        "-tau",
        "--tau",
        help="Weight for target network update.",
        type=float,
        required=False,
        default=1.0,
    )

    parser.add_argument(
        "-lr",
        "--lr",
        help="Learning rate for optimizer.",
        type=float,
        required=False,
        default=2e-4,
    )

    parser.add_argument(
        "-loss",
        "--loss_type",
        help="Type of loss fort TD error calculation.",
        type=str,
        required=False,
        default="huber",
    )

    parser.add_argument(
        "-onl",
        "--n_training_steps_per_online_update",
        help="Training frequency for online Q-network.",
        type=int,
        required=False,
        default=4,
    )

    parser.add_argument(
        "-tgt",
        "--n_training_steps_per_target_update",
        help="Update frequency for target Q-network.",
        type=int,
        required=False,
        default=40,
    )

    parser.add_argument(
        "-E",
        "--n_epochs",
        help="No. of epochs to train for.",
        type=int,
        required=False,
        default=10000,
    )

    parser.add_argument(
        "-spe",
        "--n_training_steps_per_epoch",
        help="Max. no. of training steps per epoch.",
        type=int,
        required=False,
        default=5000,
    )

    parser.add_argument(
        "-init",
        "--n_initial_samples",
        help="No. of initial samples before training begins.",
        type=int,
        required=False,
        default=2000,
    )

    parser.add_argument(
        "-eps_s",
        "--start_epsilon",
        help="Starting value of epsilon for linear schedule.",
        type=float,
        required=False,
        default=1.0,
    )

    parser.add_argument(
        "-eps_e",
        "--end_epsilon",
        help="Ending value of epsilon for linear schedule.",
        type=float,
        required=False,
        default=0.01,
    )

    parser.add_argument(
        "-eps_dur",
        "--duration_epsilon",
        help="Ending value of epsilon for linear schedule.",
        type=float,
        required=False,
        default=25000,
    )

    parser.add_argument(
        "-H",
        "--horizon",
        help="Horizon for truncation.",
        type=int,
        required=False,
        default=200,
    )
