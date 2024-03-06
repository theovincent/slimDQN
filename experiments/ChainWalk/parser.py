import argparse


def chain_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-size",
        "--chain_size",
        help="Chain size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-pr",
        "--transition_prob",
        help="The probability of success of an action (transition probability)",
        type=float,
        required=True,
    )
