import json


def load_parameters(args):
    p = {}
    for arg in vars(args):
        p[arg] = getattr(args, arg)
    return p
