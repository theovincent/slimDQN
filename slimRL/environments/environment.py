# Reference: https://github.com/MushroomRL/mushroom-rl.git

import numpy as np


class Environment:

    def __init__(self,
                 state_shape,
                 action_dim):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.single_action_space = list(range(self.action_dim))

    def reset(self, state=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, record=False):
        raise NotImplementedError

    def stop(self):
        pass
