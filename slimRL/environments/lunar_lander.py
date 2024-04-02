from typing import Tuple
from functools import partial
from gymnasium.wrappers.monitoring import video_recorder
import gymnasium as gym
import numpy as np


class LunarLanderEnv:
    def __init__(self, gamma: float):
        self.n_actions = 4
        self.gamma = gamma

        self.env = gym.make("LunarLander-v2", render_mode="rgb_array")

    def reset(self, action):
        self.state, _ = self.env.reset(seed=int(action[0]))
        self.n_steps = 0

        return np.array(self.state)

    def step(self, action):
        self.state, reward, absorbing, _, info = self.env.step(int(action[0]))
        self.n_steps += 1

        return
