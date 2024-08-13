import gymnasium as gym
import numpy as np


class LunarLander:
    def __init__(self, render_mode=None):
        self.env = gym.make("LunarLander-v2", render_mode=render_mode)
        self.observation_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

    def reset(self, key=None):
        if key is None:
            state, _ = self.env.reset()
        else:
            state, _ = self.env.reset(seed=int(key[0]))
        self.state = np.array(state)

        self.n_steps = 0

        return self.state

    def step(self, action):
        self.n_steps += 1
        self.state, reward, absorbing, _, _ = self.env.step(action)

        return self.state, reward, absorbing
