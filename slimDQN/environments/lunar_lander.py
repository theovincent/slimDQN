import gymnasium as gym
import numpy as np


class LunarLander:
    def __init__(self, render_mode=None):
        self.env = gym.make("LunarLander-v2", render_mode=render_mode)
        self.observation_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

    # Called when stored in the replay buffer
    @property
    def observation(self) -> np.ndarray:
        return np.copy(self.state)

    def reset(self):
        self.state, _ = self.env.reset()
        self.n_steps = 0

    def step(self, action):
        self.state, reward, absorbing, _, _ = self.env.step(action)
        self.n_steps += 1

        return reward, absorbing
