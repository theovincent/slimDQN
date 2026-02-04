import gymnasium as gym
import numpy as np
import jax


class MountainCar:
    def __init__(self, key, render_mode=None):
        self.env = gym.make("MountainCar-v0", render_mode=render_mode)
        self.observation_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.key = key
        self.last_action = 0

    # Called when stored in the replay buffer
    @property
    def observation(self) -> np.ndarray:
        return np.copy(self.state)

    def reset(self):
        self.state, _ = self.env.reset()
        self.n_steps = 0

    def step(self, action):
        # sticky action
        self.key, key = jax.random.split(self.key)
        action_taken = action # self.last_action if jax.random.uniform(key) < 0.25 else action
        self.last_action = action

        self.state, reward, absorbing, _, _ = self.env.step(action_taken)
        self.n_steps += 1

        return reward, absorbing