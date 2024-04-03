import gymnasium as gym
import numpy as np


class LunarLander:
    def __init__(self, horizon=1000):
        self.horizon = horizon
        self.env = gym.make("LunarLander-v2", render_mode="rgb_array")

        self._state = None
        self.observation_shape = (8,)
        self.action_shape = ()
        self.action_dim = self.env.action_space.n
        self.single_action_space = list(range(self.action_dim))

        self.timer = 0

    def reset(self, state=None):
        if state is None:
            self._state, _ = self.env.reset()
        else:
            self._state = state
        self._state = np.array(self._state)

        self.timer = 0

        return self._state, {}

    def step(self, action):
        action = action[0]
        self.timer += 1
        self._state, reward, absorbing, _, _ = self.env.step(action)

        infos = {}
        if self.timer == self.horizon:
            infos["episode_end"] = True

        return self._state, reward, absorbing, infos
