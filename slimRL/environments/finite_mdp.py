# Reference: https://github.com/MushroomRL/mushroom-rl.git

import numpy as np
from slimRL.environments.environment import Environment

class FiniteMDP(Environment):
    """
    Finite Markov Decision Process.

    """

    def __init__(self, p, rew, mu=None, horizon=np.inf, dt=1e-1):
        """
        Constructor.

        Args:
            p (np.ndarray): transition probability matrix;
            rew (np.ndarray): reward matrix;
            mu (np.ndarray, None): initial state probability distribution;
            gamma (float, .9): discount factor;
            horizon (int, np.inf): the horizon;
            dt (float, 1e-1): the control timestep of the environment.

        """
        assert p.shape == rew.shape
        assert mu is None or p.shape[0] == mu.size

        # MDP parameters
        self.p = p
        self.r = rew
        self.mu = mu

        # MDP properties
        self.horizon = horizon
        self.observation_shape = (1, )
        self.action_shape = ()
        self.action_dim = 2
        self.single_action_space = list(range(self.action_dim))
        super().__init__(self.observation_shape, self.action_dim)

        self.timer = 0

    def reset(self, state=None):
        if state is None:
            if self.mu is not None:
                self._state = np.array(
                    [np.random.choice(self.mu.size, p=self.mu)])
            else:
                self._state = np.array([np.random.choice(self.p.shape[0])])
        else:
            self._state = state
        self.timer = 0

        return self._state, {}

    def step(self, action):
        action = action[0]
        self.timer += 1
        p = self.p[self._state[0], action, :]
        if np.sum(p) == 0: # handle the case when agent starts in the goal state
            next_state = self._state
            absorbing = True
            reward = self.r[self._state[0], action, self._state[0]]
        else:
            next_state = np.array([np.random.choice(p.size, p=p)])
            absorbing = not np.any(self.p[next_state[0]])
            reward = self.r[self._state[0], action, next_state[0]]

        if self.timer == self.horizon:
            absorbing = True

        self._state = next_state

        return self._state, reward, absorbing, {}
