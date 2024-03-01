# Reference: https://github.com/MushroomRL/mushroom-rl.git

import numpy as np
from slimRL.environments.finite_mdp import FiniteMDP

class Chain(FiniteMDP):

    def __init__(self, state_n, prob, mu=None, horizon=None) -> None:
        self.p = self._compute_probabilities(state_n, prob, goal_states=[0, -1])
        self.r = self._compute_reward(state_n, goal_states=[0, -1],rew=1.0)

        super().__init__(self.p, self.r, mu, horizon)

    def _compute_probabilities(self, state_n, prob, goal_states):
        """
        Compute the transition probability matrix.
        0 = right, 1 = left
        """
        p = np.zeros((state_n, 2, state_n))

        for i in range(state_n):
            if i == 0:
                p[i, 1, i] = 1.
            else:
                p[i, 1, i] = 1. - prob
                p[i, 1, i - 1] = prob

            if i == state_n - 1:
                p[i, 0, i] = 1.
            else:
                p[i, 0, i] = 1. - prob
                p[i, 0, i + 1] = prob

        for g in goal_states:
            p[g, :, :] = 0

        return p


    def _compute_reward(self, state_n, goal_states, rew):
        r = np.zeros((state_n, 2, state_n))

        for g in goal_states:
            r[g, :, g] = rew
            if g != 0:
                r[g - 1, 0, g] = rew

            if g != state_n - 1:
                r[g + 1, 1, g] = rew

        return r


class ChainDQN(Chain):
    def __init__(self, state_n, prob, mu=None, horizon=None) -> None:
        super().__init__(state_n, prob, mu, horizon)