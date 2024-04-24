# Reference: https://github.com/MushroomRL/mushroom-rl.git

import numpy as np


class Chain:
    def __init__(self, state_n, prob, mu=None) -> None:
        self.p = self._compute_probabilities(
            state_n, prob, goal_states=[0, state_n - 1]
        )
        self.r = self._compute_reward(state_n, goal_states=[0, state_n - 1], rew=1.0)
        self.mu = mu

        # MDP properties
        self.observation_shape = (1,)
        self.n_actions = 2

    def reset(self, state=None):
        if state is None:
            if self.mu is not None:
                self.state = np.array([np.random.choice(self.mu.size, p=self.mu)])
            else:
                self.state = np.array(
                    [
                        np.random.choice(
                            [(self.p.shape[0] - 1) // 2, self.p.shape[0] // 2]
                        )
                    ]
                )
        else:
            self.state = state
        self.n_steps = 0

        return self.state

    def step(self, action):
        self.n_steps += 1
        p = self.p[self.state[0], action, :]
        if np.sum(p) == 0:  # handle the case when agent starts in the goal state
            next_state = self.state
            absorbing = True
            reward = self.r[self.state[0], action, self.state[0]]
        else:
            next_state = np.array([np.random.choice(p.size, p=p)])
            absorbing = not np.any(self.p[next_state[0]])
            reward = self.r[self.state[0], action, next_state[0]]

        self.state = next_state

        return self.state, reward, absorbing

    def _compute_probabilities(self, state_n, prob, goal_states):
        """
        Compute the transition probability matrix.
        0 = right, 1 = left
        """
        p = np.zeros((state_n, 2, state_n))

        for i in range(state_n):
            if i == 0:
                p[i, 1, i] = 1.0
            else:
                p[i, 1, i] = 1.0 - prob
                p[i, 1, i - 1] = prob

            if i == state_n - 1:
                p[i, 0, i] = 1.0
            else:
                p[i, 0, i] = 1.0 - prob
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
