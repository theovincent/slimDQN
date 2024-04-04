# Reference: https://github.com/MushroomRL/mushroom-rl.git

import numpy as np


class Chain:
    def __init__(self, state_n, prob, mu=None, horizon=np.inf) -> None:
        self.p = self._compute_probabilities(
            state_n, prob, goal_states=[0, state_n - 1]
        )
        self.r = self._compute_reward(state_n, goal_states=[0, state_n - 1], rew=1.0)
        self.mu = mu

        # MDP properties
        self.horizon = horizon
        self.observation_shape = (1,)
        self.action_shape = ()
        self.action_dim = 2
        self.single_action_space = list(range(self.action_dim))

        self.timer = 0

    def reset(self, state=None):
        if state is None:
            if self.mu is not None:
                self._state = np.array([np.random.choice(self.mu.size, p=self.mu)])
            else:
                self._state = np.array(
                    [
                        np.random.choice(
                            [(self.p.shape[0] - 1) // 2, self.p.shape[0] // 2]
                        )
                    ]
                )
        else:
            self._state = state
        self.timer = 0

        return self._state, {}

    def step(self, action):
        action = action[0]
        self.timer += 1
        p = self.p[self._state[0], action, :]
        if np.sum(p) == 0:  # handle the case when agent starts in the goal state
            next_state = self._state
            absorbing = True
            reward = self.r[self._state[0], action, self._state[0]]
        else:
            next_state = np.array([np.random.choice(p.size, p=p)])
            absorbing = not np.any(self.p[next_state[0]])
            reward = self.r[self._state[0], action, next_state[0]]

        infos = {}
        if self.timer == self.horizon:
            infos["episode_end"] = True

        self._state = next_state

        return self._state, reward, absorbing, infos

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
