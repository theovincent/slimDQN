# Reference: https://github.com/MushroomRL/mushroom-rl.git

import numpy as np
from scipy.integrate import odeint

from slimRL.environments.viewer import Viewer
    
class CarOnHill():
    """
    The Car On Hill environment as presented in:
    "Tree-Based Batch Mode Reinforcement Learning". Ernst D. et al.. 2005.

    """

    def __init__(self, horizon=200):
        """
        Constructor.

        Args:
            horizon (int, 100): horizon of the problem;

        """
        # MDP parameters
        self.max_pos = 1.
        self.max_velocity = 3.
        self._g = 9.81
        self._m = 1.
        self._discrete_actions = [-4., 4.]

        # MDP properties
        self._dt = .1
        self.horizon = horizon

        # Visualization
        self._viewer = Viewer(1, 1)
        self._state = None
        self.observation_shape = (2, )
        self.action_shape = ()
        self.action_dim = 2
        self.single_action_space = list(range(self.action_dim))

        self.timer = 0

    def reset(self, state=None):
        if state is None:
            self._state = np.array([-0.5, 0])
        else:
            self._state = state

        self.timer = 0

        return self._state, {}

    def step(self, action):
        action = action[0]
        self.timer += 1
        action = self._discrete_actions[action]
        sa = np.append(self._state, action)
        new_state = odeint(self._dpds, sa, [0, self._dt])

        self._state = new_state[-1, :-1]

        if self._state[0] < -self.max_pos or np.abs(self._state[1]) > self.max_velocity:
            reward = -1.
            absorbing = True
        elif self._state[0] > self.max_pos and np.abs(self._state[1]) <= self.max_velocity:
            reward = 1.
            absorbing = True
        else:
            reward = 0.
            absorbing = False

        infos = {}
        if self.timer == self.horizon:
            infos["episode_end"] = True

        return self._state, reward, absorbing, infos

    @staticmethod
    def _angle(x):
        if x < 0.5:
            m = 4 * x - 1
        else:
            m = 1 / ((20 * x ** 2 - 20 * x + 6) ** 1.5)

        return np.arctan(m)

    @staticmethod
    def _height(x):
        y_neg = 4 * x ** 2 - 2 * x
        y_pos = (2 * x - 1) / np.sqrt(5 * (2 * x - 1) ** 2 + 1)
        y = np.zeros_like(x)

        mask = x < .5
        neg_mask = np.logical_not(mask)
        y[mask] = y_neg[mask]
        y[neg_mask] = y_pos[neg_mask]

        y_norm = (y + 1) / 2

        return y_norm

    def _dpds(self, state_action, t):
        position = state_action[0]
        velocity = state_action[1]
        u = state_action[-1]

        if position < 0.:
            diff_hill = 2 * position + 1
            diff_2_hill = 2
        else:
            diff_hill = 1 / ((1 + 5 * position ** 2) ** 1.5)
            diff_2_hill = (-15 * position) / ((1 + 5 * position ** 2) ** 2.5)

        dp = velocity
        ds = (u - self._g * self._m * diff_hill - velocity ** 2 * self._m *
              diff_hill * diff_2_hill) / (self._m * (1 + diff_hill ** 2))

        return dp, ds, 0.


class CarOnHillDQN(CarOnHill):
    """Add functions necessary for running DQN on CarOnHill Environment"""

    def __init__(self, horizon=200):
        super().__init__(horizon)

    def render(self, record=False):
        # Slope
        self._viewer.function(0, 1, self._height)

        # Car
        car_body = [
            [-3e-2, 0],
            [-3e-2, 2e-2],
            [-2e-2, 2e-2],
            [-1e-2, 3e-2],
            [1e-2, 3e-2],
            [2e-2, 2e-2],
            [3e-2, 2e-2],
            [3e-2, 0]
        ]

        x_car = (self._state[0] + 1) / 2
        y_car = self._height(x_car)
        c_car = [x_car, y_car]
        angle = self._angle(x_car)
        self._viewer.polygon(c_car, angle, car_body, color=(32, 193, 54))

        frame = self._viewer.get_frame() if record else None

        self._viewer.display(self._dt)

        return frame