# This file was inspired by https://github.com/MushroomRL/mushroom-rl

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from scipy.integrate import odeint


from rl.utils.viewer import Viewer
from pbo.networks.base_q import BaseQ, BaseMultiHeadQ


class CarOnHillEnv:
    """
    The Car On Hill selfironment as presented in:
    "Tree-Based Batch Mode Reinforcement Learning". Ernst D. et al.. 2005.
    """

    def __init__(self, gamma: float) -> None:
        self.gamma = gamma
        self.actions_on_max = jnp.array([[-1], [1]])
        self.max_position = 1.0
        self.max_velocity = 3.0
        self._g = 9.81
        self._m = 1.0
        self._dt = 0.1

        # Visualization
        self._viewer = Viewer(1, 1)

    def reset(self, state: jnp.ndarray = None) -> jnp.ndarray:
        if state is None:
            self.state = jnp.array([-0.5, 0])
        else:
            self.state = state

        return self.state

    @partial(jax.jit, static_argnames="self")
    def boundery_conditions(self, new_state_odeint: jnp.ndarray) -> tuple:
        state = jnp.array(new_state_odeint[-1, :-1])

        too_fast = (jnp.abs(state[1]) > self.max_velocity).astype(float)
        too_far_left = (state[0] < -self.max_position).astype(float)
        too_far_right = (state[0] > self.max_position).astype(float)

        too_far_left_or_too_fast = too_far_left + too_fast - too_far_left * too_fast
        too_far_right_and_not_too_fast = too_far_right * (1 - too_fast)

        reward, absorbing = too_far_left_or_too_fast * jnp.array(
            [-1.0, 1]
        ) + too_far_right_and_not_too_fast * jnp.array([1.0, 1])

        return state, jnp.array([reward]), jnp.array([absorbing], dtype=bool)

    @partial(jax.jit, static_argnames="self")
    def state_action(self, state, action):
        return jnp.append(state, 4 * action[0])

    def step(self, action: jnp.ndarray) -> tuple:
        new_state = odeint(self._dpds, self.state_action(self.state, action), [0, self._dt])

        self.state, reward, absorbing = self.boundery_conditions(new_state)

        return self.state, reward, absorbing, {}

    def render(self, action: jnp.ndarray) -> None:
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
            [3e-2, 0],
        ]

        x_car = (np.array(self.state)[0] + 1) / 2
        y_car = self._height(x_car)
        c_car = [x_car, y_car]
        angle = self._angle(x_car)
        self._viewer.polygon(c_car, angle, car_body, color=(32, 193, 54))

        # Action
        self._viewer.force_arrow(c_car, np.array([action[0], 0]), 1, 20, 1, width=3)

        self._viewer.display(self._dt)

    @staticmethod
    def _angle(x):
        if x < 0.5:
            m = 4 * x - 1
        else:
            m = 1 / ((20 * x**2 - 20 * x + 6) ** 1.5)

        return np.arctan(m)

    @staticmethod
    def _height(x):
        y_neg = 4 * x**2 - 2 * x
        y_pos = (2 * x - 1) / np.sqrt(5 * (2 * x - 1) ** 2 + 1)
        y = np.zeros_like(x)

        mask = x < 0.5
        neg_mask = np.logical_not(mask)
        y[mask] = y_neg[mask]
        y[neg_mask] = y_pos[neg_mask]

        y_norm = (y + 1) / 2

        return y_norm

    def _dpds(self, state_action, t):
        position = state_action[0]
        velocity = state_action[1]
        u = state_action[-1]

        if position < 0.0:
            diff_hill = 2 * position + 1
            diff_2_hill = 2
        else:
            diff_hill = 1 / ((1 + 5 * position**2) ** 1.5)
            diff_2_hill = (-15 * position) / ((1 + 5 * position**2) ** 2.5)

        dp = velocity
        ds = (u - self._g * self._m * diff_hill - velocity**2 * self._m * diff_hill * diff_2_hill) / (
            self._m * (1 + diff_hill**2)
        )

        return dp, ds, 0.0

    def close(self):
        return self._viewer.close()

    def optimal_steps_to_absorbing(self, state: jnp.ndarray, max_steps: int) -> tuple:
        current_states = [state]
        step = 0

        while len(current_states) > 0 and step < max_steps:
            next_states = []
            for state_ in current_states:
                for idx_action in range(2):
                    self.state = state_
                    next_state, reward, _, _ = self.step(self.actions_on_max[idx_action])

                    if reward == 1:
                        return True, step + 1
                    elif reward == 0:
                        next_states.append(next_state)
                    ## if reward == -1 we pass

            step += 1
            current_states = next_states

        return False, step

    def optimal_v(self, state: jnp.ndarray, max_steps: int) -> float:
        success, step_to_absorbing = self.optimal_steps_to_absorbing(state, max_steps)

        if step_to_absorbing == 0:
            return 0
        else:
            return self.gamma ** (step_to_absorbing - 1) if success else -self.gamma ** (step_to_absorbing - 1)

    @partial(jax.jit, static_argnames=("self", "q"))
    def diff_q_estimate_mesh(
        self, q: BaseQ, q_params: hk.Params, states_x: jnp.ndarray, states_v: jnp.ndarray
    ) -> jnp.ndarray:
        q_mesh_ = self.q_estimate_mesh(q, q_params, states_x, states_v)

        return q_mesh_[:, :, 1] - q_mesh_[:, :, 0]

    @partial(jax.jit, static_argnames=("self", "q"))
    def q_estimate_mesh(
        self, q: BaseQ, q_params: hk.Params, states_x: jnp.ndarray, states_v: jnp.ndarray
    ) -> jnp.ndarray:
        n_boxes = states_x.shape[0] * states_v.shape[0]
        states_x_mesh, states_v_mesh = jnp.meshgrid(states_x, states_v, indexing="ij")

        states = jnp.hstack((states_x_mesh.reshape((n_boxes, 1)), states_v_mesh.reshape((n_boxes, 1))))

        idx_states_mesh, idx_actions_mesh = jnp.meshgrid(
            jnp.arange(states.shape[0]), jnp.arange(self.actions_on_max.shape[0]), indexing="ij"
        )
        states_ = states[idx_states_mesh.flatten()]
        actions_ = self.actions_on_max[idx_actions_mesh.flatten()]

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        return q(q_params, states_, actions_).reshape(
            (states_x.shape[0], states_v.shape[0], self.actions_on_max.shape[0])
        )

    @partial(jax.jit, static_argnames=("self", "q"))
    def q_multi_head_estimate_mesh(
        self, q: BaseMultiHeadQ, q_params: hk.Params, states_x: jnp.ndarray, states_v: jnp.ndarray
    ) -> jnp.ndarray:
        n_boxes = states_x.shape[0] * states_v.shape[0]
        states_x_mesh, states_v_mesh = jnp.meshgrid(states_x, states_v, indexing="ij")

        states = jnp.hstack((states_x_mesh.reshape((n_boxes, 1)), states_v_mesh.reshape((n_boxes, 1))))

        idx_states_mesh, idx_actions_mesh = jnp.meshgrid(
            jnp.arange(states.shape[0]), jnp.arange(self.actions_on_max.shape[0]), indexing="ij"
        )
        states_ = states[idx_states_mesh.flatten()]
        actions_ = self.actions_on_max[idx_actions_mesh.flatten()]

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        return q(q_params, states_, actions_).reshape(
            (states_x.shape[0], states_v.shape[0], self.actions_on_max.shape[0], q.n_heads)
        )

    def simulate(self, q: BaseQ, horizon: int, initial_state: jnp.ndarray) -> bool:
        self.reset(initial_state)
        absorbing = False
        step = 0

        while not absorbing and step < horizon:
            if q(q.params, self.state, self.actions_on_max[1]) > q(q.params, self.state, self.actions_on_max[0]):
                action = self.actions_on_max[1]
            else:
                action = self.actions_on_max[0]
            _, reward, absorbing, _ = self.step(action)

            step += 1
            self.render(action)

        self.close()

        return reward == 1

    def evaluate(self, q: BaseQ, q_params: hk.Params, horizon: int, initial_state: jnp.ndarray) -> float:
        performance = 0
        discount = 1
        self.reset(initial_state)
        absorbing = False
        step = 0

        while not absorbing and step < horizon:
            if q(q_params, self.state, self.actions_on_max[1]) > q(q_params, self.state, self.actions_on_max[0]):
                action = self.actions_on_max[1]
            else:
                action = self.actions_on_max[0]
            _, reward, absorbing, _ = self.step(action)

            performance += discount * reward[0]
            discount *= self.gamma
            step += 1

        return performance

    def v_mesh(
        self, q: BaseQ, q_params: hk.Params, horizon: int, states_x: jnp.ndarray, states_v: jnp.ndarray
    ) -> np.ndarray:
        v_mesh_ = np.zeros((len(states_x), len(states_v)))

        for idx_state_x, state_x in enumerate(states_x):
            for idx_state_v, state_v in enumerate(states_v):
                v_mesh_[idx_state_x, idx_state_v] = self.evaluate(q, q_params, horizon, jnp.array([state_x, state_v]))

        return v_mesh_

    def multi_head_evaluate(
        self, head_idx: int, q: BaseQ, q_params: hk.Params, horizon: int, initial_state: jnp.ndarray
    ) -> float:
        performance = 0
        discount = 1
        self.reset(initial_state)
        absorbing = False
        step = 0

        while not absorbing and step < horizon:
            if (
                q(q_params, self.state, self.actions_on_max[1])[0, head_idx]
                > q(q_params, self.state, self.actions_on_max[0])[0, head_idx]
            ):
                action = self.actions_on_max[1]
            else:
                action = self.actions_on_max[0]
            _, reward, absorbing, _ = self.step(action)

            performance += discount * reward[0]
            discount *= self.gamma
            step += 1

        return performance

    def multi_head_v_mesh(
        self, head_idx: int, q: BaseQ, q_params: hk.Params, horizon: int, states_x: jnp.ndarray, states_v: jnp.ndarray
    ) -> np.ndarray:
        v_mesh_ = np.zeros((len(states_x), len(states_v)))

        for idx_state_x, state_x in enumerate(states_x):
            for idx_state_v, state_v in enumerate(states_v):
                v_mesh_[idx_state_x, idx_state_v] = self.multi_head_evaluate(
                    head_idx, q, q_params, horizon, jnp.array([state_x, state_v])
                )

        return v_mesh_
