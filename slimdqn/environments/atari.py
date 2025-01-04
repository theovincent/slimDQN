"""
The environment is inspired from https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py
"""

import ale_py
from typing import Tuple
import gymnasium as gym
import numpy as np
import cv2


class AtariEnv:
    def __init__(self, name: str) -> None:
        self.name = name
        self.state_height, self.state_width = (84, 84)
        self.n_stacked_frames = 4
        self.n_skipped_frames = 4

        gym.register_envs(ale_py)  # To use ale with gym which speeds up step()
        self.env = gym.make(
            f"ALE/{self.name}-v5",
            frameskip=1,
            repeat_action_probability=0.25,
            max_num_frames_per_episode=100_000,
            continuous=False,
            continuous_action_threshold=0.0,
        ).env

        self.n_actions = self.env.action_space.n
        self.original_state_height, self.original_state_width, _ = self.env.observation_space._shape
        self.screen_buffer = [
            np.empty((self.original_state_height, self.original_state_width), dtype=np.uint8),
            np.empty((self.original_state_height, self.original_state_width), dtype=np.uint8),
        ]
        self.observation = None

    # @property
    # def observation(self) -> np.ndarray:
    #     return self.state_[:, :, -1]

    # @property
    # def state(self) -> np.ndarray:
    #     return np.array(self.state_, dtype=np.float32)

    def reset(self) -> None:
        self.env.reset()

        self.n_steps = 0

        self.env.env.ale.getScreenGrayscale(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        
        self.observation = self.pool_and_resize()

        # self.state_ = np.zeros((self.state_height, self.state_width, self.n_stacked_frames), dtype=np.uint8)
        # self.state_[:, :, -1] = self.resize()

    def step(self, action) -> Tuple[float, bool]:
        reward = 0

        for idx_frame in range(self.n_skipped_frames):
            _, reward_, terminal, _, _ = self.env.step(action)

            reward += reward_

            if idx_frame >= self.n_skipped_frames - 2:
                t = idx_frame - (self.n_skipped_frames - 2)
                self.env.env.ale.getScreenGrayscale(self.screen_buffer[t])

            if terminal:
                break

        # self.state = np.roll(self.state, -1, axis=-1)
        # self.state[..., -1] = self.pool_and_resize()
        # self.observation = self.state[:, :, -1]
        
        self.observation = self.pool_and_resize()

        self.n_steps += 1

        return reward, terminal

    def pool_and_resize(self) -> np.ndarray:
        np.maximum(self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0])

        # return self.resize()
        
        transformed_image = cv2.resize(
            self.screen_buffer[0],
            (self.state_width, self.state_height),
            interpolation=cv2.INTER_AREA,
        )
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)

    # def resize(self):
    #     return np.asarray(
    #         cv2.resize(self.screen_buffer[0], (self.state_width, self.state_height), interpolation=cv2.INTER_AREA),
    #         dtype=np.uint8,
    #     )
