from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.envs import SimpleMultiObsEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from sb3_contrib import QRDQN, TQC, TRPO
from sb3_contrib.common.wrappers.skill_observation import SkillObservationWrapper


class DummyDictEnv(gym.Env):
    """Custom Environment for testing purposes only"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        use_discrete_actions=False,
        channel_last=False,
        nested_dict_obs=False,
        vec_only=False,
    ):
        super().__init__()
        if use_discrete_actions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        N_CHANNELS = 1
        HEIGHT = 36
        WIDTH = 36

        if channel_last:
            obs_shape = (HEIGHT, WIDTH, N_CHANNELS)
        else:
            obs_shape = (N_CHANNELS, HEIGHT, WIDTH)

        self.observation_space = spaces.Dict(
            {
                # Image obs
                "img": spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                # Vector obs
                "vec": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                # Discrete obs
                "discrete": spaces.Discrete(4),
            }
        )

        # For checking consistency with normal MlpPolicy
        if vec_only:
            self.observation_space = spaces.Dict(
                {
                    # Vector obs
                    "vec": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                }
            )

        if nested_dict_obs:
            # Add dictionary observation inside observation space
            self.observation_space.spaces["nested-dict"] = spaces.Dict({"nested-dict-discrete": spaces.Discrete(4)})

    def step(self, action):
        reward = 0.0
        done = truncated = False
        return self.observation_space.sample(), reward, done, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.observation_space.seed(seed)
        return self.observation_space.sample(), {}

    def render(self):
        pass


def test_skill_obs():
    env = DummyDictEnv()
    env_wrapped = SkillObservationWrapper(env)
    check_env(env_wrapped)

