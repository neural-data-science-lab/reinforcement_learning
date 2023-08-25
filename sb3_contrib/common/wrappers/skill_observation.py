from typing import Any, Dict, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType

TimeFeatureObs = Union[np.ndarray, Dict[str, np.ndarray]]


class SkillObservationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_skills: int = 5, skill_domain: str = "continuous"):
        super().__init__(env)
        assert isinstance(
            env.observation_space, (spaces.Discrete, spaces.Box, spaces.Dict)
        ), "`SkillObservationWrapper` only supports `gym.spaces.Discrete`, `gym.spaces.Box` and " \
           "`gym.spaces.Dict` observation spaces."

        if skill_domain == "continuous":
            self.skill_space = gym.spaces.Box(low=-1, high=1, shape=(n_skills,), dtype=np.float)
        elif skill_domain == "discrete":
            self.skill_space = gym.spaces.Discrete(n=n_skills)
        else:
            raise KeyError(f"Skill domain {skill_domain} unknown.")

        # Add a time feature to the observation
        if isinstance(env.observation_space, spaces.Dict):
            env.observation_space.spaces["skill"] = self.skill_space
        else:
            env.observation_space = gym.spaces.Dict({
                "observation": env.observation_space,
                "skill": self.skill_space
            })

        self.current_skill = self.skill_space.sample()

    def reset(self, skill=None, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_skill = skill or self.skill_space.sample()
        return self._get_obs(obs), info

    def step(self, action: ActType):
        # make sure there is no reward
        obs, _, done, truncated, info = self.env.step(action)
        return self._get_obs(obs), 0, done, truncated, info

    def _get_obs(self, obs, skill=None):
        """
        Add skill to the current observation.
        """
        return {
            "observation": obs,
            "skill": self.current_skill
        }
