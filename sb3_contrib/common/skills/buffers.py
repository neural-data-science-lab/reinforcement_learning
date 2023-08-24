import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from sb3_contrib.common.skills.type_aliases import (
    ReplayBufferSamplesZ,
    ReplayBufferSamplesZExternalDisc,
    ReplayBufferSamplesZExternalDiscTraj,
)
from stable_baselines3.common.vec_env import VecNormalize


class ReplayBufferZ(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            prior: th.distributions,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
    ):
        super(ReplayBufferZ, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=observation_space.dtype,
        )
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs) + self.obs_shape,
                dtype=observation_space.dtype,
            )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.prior = prior
        z_size = self.prior.event_shape
        self.zs = np.zeros((self.buffer_size, z_size[0]), dtype=np.float32)

        self.ep_index = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        if psutil is not None:
            total_memory_usage = (
                    self.observations.nbytes
                    + self.actions.nbytes
                    + self.rewards.nbytes
                    + self.dones.nbytes
                    + self.zs.nbytes
                    + self.ep_index.nbytes
            )
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            total_memory_usage /= 1e9
            mem_available /= 1e9
            print(total_memory_usage, mem_available)
            if total_memory_usage > mem_available:
                # Convert to GB

                total_memory_usage /= 1e9
                mem_available /= 1e9
                print(total_memory_usage, mem_available)
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            z: np.ndarray,
            ep_index: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(
                next_obs
            ).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.zs[self.pos] = np.array(z).copy()
        self.ep_index[self.pos] = np.array(ep_index).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
            self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (
                                 np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
                         ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
            self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, 0, :], env
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
            self.zs[batch_inds],
            self.ep_index[batch_inds]
        )
        return ReplayBufferSamplesZ(*tuple(map(self.to_torch, data)))


class ReplayBufferZExternalDisc(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            prior: th.distributions,
            disc_shape: np.ndarray,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
    ):
        super(ReplayBufferZExternalDisc, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=observation_space.dtype,
        )
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs) + self.obs_shape,
                dtype=observation_space.dtype,
            )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.prior = prior
        z_size = self.prior.event_shape
        self.zs = np.zeros((self.buffer_size, z_size[0]), dtype=np.float32)
        self.disc_shape = disc_shape
        self.disc_obs = np.zeros((self.buffer_size, self.n_envs) + tuple(self.disc_shape),
                                 dtype=np.float32)

        self.ep_index = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        if psutil is not None:
            total_memory_usage = (
                    self.observations.nbytes
                    + self.actions.nbytes
                    + self.rewards.nbytes
                    + self.dones.nbytes
                    + self.zs.nbytes
                    + self.disc_obs.nbytes
                    + self.ep_index.nbytes
            )
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            total_memory_usage /= 1e9
            mem_available /= 1e9
            print(total_memory_usage, mem_available)
            if total_memory_usage > mem_available:
                # Convert to GB

                total_memory_usage /= 1e9
                mem_available /= 1e9
                print(total_memory_usage, mem_available)
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            z: np.ndarray,
            disc_obs: np.ndarray,
            ep_index: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(
                next_obs
            ).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.zs[self.pos] = np.array(z).copy()
        self.disc_obs[self.pos] = np.array(disc_obs).copy()
        self.ep_index[self.pos] = np.array(ep_index).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
            self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (
                                 np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
                         ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
            self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamplesZExternalDisc:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, 0, :], env
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
            self.zs[batch_inds],
            self.disc_obs[batch_inds, 0, :],
            self.ep_index[batch_inds]
        )
        return ReplayBufferSamplesZExternalDisc(*tuple(map(self.to_torch, data)))

    def sample_trajectories(self, batch_size, disc_only=False):
        if not self.full:
            ind_dones = np.concatenate([[-1], np.where(self.dones)[0]])
        else:
            ind_dones = np.where(self.dones)[0]

        ind_dones_idx = np.random.randint(0, len(ind_dones) - 1, size=batch_size + 1)
        trajs_bound = np.concatenate([ind_dones[ind_dones_idx, None], ind_dones[ind_dones_idx + 1, None]], axis=1)
        l_big_traj = np.diff(ind_dones).max()
        disc_trajs = np.zeros((batch_size, l_big_traj) + self.disc_shape, dtype=np.float32)
        z_size = self.prior.event_shape[0]
        z_trajs = np.zeros((batch_size, l_big_traj, z_size), dtype=np.float32)
        if not disc_only:
            obs_trajs = np.zeros(
                (batch_size, l_big_traj) + self.obs_shape, dtype=np.float32
            )
            next_obs_trajs = np.zeros(
                (batch_size, l_big_traj) + self.obs_shape, dtype=np.float32
            )
            reward_trajs = np.zeros((batch_size, l_big_traj, 1), dtype=np.float32)

            done_trajs = np.zeros((batch_size, l_big_traj, 1), dtype=np.float32)
            action_trajs = np.zeros((batch_size, l_big_traj, self.action_dim), dtype=np.float32)

        lenghts = np.zeros(batch_size)
        for i in range(batch_size):

            start, end = trajs_bound[i] + 1
            l = end - start
            disc_traj = np.swapaxes(self.disc_obs[start:end], 0, 1)
            z_traj = self.zs[start:end]
            if not disc_only:
                next_obs_traj = self.next_observations[start:end, 0, :]
                obs_traj = self.observations[start:end, 0, :]
                reward_traj = self.rewards[start:end]
                done_traj = self.dones[start:end]
                action_traj = self.actions[start:end, 0, :]

                # print(disc_traj.shape, z_traj.shape, next_o
                # s_traj.shape, obs_traj.shape, reward_traj.shape)

            disc_trajs[i] = np.pad(disc_traj, ((0, 0), (0, l_big_traj - l), (0, 0)), constant_values=-1)
            z_trajs[i] = np.pad(z_traj, ((0, l_big_traj - l), (0, 0)), constant_values=-1)
            if not disc_only:
                next_obs_trajs[i] = np.pad(next_obs_traj, ((0, l_big_traj - l), (0, 0)), constant_values=-np.inf)
                obs_trajs[i] = np.pad(obs_traj, ((0, l_big_traj - l), (0, 0)), constant_values=-np.inf)
                reward_trajs[i] = np.pad(reward_traj, ((0, l_big_traj - l), (0, 0)), constant_values=-np.inf)
                done_trajs[i] = np.pad(done_traj, ((0, l_big_traj - l), (0, 0)), constant_values=-1)
                action_trajs[i] = np.pad(action_traj, ((0, l_big_traj - l), (0, 0)), constant_values=-np.inf)
            lenghts[i] = l

        if not disc_only:

            return obs_trajs, next_obs_trajs, disc_trajs, reward_trajs, z_trajs, done_trajs, action_trajs, lenghts

        else:
            return disc_trajs, z_trajs, lenghts

    def get_current_traj(self):
        ind_dones = np.where(self.dones)[0]
        if self.full and len(ind_dones[ind_dones < self.pos]) == 0:
            # if full and if episode starts at the end of buffer
            start_ind = ind_dones[-1]
            traj_0 = self.disc_obs[start_ind + 1:self.buffer_size]
            traj_1 = self.disc_obs[:self.pos + 1]
            traj = np.concatenate([traj_0, traj_1])
        else:

            if len(ind_dones) == 0:
                # if first episode
                start_ind = -1
            else:
                start_ind = ind_dones[ind_dones < self.pos][-1]
            traj = self.disc_obs[start_ind + 1:self.pos + 1]

        traj = np.swapaxes(traj, 0, 1)

        lenght = traj.shape[1] - 1

        return th.Tensor(traj[0]), lenght


class ReplayBufferZExternalDiscTraj(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
            self,
            buffer_size: int,
            max_steps: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            prior: th.distributions,
            disc_shape: np.ndarray,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
    ):
        super(ReplayBufferZExternalDiscTraj, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available
        self.max_steps = max_steps
        self.n_episodes = 0
        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.full(
            (self.buffer_size, self.max_steps, self.n_envs) + self.obs_shape, -np.inf,
            dtype=np.float32,
        )
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.full(
                (self.buffer_size, self.max_steps, self.n_envs) + self.obs_shape, -np.inf,
                dtype=np.float32,
            )
        self.actions = np.full(
            (self.buffer_size, self.max_steps, self.n_envs, self.action_dim), -np.inf, dtype=np.float32
        )
        self.rewards = np.full((self.buffer_size, self.max_steps, self.n_envs), -np.inf, dtype=np.float32)
        self.dones = np.full((self.buffer_size, self.max_steps, self.n_envs), -1, dtype=np.float32)
        self.prior = prior
        z_size = self.prior.event_shape
        self.zs = np.full((self.buffer_size, self.max_steps, z_size[0]), -1, dtype=np.float32)
        self.disc_shape = disc_shape
        self.disc_obs = np.full((self.buffer_size, self.max_steps, self.n_envs) + tuple(self.disc_shape), -np.inf,
                                dtype=np.float32)
        self.current_episode = 0
        self.lenghts = np.zeros((self.buffer_size), dtype=np.float32)
        if psutil is not None:
            total_memory_usage = (
                    self.observations.nbytes
                    + self.actions.nbytes
                    + self.rewards.nbytes
                    + self.dones.nbytes
                    + self.zs.nbytes
                    + self.disc_obs.nbytes
                    + self.lenghts.nbytes
            )
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            total_memory_usage /= 1e9
            mem_available /= 1e9
            print(total_memory_usage, mem_available)
            if total_memory_usage > mem_available:
                # Convert to GB

                total_memory_usage /= 1e9
                mem_available /= 1e9
                print(total_memory_usage, mem_available)
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            z: np.ndarray,
            disc_obs: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference

        idx = np.unravel_index(self.pos, (self.buffer_size, self.max_steps))

        self.observations[idx] = np.array(obs).copy()

        if self.optimize_memory_usage:
            assert self.optimize_memory_usage == False, "Not Implemented"
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(
                next_obs
            ).copy()
        else:
            self.next_observations[idx] = np.array(next_obs).copy()

        self.actions[idx] = np.array(action).copy()
        self.rewards[idx] = np.array(reward).copy()
        self.dones[idx] = np.array(done).copy()
        self.zs[idx] = np.array(z).copy()
        self.disc_obs[idx] = np.array(disc_obs).copy()

        # current_episode = self.pos//self.max_steps
        # if self.pos%self.max_steps == 0:
        #    self.lenghts[self.current_episode]=0
        #    print("reset_episode",current_episode)

        if self.current_episode * self.max_steps == self.pos:
            self.lenghts[self.current_episode] = 0

        self.lenghts[self.current_episode] += 1

        if done.item():
            if self.pos % self.max_steps:
                self.current_episode += 1
                self.pos = self.current_episode * self.max_steps - 1

        self.pos += 1

        if self.pos == self.buffer_size * self.max_steps:
            self.full = True
            self.pos = 0
            self.current_episode = 0

    def sample(
            self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.current_episode
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        return self._get_samples(batch_inds, env=env)

    def _get_samples(
            self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamplesZExternalDisc:
        if self.optimize_memory_usage:
            assert self.optimize_memory_usage == False, "Not implemented"
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, :, 0, :], env
            )
        data = (
            self._normalize_obs(self.observations[batch_inds, :, 0, :], env),
            self.actions[batch_inds, :, 0, :],
            next_obs,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
            self.zs[batch_inds],
            self.disc_obs[batch_inds, :, 0, :],
            self.lenghts[batch_inds],
        )
        return ReplayBufferSamplesZExternalDiscTraj(*tuple(map(self.to_torch, data)))

    def get_current_traj(self):
        if self.current_episode * self.max_steps == self.pos:
            self.lenghts[self.current_episode] = 0

        _, _, _, _, _, _, traj, lenght = self._get_samples(np.array(self.current_episode))
        return traj, lenght
