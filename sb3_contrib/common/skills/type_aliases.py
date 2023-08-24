from typing import NamedTuple

import torch as th


class RolloutReturnZ(NamedTuple):
    episode_reward: float
    episode_timesteps: int
    n_episodes: int
    continue_training: bool
    z: th.Tensor


class ReplayBufferSamplesZ(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    zs: th.Tensor
    ep_index: th.Tensor


class ReplayBufferSamplesZExternalDisc(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    zs: th.Tensor
    disc_obs: th.Tensor
    ep_index: th.Tensor


class ReplayBufferSamplesZExternalDiscTraj(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    zs: th.Tensor
    disc_obs: th.Tensor
    lengths: th.Tensor
