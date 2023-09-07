import math
import sys
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from gymnasium.core import ObsType
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, RolloutReturn, \
    TrainFrequencyUnit, ReplayBufferSamples, DictReplayBufferSamples
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, should_collect_more_steps
from stable_baselines3 import DDPG
from sb3_contrib.cic.policies import CICPolicy
from sb3_contrib.common.wrappers.skill_observation import SkillObservationWrapper

SelfCicDDPG = TypeVar("SelfCicDDPG", bound="CicDDPG")


class CicDDPG(DDPG):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MultiInputPolicy": CICPolicy,
    }
    policy: CICPolicy

    def __init__(
            self,
            env: Union[GymEnv, str],
            policy: str = "MultiInputPolicy",
            learning_rate: Union[float, Schedule] = 1e-3,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 100,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
            gradient_steps: int = -1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            alpha: float = 0.5,
            cpc_temp: float = 0.5,
    ):
        # TODO: check env is wrapped in skill_obs_wrapper
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
        )
        self.cpc_temp = cpc_temp
        self.alpha = alpha
        self.reward_rms = RMS(epsilon=1e-4, shape=(1,), device=self.device)

        self.apt_args = APTArgs()

    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.discriminator = self.policy.discriminator

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """collect rollout as usual but overwrite with intrinsic rewards"""
        rollout_start_pos = self.replay_buffer.pos

        rollout_return = super().collect_rollouts(env, callback, train_freq, replay_buffer, action_noise,
                                                  learning_starts, log_interval)

        rollout_stop_pos = self.replay_buffer.pos

        # TODO: check what happens when buffer is full
        assert rollout_start_pos < rollout_stop_pos
        replay_data = self.get_samples(batch_inds=np.arange(start=rollout_start_pos, stop=rollout_stop_pos))

        # overwrite the zero rewards with intrinsic reward
        self.replay_buffer.rewards[rollout_start_pos:rollout_stop_pos] = self.compute_intrinsic_reward(
            replay_data.observations, replay_data.next_observations).cpu().numpy()

        return rollout_return

    def get_samples(self, batch_inds: np.ndarray) -> DictReplayBufferSamples:
        obs_ = self.replay_buffer._normalize_obs({key: obs[batch_inds, :, :] for key, obs in self.replay_buffer.observations.items()},
                                   env=None)
        next_obs_ = self.replay_buffer._normalize_obs(
            {key: obs[batch_inds, :, :] for key, obs in self.replay_buffer.next_observations.items()}, env=None
        )

        # Convert to torch tensor
        observations = {key: self.replay_buffer.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.replay_buffer.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.replay_buffer.to_torch(self.replay_buffer.actions[batch_inds]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.replay_buffer.to_torch(
                self.replay_buffer.dones[batch_inds] * (1 - self.replay_buffer.timeouts[batch_inds])).reshape(
                -1, 1
            ),
            rewards=self.replay_buffer.to_torch(self.replay_buffer._normalize_reward(self.replay_buffer.rewards[batch_inds].reshape(-1, 1), env=None)),
        )

    def train(self, gradient_steps: int, batch_size: int = 100):
        print("TRAIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.update_cpc(gradient_steps, batch_size)
        super().train(gradient_steps, batch_size)

    def compute_intrinsic_reward(self, obs, obs_next):
        with th.no_grad():
            # TODO: or just knn_reward?
            cpc, _ = self.compute_cpc_loss(obs, obs_next)
            return self.alpha * self.compute_knn_reward(obs, obs_next) + (1 - self.alpha) * cpc

    def compute_cpc_loss(self, obs, next_obs, eps=1e-6):
        queries, keys = [], []
        for env in range(self.replay_buffer.n_envs):
            query, key = self.discriminator({key: value[:, env] for key, value in obs.items()},
                                            {key: value[:, env] for key, value in next_obs.items()})
            queries.append(query)
            keys.append(key)
        query, key = th.stack(queries, dim=1), th.stack(keys, dim=1)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        inner = th.einsum('...i,...i->...', query, key)
        outer = th.einsum('...i,...j->...ij', query, key)
        sim = th.exp(outer / self.cpc_temp)
        neg = sim.sum(dim=-1)
        row_sub = th.Tensor(neg.shape).fill_(math.e ** (1 / self.cpc_temp)).to(neg.device)
        neg = th.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = th.exp(inner / self.cpc_temp)
        loss = -th.log(pos / (neg + eps)).mean()
        return loss, outer / self.cpc_temp

    def compute_knn_reward(self, obs, next_obs):
        sources, targets = [], []
        rewards = []
        for env in range(self.replay_buffer.n_envs):
            source, target = self.discriminator.transition_forward(
                {key: value[:, env] for key, value in obs.items()},
                {key: value[:, env] for key, value in next_obs.items()}
            )
            source = self.discriminator.state_net(obs["observation"][:, env]).unsqueeze(1)
            target = self.discriminator.state_net(next_obs["observation"][:, env]).unsqueeze(1)
            b1, b2 = source.size(0), target.size(0)
            # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
            sim_matrix = th.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)
            reward, _ = sim_matrix.topk(self.apt_args.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

            if not self.apt_args.knn_avg:  # only keep k-th nearest neighbor
                reward = reward[:, -1]
                reward = reward.reshape(-1, 1)  # (b1, 1)
                if self.apt_args.rms:
                    moving_mean, moving_std = self.reward_rms(reward)
                    reward = reward / moving_std
                reward = th.max(reward - self.apt_args.knn_clip, th.zeros_like(reward).to(self.device))  # (b1, )
            else:  # average over all k nearest neighbors
                reward = reward.reshape(-1, 1)  # (b1 * k, 1)
                if self.apt_args.rms:
                    moving_mean, moving_std = self.reward_rms(reward)
                    reward = reward / moving_std
                reward = th.max(reward - self.apt_args.knn_clip, th.zeros_like(reward).to(self.device))
                reward = reward.reshape((b1, self.apt_args.knn_k))  # (b1, k)
                reward = reward.mean(dim=1)  # (b1,)
            reward = th.log(reward + 1.0)
            rewards.append(reward)
        return th.stack(rewards, dim=1)

    def update_cpc(self, gradient_steps: int, batch_size: int = 100):
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        obs = {key: value.unsqueeze(1) for key, value in replay_data.observations.items()}
        obs_next = {key: value.unsqueeze(1) for key, value in replay_data.next_observations.items()}
        for _ in range(gradient_steps):
            self.discriminator.optimizer.zero_grad()
            disc_loss, _ = self.compute_cpc_loss(obs, obs_next)
            disc_loss.backward()
            self.discriminator.optimizer.step()


class RMS(object):
    def __init__(self, epsilon=1e-4, shape=(1,), device="cpu"):
        self.M = th.zeros(shape).to(device)
        self.S = th.ones(shape).to(device)
        self.n = epsilon

    @th.no_grad()
    def __call__(self, x):
        bs = x.size(0)
        delta = th.mean(x, dim=0) - self.M
        new_m = self.M + delta * bs / (self.n + bs)
        new_s = (self.S * self.n + th.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_m
        self.S = new_s
        self.n += bs

        return self.M, self.S


class APTArgs:
    def __init__(self, knn_k=16, knn_avg=True, rms=True, knn_clip=0.0005):
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.rms = rms
        self.knn_clip = knn_clip


def slice_dict(d: Dict, d_slice: List[slice]):
    return {key: d[key][d_slice] for key in d}
