import math
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, RolloutReturn, \
    TrainFrequencyUnit
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, should_collect_more_steps
from stable_baselines3 import DDPG
from sb3_contrib.cic.policies import Actor, MultiInputPolicy, CICPolicy
from sb3_contrib.common.wrappers.skill_observation import SkillObservationWrapper

SelfCicDDPG = TypeVar("SelfCicDDPG", bound="CicDDPG")


class CicDDPG(DDPG):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: CICPolicy

    def __init__(
            self,
            env: Union[GymEnv, str],
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
            policy=CICPolicy,
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
        """collect rollout as usual but overwrite the rewards with intrinsic reward"""
        rollout_start_pos = self.replay_buffer.pos

        rollout_return = super().collect_rollouts(env, callback, train_freq, replay_buffer, action_noise,
                                                  learning_starts, log_interval)

        rollout_stop_pos = self.replay_buffer.pos

        # overwrite the zero rewards with intrinsic reward
        self.replay_buffer.rewards[rollout_start_pos:rollout_stop_pos] = self.compute_intrinsic_reward(
            self.replay_buffer.observations[rollout_start_pos:rollout_stop_pos],
            self.replay_buffer.next_observations[rollout_start_pos:rollout_stop_pos]
        )

        return rollout_return

    def train(self, gradient_steps: int, batch_size: int = 100):
        self.update_cpc(gradient_steps, batch_size)
        super().train(gradient_steps, batch_size)

    def compute_intrinsic_reward(self, obs, obs_next):
        with th.no_grad():
            # TODO: or just knn_reward?
            cpc, _ = self.compute_cpc_loss(obs, obs_next)
            return self.alpha * self.compute_knn_reward(obs, obs_next) + (1 - self.alpha) * cpc

    def compute_cpc_loss(self, obs, next_obs):
        eps = 1e-6
        query, key = self.discriminator.forward(obs, next_obs)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = th.mm(query, key.T)  # (b,b)
        sim = th.exp(cov / self.cpc_temp)
        neg = sim.sum(dim=-1)  # (b,)
        row_sub = th.Tensor(neg.shape).fill_(math.e ** (1 / self.cpc_temp)).to(neg.self.device)
        neg = th.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = th.exp(th.sum(query * key, dim=-1) / self.cpc_temp)  # (b,)
        loss = -th.log(pos / (neg + eps))  # (b,)
        return loss, cov / self.cpc_temp

    def compute_knn_reward(self, obs, next_obs):
        source = self.discriminator.state_net(obs)
        target = self.discriminator.state_net(next_obs)

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
        return reward

    def update_cpc(self, gradient_steps: int, batch_size: int = 100):
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        for _ in range(gradient_steps):
            self.discriminator.optimizer.zero_grad()
            disc_loss, _ = self.compute_cpc_loss(replay_data.observations, replay_data.next_observations)
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
