from typing import Any, Dict, List, Optional, Type, Union, Tuple

import torch as th
from gymnasium import spaces
from stable_baselines3.td3.policies import TD3Policy
from torch import nn

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, BaseModel
from stable_baselines3.common.preprocessing import get_action_dim, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule

from sb3_contrib.cic.torch_layers import CICExtractor


class CICDiscriminator(BaseModel):
    """
    Discriminator network(s) for CIC. Computes query h_z and key h_tau.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: CICExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        skill_space = observation_space.spaces["skill"]
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        skill_dim = get_action_dim(skill_space)
        features_dim = features_extractor.get_state_features_dim()

        self.share_features_extractor = share_features_extractor
        self.state_net = nn.Sequential(*create_mlp(features_dim, features_dim, net_arch, activation_fn))
        self.key_net = nn.Sequential(*create_mlp(2 * features_dim, skill_dim, net_arch, activation_fn))
        self.query_net = nn.Sequential(*create_mlp(skill_dim, skill_dim, net_arch, activation_fn))

    def extract_features(self, obs: th.Tensor, features_extractor: CICExtractor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Preprocess the observation if needed and extract features.

         :param obs: The observation
         :param features_extractor: The features extractor to use.
         :return: The extracted features
        """
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor.split_forward(preprocessed_obs)

    def forward(self, obs: th.Tensor, obs_next: th.Tensor) -> Tuple[th.Tensor, ...]:
        skill, state_features = self.extract_features(obs, self.features_extractor)
        _, next_state_features = self.extract_features(obs_next, self.features_extractor)
        query = self.query_net(skill)
        key = self.key_net(th.cat([self.state_net(state_features), self.state_net(next_state_features)], dim=-1))
        return query, key

    def transition_forward(self, obs: th.Tensor, obs_next: th.Tensor) -> Tuple[th.Tensor, ...]:
        _, state_features = self.extract_features(obs, self.features_extractor)
        _, next_state_features = self.extract_features(obs_next, self.features_extractor)
        return self.state_net(state_features), self.state_net(next_state_features)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            observation_space=self.observation_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            normalize_images=self.normalize_images,
        )


class CICPolicy(TD3Policy):
    """
    Policy class (with both actor and critic) for CIC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """
    discriminator: CICDiscriminator

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CICExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule=lr_schedule)
        # imo the discriminator should share feature extractor with target networks
        self.discriminator = self.make_discriminator(features_extractor=self.actor_target.features_extractor)

        self.discriminator.optimizer = self.optimizer_class(
            self.discriminator.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # From super method: "Target networks should always be in eval mode"
        # self.actor_target.set_training_mode(False)
        # self.critic_target.set_training_mode(False)
        # TODO: think about what this means for the discriminator

    def make_discriminator(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CICDiscriminator:
        discriminator_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CICDiscriminator(**discriminator_kwargs).to(self.device)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        super().set_training_mode(mode)
        self.discriminator.set_training_mode(mode)
