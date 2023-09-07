import sys
from typing import Dict, List, Tuple, Type, Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space, get_action_dim
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device


class CICExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Skill-augmented Dict observation spaces.
    Builds a features extractor for each key of the space. Using a special extractor for the skill (called projection)
    Input from each space is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        assert "skill" in observation_space.spaces.keys()
        self.skill_dim = get_action_dim(observation_space.spaces["skill"])

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            elif key == "skill":
                extractors[key] = nn.Flatten()
                total_concat_size += self.skill_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

    def state_forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            if key != "skill":
                encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=-1)

    def skill_forward(self, observations: TensorDict) -> th.Tensor:
        return self.extractors["skill"](observations["skill"])

    def split_forward(self, observations: TensorDict) -> Tuple[th.Tensor, th.Tensor]:
        """return tuple of extracted features without concatenation (skill_features, state_features)"""
        return self.skill_forward(observations), self.state_forward(observations)

    def get_features_dim(self):
        return self._features_dim

    def get_state_features_dim(self):
        return self._features_dim - self.skill_dim


def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], List[int]]:
    """
    Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.

    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.

    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).

    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.

    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).

    :param net_arch: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return: The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch
