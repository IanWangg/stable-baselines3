from typing import Any, Dict, List, Optional, Type, Union, Tuple

import torch as th
from gym import spaces
from torch import nn
import torch.nn.functional as F
import numpy as np

from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor

from abc import ABC, abstractmethod
from functools import partial

def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1

class TargetModel(nn.Module, ABC):
    def __init__(self, state_shape=(1, 84, 84)):
        super(TargetModel, self).__init__()
        self.state_shape = state_shape

        c, w, h = state_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.encoded_features = nn.Linear(in_features=flatten_size, out_features=512)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs=(1, 84, 84)):
        x = inputs
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu((self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.encoded_features(x)


class PredictorModel(nn.Module, ABC):

    def __init__(self, state_shape=(1, 84, 84)):
        super(PredictorModel, self).__init__()
        self.state_shape = state_shape

        c, w, h = state_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc1 = nn.Linear(in_features=flatten_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.encoded_features = nn.Linear(in_features=512, out_features=512)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu((self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.encoded_features(x)

class ActorDoubleCriticCnnPolicy(ActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
    
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.int_value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1) # add a value head
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.int_value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        
        self.rnd_predictor = RNDPred().to(self.device) # rnd predictor
        self.rnd_target = RNDTarget().to(self.device) # rnd target, fixed
        for param in self.rnd_target.parameters():
            param.requires_grad = False
        self.rnd_optimizer = th.optim.Adam(self.rnd_predictor.parameters(), lr=1e-4)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        int_values = self.int_value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, int_values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        int_values = self.int_value_net(latent_vf)
        entropy = distribution.entropy()
        return values, int_values, log_prob, entropy

    # predict int and ext value
    def predict_all_values(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf), self.int_value_net(latent_vf)

    def predict_int_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.int_value_net(latent_vf)


class DoubleActorDoubleCriticCnnPolicy(ActorDoubleCriticCnnPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
    
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Get the explore policy constructed as in RND (action_net)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        # Construct an additional policy for PPO training (exploit_action_net)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.exploit_action_net, self.exploit_log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.exploit_action_net, self.exploit_log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.exploit_action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.int_value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1) # add a value head
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.exploit_action_net: 0.01,
                self.value_net: 1,
                self.int_value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        
        self.rnd_predictor = RNDPred().to(self.device) # rnd predictor
        self.rnd_target = RNDTarget().to(self.device) # rnd target, fixed
        for param in self.rnd_target.parameters():
            param.requires_grad = False
        self.rnd_optimizer = th.optim.Adam(self.rnd_predictor.parameters(), lr=1e-4)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        int_values = self.int_value_net(latent_vf)
        distribution, exploit_distribution = self._get_action_dist_from_latent(latent_pi)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        exploit_actions = exploit_distribution.get_actions(deterministic=deterministic)
        exploit_log_prob = exploit_distribution.log_prob(exploit_actions)

        actions = actions.reshape((-1, *self.action_space.shape))
        exploit_actions = exploit_actions.reshape((-1, *self.action_space.shape))

        return actions, exploit_actions, values, int_values, log_prob, exploit_log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        int_values = self.int_value_net(latent_vf)
        entropy = distribution.entropy()
        return values, int_values, log_prob, entropy

    # predict int and ext value
    def predict_all_values(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf), self.int_value_net(latent_vf)

    def predict_int_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.int_value_net(latent_vf)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        exploit_mean_actions = self.exploit_action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return (
                self.action_dist.proba_distribution(mean_actions, self.log_std),
                self.action_dist.proba_distribution(exploit_mean_actions, self.exploit_log_std)
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return (
                self.action_dist.proba_distribution(action_logits=mean_actions),
                self.action_dist.proba_distribution(action_logits=exploit_mean_actions)
            )
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return (
                self.action_dist.proba_distribution(action_logits=mean_actions),
                self.action_dist.proba_distribution(action_logits=exploit_mean_actions)
            )
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return (
                self.action_dist.proba_distribution(action_logits=mean_actions),
                self.action_dist.proba_distribution(action_logits=exploit_mean_actions)
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return (
                self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi),
                self.action_dist.proba_distribution(exploit_mean_actions, self.exploit_log_std, latent_pi)
            )
        else:
            raise ValueError("Invalid action distribution")



RNDPred = PredictorModel
RNDTarget = TargetModel
RNDActorCritic = ActorDoubleCriticCnnPolicy