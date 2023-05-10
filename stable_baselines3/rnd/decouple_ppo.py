import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union, Tuple, List

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F
from copy import copy, deepcopy
import time
import sys

# For rewrite RND
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import RolloutRNDBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.running_mean_std import RunningMeanStd

# For PPO part
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.rnd.policies import RNDActorCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

# For component models
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.rnd.rnd import RND
from stable_baselines3.common.evaluation import evaluate_policy

SelfDecouple = TypeVar("SelfDecouplePPO", bound="DecouplePPO")

class DecouplePPO(RND):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "CnnPolicy": RNDActorCritic,
    }

    def __init__(
        # General args
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],

        # Decouple Policy Part
        exploit_iterations_per_explore_session: int,
        exploit_env: GymEnv,
        eval_env: GymEnv,

        # RND Part
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        int_gamma: float = 0.999,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,

        # DQN Part
        learning_starts: int = 50000,
        tau: float = 1.0,
        train_freq: Union[int, Tuple[int, str]] = 4,
        buffer_size: int = 100000,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        exploit_policy_max_grad_norm: float = 10,
        exploit_policy_kwargs: Optional[Dict[str, Any]] = None,

        # Logging args
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,

        # rollin
        rollin: bool = False,
    ):
        self.exploit_iterations_per_explore_session = exploit_iterations_per_explore_session

        # the object itself is an exploratory agent
        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size, # batch_size
            n_epochs,
            gamma,
            int_gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            normalize_advantage,
            ent_coef,
            vf_coef,
            max_grad_norm,
            use_sde,
            sde_sample_freq,
            target_kl,
            tensorboard_log,
            policy_kwargs,
            1, # verbose
            seed,
            device,
            _init_setup_model,
        )

        # define the exploitation solver
        # parameters adapted from rl-baselinse3 zoo
        self.exploit_policy = PPO(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size, # batch_size
            n_epochs,
            gamma,
            int_gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            normalize_advantage,
            ent_coef,
            vf_coef,
            max_grad_norm,
            use_sde,
            sde_sample_freq,
            target_kl,
            tensorboard_log,
            policy_kwargs,
            1, # verbose
            seed,
            device,
            _init_setup_model,
        )

        self.observation_shape = (1, 84, 84)
        self.exploit_policy_evaluation = []
        self.eval_env = eval_env
        self.exploit_env = exploit_env

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        for i in range(0, self.env.num_envs, self.exploit_env.num_envs):
            replay_buffer.add(
                self._last_original_obs[i:i+self.exploit_env.num_envs, :],
                next_obs[i:i+self.exploit_env.num_envs, :],
                buffer_action[i:i+self.exploit_env.num_envs, :],
                reward_[i:i+self.exploit_env.num_envs],
                dones[i:i+self.exploit_env.num_envs],
                infos[i:i+self.exploit_env.num_envs],
            )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    # override the collect_rollouts for RND
    # The only changes is that each time we collect a transition, we also store it to the replay buffer of DQN
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutRNDBuffer,
        n_rollout_steps: int,
        replay_buffer=None,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, int_values, log_probs = self.policy(obs_tensor)
                exploit_probs = self.exploit_policy.policy._action_probs(obs_tensor)
                exploit_log_probs = th.log(exploit_probs)
                exploit_log_probs = th.gather(exploit_log_probs, dim=1, index=actions.long().unsqueeze(1))
                exploit_log_probs = exploit_log_probs.reshape(*log_probs.shape)
                # log_probs = (log_probs + exploit_log_probs) / 2 # mix the exploit log_prob into the PPO training mechanism

                log_probs = exploit_log_probs # probably we should directly use this?
                # print(exploit_log_probs.shape)
                # raise NotImplementedError

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # Calculate the intrinsic reward
            with th.no_grad():
                # print(new_obs.shape, "The original env output shape")
                # print(new_obs[:, -1, :, :].shape, "The attempted extracted frames shape")
                new_obs_normalized = np.clip((new_obs[:, -1:, :, :].copy() - self.state_rms.mean) / (self.state_rms.var ** 0.5), -5, 5,
                              dtype="float32")
                # print(new_obs[:, -1:, :, :].copy().shape, "The extracted shape")
                # print(self.state_rms.mean.shape)
                # print(self.state_rms.var.shape)
                # print(new_obs_normalized.shape, "The normalized shape ")
                next_obs_tensor = obs_as_tensor(new_obs_normalized, self.device)
                # latest_frame_tensor = next_obs_tensor[:, -1, :, :]
                # print(next_obs_tensor.shape, "2")
                latest_frame_tensor = next_obs_tensor
                # rnd_pred, rnd_target = self.rnd_model(latest_frame_tensor)
                rnd_pred = self.policy.rnd_predictor(latest_frame_tensor)
                rnd_target = self.policy.rnd_target(latest_frame_tensor)
                int_rewards = (rnd_pred - rnd_target).pow(2).mean(1)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value, terminal_int_value = self.policy.predict_all_values(terminal_obs)
                        # print(terminal_value.shape)
                        terminal_value, terminal_int_value = terminal_value[0], terminal_int_value[0]
                    rewards[idx] += self.gamma * terminal_value
                    int_rewards[idx] += self.int_gamma * terminal_int_value
            
            # add to rollout buffer belonging to RND
            rollout_buffer.add(
                self._last_obs,
                new_obs,
                actions, 
                rewards,
                int_rewards,
                self._last_episode_starts, 
                values, 
                int_values,
                log_probs
            )

            # add to replay buffer belonging to DQN
            self._store_transition(self.exploit_policy.replay_buffer, actions.copy(), new_obs.copy(), rewards.copy(), dones.copy(), copy(infos))

            self._last_obs = new_obs
            self._last_episode_starts = dones

        # normalize the intrinsic rewards
        rollout_buffer.int_rewards = self.normalize_int_rewards(rollout_buffer.int_rewards)

        with th.no_grad():
            # Compute value for the last timestep
            values, int_values = self.policy.predict_all_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, 
            last_int_values=int_values,
            dones=dones
        )

        callback.on_rollout_end()

        return True

    def sync_time_steps(self):
        current_time_steps = max(
            self.num_timesteps,
            self.exploit_policy.num_timesteps,
        )

        self.num_timesteps = current_time_steps
        self.exploit_policy.num_timesteps = current_time_steps

    def total_num_timesteps(self):
        return self.num_timesteps + self.exploit_policy.num_timesteps

    def learn(
        self: SelfDecouple,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDecouple:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        total_timesteps, callback = self.exploit_policy._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar=False,
        )

        iteration = 0
        performance = []
        eval_length = []
        DQN_losses = []
        callback.on_training_start(locals(), globals())

        while self.total_num_timesteps() < total_timesteps:
            print("Collecting Data")
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    # print(self.ep_info_buffer)
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.total_num_timesteps(), exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

                # print(self.exploit_policy.ep_info_buffer)
                if performance:
                    print(f"============== Last Evaluation Performance : {np.mean(performance[-1])} ==============")
                    print(f"============== Last Evaluation Length : {np.mean(eval_length[-1])} ==============")

            print("Training")
            self.train()

            for exploit_train_step in range(int(128 * 128 / 32 * 4)):
                self.exploit_policy.train(gradient_steps=self.exploit_policy.gradient_steps, batch_size=32)

            if iteration % 10 == 0:
                for step in range(0, self.exploit_iterations_per_explore_session, self.exploit_policy.n_envs):
                    rollout = self.exploit_policy.collect_rollouts(
                        self.exploit_env,
                        train_freq=self.exploit_policy.train_freq,
                        action_noise=self.exploit_policy.action_noise,
                        callback=callback,
                        learning_starts=self.exploit_policy.learning_starts,
                        replay_buffer=self.exploit_policy.replay_buffer,
                        log_interval=log_interval,
                    )
                
                    self.exploit_policy.train(gradient_steps=self.exploit_policy.gradient_steps, batch_size=32)
                    np.save(
                        f"/home/ywang3/workplace/theory_inspired/decouple/stable-baselines3/decouple_results/DQN_loss", 
                        self.exploit_policy.Q_loss
                    )

                results, length = evaluate_policy(
                    model=self.exploit_policy,
                    env=self.eval_env,
                    n_eval_episodes=10,
                    return_episode_rewards=True,
                )
                performance.append(results)
                eval_length.append(length)
                np.save(f'/home/ywang3/workplace/theory_inspired/decouple/stable-baselines3/decouple_results/DQN_{tb_log_name}', performance)
                # print(f"============ Last Evaluation Performance of Exploit Policy : {performance[-1]} =============")
                # print(f"============ Last Evaluation Performance of Explore Policy : {explore_results} =============")


        callback.on_training_end()

        return self