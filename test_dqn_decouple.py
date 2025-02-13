import gym
import stable_baselines3
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.dqn.dqn_decouple import DQN_Decouple
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

from typing import Callable

if __name__ == '__main__':
    env_name = 'MontezumaRevengeNoFrameskip-v4'
    
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.
        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.
            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func

    env = make_atari_env(
        env_name, 
        # wrapper_class=sparse_reward_wrapper,
        # wrapper_kwargs={'montezuma': False},
        n_envs=1,
        seed=1,
    )
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    eval_env = make_atari_env(
        env_name, 
        # wrapper_class=sparse_reward_wrapper,
        # wrapper_kwargs={'montezuma': False},
        n_envs=1,
        seed=1,
        vec_env_cls=SubprocVecEnv
    )
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    print(env.reset().shape)
    # eval_env = make_vec_env(env_name, n_envs=1)
    # env = VecNormalize(env, norm_reward=False, clip_reward=100)
    # eval_env = VecNormalize(eval_env, training=False, norm_reward=False, clip_reward=100)
    # eval_env.obs_rms = env.obs_rms
    
    
    model = DQN_Decouple(
        env=env,
        eval_env=eval_env,
        policy='CnnPolicy',
        learning_rate=1e-4,
        tensorboard_log='./MR',
        train_freq=4,
        batch_size=32,
        buffer_size=1000000,
        learning_starts=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        target_update_interval=1000,
        lam=10.0,
        optimize_memory_usage=False,
        verbose=1,
    )
    


    model.learn(
        total_timesteps=int(100_000_000),
        tb_log_name=f'DQN_MR_Decouple',
        reset_num_timesteps=True,
        progress_bar=True,
    )
