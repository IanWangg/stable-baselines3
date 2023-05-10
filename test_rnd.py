import gym
import stable_baselines3
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.rnd.rnd import RND
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

from typing import Callable

if __name__ == '__main__':
    env_name = 'BreakoutNoFrameskip-v4'
    # env_name = 'BreakoutNoFrameskip-v4'
    # env_name = 'VentureNoFrameskip-v4'
    
    # ENIAC : 32 env, 16 steps, 4 epoch 
    # OSPOE : 32 env, 32 steps, 6 epoch
    
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
        n_envs=8,
        seed=1,
        vec_env_cls=SubprocVecEnv
    )
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    print(env.reset().shape)
    # eval_env = make_vec_env(env_name, n_envs=1)
    # env = VecNormalize(env, norm_reward=False, clip_reward=100)
    # eval_env = VecNormalize(eval_env, training=False, norm_reward=False, clip_reward=100)
    # eval_env.obs_rms = env.obs_rms
    
    
    model = RND(
        env=env,
        policy='CnnPolicy',
        learning_rate=linear_schedule(2.5e-4),
        vf_coef=0.5,
        ent_coef=0.001,
        tensorboard_log='./Breakout',
        clip_range=linear_schedule(0.1),
        n_steps=128,
        n_epochs=4,
        batch_size=256,
        verbose=1,
    )
    


    model.learn(
        total_timesteps=int(10_000_000),
        tb_log_name=f'RND_Breakout',
        reset_num_timesteps=True,
        progress_bar=True,
    )
