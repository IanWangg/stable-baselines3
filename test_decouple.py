import gym
import stable_baselines3
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.rnd.decouple import DecoupleDQN
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
    
    exploit_env = make_atari_env(
        env_name, 
        # wrapper_class=sparse_reward_wrapper,
        # wrapper_kwargs={'montezuma': False},
        n_envs=1,
        seed=1,
        vec_env_cls=SubprocVecEnv
    )
    exploit_env = VecTransposeImage(exploit_env)
    exploit_env = VecFrameStack(exploit_env, n_stack=4)

    eval_env = make_atari_env(
        env_name, 
        n_envs=5,
        seed=1,
        vec_env_cls=SubprocVecEnv
    )
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    
    model = DecoupleDQN(
        env=env,
        policy='CnnPolicy',
        exploit_env=exploit_env,
        eval_env=eval_env,
        exploit_iterations_per_explore_session=int(1e4),
        # learning_rate=linear_schedule(1e-4),
        learning_rate=1e-4,
        vf_coef=0.5,
        ent_coef=0.001,
        int_gamma=0.99,
        gamma=0.99,
        clip_range=linear_schedule(0.1),
        n_steps=128,
        n_epochs=4,

        learning_starts=0,
        batch_size=256,
        buffer_size=1000000,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1,
        exploration_final_eps=0.01,
        tensorboard_log='./Breakout',
        verbose=1,
    )
    


    model.learn(
        total_timesteps=int(10_000_000),
        tb_log_name=f'Decouple_Breakout',
        reset_num_timesteps=True,
        progress_bar=True,
    )
