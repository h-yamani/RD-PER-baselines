import argparse
from typing import Callable, List

import gym

from algorithms.common.env.multiprocessing_env import SubprocVecEnv


def set_env(
        env: gym.Env, args: argparse.Namespace, env_wrappers: List[gym.Wrapper] = None
) -> gym.Env:
    """Set environment according to user's config."""
    # print(self.args.max_episode_steps)
    if env._max_episode_steps:
        args.max_episode_steps = env._max_episode_steps
    else:
        env._max_episode_steps = args.max_episode_steps

    if env_wrappers:
        for env_wrapper in env_wrappers:
            env = env_wrapper(env)
    print(env._max_episode_steps)
    return env


def env_generator(
        env_name: str, args: argparse.Namespace, env_wrappers: List[gym.Wrapper] = None
) -> Callable:
    """Return env creating function (with normalizers)."""

    def _thunk(rank: int):
        env = gym.make(env_name)
        env.seed(args.seed + rank + 1)
        env = set_env(env, args, env_wrappers)
        return env

    return _thunk


def make_envs(env_gen: Callable, n_envs: int = 8) -> SubprocVecEnv:
    """Make multiple environments running on multiprocssors."""
    envs = [env_gen(i) for i in range(n_envs)]
    envs = SubprocVecEnv(envs)
    return envs
