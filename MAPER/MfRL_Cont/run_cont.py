"""
Policy-based Reinforcement Learning for Continuous Atari Environments
Currently, SAC, TD3, ans DDPG are available
"""
import copy
import importlib
import random
import time

import gym
import numpy as np
import torch

import algorithms.common.env.utils as env_utils
from args import args


torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)


# configurations
def main():
    """Main."""
    # env initialization
    env_test, env_train = generate_envs()
    #env_test.seed(args.seed+100)


    # run
    module_path = "trainers." + "trainer"

    envs = [env_train, env_test]

    trainers = importlib.import_module(module_path)
    trainers.run(envs, args)


def generate_envs():
    env_train = gym.make(args.envname)
    env_train = env_utils.set_env(env_train, args)
    env_train.action_space.seed(args.seed)
    env_test = copy.deepcopy(env_train)


    return env_test, env_train


if __name__ == "__main__":
    training_begin = time.time()
    main()
    training_end = time.time()
    print("{} seconds".format(training_end-training_begin))
