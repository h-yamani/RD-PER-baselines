import argparse
import itertools
import os
import socket

import gym
import numpy as np
import torch

from PER_SAC import PER_SAC
from SAC import SAC
from LA3P_SAC import LA3P_SAC
import utils
import csv
import time


# SAC tuned hyper-parameters are imported from the original paper: https://arxiv.org/abs/1801.01290
def reward_scale_dict(args):
    if args.env == "Humanoid-v2":
        args.reward_scale = 1
    else:
        args.reward_scale = 1

    return args


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate_policy(agent, env_name, seed, eval_episodes=10):
    # Set seeds
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    eval_env = gym.make(env_name)

    avg_reward = 0.

    for _ in range(eval_episodes):
        state,_ = eval_env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)

            next_state, reward, done,truncated, _ = eval_env.step(action)
            episode_reward += reward

            state = next_state

        avg_reward += episode_reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LA3P + Soft Actor-Critic (SAC)')

    parser.add_argument('--policy', default="LA3P_SAC", help='Algorithm (default: LA3P_SAC)')
    parser.add_argument('--policy_type', default="Gaussian", help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--env', default="HalfCheetah-v4", help='OpenAI Gym environment name')
    parser.add_argument('--seed', type=int, default=571, help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 0)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ordinal for multi-GPU computers (default: 0)')
    parser.add_argument('--start_steps', type=int, default=1000, metavar='N', help='Number of exploration time steps sampling random actions (default: 25000)')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='Size of the experience replay buffer (default: ''1000000)')
    parser.add_argument("--prioritized_fraction", default=0.5, type=float, help='Fraction of prioritized sampled batch of transitions')
    parser.add_argument('--eval_freq', type=int, default=1e4, metavar='N', help='evaluation period in number of time steps (default: 1000)')
    parser.add_argument('--num_steps', type=int, default=1000000, metavar='N', help='Maximum number of steps (default: 1000000)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='Batch size (default: 256)')
    parser.add_argument('--hard_update', action="store_false",  help='Hard update the target networks (default: True)') #metavar='G',
    parser.add_argument('--train_freq', type=int, default=1, metavar='N', help='Frequency of the training (default: 1)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N', help='Model updates per training time step (default: 1)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N', help='Number of critic function updates per training time step (default: 1)')
    parser.add_argument('--alpha', type=float, default=0.2, help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)') #metavar='G', #0.2
    parser.add_argument('--automatic_entropy_tuning', action="store_true", help='Automatically adjust α (default: True)') #metavar='G',
    parser.add_argument('--reward_scale', type=float, default=1.0, help='Scale of the environment rewards (default: 5)') #, metavar='N'
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='Discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G', help='Learning rate in soft/hard updates of the target networks (default: 0.005)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='G', help='Learning rate (default: 0.0003)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N', help='Hidden unit size in neural networks (default: 256)')

    args = parser.parse_args()

    # Adjust the hyper-parameters with respect to the environment
    args = reward_scale_dict(args)

    # Target update specific parameters
    print(f"\nEnvironment: {args.env}\n")

    print(f"Policy type: {args.policy_type}\n")

    if args.hard_update:
        print(f"Update: HARD\n")
    else:
        print(f"Update: SOFT\n")

    print(f"Tau: {args.tau}")
    print(f"Target update interval: {args.target_update_interval}")
    print(f"Updates per step: {args.updates_per_step}\n")

    print(f"Reward scale: {args.reward_scale}\n")
    print(f"Start time steps: {args.start_steps}\n")

    file_name = f"{args.policy}_{args.env}_{args.seed}"

    if not os.path.exists(f"./results/{args.prioritized_fraction}"):
        os.makedirs(f"./results/{args.prioritized_fraction}")
    if not os.path.exists(f"./results/csv"):
        os.makedirs(f"./results/csv")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Environment
    env = gym.make(args.env)

    # Set seeds
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Agent
    if args.policy == "LA3P_SAC":
        agent = LA3P_SAC(env.observation_space.shape[0], env.action_space, args, device)
    if args.policy == "PER_SAC":
        agent = PER_SAC(env.observation_space.shape[0], env.action_space, args, device)
    else:
        agent = SAC(env.observation_space.shape[0], env.action_space, args, device)

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    # Memory
    if "LA3P" in args.policy:
        replay_buffer = utils.ActorPrioritizedReplayBuffer(state_dim, action_dim, max_size=args.buffer_size, device=device)
    if args.policy == "PER_SAC":
        replay_buffer = utils.PrioritizedReplayBuffer(state_dim, action_dim, max_size=args.buffer_size, device=device)
    else:
        replay_buffer = utils.ExperienceReplayBuffer(state_dim, action_dim, max_size=args.buffer_size, device=device)

    # Training Loop
    total_time_steps = 0
    updates = 0

    # Evaluate untrained policy
    evaluations = [evaluate_policy(agent, args.env, args.seed)]
    historical_reward = {"step": [], "episode_reward": []}

    state, _ = env.reset(seed=args.seed)
    done = False
    episode_reward = 0
    episode_steps = 0

    done = False
    truncated = False
    episode_reward = 0
    episode_num= 0
    episode_timesteps = 0

    done = False
    truncated = False
    startTime = time.time()
    for t in range(int(args.num_steps)):
        episode_timesteps += 1

        if t < args.start_steps:
            print(f"Running Exploration Steps {t}/{args.start_steps}")
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, evaluate=False)

        next_state, reward, done, truncated, _ = env.step(action)

        episode_reward += reward

        reward *= args.reward_scale

        done_bool = float(done) if episode_timesteps <= 1000 else 0

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        if t >= args.start_steps:
            if "LA3P_SAC" in args.policy:
                agent.update_parameters(replay_buffer, updates, args.prioritized_fraction, args.batch_size)
            else:
                agent.update_parameters(replay_buffer, updates, args.batch_size)
        if done or truncated:
            print(
                f"Total T: {t+1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps}" f" Reward: {episode_reward:.3f} Episode time : {time.time()- startTime}")
            startTime = time.time()
            historical_reward["step"].append(t+1)
            historical_reward["episode_reward"].append(episode_reward)

            # Specify the file name for the .csv file

            rl_data = historical_reward

            # Open the .csv file in write mode
            with open(f"./results/{args.prioritized_fraction}/{file_name}.csv", mode='w', newline='') as file:
                # Create a CSV writer object
                writer = csv.writer(file)

                # Write the header row
                writer.writerow(["Step", "episode_reward"])

                # Write each data row
                for step, episode_reward in zip(rl_data["step"], rl_data["episode_reward"]):
                    writer.writerow([step, episode_reward])
            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1


            # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(evaluate_policy(agent, args.env, args.seed))
            np.save(f"./results/{args.prioritized_fraction}/{file_name}", evaluations)
            # Save evaluations to a CSV file
            file_path = f"./results/csv/{file_name}_eval_.csv"
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the header row
                writer.writerow(["Step", "episode_reward"])
                # Write evaluation results

                for step, episode_reward in zip(historical_reward["step"], evaluations):
                    writer.writerow([(step -1000)*10, episode_reward])



