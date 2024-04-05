import argparse
import os
import socket

import gym
import numpy as np
import torch

import TD3
import LA3P_TD3
import utils
import csv


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate_policy(agent, env_name, seed, eval_episodes=10):
    # Set seeds
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eval_env = gym.make(env_name)
    # eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state,_ = eval_env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state)

            next_state, reward, done,truncated, _ = eval_env.step(action)
            episode_reward += reward

            state = next_state

        avg_reward += episode_reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    return avg_reward
# Pendulum-v1, BipedalWalker-v3, HalfCheetah-v4,  Humanoid-v4, Swimmer-v4, Ant-v4, InvertedPendulum-v4, Walker2d-v4, LunarLander-v2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LA3P + Twin Delayed Deep Deterministic Policy Gradient (TD3)')

    parser.add_argument("--policy", default="LA3P_TD3", help='Algorithm (default: LA3P_TD3)')
    parser.add_argument("--env", default="HalfCheetah-v4", help='OpenAI Gym environment name')
    parser.add_argument("--seed", default=10, type=int, help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 0)')
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')
    parser.add_argument("--start_time_steps", default=0, type=int, metavar='N', help='Number of exploration time steps sampling random actions (default: 25000)')
    parser.add_argument("--buffer_size", default=1000000, type=int, help='Size of the experience replay buffer (default: 1000000)')
    parser.add_argument("--prioritized_fraction", default=0.5, type=float, help='Fraction of prioritized sampled batch of transitions')
    parser.add_argument("--eval_freq", default=1e4, metavar='N', help='Evaluation period in number of time steps (default: 1000)')
    parser.add_argument("--max_time_steps", default=1000000, type=int, metavar='N', help='Maximum number of steps (default: 1000000)')
    parser.add_argument("--exploration_noise", default=0.1, metavar='G', help='Std of Gaussian exploration noise')
    parser.add_argument("--batch_size", default=256, metavar='N', help='Batch size (default: 256)')
    parser.add_argument("--discount", default=0.99, metavar='G', help='Discount factor for reward (default: 0.99)')
    parser.add_argument("--tau", default=0.005, type=float, metavar='G', help='Learning rate in soft/hard updates of the target networks (default: 0.005)')
    parser.add_argument("--policy_noise", default=0.2, metavar='G', help='Noise added to target policy during critic update')
    parser.add_argument("--noise_clip", default=0.5, metavar='G', help='Range to clip target policy noise')
    parser.add_argument("--policy_freq", default=2, type=int, metavar='N', help='Frequency of delayed policy updates')
    parser.add_argument("--save_model", action="store_true", help='Save model and optimizer parameters')
    parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')

    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists(f"./results/{args.prioritized_fraction}"):
        os.makedirs(f"./results/{args.prioritized_fraction}")
    if not os.path.exists(f"./results/csv") :
        os.makedirs(f"./results/csv")
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise,
        "noise_clip": args.noise_clip,
        "policy_freq": args.policy_freq,
        "device": device
    }

    # Initialize the algorithm
    if args.policy == "LA3P_TD3":
        agent = LA3P_TD3.LA3P_TD3(**kwargs)
    else:
        agent = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        agent.load(f"./models/{policy_file}")

    if "LA3P" in args.policy:
        replay_buffer = utils.ActorPrioritizedReplayBuffer(state_dim, action_dim, max_size=args.buffer_size, device=device)
    else:
        replay_buffer = utils.ExperienceReplayBuffer(state_dim, action_dim, max_size=args.buffer_size, device=device)

    # Evaluate the untrained policy
    evaluations = [evaluate_policy(agent, args.env, args.seed)]

    state, _ = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    episode_time_steps = 0
    episode_num = 0

    historical_reward = {"step": [], "episode_reward": []}

    for t in range(int(args.max_time_steps)):
        episode_time_steps += 1

        # Sample action from the action space or policy
        if t < args.start_time_steps:
            print(f"Running Exploration Steps {t}/{args.start_time_steps}")
            action = env.action_space.sample()
        else:
            action = (
                    agent.select_action(state)
                    + np.random.normal(0, max_action * args.exploration_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Take the selected action
        next_state, reward, done, truncated, _ = env.step(action)
        done_bool = float(done) if episode_time_steps <= 1000 else 0

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train the agent after collecting a sufficient number of samples
        if t >= args.start_time_steps:
            if "LA3P" in args.policy:
                agent.update_parameters(replay_buffer, args.prioritized_fraction, args.batch_size)
            else:
                agent.update_parameters(replay_buffer, args.batch_size)

        if done or truncated:
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_time_steps} Reward: {episode_reward:.3f}")

            historical_reward["step"].append(t + 1)
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

                    writer.writerow([step , episode_reward])

            # Reset the environment
            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_time_steps = 0
            episode_num += 1

        # Evaluate the agent over a number of episodes
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

            if args.save_model:
                agent.save(f"./models/{file_name}")
