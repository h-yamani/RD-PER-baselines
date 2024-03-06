import numpy as np
import torch
import gym
import argparse
import os
import time

import utils
import TD3
import LAP_TD3
import PAL_TD3
import PER_TD3
import csv


# Runs policy for X episodes and returns average reward
def eval_policy(policy, env_name, seed, eval_episodes=10):
	env.action_space.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	eval_env = gym.make(env_name)

	avg_reward = 0.0

	for _ in range(eval_episodes):
		state,_ = eval_env.reset()
		episode_reward = 0
		done = False
		truncated = False

		while not (done or truncated):
			action = policy.select_action(state, test=True)
			next_state, reward, done, truncated, _ = eval_env.step(action)
			episode_reward += reward
			state = next_state

		avg_reward += episode_reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


# Pendulum-v1, BipedalWalker-v3, HalfCheetah-v4,  Humanoid-v4, Swimmer-v4, Ant-v2, InvertedPendulum-v4, Walker2d-v4, LunarLander-v2
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--algorithm", default="LAP_TD3")			# Algorithm nameu
	parser.add_argument("--env", default="Hopper-v4")			# OpenAI gym environment name
	parser.add_argument("--seed", default=571, type=int)				# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')
	parser.add_argument("--eval_freq", default=1e4, type=int)		# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)	# Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)				# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=32, type=int)		# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)					# Discount factor
	parser.add_argument("--tau", default=0.005)						# Target network update rate
	parser.add_argument("--policy_noise", default=0.2)				# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)				# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)		# Frequency of delayed policy updates
	parser.add_argument("--alpha", default=0.4)						# Priority = TD^alpha (only used by LAP/PAL)
	parser.add_argument("--min_priority", default=1, type=int)		# Minimum priority (set to 1 in paper, only used by LAP/PAL)
	args = parser.parse_args()

	file_name = "%s_%s_%s" % (args.algorithm, args.env, str(args.seed))
	print("---------------------------------------")
	print(f"Settings: {file_name}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")
	if not os.path.exists(f"./results/csv"):
		os.makedirs(f"./results/csv")

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
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq
	}

	# Initialize policy and replay buffer
	if args.algorithm == "TD3": 
		policy = TD3.TD3(**kwargs)
		replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	elif args.algorithm == "PER_TD3": 
		policy = PER_TD3.PER_TD3(**kwargs)
		replay_buffer = utils.PrioritizedReplayBuffer(state_dim, action_dim)
	
	kwargs["alpha"] = args.alpha
	kwargs["min_priority"] = args.min_priority

	if args.algorithm == "LAP_TD3": 
		policy = LAP_TD3.LAP_TD3(**kwargs)
		replay_buffer = utils.PrioritizedReplayBuffer(state_dim, action_dim)

	elif args.algorithm == "PAL_TD3":
		policy = PAL_TD3.PAL_TD3(**kwargs)
		replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, _ = env.reset(seed=args.seed)
	done = False
	truncated = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	historical_reward = {"step": [], "episode_reward": []}
	startTime = time.time()

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			print(f"Running Exploration Steps {t}/{args.start_timesteps}")
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, truncated, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < 1000 else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps: #>=
			policy.train(replay_buffer, args.batch_size)

		if done or truncated:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}  Episode time : {time.time()- startTime}")
			startTime = time.time()
			historical_reward["step"].append(t)
			historical_reward["episode_reward"].append(episode_reward)

			rl_data = historical_reward

			# Open the .csv file in write mode
			with open(f"./results/{file_name}.csv", mode='w', newline='') as file:
				# Create a CSV writer object
				writer = csv.writer(file)

				# Write the header row
				writer.writerow(["Step", "episode_reward"])

				# Write each data row
				for step, episode_reward in zip(rl_data["step"], rl_data["episode_reward"]):
					writer.writerow([step + 1, episode_reward])

			state, _ = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save("./results/%s" % (file_name), evaluations)
			# Save evaluations to a CSV file
			file_path = f"./results/csv/{file_name}_eval_.csv"
			with open(file_path, mode='w', newline='') as file:
				writer = csv.writer(file)
				# Write the header row
				writer.writerow(["Step", "episode_reward"])
				# Write evaluation results

				for step, episode_reward in zip(historical_reward["step"], evaluations):
					writer.writerow([(step -999)*10, episode_reward])