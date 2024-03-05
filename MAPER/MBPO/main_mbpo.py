import argparse
import logging
import os
import pathlib
from itertools import count

import gym
import numpy as np
# from tf_models.constructor import construct_model, format_samples_for_training
import pandas as pd
import torch
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sac.replay_memory import ReplayMemory, PrioritizedReplayMemory
from sac.sac import SAC
from sample_env import EnvSampler

os.chdir(pathlib.Path(__file__).parent.absolute())


def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env_name', default="Hopper-v2",
                        help='Mujoco Gym environment (default: Hopper-v2)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(?) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')

    parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                        help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                        help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                        help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')

    parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                        help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                        help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                        help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                        help='steps per epoch')
    parser.add_argument('--rollout_min_epoch', type=int, default=20, metavar='A',
                        help='rollout min epoch')
    parser.add_argument('--rollout_max_epoch', type=int, default=150, metavar='A',
                        help='rollout max epoch')
    parser.add_argument('--rollout_min_length', type=int, default=1, metavar='A',
                        help='rollout min length')
    parser.add_argument('--rollout_max_length', type=int, default=15, metavar='A',
                        help='rollout max length')
    parser.add_argument('--num_epoch', type=int, default=1000, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                        help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                        help='exploration steps initially')
    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')
    parser.add_argument('--debug', default=False, action="store_true",
                        help='Test Mode')
    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    parser.add_argument('--replay_type', default="MaPER",
                        help='Replay_Type')
    parser.add_argument('--suffix', default=9, type=int,
                        help='suffix')
    parser.add_argument('--alpha_for_PER', default=0.5, type=float,
                        help='alpha_for_PER')
    parser.add_argument('--beta_for_PER', default=1.0, type=float,
                        help='beta_for_PER')
    parser.add_argument('--save_folder', default=None, type=str,
                        help='save_folder')
    parser.add_argument('--eval_freq', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--eval_episodes', default=1, type=int, help='evaluation episode #')
    parser.add_argument('--temp', default=50.0, type=float, help='temperature')
    return parser.parse_args()


def train(args, env_sampler, predict_env, agent, env_pool, model_pool):
    if args.save_folder is None:
        args.save_folder = './exps/'
    base_filename = f'{args.env_name}-{args.replay_type}-{args.suffix}'
    rollout_length = 1
    sum_rewards = []
    total_steps = []
    total_times = []
    total_step = 0
    exploration_before_start(args, env_sampler, env_pool, agent)
    env_sampler.current_state = None
    sum_reward = 0
    done = False
    while not done:
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
        sum_reward += reward
    logging.info("total_step: " + str(total_step) + " " + str(sum_reward))
    print("total_step: " + str(total_step) + ", Cumulative Reward: " + str(sum_reward))
    if total_step == 0:
        sum_rewards.append(sum_reward)
        total_steps.append(total_step)
        dp = pd.DataFrame([])
        dp['sum_rewards'] = np.nan
        dp['total_steps'] = np.nan
        dp['sum_rewards'] = pd.Series(sum_rewards)
        dp['total_steps'] = pd.Series(total_steps)
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        csv_path = os.path.join(args.save_folder, base_filename + f'.csv')
        dp.to_csv(csv_path)
    counter = 0
    for epoch_step in range(args.num_epoch):
        start_step = total_step
        train_policy_steps = 0
        for i in count():
            cur_step = total_step - start_step
            beta = np.min([args.beta_for_PER + total_step * (1.0 - args.beta_for_PER) / (args.num_epoch * 1000), 1.0])
            if cur_step >= start_step + args.epoch_length and len(env_pool) > args.min_pool_size:
                break

            if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                print("Train Predict Model Start!")
                train_predict_model(args, env_pool, predict_env, agent=agent)
                print("Train Predict Model Finish!")
                new_rollout_length = set_rollout_length(args, epoch_step)
                if rollout_length != new_rollout_length:
                    rollout_length = new_rollout_length
                    model_pool = resize_model_pool(args, rollout_length, model_pool)

                rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length, beta)

            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
            env_pool.push(cur_state, action, reward, next_state, done)

            if len(env_pool) > args.min_pool_size:
                print(f"{total_step}, {args.eval_freq}: Train Agent start!")
                train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool,
                                                           model_pool, agent, beta)
                print(f"{total_step}, {args.eval_freq}: Train Agent Finish!")
            if total_step > 0 and total_step % args.eval_freq == 0:
                print("Eval Start")
                env_sampler.current_state = None
                sum_reward = 0
                done = False
                while not done:
                    cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
                    sum_reward += reward
                print("Eval Finished")
                sum_rewards.append(sum_reward)
                total_steps.append(total_step)
                print("total_step: " + str(total_step) + ", Cumulative Reward: " + str(sum_reward))
                dp = pd.DataFrame([])
                dp['sum_rewards'] = np.nan
                dp['total_steps'] = np.nan
                dp['sum_rewards'] = pd.Series(sum_rewards)
                dp['total_steps'] = pd.Series(total_steps)

                if not os.path.exists(args.save_folder):
                    os.makedirs(args.save_folder)
                csv_path = os.path.join(args.save_folder, base_filename + f'.csv')
                print("csv_path: ", csv_path)
                print("results: ", sum_rewards)
                dp.to_csv(csv_path)
            if done:
                counter += 1
            total_step += 1


def exploration_before_start(args, env_sampler, env_pool, agent):
    for i in range(args.init_exploration_steps):
        if i % 100 == 0:
            print(f'exploration_before_start={i}/{args.init_exploration_steps}')
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
        env_pool.push(cur_state, action, reward, next_state, done)


def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                              / (args.rollout_max_epoch - args.rollout_min_epoch) * (
                                      args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env, agent=None):
    # Get all samples from environment
    if 'MaPER' in args.replay_type:
        state, action, reward, next_state, done = env_pool.dump(len(env_pool))
    else:
        state, action, reward, next_state, done = env_pool.sample(len(env_pool))

    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    predict_env.model.train(inputs, labels, agent, batch_size=256, holdout_ratio=0.2)


def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    if 'MaPER' in args.replay_type:
        sample_all = model_pool.return_all()
        new_model_pool = PrioritizedReplayMemory(args, new_pool_size)
        new_model_pool._max_priority = model_pool._max_priority
        new_model_pool.buffer = sample_all
        data1 = model_pool._it_sum.value
        data2 = model_pool._it_min.value
        new_model_pool._it_sum.value[:len(data1)] = data1
        new_model_pool._it_min.value[:len(data2)] = data2
    else:
        sample_all = model_pool.return_all()
        new_model_pool = ReplayMemory(args, new_pool_size)
        new_model_pool.push_batch(sample_all)

    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length, beta):
    if 'MaPER' in args.replay_type:
        state, action, reward, next_state, done = env_pool.dump(args.rollout_batch_size)
    else:
        state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(state, action)
        # TODO: Push a batch of samples
        model_pool.push_batch(
            [(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent, beta):
    if total_step % args.train_every_n_steps > 0:
        return 0

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size
        env_indices = None
        model_indices = None
        if 'MaPER' in args.replay_type:
            env_state, env_action, env_reward, env_next_state, env_done, env_weight, env_indices = env_pool.sample(
                int(env_batch_size), beta)
        else:
            env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))
            env_weight = np.ones_like(env_reward)
        if model_batch_size > 0 and len(model_pool) > 0:
            if 'MaPER' in args.replay_type:
                model_state, model_action, model_reward, model_next_state, model_done, model_weight, model_indices = model_pool.sample(
                    int(model_batch_size), beta)
            else:
                model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample(
                    int(model_batch_size))
                model_weight = np.ones_like(model_reward)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate(
                (env_state, model_state), axis=0), \
                                                                                    np.concatenate(
                                                                                        (env_action, model_action),
                                                                                        axis=0), np.concatenate(
                (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                                                                                    np.concatenate((env_next_state,
                                                                                                    model_next_state),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
            batch_weight = np.concatenate([env_weight.reshape(-1), model_weight.reshape(-1)], axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done
            batch_weight = env_weight

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        priorities = agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done),
                                             args.policy_train_batch_size, i, batch_weight)
        if env_indices is not None:
            env_pool.update_priorities(env_indices, priorities[:int(env_batch_size)])
        if model_indices is not None:
            model_pool.update_priorities(model_indices, priorities[int(env_batch_size):])

    return args.num_train_repeat


from gym.spaces import Box


class SingleEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleEnvWrapper, self).__init__(env)
        obs_dim = env.observation_space.shape[0]
        obs_dim += 2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]  # Need this in the obs for determining when to stop
        obs = np.append(obs, [torso_height, torso_ang])

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]
        obs = np.append(obs, [torso_height, torso_ang])
        return obs


def main(args=None):
    # Initial environment

    if args is None:
        args = readParser()

    print(args)
    env = gym.make(args.env_name)
    print("Current Working Directory={}".format(os.getcwd()))

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    args.state_size = state_size
    args.action_size = action_size
    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    if args.model_type == 'pytorch':
        env_model = EnsembleDynamicsModel(args, args.num_networks, args.num_elites, state_size, action_size,
                                          args.reward_size,
                                          args.pred_hidden_size,
                                          use_decay=args.use_decay)
    else:
        env_model = construct_model(obs_dim=state_size, act_dim=action_size, hidden_dim=args.pred_hidden_size,
                                    num_networks=args.num_networks,
                                    num_elites=args.num_elites)
    # Predict environments
    predict_env = PredictEnv(args, env_model, args.env_name, args.model_type)
    if 'MaPER' in args.replay_type:
        env_pool = PrioritizedReplayMemory(args, args.replay_size)
    else:
        env_pool = ReplayMemory(args, args.replay_size)

    # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    if 'MaPER' in args.replay_type:
        model_pool = PrioritizedReplayMemory(args, new_pool_size)
    else:
        model_pool = ReplayMemory(args, new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(env)
    train(args, env_sampler, predict_env, agent, env_pool, model_pool)


if __name__ == '__main__':
    main()
