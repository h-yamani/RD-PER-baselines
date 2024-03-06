import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from algorithms.common.helper_functions import make_one_hot


class AgentBase(nn.Module):
    def __init__(self):
        self.device = device
        super().__init__()
        if not "MULTIPLE_LEARN" in self.hyper_params:
            self.hyper_params["MULTIPLE_LEARN"] = 1
        if self.args.savealgname == '':
            self.fname = "{}-{}-{}.csv".format(self.args.envname, self.args.algo, self.args.seed)
        else:
            self.fname = f'{self.args.envname}-{self.args.savealgname}-{self.args.seed}.csv'
        pathlib.Path('./exps').mkdir(exist_ok=True)
        pathlib.Path('./exps/' + self.args.savefolder).mkdir(exist_ok=True)
        pathlib.Path('./exps/evaluation').mkdir(exist_ok=True)
        self.evaluation_save_path = os.path.join('./exps/evaluation', 'eval_' + self.fname)
        self.save_path = os.path.join('./exps/' + self.args.savefolder, self.fname)
        my_dict = self.args.__dict__
        with open(self.save_path.replace(".csv", ".txt"), "w") as f:
            for key in my_dict:
                elt = "{} : {}\n".format(key, my_dict[key])
                print(elt)
                f.write(elt)
            for key in self.hyper_params:
                elt = "{} : {}\n".format(key, self.hyper_params[key])
                print(elt)
                f.write(elt)
        self.total_step = 0
        self.episode_step = 0
        self.update_step = 0
        self.i_episode = 0
        self.fraction = 0.0
        self.total_scores = []
        self.total_rew_diff1 = []
        self.total_rew_diff2 = []
        self.total_tran_diff1 = []
        self.total_tran_diff2 = []
        self.total_eval_td1 = []
        self.total_eval_td2 = []
        self.true_returns = []
        self.total_q1 = []
        self.total_q2 = []
        self.total_stdev = []
        self.total_steps = []
        self.total_mean_td = []
        self.total_mean_q = []
        self.total_mean_r = []
        self.total_mean_s = []
        self.total_mean_a = []
        self.total_mean_o = []
        self.total_num = []
        self.sum_of_rewards = 0.0
        self.test_count = 1
        self.timestep = 0
        self.num_mb = 0
        self.resetmean()

    def averagemean(self):
        self.mean_td /= self.hyper_params["MULTIPLE_LEARN"]
        self.mean_q /= self.hyper_params["MULTIPLE_LEARN"]
        self.mean_r /= self.hyper_params["MULTIPLE_LEARN"]
        self.mean_s /= self.hyper_params["MULTIPLE_LEARN"]

    def resetmean(self):
        self.mean_td = 0.0
        self.mean_q = 0.0
        self.mean_r = 0.0
        self.mean_s = 0.0
        self.mean_a = 0.0
        self.mean_o = 0.0
        self.num_mb = 0

    def concat(self,
               in_1: torch.Tensor, in_2: torch.Tensor, n_category: int = -1
               ) -> torch.Tensor:
        """Concatenate state and action tensors properly depending on the action."""
        in_2 = make_one_hot(in_2, n_category) if n_category > 0 else in_2

        if len(in_2.size()) == 1:
            in_2 = in_2.unsqueeze(0)

        in_concat = torch.cat((in_1, in_2), dim=-1)

        return in_concat

    def save_results(self):
        self.true_returns.append(np.mean(self.episode_returns))
        self.total_q1.append(np.mean(self.q1))
        self.total_q2.append(np.mean(self.q2))
        self.total_steps.append(self.total_step)
        self.total_scores.append(np.mean(self.sum_of_rewards))
        self.total_stdev.append(np.std(self.sum_of_rewards))
        self.total_mean_q.append(self.mean_q)
        self.total_mean_td.append(self.mean_td)
        self.total_mean_r.append(self.mean_r)
        self.total_mean_s.append(self.mean_s)
        self.total_mean_a.append(self.mean_a)
        self.total_mean_o.append(self.mean_o)
        self.total_num.append(self.num_mb)

        self.total_rew_diff1.append(np.mean(self.episode_reward_diff1))
        self.total_rew_diff2.append(np.mean(self.episode_reward_diff2))
        self.total_tran_diff1.append(np.mean(self.episode_tran_diff1))
        self.total_tran_diff2.append(np.mean(self.episode_tran_diff2))
        self.total_eval_td1.append(np.mean(self.episode_td1))
        self.total_eval_td2.append(np.mean(self.episode_td2))

        dataframe1 = pd.DataFrame([])
        #dataframe1['Step'] = np.nan
        #dataframe1['episode_reward'] = np.nan
        dataframe = pd.DataFrame([])
        dataframe['all_scores'] = np.nan
        dataframe['all_stdev'] = np.nan
        dataframe['total_steps'] = np.nan
        dataframe['mean_q'] = np.nan
        dataframe['mean_td'] = np.nan
        dataframe['mean_s'] = np.nan
        dataframe['mean_a'] = np.nan
        dataframe['mean_o'] = np.nan
        dataframe['num_mb'] = np.nan
        dataframe['true_returns'] = np.nan
        dataframe['eval_q1'] = np.nan
        dataframe['eval_q2'] = np.nan
        dataframe['total_rew_diff1'] = np.nan
        dataframe['total_rew_diff2'] = np.nan
        dataframe['total_tran_diff1'] = np.nan
        dataframe['total_tran_diff2'] = np.nan
        dataframe['total_td1_diff1'] = np.nan
        dataframe['total_td2_diff2'] = np.nan
        dataframe['true_returns'] = pd.Series(np.array(self.true_returns))
        dataframe['eval_q1'] = pd.Series(np.array(self.total_q1))
        dataframe['eval_q2'] = pd.Series(np.array(self.total_q2))
        dataframe['all_scores'] = pd.Series(np.array(self.total_scores))
        dataframe['all_stdev'] = pd.Series(np.array(self.total_stdev))
        dataframe['total_steps'] = pd.Series(np.array(self.total_steps))
        dataframe['mean_q'] = pd.Series(np.array(self.total_mean_q))
        dataframe['mean_td'] = pd.Series(np.array(self.total_mean_td))
        dataframe['mean_r'] = pd.Series(np.array(self.total_mean_r))
        dataframe['mean_s'] = pd.Series(np.array(self.total_mean_s))
        dataframe['mean_a'] = pd.Series(np.array(self.total_mean_a))
        dataframe['mean_o'] = pd.Series(np.array(self.total_mean_o))

        dataframe['total_rew_diff1'] = pd.Series(np.array(self.total_rew_diff1))
        dataframe['total_rew_diff2'] = pd.Series(np.array(self.total_rew_diff2))
        dataframe['total_tran_diff1'] = pd.Series(np.array(self.total_tran_diff1))
        dataframe['total_tran_diff2'] = pd.Series(np.array(self.total_tran_diff2))
        dataframe['total_td1_diff1'] = pd.Series(np.array(self.total_eval_td1))
        dataframe['total_td2_diff2'] = pd.Series(np.array(self.total_eval_td2))

        dataframe['total_num'] = pd.Series(np.array(self.total_num))
        dataframe1['Step'] = pd.Series(np.array(self.total_steps))
        dataframe1['episode_reward'] = pd.Series(np.array(self.total_scores))
        dataframe.to_csv(self.save_path)
        dataframe1.to_csv(self.evaluation_save_path, header=True, index=False)

    def interimtest(self):
        """Common test routine."""
        self.sum_of_rewards = []
        self.episode_returns = []
        self.episode_reward_diff1 = []
        self.episode_reward_diff2 = []
        self.episode_tran_diff1 = []
        self.episode_tran_diff2 = []
        self.episode_td1 = []
        self.episode_td2 = []
        self.q1 = []
        self.q2 = []
        total_data = {}
        total_data['fig'] = []
        total_data['fake_rew'] = []
        total_data['real_rew'] = []
        total_data['q'] = []

        for i_episode in range(self.args.interim_test_num):
            state, _ = self.env_test.reset()
            done = False
            sum_of_rewards = 0.0
            step = 0
            self.episode_step_test = 0
            episode_return = 0
            diff_rewards1 = []
            diff_rewards2 = []
            diff_states1 = []
            diff_states2 = []
            diff_td1 = []
            diff_td2 = []
            while not done:
                self.episode_step_test += 1
                action = self.select_action_test(state)
                next_state, reward, done, _ = self.test_step(action)
                state = next_state
                sum_of_rewards += reward
                episode_return += reward * np.power(0.98, step)
                step = step + 1
                if self.episode_step_test >= self.args.max_episode_steps:
                    break
            self.episode_returns.append(episode_return)
            self.sum_of_rewards.append(sum_of_rewards)
            print(
                "[INFO] Test %d\t total step: %d\ttotal score: %d, step: %d" %
                (i_episode, self.total_step, sum_of_rewards, self.episode_step_test)
            )



    def train_step(self, action):
        """Take an action and return the response of the env."""
        next_state, reward, done, truncated, info = self.env_train.step(action)
        done_bool = (
            True if self.episode_step == self.args.max_episode_steps else done
        )
        # done_pool = True, or False
        transition = [self.curr_state, action, reward, next_state, done_bool]

        return next_state, reward, done_bool, transition, info

    def test_step(self, action):
        """Take an action and return the response of the env."""
        next_state, reward, done, truncated, info = self.env_test.step(action)
        done_bool = (
            True if self.episode_step_test == self.args.max_episode_steps else done
        )

        return next_state, reward, done_bool, info

    def train(self):
        """Train the agent."""
        # logger
        historical_reward = {"step": [], "episode_reward": []}
        self.i_episode = 0
        end_training = False
        self.tstep = 0
        maxer = 1
        while True:
            t_begin = time.time()
            self.i_episode += 1
            state, _ = self.env_train.reset()
            done = False
            sum_of_rewards = 0
            self.episode_step = 0
            self.timestep = 0
            rewards = []

            while not done:
                if int(self.args.evalstep) > 0 and self.total_step % int(self.args.evalstep) == 0:
                    with torch.no_grad():
                        self.interimtest()
                        self.save_results()

                action = self.select_action_train(state)
                next_state, reward, done, transition, _ = self.train_step(action)
                self._add_transition_to_memory(transition)

                self.total_step += 1
                self.episode_step += 1
                self.timestep += 1
                state = next_state
                sum_of_rewards += reward
                rewards.append(reward)

                if self.total_step >= self.hyper_params['INITIAL_RANDOM_ACTION']:
                    if self.tstep % self.hyper_params["TRAIN_FREQ"] == 0:
                        self.resetmean()
                        self.update_model()
                        self.averagemean()
                        self.tstep = 0
                    self.tstep += 1

                if self.hyper_params["TOTAL_STEPS"] <= self.total_step:
                    end_training = True
                    break
                if self.episode_step >= self.args.max_episode_steps:
                    break
            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step
            self.write_log(self.i_episode, sum_of_rewards, avg_time_cost)

            historical_reward["step"].append(self.total_step)
            historical_reward["episode_reward"].append(sum_of_rewards)

            # Specify the file name for the .csv file
            file_name = self.fname

            rl_data = historical_reward

            # Open the .csv file in write mode
            with open(file_name, mode='w', newline='') as file:
                # Create a CSV writer object
                writer = csv.writer(file)

                # Write the header row
                writer.writerow(["Step", "episode_reward"])

                # Write each data row
                for step, episode_reward in zip(rl_data["step"], rl_data["episode_reward"]):
                    writer.writerow([step, episode_reward])
            if end_training == True:
                break
        # termination
        self.env_train.close()
        self.env_test.close()

    def write_log(
            self,
            i: int,
            score: float = 0.0,
            avg_time_cost: float = 0.0,
    ):
        """Write log about loss and score"""
        # total_loss = loss.sum()
        print(
            "[INFO] episode %d, episode_step %d, total step %d, total score: %d (spent %.6f sec/step)"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                avg_time_cost,
            )
        )



    def set_beta_fraction(self):
        if not 'RANDOM' in self.args.algo:
            self.fraction = min(float(self.total_step) / self.hyper_params["TOTAL_STEPS"], 1.0)
            self.beta = self.hyper_params["PER_BETA"] + self.fraction * (1.0 - self.hyper_params["PER_BETA"])

    def get_sample(self):
        if 'RANDOM' in self.args.algo:
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones, indices = experiences
            weights = torch.Tensor(np.ones([self.hyper_params["BATCH_SIZE"], 1])).cuda()
        else:
            experiences = self.memory.sample(self.beta)
            states, actions, rewards, next_states, dones, weights, indices = experiences
        states = states.detach()
        actions = actions.detach()
        next_states = next_states.detach()
        rewards = rewards.detach()
        weights = weights.detach()
        dones = dones.detach()

        return states, actions, rewards, next_states, dones, indices, weights
