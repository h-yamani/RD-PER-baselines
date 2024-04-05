import numpy as np
import torch
import torch.optim as optim

import algorithms.common.helper_functions as common_utils
from algorithms.common.networks.mlp import FlattenMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from algorithms.sac.sac import SACAgent


class MaSAC(SACAgent):
    def __init__(self, envs, args):
        SACAgent.__init__(self, envs, args, False)
        self.make_critic()

    def make_critic(self):

        # create q_critic
        self.qf_1 = FlattenMLP(
            input_size=self.state_dim + self.action_dim, output_size=1 + 1 + self.state_dim,
            hidden_sizes=self.hyper_params["CRITIC_SIZE"]
        ).to(device)
        self.qf_2 = FlattenMLP(
            input_size=self.state_dim + self.action_dim, output_size=1 + 1 + self.state_dim,
            hidden_sizes=self.hyper_params["CRITIC_SIZE"]
        ).to(device)
        # create q_critic
        self.qf_1_target = FlattenMLP(
            input_size=self.state_dim + self.action_dim, output_size=1 + 1 + self.state_dim,
            hidden_sizes=self.hyper_params["CRITIC_SIZE"]
        ).to(device)
        self.qf_2_target = FlattenMLP(
            input_size=self.state_dim + self.action_dim, output_size=1 + 1 + self.state_dim,
            hidden_sizes=self.hyper_params["CRITIC_SIZE"]
        ).to(device)
        self.qf_1_target.load_state_dict(self.qf_1.state_dict())
        self.qf_2_target.load_state_dict(self.qf_2.state_dict())

        self.qf_1_optim = optim.Adam(
            self.qf_1.parameters(),
            lr=self.hyper_params["LR_QF1"],
            weight_decay=self.hyper_params["WEIGHT_DECAY"],
        )
        self.qf_2_optim = optim.Adam(
            self.qf_2.parameters(),
            lr=self.hyper_params["LR_QF2"],
            weight_decay=self.hyper_params["WEIGHT_DECAY"],
        )

    def update_model(self):
        for grad_step in range(self.hyper_params["MULTIPLE_LEARN"]):
            states, actions, rewards, next_states, dones, indices, weights = self.get_sample()
            qf1, rew1, next_states1 = self.div(self.qf_1(states, actions))
            qf2, rew2, next_states2 = self.div(self.qf_2(states, actions))
            diff_rew1 = 0.5 * torch.pow(rew1.reshape(-1, 1) - rewards.reshape(-1, 1), 2.0).reshape(-1, 1)
            diff_rew2 = 0.5 * torch.pow(rew2.reshape(-1, 1) - rewards.reshape(-1, 1), 2.0).reshape(-1, 1)
            diff_next_states1 = 0.5 * torch.mean(
                torch.pow(next_states1.reshape(-1, self.state_dim) - next_states.reshape(-1, self.state_dim), 2.0),
                -1).reshape(-1, 1)
            diff_next_states2 = 0.5 * torch.mean(
                torch.pow(next_states2.reshape(-1, self.state_dim) - next_states.reshape(-1, self.state_dim), 2.0),
                -1).reshape(-1, 1)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.actor(next_states)
                qf1_next_target, _, _ = self.div(self.qf_1_target(next_states, next_state_action))
                qf2_next_target, _, _ = self.div(self.qf_2_target(next_states, next_state_action))
                min_qf_next_target = torch.min(qf1_next_target.reshape(-1, 1), qf2_next_target.reshape(-1, 1))
                min_qf_next_target = min_qf_next_target - self.log_alpha.exp().detach() * next_state_log_pi
                masks = 1 - dones
                rew = (rew1.reshape(-1, 1) + rew1.reshape(-1, 1)) / 2
                next_q_value = rew + masks * self.hyper_params["GAMMA"] * min_qf_next_target

            diff_td1 = 0.5 * torch.pow(qf1.reshape(-1, 1) - next_q_value.reshape(-1, 1), 2.0).reshape(-1, 1)
            diff_td2 = 0.5 * torch.pow(qf2.reshape(-1, 1) - next_q_value.reshape(-1, 1), 2.0).reshape(-1, 1)

            diff1 = diff_td1 + self.scale_r * diff_rew1 + self.scale_s * diff_next_states1
            diff2 = diff_td2 + self.scale_r * diff_rew2 + self.scale_s * diff_next_states2
            qf1_loss = torch.mean(diff1 * weights.detach())
            self.qf_1_optim.zero_grad()
            qf1_loss.backward()
            self.qf_1_optim.step()
            qf2_loss = torch.mean(diff2 * weights.detach())
            self.qf_2_optim.zero_grad()
            qf2_loss.backward()
            self.qf_2_optim.step()
            pi, log_pi, _ = self.actor(states)
            qf1_pi, _, _ = self.div(self.qf_1(states, pi))
            qf2_pi, _, _ = self.div(self.qf_2(states, pi))
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            policy_loss = torch.mean((torch.exp(self.log_alpha).detach() * log_pi - min_qf_pi) * weights)
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

            if self.hyper_params["AUTO_ENTROPY_TUNING"]:
                alpha_loss = (
                        -weights * self.log_alpha * (log_pi + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            numpy_td = torch.cat([diff_td1, diff_td2], -1)
            numpy_td = torch.mean(numpy_td, 1)
            numpy_td = numpy_td.view(-1, 1)
            numpy_td = numpy_td[:, 0].detach().data.cpu().numpy()

            numpy_r = torch.cat([diff_rew1, diff_rew2], -1)
            numpy_r = torch.mean(numpy_r, 1)
            numpy_r = numpy_r.view(-1, 1)
            numpy_r = numpy_r[:, 0].detach().data.cpu().numpy()

            numpy_s = torch.cat([diff_next_states1, diff_next_states2], -1)
            numpy_s = torch.mean(numpy_s, 1)
            numpy_s = numpy_s.view(-1, 1)
            numpy_s = numpy_s[:, 0].detach().data.cpu().numpy()

            numpy_next_q_value = next_q_value[:, 0].detach().data.cpu().numpy()

            """if any([f in self.args.algo for f in ['PER', 'NERS', 'ERO']]):
                if 'PER' in self.args.algo:
                    new_priorities = np.array(
                        [numpy_next_q_value,
                         numpy_td])
                else:"""
            new_priorities = np.array(
                        [numpy_next_q_value,
                         numpy_td + self.scale_s * numpy_s + self.scale_r * numpy_r])

            indices = np.array(indices)
            self.memory.update_priorities(indices, new_priorities)
            self.mean_td += np.mean(numpy_td)
            self.mean_q += np.mean(numpy_next_q_value)
            self.mean_r += np.mean(numpy_r)
            self.mean_s += np.mean(numpy_s)
            common_utils.soft_update(self.qf_1, self.qf_1_target, self.hyper_params["TAU"])
            common_utils.soft_update(self.qf_2, self.qf_2_target, self.hyper_params["TAU"])
        if self.update_step == 0:
            self.scale_r = np.mean(numpy_td) / (np.mean(numpy_s))
            self.scale_s = np.mean(numpy_td) / (np.mean(numpy_s))

        self.update_step += 1
        self.set_beta_fraction()
