import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import algorithms.common.helper_functions as common_utils
from algorithms.common.networks.mlp import FlattenMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from algorithms.td3.td3 import TD3Agent


class MaTD3(TD3Agent):
    def __init__(self, envs, args):
        TD3Agent.__init__(self, envs, args, False)
        self.make_critic()

    def make_critic(self):
        # create critic
        self.critic1 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1 + 1 + self.state_dim,
            hidden_sizes=self.hyper_params["CRITIC_SIZE"],
        ).to(device)

        self.critic2 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1 + 1 + self.state_dim,
            hidden_sizes=self.hyper_params["CRITIC_SIZE"],
        ).to(device)

        self.critic_target1 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1 + 1 + self.state_dim,
            hidden_sizes=self.hyper_params["CRITIC_SIZE"],
        ).to(device)

        self.critic_target2 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1 + 1 + self.state_dim,
            hidden_sizes=self.hyper_params["CRITIC_SIZE"],
        ).to(device)

        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        # concat critic parameters to use one optim
        self.critic1_optim = optim.Adam(
            self.critic1.parameters(),
            lr=self.hyper_params["LR_CRITIC"],
            weight_decay=self.hyper_params["WEIGHT_DECAY"],
        )
        self.critic2_optim = optim.Adam(
            self.critic2.parameters(),
            lr=self.hyper_params["LR_CRITIC"],
            weight_decay=self.hyper_params["WEIGHT_DECAY"],
        )

    def update_model(self):
        """Train the model after each episode."""
        for grad_step in range(self.hyper_params["MULTIPLE_LEARN"]):
            states, actions, rewards, next_states, dones, indices, weights = self.get_sample()
            masks = 1 - dones
            masks = masks.reshape(-1, 1)
            values1, rew1, next_states1 = self.div(self.critic1(states.detach(), actions.detach()))
            values2, rew2, next_states2 = self.div(self.critic2(states.detach(), actions.detach()))
            diff_rew1 = 0.5 * torch.pow(rew1.reshape(-1, 1) - rewards.reshape(-1, 1), 2.0).reshape(-1, 1)
            diff_rew2 = 0.5 * torch.pow(rew2.reshape(-1, 1) - rewards.reshape(-1, 1), 2.0).reshape(-1, 1)
            diff_next_states1 = 0.5 * torch.mean(
                torch.pow(next_states1.reshape(-1, self.state_dim) - next_states.reshape(-1, self.state_dim),
                          2.0), -1).reshape(-1, 1)
            diff_next_states2 = 0.5 * torch.mean(
                torch.pow(next_states2.reshape(-1, self.state_dim) - next_states.reshape(-1, self.state_dim),
                          2.0), -1).reshape(-1, 1)
            noise = torch.FloatTensor(self.target_policy_noise.sample()).to(device)
            clipped_noise = torch.clamp(
                noise,
                -self.hyper_params["TARGET_POLICY_NOISE_CLIP"],
                self.hyper_params["TARGET_POLICY_NOISE_CLIP"],
            )
            with torch.no_grad():

                next_actions = self.actor_target(next_states)
                next_actions = (next_actions + clipped_noise).clamp(-1.0, 1.0)
                next_values1, _, _ = self.div(self.critic_target1(next_states, next_actions))
                next_values2, _, _ = self.div(self.critic_target2(next_states, next_actions))
                next_values = torch.min(next_values1, next_values2).reshape(-1, 1)
                rew = (rew1.reshape(-1, 1) + rew1.reshape(-1, 1)) / 2
                curr_returns = rew.reshape(-1, 1)
                curr_returns = curr_returns + self.hyper_params["GAMMA"] * next_values * masks

            # critic loss
            diff_td1 = F.mse_loss(values1.reshape(-1, 1), curr_returns, reduction='none')
            diff_td2 = F.mse_loss(values2.reshape(-1, 1), curr_returns, reduction='none')
            critic1_loss = (diff_td1 + self.scale_r * diff_rew1 + self.scale_s * diff_next_states1) * weights.detach()
            critic2_loss = (diff_td2 + self.scale_r * diff_rew2 + self.scale_s * diff_next_states2) * weights.detach()

            # train critic

            self.critic1_optim.zero_grad()
            torch.mean(critic1_loss).backward()
            self.critic1_optim.step()
            self.critic2_optim.zero_grad()
            torch.mean(critic2_loss).backward()
            self.critic2_optim.step()
            # policy loss

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
            numpy_next_q_value = curr_returns[:, 0].detach().data.cpu().numpy()

            self.mean_td += np.mean(numpy_td)
            self.mean_q += np.mean(numpy_next_q_value)
            self.mean_r += np.mean(numpy_r)
            self.mean_s += np.mean(numpy_s)

            # Obtain TD erros and Q for priorities update
            if any([f in self.args.algo for f in ['PER', 'NERS', 'ERO']]):
                new_priorities = np.array(
                    [numpy_next_q_value,
                     numpy_td + self.scale_s * numpy_s + self.scale_r * numpy_r])
                indices = np.array(indices)
                self.memory.update_priorities(indices, new_priorities)

            if grad_step % self.hyper_params["POLICY_UPDATE_FREQ"] == 0:
                # train actor
                actions = self.actor(states.detach())
                actor_val, _, _ = self.div(self.critic1(states.detach(), actions))
                actor_loss = -(weights * actor_val).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # update target networks
                tau = self.hyper_params["TAU"]
                common_utils.soft_update(self.critic1, self.critic_target1, tau)
                common_utils.soft_update(self.critic2, self.critic_target2, tau)
                common_utils.soft_update(self.actor, self.actor_target, tau)

        if self.update_step == 0:
            self.scale_r = np.mean(numpy_td) / (np.mean(numpy_s))
            self.scale_s = np.mean(numpy_td) / (np.mean(numpy_s))
        self.update_step += 1
        self.set_beta_fraction()
