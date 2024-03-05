import numpy as np
import torch
import torch.nn.functional as F
from sac.model import GaussianPolicy, QNetwork, DeterministicPolicy
from sac.utils import soft_update, hard_update
from torch.optim import Adam


class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        torch.autograd.set_detect_anomaly(True)
        self.count = 0
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'RANDOM' in self.args.replay_type:
            output_dim = 1
        elif 'MaPER' in self.args.replay_type:
            output_dim = args.state_size + 2
        else:
            assert False
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, output_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.td_errors = []
        self.t_errors = []
        self.r_errors = []
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, output_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        self.prev_val_rew = 1.0
        self.prev_val_dyn = 1.0
        self.prev_val_tde = 1.0

        self.curr_val_rew = 1.0
        self.curr_val_dyn = 1.0
        self.curr_val_tde = 1.0

        self.qfv_coef = 1.0
        self.rew_coef = 0.0
        self.dyn_coef = 0.0
        self.temp = self.args.temp

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates, batch_weight):
        coef_rew = np.exp(self.curr_val_rew / (self.temp * self.prev_val_rew + 1e-9))
        coef_dyn = np.exp(self.curr_val_dyn / (self.temp * self.prev_val_dyn + 1e-9))
        coef_tde = np.exp(self.curr_val_tde / (self.temp * self.prev_val_tde + 1e-9))
        coef_sum = (coef_rew + coef_dyn + coef_tde) / 3
        coef_rew /= coef_sum
        coef_dyn /= coef_sum
        coef_tde /= coef_sum

        # Sample a batch from memory
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        batch_weight = torch.FloatTensor(batch_weight).to(self.device).reshape(-1, 1)

        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target[:, 0].reshape(-1, 1),
                                           qf2_next_target[:, 0].reshape(-1, 1)) - self.alpha * next_state_log_pi
            if 'MaPER' in self.args.replay_type:
                estimated_reward = (qf1[:, 1] + qf2[:, 1]) / 2
                estimated_reward = estimated_reward.reshape(-1, 1)
            else:
                estimated_reward = reward_batch
            next_q_value = estimated_reward + mask_batch * self.gamma * (min_qf_next_target)
        qf1_loss = F.mse_loss(qf1[:, 0].reshape(-1, 1), next_q_value,
                              reduction='none')  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2[:, 0].reshape(-1, 1), next_q_value,
                              reduction='none')  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        if qf1.shape[1] == 1:
            total_loss = qf1_loss + qf2_loss
        elif qf1.shape[1] > 1:
            if 'MaPER' in self.args.replay_type:
                rew1_loss = F.mse_loss(qf1[:, 1].reshape(-1, 1), reward_batch, reduction='none')
                rew2_loss = F.mse_loss(qf2[:, 1].reshape(-1, 1), reward_batch, reduction='none')
                dyn1_loss = torch.sum(F.mse_loss(qf1[:, 2:], next_state_batch, reduction='none'), dim=1).reshape(-1, 1)
                dyn2_loss = torch.sum(F.mse_loss(qf2[:, 2:], next_state_batch, reduction='none'), dim=1).reshape(-1, 1)

                norm_rew1_loss = rew1_loss
                norm_rew2_loss = rew2_loss
                norm_dyn1_loss = dyn1_loss
                norm_dyn2_loss = dyn2_loss

                self.prev_val_rew = self.curr_val_rew
                self.prev_val_dyn = self.curr_val_dyn
                self.prev_val_tde = self.curr_val_tde
                self.curr_val_rew = torch.mean(rew1_loss + rew2_loss).item()
                self.curr_val_dyn = torch.mean(dyn1_loss + dyn2_loss).item()
                self.curr_val_tde = torch.mean(qf1_loss + qf2_loss).item()
            else:
                assert False
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi[:, 0].reshape(-1, 1), qf2_pi[:, 0].reshape(-1, 1))

        policy_loss = (((
                                self.alpha * log_pi) - min_qf_pi) * batch_weight).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        if 'MaPER' in self.args.replay_type:
            total_td = torch.mean((qf1_loss + qf2_loss) * batch_weight)
            total_rew = torch.mean((norm_rew1_loss + norm_rew2_loss) * batch_weight)
            total_dyn = torch.mean((norm_dyn1_loss + norm_dyn2_loss) * batch_weight)
            weighted_total_loss = total_td * coef_tde + total_dyn * coef_dyn + total_rew * coef_rew
            self.critic_optim.zero_grad()
            weighted_total_loss.backward()
            self.critic_optim.step()
        else:
            weighted_total_loss = total_loss * batch_weight
            self.critic_optim.zero_grad()
            (torch.mean(weighted_total_loss)).backward()
            self.critic_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(batch_weight * (self.log_alpha * (log_pi + self.target_entropy).detach())).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        if 'MaPER' in self.args.replay_type:
            priorities = ((
                                  qf1_loss + qf2_loss + norm_dyn1_loss + norm_dyn2_loss + rew1_loss + rew2_loss) / 2).detach().cpu().numpy().reshape(
                -1)
        else:
            priorities = ((qf1_loss + qf2_loss) / 2).detach().cpu().numpy().reshape(-1)

        if self.count <= 64 and 'MaPER' in self.args.replay_type:
            self.td_errors.append(self.curr_val_tde)
            self.t_errors.append(self.curr_val_dyn)
            self.r_errors.append(self.curr_val_rew)
            if self.count == 64:
                self.qfv_coef = np.mean(self.td_errors)
                self.dyn_coef = self.qfv_coef / np.mean(self.t_errors)
                self.rew_coef = self.qfv_coef / np.mean(self.r_errors)
        self.count += 1
        return priorities
