# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import torch
from model import DQN, MaDQN
from torch import optim
from torch.nn.utils import clip_grad_norm_


class Agent():
    def __init__(self, args, env):
        self.args = args
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip
        self.start = 0
        # self.online_net = DQN(args, self.action_space).to(device=args.device)
        self.scale_td = 1
        self.scale_r = 1
        self.scale_t = 1

        self.curr_td = 1.0
        self.curr_r = 1.0
        self.curr_t = 1.0
        self.prev_td = 1.0
        self.prev_r = 1.0
        self.prev_t = 1.0
        self.td_errors = []
        self.r_errors = []
        self.t_errors = []
        if 'MaPER' in self.args.replay_type:
            self.online_net = MaDQN(args, self.action_space).to(device=args.device)
        else:
            self.online_net = DQN(args, self.action_space).to(device=args.device)

        self.online_net.train()
        if 'MaPER' in self.args.replay_type:
            self.target_net = MaDQN(args, self.action_space).to(device=args.device)
        else:
            self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            if 'MaPER' in self.args.replay_type:
                return (self.online_net(state.unsqueeze(0))[0] * self.support).sum(2).argmax(1).item()
            else:
                return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def learn(self, mem):
        # Sample transitions
        if self.start >= 2:
            coef_td = np.exp((self.curr_td / (self.args.temp * self.prev_td)))
            coef_r = np.exp((self.curr_r / (self.args.temp * self.prev_r)))
            coef_t = np.exp((self.curr_t / (self.args.temp * self.prev_t)))
            coef_sum = (coef_td + coef_r + coef_t) / 3
            coef_td = coef_td / coef_sum
            coef_r = coef_r / coef_sum
            coef_t = coef_t / coef_sum

        else:
            coef_td = 1.0
            coef_r = 1.0
            coef_t = 1.0
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

        # Calculate current state probabilities (online network noise already sampled)
        if 'MaPER' in self.args.replay_type:
            log_ps, t, r, f = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
            _, _, _, nf = self.online_net(next_states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        else:
            log_ps = self.online_net(states, log=True)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            if 'MaPER' in self.args.replay_type:
                pns, tt, tr, tf = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            else:
                pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(
                1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.target_net.reset_noise()  # Sample new target net noise
            if 'MaPER' in self.args.replay_type:
                pns, tnr, tnt, tnf = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            else:
                pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(
                self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
        if 'MaPER' in self.args.replay_type:
            tran_error = torch.sum(torch.pow(t - nf.detach(), 2.0), -1).reshape(-1) / 2
            r_error = torch.sum(torch.pow(r - returns, 2.0), -1).reshape(-1) / 2
        with torch.no_grad():
            # Compute Tz (Bellman operator T applied to z)                        
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(
                0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        td_error = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))

        if 'MaPER' in self.args.replay_type:
            self.prev_td = self.curr_td
            self.prev_r = self.curr_r
            self.prev_t = self.curr_t
            self.curr_td = torch.mean(td_error).detach().cpu().numpy()
            self.curr_r = torch.mean(r_error).detach().cpu().numpy()
            self.curr_t = torch.mean(tran_error).detach().cpu().numpy()
            loss = coef_td * td_error + self.scale_t * coef_t * tran_error + self.scale_r * coef_r * r_error

        else:
            loss = td_error
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimiser.step()
        if 'MaPER' in self.args.replay_type:
            # Update priorities of sampled transitions
            mem.update_priorities(idxs, loss.detach().cpu().numpy())
        else:
            mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions
        self.start += 1
        if self.start <= 64 and 'MaPER' in self.args.replay_type:
            self.td_errors.append(torch.mean(td_error).item())
            self.r_errors.append(torch.mean(r_error).item())
            self.t_errors.append(torch.mean(tran_error).item())
            if self.start == 64 and 'MaPER' in self.args.replay_type:
                mean_td_errors = np.mean(self.td_errors)
                mean_t_errors = np.mean(self.t_errors)
                mean_r_errors = np.mean(self.r_errors)

                self.scale_td = mean_td_errors
                self.scale_t = mean_td_errors / mean_t_errors
                self.scale_r = mean_td_errors / mean_r_errors
        self.start + 1
        return

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            if 'MaPER' in self.args.replay_type:
                return (self.online_net(state.unsqueeze(0))[0] * self.support).sum(2).max(1)[0].item()
            else:
                return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
