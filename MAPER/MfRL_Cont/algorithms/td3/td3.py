import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import algorithms.common.helper_functions as common_utils
from algorithms.common.networks.mlp import MLP, FlattenMLP
from algorithms.common.noise import GaussianNoise
from trainers.buffer import Buffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from algorithms.common.abstract.agent import AgentBase


class TD3Agent(AgentBase):
    def __init__(self, envs, args, make_critic=True):
        self.args = args
        from algorithms.params import td3_exp_hyper_params
        from algorithms.params import td3_test_hyper_params
        if self.args.test == True:
            self.hyper_params = td3_test_hyper_params
        else:
            self.hyper_params = td3_exp_hyper_params
        self.env_test = envs[1]
        self.env_train = envs[0]
        super().__init__()

        self.state_dim = self.env_train.observation_space.shape[0]
        self.action_dim = self.env_train.action_space.shape[0]
        self.action_scale = torch.FloatTensor(
            (self.env_train.action_space.high - self.env_train.action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (self.env_train.action_space.high + self.env_train.action_space.low) / 2.)
        self.make_actor()
        if make_critic:
            self.make_critic()
        self.update_step = 0
        self.scale_r = 1.0
        self.scale_s = 1.0

    def make_actor(self):
        self.exploration_noise = GaussianNoise(
            self.action_dim, self.hyper_params["EXPLORATION_NOISE"], self.hyper_params["EXPLORATION_NOISE"]
        )

        self.target_policy_noise = GaussianNoise(
            self.action_dim,
            self.hyper_params["TARGET_POLICY_NOISE"],
            self.hyper_params["TARGET_POLICY_NOISE"],
        )
        self.actor = MLP(
            input_size=self.state_dim,
            output_size=self.action_dim,
            hidden_sizes=self.hyper_params["ACTOR_SIZE"],
            output_activation=torch.tanh,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
        ).to(device)

        self.actor_target = MLP(
            input_size=self.state_dim,
            output_size=self.action_dim,
            hidden_sizes=self.hyper_params["ACTOR_SIZE"],
            output_activation=torch.tanh,
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        # create optimizers
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.hyper_params["LR_ACTOR"],
            weight_decay=self.hyper_params["WEIGHT_DECAY"],
        )

        self.curr_state = np.zeros((1,))
        self.beta = self.hyper_params["PER_BETA"]
        buffer = Buffer(
            args=self.args,
            buffersize=self.hyper_params["BUFFER_SIZE"],
            batchsize=self.hyper_params["BATCH_SIZE"],
            alpha=self.hyper_params["PER_ALPHA"],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gamma=self.hyper_params["GAMMA"],
        )
        self.memory = buffer.return_buffer()

    def make_critic(self):

        # create critic
        self.critic1 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.hyper_params["CRITIC_SIZE"],
        ).to(device)

        self.critic2 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.hyper_params["CRITIC_SIZE"],
        ).to(device)

        self.critic_target1 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.hyper_params["CRITIC_SIZE"],
        ).to(device)

        self.critic_target2 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.hyper_params["CRITIC_SIZE"],
        ).to(device)

        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

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

    def div(self, target):
        return target[:, 0], target[:, 1], target[:, 2:]

    def select_action_train(self, state: np.ndarray) -> np.ndarray:
        self.curr_state = state
        if self.total_step < self.hyper_params['INITIAL_RANDOM_ACTION']:
            return np.array(self.env_train.action_space.sample())
        else:
            state = torch.FloatTensor(state).to(device)
            selected_action = self.actor(state).detach().cpu().numpy()
            noise = self.exploration_noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
            return selected_action

    def select_action_test(self, state: np.ndarray) -> np.ndarray:
        self.curr_state = state
        state = torch.FloatTensor(state).to(device)
        selected_action = self.actor(state).detach().cpu().numpy()
        return selected_action

    def _add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        self.memory.add(transition, self.timestep, self.i_episode)

    def update_model(self):
        """Train the model after each episode."""
        for grad_step in range(self.hyper_params["MULTIPLE_LEARN"]):
            states, actions, rewards, next_states, dones, indices, weights = self.get_sample()
            masks = 1 - dones

            # get actions with noise
            noise = torch.FloatTensor(self.target_policy_noise.sample()).to(device)
            clipped_noise = torch.clamp(
                noise,
                -self.hyper_params["TARGET_POLICY_NOISE_CLIP"],
                self.hyper_params["TARGET_POLICY_NOISE_CLIP"],
            )
            with torch.no_grad():
                next_actions = (self.actor_target(next_states) + clipped_noise).clamp(-1.0, 1.0)
                # min (Q_1', Q_2')
                next_values1 = self.critic_target1(next_states, next_actions)
                next_values2 = self.critic_target2(next_states, next_actions)
                next_values = torch.min(next_values1, next_values2)

                # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
                #       = r                       otherwise
                curr_returns = rewards + self.hyper_params["GAMMA"] * next_values * masks

            # critic loss
            values1 = self.critic1(states, actions)
            values2 = self.critic2(states, actions)
            diff1 = F.mse_loss(values1, curr_returns, reduction='none')
            diff2 = F.mse_loss(values2, curr_returns, reduction='none')
            critic1_loss = torch.mean(diff1 * weights)
            critic2_loss = torch.mean(diff2 * weights)

            self.critic1_optim.zero_grad()
            critic1_loss.backward()
            self.critic1_optim.step()

            self.critic2_optim.zero_grad()
            critic2_loss.backward()
            self.critic2_optim.step()
            # policy loss

            numpy_next_q_value = curr_returns[:, 0].data.cpu().numpy()
            numpy_td = diff1[:, 0].detach().data.cpu().numpy()

            # Obtain TD erros and Q for priorities update
            if any([f in self.args.algo for f in ['PER', 'NERS', 'ERO']]):
                new_priorities = np.array(
                    [numpy_next_q_value,
                     numpy_td])
                indices = np.array(indices)
                self.memory.update_priorities(indices, new_priorities)

            if grad_step % self.hyper_params["POLICY_UPDATE_FREQ"] == 0:
                # train actor
                actor_val = self.critic1(states, self.actor(states))
                actor_loss = -torch.mean(weights * actor_val)
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                tau = self.hyper_params["TAU"]
                common_utils.soft_update(self.critic1, self.critic_target1, tau)
                common_utils.soft_update(self.critic2, self.critic_target2, tau)
                common_utils.soft_update(self.actor, self.actor_target, tau)

            self.mean_td += np.mean(numpy_td)
            self.mean_q += np.mean(numpy_next_q_value)
            self.mean_r += np.mean(numpy_td)
            self.mean_s += np.mean(numpy_td)

        self.update_step += 1
        self.set_beta_fraction()
