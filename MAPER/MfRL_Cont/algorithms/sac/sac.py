import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import algorithms.common.helper_functions as common_utils
from algorithms.common.abstract.agent import AgentBase
from algorithms.common.networks.mlp import FlattenMLP, TanhGaussianDistParams
from algorithms.params import sac_exp_hyper_params
from algorithms.params import sac_test_hyper_params
from trainers.buffer import Buffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SACAgent(AgentBase):
    """SAC agent interacting with environment.
    """

    def __init__(
            self, envs, args, make_critic=True
    ):
        """Initialization."""
        self.args = args

        if self.args.test == True:
            self.hyper_params = sac_test_hyper_params
        else:
            self.hyper_params = sac_exp_hyper_params

        self.env_test = envs[1]
        self.env_train = envs[0]

        self.state_dim = self.env_train.observation_space.shape[0]
        self.action_dim = self.env_train.action_space.shape[0]
        self.action_scale = torch.FloatTensor(
            (self.env_train.action_space.high - self.env_train.action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (self.env_train.action_space.high + self.env_train.action_space.low) / 2.)
        AgentBase.__init__(self)
        self.make_actor()
        if make_critic:
            self.make_critic()
        self.update_step = 0
        self.scale_r = 1.0
        self.scale_s = 1.0
        self.scale_a = 1.0

    def make_actor(self):
        # target entropy
        target_entropy = -np.prod((self.action_dim,)).item()  # heuristic
        self.actor = TanhGaussianDistParams(
            input_size=self.state_dim, output_size=self.action_dim, hidden_sizes=self.hyper_params["ACTOR_SIZE"],
            action_scale=self.action_scale, action_bias=self.action_bias,
        ).to(device)
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.hyper_params["LR_ACTOR"],
            weight_decay=self.hyper_params["WEIGHT_DECAY"],
        )
        self.curr_state = np.zeros((1,))
        # automatic entropy tuning
        if self.hyper_params["AUTO_ENTROPY_TUNING"]:
            self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.hyper_params["LR_ENTROPY"]
            )
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
        # create q_critic

    def div(self, target):
        return target[:, 0], target[:, 1], target[:, 2:]

    def make_critic(self):
        self.qf_1 = FlattenMLP(
            input_size=self.state_dim + self.action_dim, output_size=1, hidden_sizes=self.hyper_params["CRITIC_SIZE"],
        ).to(device)
        self.qf_2 = FlattenMLP(
            input_size=self.state_dim + self.action_dim, output_size=1, hidden_sizes=self.hyper_params["CRITIC_SIZE"],
        ).to(device)
        # create q_critic
        self.qf_1_target = FlattenMLP(
            input_size=self.state_dim + self.action_dim, output_size=1, hidden_sizes=self.hyper_params["CRITIC_SIZE"],
        ).to(device)
        self.qf_2_target = FlattenMLP(
            input_size=self.state_dim + self.action_dim, output_size=1, hidden_sizes=self.hyper_params["CRITIC_SIZE"],
        ).to(device)
        self.qf_1_target.load_state_dict(self.qf_1.state_dict())
        self.qf_2_target.load_state_dict(self.qf_2.state_dict())
        # create optimizers

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

    def select_action_train(self, state):
        """Select an action from the input space."""
        self.curr_state = state
        state = self._preprocess_state(state)
        if self.total_step < self.hyper_params['INITIAL_RANDOM_ACTION']:
            return np.array(self.env_train.action_space.sample())
        else:
            action, log_prob, mean = self.actor(state)
            return action.detach().cpu().numpy()

    def select_action_test(self, state):
        """Select an action from the input space."""
        self.curr_state = state
        state = self._preprocess_state(state)
        action, log_prob, mean = self.actor(state)
        return mean.detach().cpu().numpy()

    # pylint: disable=no-self-use
    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""

        state = torch.FloatTensor(np.array(state)).to(device)

        return state

    def _add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        self.memory.add(transition, self.timestep, self.i_episode)

    def update_model(self):
        for grad_step in range(self.hyper_params["MULTIPLE_LEARN"]):
            """Train the model after each episode."""
            states, actions, rewards, next_states, dones, indices, weights = self.get_sample()

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.actor(next_states)
                qf1_next_target = self.qf_1_target(next_states, next_state_action)
                qf2_next_target = self.qf_2_target(next_states, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target,
                                               qf2_next_target)

                min_qf_next_target = min_qf_next_target - self.log_alpha.exp() * next_state_log_pi
                masks = 1 - dones
                next_q_value = rewards + masks * self.hyper_params["GAMMA"] * min_qf_next_target

            qf1 = self.qf_1(states, actions)
            qf2 = self.qf_2(states, actions)

            diff1 = F.mse_loss(qf1, next_q_value, reduction='none')
            diff2 = F.mse_loss(qf2, next_q_value, reduction='none')
            qf1_loss = torch.mean(
                diff1 * weights)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            self.qf_1_optim.zero_grad()
            qf1_loss.backward()
            self.qf_1_optim.step()

            qf2_loss = torch.mean(diff2 * weights)
            self.qf_2_optim.zero_grad()
            qf2_loss.backward()
            self.qf_2_optim.step()
            pi, log_pi, _ = self.actor(states)
            qf1_pi = self.qf_1(states, pi)
            qf2_pi = self.qf_2(states, pi)
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

            critic_loss_element_wise = torch.cat([diff1, diff2], -1)
            critic_loss_element_wise, _ = torch.max(critic_loss_element_wise, 1)
            critic_loss_element_wise = critic_loss_element_wise.view(-1, 1)

            numpy_next_q_value = next_q_value[:, 0].detach().data.cpu().numpy()
            numpy_td = critic_loss_element_wise[:, 0].detach().data.cpu().numpy()

            if any([f in self.args.algo for f in ['PER', 'NERS', 'ERO']]):
                new_priorities = np.array(
                    [numpy_next_q_value,
                     numpy_td])
                indices = np.array(indices)
                self.memory.update_priorities(indices, new_priorities)

            self.mean_td += np.mean(numpy_td)
            self.mean_q += np.mean(numpy_next_q_value)
            self.mean_r += np.mean(numpy_td)
            self.mean_s += np.mean(numpy_td)
            common_utils.soft_update(self.qf_1, self.qf_1_target, self.hyper_params["TAU"])
            common_utils.soft_update(self.qf_2, self.qf_2_target, self.hyper_params["TAU"])
        self.update_step += 1
        self.set_beta_fraction()
