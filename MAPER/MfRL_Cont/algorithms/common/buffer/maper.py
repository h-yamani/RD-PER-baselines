import random
from typing import List, Tuple

import numpy as np
import torch

from algorithms.common.buffer.replay_buffer import ReplayBuffer
from algorithms.common.buffer.segment_tree import MinSegmentTree, SumSegmentTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# SEES + PER
class MaPER(ReplayBuffer):
    """Create Prioritized Replay buffer.

    Refer to OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

    Attributes:
        alpha (float): alpha parameter for prioritized replay buffer
        epsilon_d (float): small positive constants to add to the priorities
        tree_idx (int): next index of tree
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        _max_priority (float): max priority
    """

    def __init__(
            self, algo: str,
            buffer_size: int,
            batch_size: int = 32,
            gamma: float = 0.99,
            n_step: int = 1,
            alpha: float = 0.6,
            epsilon_d: float = 1e-6,
            demo: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = None,
    ):
        """Initialization.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            alpha (float): alpha parameter for prioritized replay buffer

        """
        ReplayBuffer.__init__(self,
                              buffer_size, batch_size, gamma, n_step, demo
                              )
        self.algo = algo
        assert alpha >= 0
        self.alpha = alpha
        self.epsilon_d = epsilon_d
        self.epsilon = 1e-6
        self.tree_idx = 0
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self._max_priority = 1.0

    def add(self, transition, timestep, episode):
        """Add experience and priority."""
        n_step_transition = super().add(transition, timestep, episode)
        if n_step_transition:
            self.sum_tree[self.tree_idx] = self._max_priority ** self.alpha
            self.min_tree[self.tree_idx] = self._max_priority ** self.alpha

            self.tree_idx += 1
            if self.tree_idx % self.buffer_size == 0:
                self.tree_idx = self.demo_size

        return n_step_transition

    def _sample_proportional(self, batch_size: int) -> list:
        """Sample indices based on proportional."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def sample(self, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        indices = np.array(self._sample_proportional(self.batch_size))
        weights = self.get_weight(indices, beta=1.0)
        states, actions, rewards, next_states, dones, indices = super().sample(indices)
        return states, actions, rewards, next_states, dones, weights, indices

    def get_weight(self, indices, beta=1.0):
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        # calculate weights
        weights_ = []
        for i in indices:
            p_sample = self.sum_tree[i] / self.sum_tree.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights_.append(weight / max_weight)
        weights = np.array(weights_)
        weights = torch.FloatTensor(weights.reshape(-1, 1)).to(device)
        return weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        # Newly Inserted
        # Update Q-values and uncertainties
        q = priorities[0, :]
        s = priorities[1, :]
        priorities = s
        priorities = priorities + self.epsilon_d
        indices = indices.astype(int)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)
