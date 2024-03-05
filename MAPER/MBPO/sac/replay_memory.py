import random
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sac.segment_tree import SumSegmentTree, MinSegmentTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, args, capacity):
        self.args = args
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.eps = 1e-9

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position: self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position: len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position

    def sample(self, batch_size, beta=1.0, episode=1):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_all_batch(self, batch_size, beta=1.0, episode=1):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)

    def update_priorities(self, indices, priorities):
        return


class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, args, capacity):
        super().__init__(args, capacity)
        self.args = args
        self._alpha = self.args.alpha_for_PER
        self.capacity = capacity
        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        idx = self.position
        super().push(state, action, reward, next_state, done)
        weight = self._max_priority
        self._it_sum[idx] = weight ** self._alpha
        self._it_min[idx] = weight ** self._alpha

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position: self.position + len(batch)] = batch
            weight = self._max_priority
            for i in range(self.position, self.position + len(batch)):
                self._it_sum[i] = weight ** self._alpha
                self._it_min[i] = weight ** self._alpha
            self.position += len(batch)
        else:
            self.buffer[self.position: len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            weight = self._max_priority
            for i in range(self.position, len(self.buffer)):
                self._it_sum[i] = weight ** self._alpha
                self._it_min[i] = weight ** self._alpha
            for i in range(0, len(batch) - len(self.buffer) + self.position):
                self._it_sum[i] = weight ** self._alpha
                self._it_min[i] = weight ** self._alpha

            self.position = len(batch) - len(self.buffer) + self.position

    def _sample_proportional(self, batchsize):
        res = []
        for _ in range(batchsize):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self.buffer))
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def cal_weight(self, indices, beta):
        # calculate weights
        # get max weight
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        weights_ = []
        for i in indices:
            p_sample = self._it_sum[i] / self._it_sum.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights_.append(weight / max_weight)
        weights = np.array(weights_)
        weights = weights / weights.max()
        return weights

    def sample(self, batch_size, beta=1.0, episode=1):
        assert beta >= 0.0
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        indices = self._sample_proportional(batch_size)
        weights = self.cal_weight(indices, beta)
        batch = [self.buffer[i] for i in indices]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, weights, indices

    def dump(self, batch_size=None):
        if batch_size is None:
            idxes = np.random.randint(0, len(self.buffer), len(self.buffer))
        else:
            idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_all_batch(self, batch_size, beta=1.0, episode=1):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        weights = self.cal_weight(idxes, beta)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, weights, idxes

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        if indices is None:
            return
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            priority += self.eps
            assert priority > 0
            assert 0 <= idx < len(self)

            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


# Newly Inserted : NERS
class Likelihood(nn.Module):
    def __init__(self, args, capacity):
        super(Likelihood, self).__init__()
        self.args = args
        self.capacity = capacity
        self.buffer_slow = []
        self.buffer_fast = []
        self.position_slow = 0
        self.position_fast = 0
        self.eps = 1e-9
        self.temperature = 7.5
        self.episode = 0
        self.step = 0
        self.train_start = 10

    def make_network_for_sample(self):
        print("Make a Network for Sampling")
        # dim_of_curr_state, next_state, action, expected return, and td-errors
        hiddensize = 256
        self.input_size = self.args.state_size + self.args.action_size
        self.net = nn.Sequential(nn.Linear(self.input_size, hiddensize), nn.ReLU(),
                                 nn.Linear(hiddensize, 1), nn.ReLU()).to(device)

        self.optim = optim.Adam(
            self.parameters(),
            lr=0.001,
            weight_decay=0.0,
        )

    def push(self, state, action, reward, next_state, done, episode=0):
        if len(self.buffer_slow) == 0:
            self.make_network_for_sample()
        if len(self.buffer_slow) < self.capacity:
            self.buffer_slow.append(None)
        if len(self.buffer_fast) < int(self.capacity / 100):
            self.buffer_fast.append(None)
        self.buffer_slow[self.position_slow] = (state, action, reward, next_state, done)
        self.buffer_fast[self.position_fast] = (state, action, reward, next_state, done)
        self.position_slow = (self.position_slow + 1) % self.capacity
        self.position_fast = (self.position_fast + 1) % (int(self.capacity / 100))

    def sample(self, batch_size, beta=1.0, episode=1):
        if batch_size > len(self.buffer_slow):
            batch_size = len(self.buffer_slow)

        batch_slow = random.sample(self.buffer_slow, int(batch_size))
        batch_fast = random.sample(self.buffer_fast, int(batch_size))
        slow_states, slow_actions, slow_rewards, slow_next_states, slow_dones = map(np.stack, zip(*batch_slow))
        fast_states, fast_actions, fast_rewards, fast_next_states, fast_dones = map(np.stack, zip(*batch_fast))

        weights = torch.ones_like(torch.Tensor(fast_rewards).reshape(-1)).to(device)
        if episode >= self.train_start:
            slow_input = torch.cat((torch.Tensor(slow_states), torch.Tensor(slow_actions)), -1).to(device)
            fast_input = torch.cat((torch.Tensor(fast_states), torch.Tensor(fast_actions)), -1).to(device)

            slow_w = self.net(slow_input.detach()).reshape(-1)
            fast_w = self.net(fast_input.detach()).reshape(-1)

            deriv_slow_f = torch.log(2 * slow_w / (1 + slow_w + 1e-9) + 1e-9) - 1
            deriv_fast_f = torch.log(2 * fast_w / (1 + fast_w + 1e-9) + 1e-9) - 1
            conju_slow_f = -torch.log(2 - torch.exp(deriv_slow_f))

            l1 = torch.mean(conju_slow_f)
            l2 = torch.mean(deriv_fast_f)
            loss = l1 - l2
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            weights = torch.pow(slow_w, 1 / self.temperature) / (
                torch.mean(torch.pow(slow_w, 1 / self.temperature)))
        return slow_states, slow_actions, slow_rewards, slow_next_states, slow_dones, weights.detach().cpu(), None

    def __len__(self):
        return len(self.buffer_fast)
