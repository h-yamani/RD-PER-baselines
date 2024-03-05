# -*- coding: utf-8 -*-
"""Replay buffer for baselines."""

from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch

from algorithms.common.helper_functions import get_n_step_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Attributes:
        obs_buf (np.ndarray): observations
        acts_buf (np.ndarray): actions
        rews_buf (np.ndarray): rewards
        next_obs_buf (np.ndarray): next observations
        done_buf (np.ndarray): dones
        n_step_buffer (deque): recent n transitions
        n_step (int): step size for n-step transition
        gamma (float): discount factor
        buffer_size (int): size of buffers
        batch_size (int): batch size for training
        demo_size (int): size of demo transitions
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition
    """

    def __init__(
            self,
            buffer_size: int, #int(1e6)
            batch_size: int = 32,
            gamma: float = 0.99,
            n_step: int = 1,
            demo: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = None,
    ):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            gamma (float): discount factor
            n_step (int): step size for n-step transition
            demo (list): transitions of human play
        """
        assert 0 < batch_size <= buffer_size
        assert 0.0 <= gamma <= 1.0
        assert 1 <= n_step <= buffer_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.obs_buf: np.ndarray = None
        self.acts_buf: np.ndarray = None
        self.rews_buf: np.ndarray = None
        self.next_obs_buf: np.ndarray = None
        self.done_buf: np.ndarray = None
        self.timesteps = np.zeros((self.buffer_size), dtype=np.int32)
        self.n_step_buffer: Deque = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma
        self.size = 0

        self.demo_size = len(demo) if demo else 0
        self.demo = demo
        self.length = 0
        self.idx = self.demo_size
        self.qval_list = np.zeros(self.buffer_size)
        self.unceratinty_list = np.zeros(self.buffer_size)
        self.priorities = np.zeros((self.buffer_size), dtype=np.float32)
        # demo may have empty tuple list [()]
        if self.demo and self.demo[0]:
            self.buffer_size += self.demo_size
            self.length += self.demo_size
            for idx, d in enumerate(self.demo):
                state, action, reward, next_state, done = d
                if idx == 0:
                    self._initialize_buffers(state, action)
                self.obs_buf[idx] = state
                self.acts_buf[idx] = np.array(action)
                self.rews_buf[idx] = reward
                self.next_obs_buf[idx] = next_state
                self.done_buf[idx] = done

    def add(
            self, transition, timestep, episode):
        """Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        """
        self.n_step_buffer.append(transition)
        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        if self.length == 0:
            state, action = transition[:2]
            if np.array(action).shape == ():
                action = np.array([action])
            self._initialize_buffers(state, action)

        # add a multi step transition
        reward, next_state, done = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action = self.n_step_buffer[0][:2]

        self.obs_buf[self.idx] = curr_state
        self.acts_buf[self.idx] = action
        self.rews_buf[self.idx] = reward
        self.next_obs_buf[self.idx] = next_state
        self.done_buf[self.idx] = done
        self.timesteps[self.idx] = timestep
        self.qval_list[self.idx] = 1.0
        self.unceratinty_list[self.idx] = 1.0
        self.priorities[self.idx] = 1.0
        self.idx = (self.idx + 1) % self.buffer_size
        self.length = min(self.length + 1, self.buffer_size)
        self.size = np.min([self.size + 1, self.buffer_size])

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def extend(
            self, transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]
    ):
        """Add experiences to memory."""
        for transition in transitions:
            self.add(transition)

    def sample(self, beta=0.4):
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size

        indices = np.random.choice(self.size, size=self.batch_size, replace=False)

        states = self.obs_buf[indices]
        actions = self.acts_buf[indices]
        rewards = self.rews_buf[indices].reshape(-1, 1)
        next_states = self.next_obs_buf[indices]
        dones = self.done_buf[indices].reshape(-1, 1)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        return states, actions, rewards, next_states, dones, indices

    def _initialize_buffers(self, state: np.ndarray, action: np.ndarray) -> None:
        """Initialze buffers for state, action, resward, next_state, done."""
        # In case action of demo is not np.ndarray

        self.obs_buf = np.zeros(
            [self.buffer_size] + list(state.shape), dtype=state.dtype
        )
        self.acts_buf = np.zeros(
            [self.buffer_size] + list(action.shape), dtype=action.dtype
        )
        self.rews_buf = np.zeros([self.buffer_size], dtype=float)
        self.next_obs_buf = np.zeros(
            [self.buffer_size] + list(state.shape), dtype=state.dtype
        )
        self.done_buf = np.zeros([self.buffer_size], dtype=float)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return self.length
