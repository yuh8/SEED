import numpy as np
from .reward_utils import discounted_cumulative_sums
from .CONSTS import (ATOM_LIST, CHARGES, BOND_NAMES,
                     MAX_NUM_ATOMS, FEATURE_DEPTH)


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, T, gamma=0.99, lam=0.95):
        num_act = len(ATOM_LIST) * len(CHARGES)
        num_bond = (MAX_NUM_ATOMS - 1) * len(BOND_NAMES)
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (T, MAX_NUM_ATOMS, MAX_NUM_ATOMS, FEATURE_DEPTH + 1), dtype=np.float32
        )
        self.action_buffer = np.zeros(T, dtype=np.int32)
        self.advantage_buffer = np.zeros(T, dtype=np.float32)
        self.reward_buffer = np.zeros(T, dtype=np.float32)
        self.return_buffer = np.zeros(T, dtype=np.float32)
        self.value_buffer = np.zeros(T, dtype=np.float32)
        self.logprobability_buffer = np.zeros(T, dtype=np.float32)
        self.logits_buffer = np.zeros((T, num_act + num_bond),
                                      dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability, logits):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.logits_buffer[self.pointer] = logits
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = \
            discounted_cumulative_sums(deltas, self.gamma * self.lam)

        # self.advantage_buffer[path_slice] = deltas
        self.return_buffer[path_slice] = \
            discounted_cumulative_sums(rewards, self.gamma)[:-1]
        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (np.mean(self.advantage_buffer),
                                         np.std(self.advantage_buffer))
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (self.observation_buffer,
                self.action_buffer,
                self.value_buffer,
                self.advantage_buffer,
                self.return_buffer,
                self.logprobability_buffer,
                self.logits_buffer)
