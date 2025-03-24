import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, n_branches, buffer_size=1000):
        self.buffer = deque(maxlen=buffer_size)
        self.n_branches = n_branches

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample_batch = random.sample(
            self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*sample_batch)

        # Stack states and next_states assuming they are numpy arrays with consistent shapes.
        states = np.stack(states)             # shape: (batch_size, state_dim)
        next_states = np.stack(next_states)     # shape: (batch_size, state_dim)
        
        # 'actions' is expected to be an array-like of length num_branches per experience.
        # Stack to form an array of shape: (batch_size, num_branches)
        actions_array = np.stack(actions)
        
        # Split actions_array into a list of arrays, one per branch.
        # Each branch_actions[i] will have shape: (batch_size, 1)
        # branch_actions = [actions_array[:, i].reshape(-1, 1) for i in range(self.n_branches)]
        branch_actions = np.full((self.n_branches, batch_size), -1)
        for i in range(self.n_branches):
            for j in range(batch_size):
                branch_actions[i, j] = actions_array[j]

        rewards = np.array(rewards, dtype=np.float32)   # shape: (batch_size,)
        dones = np.array(dones, dtype=np.bool_)           # shape: (batch_size,)
        
        return states, branch_actions, rewards, next_states, dones
    
    # def sample(self, batch_size):
    #     sample_batch = random.sample(
    #         self.buffer, min(len(self.buffer), batch_size))
    #     states, actions, rewards, next_states, dones = zip(*sample_batch)

    #     # Stack states and next_states assuming they are numpy arrays with consistent shapes.
    #     states = np.stack(states)             # shape: (batch_size, state_dim)
    #     next_states = np.stack(next_states)     # shape: (batch_size, state_dim)
        
    #     # 'actions' is expected to be an array-like of length num_branches per experience.
    #     # Stack to form an array of shape: (batch_size, num_branches)
    #     actions_array = np.stack(actions)

    #     rewards = np.array(rewards, dtype=np.float32)   # shape: (batch_size,)
    #     dones = np.array(dones, dtype=np.bool_)           # shape: (batch_size,)
        
    #     return states, actions_array, rewards, next_states, dones

    def size(self):
        return len(self.buffer)
