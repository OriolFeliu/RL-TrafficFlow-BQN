import random
import numpy as np
import torch
from agent.agent_models import BQN
from agent.base_agent import BaseAgent
import torch.optim as optim
import torch.nn as nn


class BQNAgent(BaseAgent):
    def __init__(self,  state_size, action_size, n_branches, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, hidden_size=64, lr=1e-3, gamma=0.9):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.n_branches = n_branches

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-Network and target network
        self.model = BQN(state_size, action_size, n_branches,
                         hidden_size).to(self.device)
        self.target_model = BQN(state_size, action_size, n_branches,
                                hidden_size).to(self.device)
        self.update_target_model()  # initialize target network

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        # Copy weights from the main network to the target network
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return [random.randrange(self.action_size)
                    for _ in range(self.n_branches)]

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_branches = self.model(state_tensor)
        return [int(torch.argmax(q_branch).item()) for q_branch in q_branches]

    def train(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Transform to torch tensors
        states = torch.FloatTensor(states).to(self.device)
        # actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        ####################

        # q_values = self.model(states).gather(1, actions)

        # # Next Q values from target network (detach to avoid gradient flow)
        # next_q_values = self.target_model(
        #     next_states).detach().max(dim=1, keepdim=True)[0]
        # # Compute target Q values using the Bellman equation
        # target = rewards + (1 - dones) * self.gamma * next_q_values

        # loss = self.criterion(q_values, target)

        ####################

        ###################
        q_branches = self.model(states)
        next_q_branches = self.target_model(next_states)

        loss = 0
        for branch in range(self.n_branches):
            action_branch = actions[branch]
            action_branch = torch.LongTensor(
                action_branch).unsqueeze(1).to(self.device)
            q_values = q_branches[branch].gather(1, action_branch)

            with torch.no_grad():
                next_q_values = next_q_branches[branch].max(1, keepdim=True)[0]
                q_targets = rewards + self.gamma * (1 - dones) * next_q_values

            loss += self.criterion(q_values, q_targets)

        ###############################

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
