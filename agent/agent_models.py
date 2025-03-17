import torch.nn as nn
import torch.nn.functional as F


# BQN Network
class BQN(nn.Module):
    def __init__(self, state_size, action_size, n_branches, hidden_size):
        super().__init__()

        self.n_branches = n_branches

        # Shared layers (the trunk)
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)

        # Value branch
        self.value_stream = nn.Linear(hidden_size, 1)

        # Create a branch (head) for each action branch
        self.advantage_streams = nn.ModuleList([nn.Linear(hidden_size, action_size)
                                                for _ in range(n_branches)])

    def forward(self, x):
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))

        V = self.value_stream(x)

        # Advantages for branching nodes
        A_branches = [self.advantage_streams[branch](x)
                      for branch in range(self.n_branches)]

        # Advantages for branching nodes
        Q_branches = [V + (A - A.mean(dim=1, keepdim=True))
                      for A in A_branches]

        return Q_branches
