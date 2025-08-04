import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent
from utils.constants import ACTIONS, NUM_ACTIONS

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_ACTIONS),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class PPOAgent(BaseAgent):
    def __init__(self, name, lr=1e-3, gamma=0.99):
        super().__init__(name)
        self.gamma = gamma
        self.policy = PolicyNetwork(input_dim=5)  # [x_self, y_self, x_opp, y_opp, ball_owner]
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []

    def preprocess_state(self, obs):
        pos_self = obs['positions'][self.name]
        pos_opp = obs['positions'][self.get_opponent_name()]
        ball = 1 if obs['ball_owner'] == self.name else 0
        return torch.tensor([*pos_self, *pos_opp, ball], dtype=torch.float32)

    def act(self, obs):
        state_tensor = self.preprocess_state(obs)
        probs = self.policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()
        self.last_action_dist = dist
        self.last_state = state_tensor
        return ACTIONS[action_idx]

    def observe(self, obs, actions, reward, next_obs, done):
        r = reward[self.name]
        self.memory.append((self.last_state, self.last_action_dist, r))
        if done:
            self.update_policy()
            self.memory.clear()

    def update_policy(self):
        G = 0
        returns = []
        for (_, _, r) in reversed(self.memory):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for (state, dist, _), G in zip(self.memory, returns):
            log_prob = dist.log_prob(dist.sample())  # REINFORCE-style
            loss -= log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def set_opponent(self, opponent_name):
        self.opponent = opponent_name

    def get_opponent_name(self):
        return self.opponent

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
