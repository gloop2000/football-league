import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent
from utils.constants import ACTIONS, NUM_ACTIONS

# Basic MLP Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_ACTIONS)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent(BaseAgent):
    def __init__(self, name, lr=1e-3, gamma=0.99, epsilon=0.1, buffer_size=10000, batch_size=64, device='cpu'):
        super().__init__(name)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = device

        self.buffer = deque(maxlen=buffer_size)

        self.q_net = QNetwork(input_dim=5).to(device)  # input: posA(x,y), posB(x,y), ball_owner
        self.target_net = QNetwork(input_dim=5).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.steps = 0
        self.update_target_freq = 100

    def get_state_vector(self, obs):
        pos_a = obs['positions'][self.name]
        pos_b = obs['positions'][self.get_opponent_name()]
        ball_owner = 1 if obs['ball_owner'] == self.name else 0
        return torch.tensor(pos_a + pos_b + [ball_owner], dtype=torch.float32).to(self.device)

    def act(self, obs):
        state_vec = self.get_state_vector(obs)
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        with torch.no_grad():
            q_values = self.q_net(state_vec)
        action_idx = q_values.argmax().item()
        return ACTIONS[action_idx]

    def observe(self, obs, actions, reward, next_obs, done):
        state = self.get_state_vector(obs)
        next_state = self.get_state_vector(next_obs)
        action = ACTIONS.index(actions[self.name])
        r = reward[self.name]

        self.buffer.append((state, action, r, next_state, done))
        self.train_step()

        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        q_values = self.q_net(states)
        q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * max_next_q * (~dones)

        loss = self.loss_fn(q_action, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())

    def set_opponent(self, opponent_name):
        self.opponent = opponent_name

    def get_opponent_name(self):
        return self.opponent

