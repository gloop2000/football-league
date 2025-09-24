import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent
from utils.constants import ACTIONS, NUM_ACTIONS

# Actor network
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

# Critic network (value function)
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class PPOAgent(BaseAgent):
    def __init__(self, name, lr=3e-4, gamma=0.99, lam=0.95, clip_epsilon=0.2,
                 epochs=4, batch_size=64, entropy_coef=0.01):
        super().__init__(name)
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

        self.input_dim = 5  # [x_self, y_self, x_opp, y_opp, ball_owner]
        self.policy = PolicyNetwork(self.input_dim)
        self.value_fn = ValueNetwork(self.input_dim)

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_fn.parameters(), lr=lr)

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
        action_idx = dist.sample()
        action = ACTIONS[action_idx.item()]

        self.memory.append({
            'state': state_tensor,
            'action': action_idx,
            'log_prob': dist.log_prob(action_idx)
        })
        return action

    def observe(self, obs, actions, reward, next_obs, done):
        r = reward[self.name]
        self.memory[-1]['reward'] = r
        self.memory[-1]['done'] = done
        if done:
            self.update_policy()

    def update_policy(self):
        # Convert memory to tensors
        states = torch.stack([m['state'] for m in self.memory])
        actions = torch.stack([m['action'] for m in self.memory])
        rewards = [m['reward'] for m in self.memory]
        dones = [m['done'] for m in self.memory]
        log_probs_old = torch.stack([m['log_prob'] for m in self.memory])

        # Compute advantages with GAE
        values = self.value_fn(states).squeeze().detach()
        advantages, returns = self.compute_gae(rewards, values, dones)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            # New probabilities
            probs = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Ratio for clipping
            ratio = torch.exp(log_probs - log_probs_old.detach())

            # PPO surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            # Value loss
            value_loss = (returns - self.value_fn(states).squeeze()).pow(2).mean()

            # Update actor
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            # Update critic
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

        self.memory.clear()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = torch.cat([values, torch.tensor([0.0])])
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values[:-1]  # ✅ element-wise sum
        return advantages, returns


    def set_opponent(self, opponent_name):
        self.opponent = opponent_name

    def get_opponent_name(self):
        return self.opponent

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'value_fn': self.value_fn.state_dict()
        }, path)

    def load(self, path):
        data = torch.load(path)
        self.policy.load_state_dict(data['policy'])
        self.value_fn.load_state_dict(data['value_fn'])
