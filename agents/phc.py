import numpy as np
import random
from collections import deque
from agents.base_agent import BaseAgent
from utils.constants import ACTIONS, NUM_ACTIONS

class PHCAgent(BaseAgent):
    def __init__(self, name, alpha=0.1, delta_win=0.001, delta_lose=0.01, 
                 window_size=100, gamma=0.99):
        super().__init__(name)
        self.q_table = {}        # state_key -> [NUM_ACTIONS x NUM_ACTIONS] Q-values
        self.pi = {}             # state_key -> action probabilities
        self.alpha = alpha       # Q-learning rate
        self.delta_win = delta_win    # Policy update rate when winning
        self.delta_lose = delta_lose  # Policy update rate when losing
        self.gamma = gamma       # Discount factor
        
        # WOLF mechanism components
        self.window_size = window_size
        self.recent_rewards = deque(maxlen=window_size)
        self.equilibrium_threshold = 0.0  # Expected reward at equilibrium
        self.total_episodes = 0
        
        # Performance tracking
        self.episode_reward = 0.0
        self.episode_steps = 0

    def get_state_key(self, obs):
        pos_self = tuple(obs['positions'][self.name])
        pos_opp = tuple(obs['positions'][self.get_opponent_name()])
        ball = obs['ball_owner']
        return (pos_self, pos_opp, ball)

    def init_state_if_needed(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros((NUM_ACTIONS, NUM_ACTIONS))
            # Initialize with uniform random policy
            self.pi[state_key] = np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def act(self, obs):
        state_key = self.get_state_key(obs)
        self.init_state_if_needed(state_key)
        
        # Sample action according to current policy
        action = np.random.choice(ACTIONS, p=self.pi[state_key])
        return action

    def observe(self, obs, actions, reward, next_obs, done):
        state_key = self.get_state_key(obs)
        next_state_key = self.get_state_key(next_obs)
        self.init_state_if_needed(state_key)
        self.init_state_if_needed(next_state_key)

        a_idx = ACTIONS.index(actions[self.name])
        o_idx = ACTIONS.index(actions[self.get_opponent_name()])

        # Q-learning update
        q = self.q_table[state_key]
        q_next = self.q_table[next_state_key]

        if done:
            target = reward[self.name]
        else:
            # Use expected value over opponent's policy for next state
            next_pi_opp = self.get_opponent_policy_estimate(next_state_key)
            expected_next_q = np.sum(np.max(q_next, axis=0) * next_pi_opp)
            target = reward[self.name] + self.gamma * expected_next_q

        q[a_idx, o_idx] += self.alpha * (target - q[a_idx, o_idx])

        # Track episode performance
        self.episode_reward += reward[self.name]
        self.episode_steps += 1

        # Update policy using WOLF principle
        self.update_policy_wolf(state_key)

        # Episode end processing
        if done:
            self.end_episode()

    def get_opponent_policy_estimate(self, state_key):
        """Estimate opponent's policy (assume uniform for random opponent)"""
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def is_winning(self):
        """WOLF mechanism: determine if agent is winning or losing"""
        if len(self.recent_rewards) < self.window_size // 2:
            return False  # Not enough data, assume losing (learn fast)
        
        recent_avg = np.mean(list(self.recent_rewards)[-self.window_size//2:])
        return recent_avg > self.equilibrium_threshold

    def update_equilibrium_threshold(self):
        """Update the equilibrium threshold based on recent performance"""
        if len(self.recent_rewards) >= self.window_size:
            # Set threshold as moving average of rewards
            self.equilibrium_threshold = np.mean(self.recent_rewards) * 0.9
        else:
            # Conservative threshold early in training
            self.equilibrium_threshold = -0.1

    def update_policy_wolf(self, state_key):
        """Update policy using WOLF principle"""
        q = self.q_table[state_key]
        
        # Compute expected Q-values for each action (average over opponent actions)
        expected_q = np.mean(q, axis=1)
        
        # Find the best action
        best_action = np.argmax(expected_q)
        
        # Determine learning rate based on WOLF principle
        if self.is_winning():
            delta = self.delta_win   # Learn slowly when winning
        else:
            delta = self.delta_lose  # Learn fast when losing
        
        # Update policy: increase probability of best action, decrease others
        old_pi = self.pi[state_key].copy()
        
        for i in range(NUM_ACTIONS):
            if i == best_action:
                self.pi[state_key][i] += delta
            else:
                # Distribute the decrease among other actions
                self.pi[state_key][i] -= delta / (NUM_ACTIONS - 1)
        
        # Ensure probabilities remain valid
        self.pi[state_key] = np.clip(self.pi[state_key], 1e-8, 1.0)
        
        # Normalize to maintain probability distribution
        self.pi[state_key] /= np.sum(self.pi[state_key])

    def end_episode(self):
        """Process end of episode for WOLF mechanism"""
        # Calculate average reward for this episode
        if self.episode_steps > 0:
            avg_episode_reward = self.episode_reward / self.episode_steps
        else:
            avg_episode_reward = 0.0
        
        # Add to recent rewards for WOLF mechanism
        self.recent_rewards.append(avg_episode_reward)
        
        # Update equilibrium threshold
        self.update_equilibrium_threshold()
        
        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.total_episodes += 1

    def get_policy_entropy(self, state_key):
        """Calculate policy entropy for analysis"""
        pi = self.pi[state_key]
        entropy = -np.sum(pi * np.log(pi + 1e-8))
        return entropy

    def get_learning_rate_info(self):
        """Get current learning rate status for debugging"""
        return {
            'is_winning': self.is_winning(),
            'current_delta': self.delta_win if self.is_winning() else self.delta_lose,
            'equilibrium_threshold': self.equilibrium_threshold,
            'recent_avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            'episodes': self.total_episodes
        }

    def set_opponent(self, opponent_name):
        self.opponent = opponent_name

    def get_opponent_name(self):
        return self.opponent

    def save(self, path):
        import pickle
        data = {
            'q_table': self.q_table,
            'pi': self.pi,
            'recent_rewards': list(self.recent_rewards),
            'equilibrium_threshold': self.equilibrium_threshold,
            'total_episodes': self.total_episodes
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.pi = data['pi']
        self.recent_rewards = deque(data['recent_rewards'], maxlen=self.window_size)
        self.equilibrium_threshold = data['equilibrium_threshold']
        self.total_episodes = data['total_episodes']