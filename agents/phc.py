import numpy as np
import random
from agents.base_agent import BaseAgent
from utils.constants import ACTIONS, NUM_ACTIONS

class PHCAgent(BaseAgent):
    def __init__(self, name, alpha=0.1, delta=0.01):
        super().__init__(name)
        self.q_table = {}        # state_key -> [NUM_ACTIONS x NUM_ACTIONS] Q-values
        self.pi = {}             # state_key -> action probabilities
        self.alpha = alpha       # Q-learning rate
        self.delta = delta       # Policy update rate

    def get_state_key(self, obs):
        pos_self = tuple(obs['positions'][self.name])
        pos_opp = tuple(obs['positions'][self.get_opponent_name()])
        ball = obs['ball_owner']
        return (pos_self, pos_opp, ball)

    def init_state_if_needed(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros((NUM_ACTIONS, NUM_ACTIONS))
            self.pi[state_key] = np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def act(self, obs):
        state_key = self.get_state_key(obs)
        self.init_state_if_needed(state_key)
        return np.random.choice(ACTIONS, p=self.pi[state_key])

    def observe(self, obs, actions, reward, next_obs, done):
        state_key = self.get_state_key(obs)
        next_state_key = self.get_state_key(next_obs)
        self.init_state_if_needed(state_key)
        self.init_state_if_needed(next_state_key)

        a_idx = ACTIONS.index(actions[self.name])
        o_idx = ACTIONS.index(actions[self.get_opponent_name()])

        q = self.q_table[state_key]
        q_next = self.q_table[next_state_key]

        target = reward[self.name] + (0 if done else np.max(q_next))
        q[a_idx, o_idx] += self.alpha * (target - q[a_idx, o_idx])

        self.update_policy(state_key)

    def update_policy(self, state_key):
        q = self.q_table[state_key]
        expected_q = np.mean(q, axis=1)  # average over opponent actions
        best_action = np.argmax(expected_q)
        for i in range(NUM_ACTIONS):
            if i == best_action:
                self.pi[state_key][i] += self.delta
            else:
                self.pi[state_key][i] -= self.delta / (NUM_ACTIONS - 1)
        self.pi[state_key] = np.clip(self.pi[state_key], 0, 1)
        self.pi[state_key] /= self.pi[state_key].sum()

    def set_opponent(self, opponent_name):
        self.opponent = opponent_name

    def get_opponent_name(self):
        return self.opponent

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump((self.q_table, self.pi), f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.q_table, self.pi = pickle.load(f)