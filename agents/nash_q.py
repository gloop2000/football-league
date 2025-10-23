import numpy as np
import random
import nashpy as nash
from agents.base_agent import BaseAgent
from utils.constants import ACTIONS, NUM_ACTIONS

class NashQAgent(BaseAgent):
    def __init__(self, name, alpha=0.1, gamma=0.9, epsilon=0.1, update_every=5):
        super().__init__(name)
        self.q_table = {}              # state_key → Q-matrix
        self.strategy_cache = {}       # state_key → (my_strategy, opp_strategy)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_every = update_every
        self.step_counter = 0

    def get_state_key(self, obs):
        pos_self = tuple(obs['positions'][self.name])
        pos_opp = tuple(obs['positions'][self.get_opponent_name()])
        ball = obs['ball_owner']
        return (pos_self, pos_opp, ball)

    def init_state_if_needed(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros((NUM_ACTIONS, NUM_ACTIONS))
            self.strategy_cache[state_key] = (np.ones(NUM_ACTIONS)/NUM_ACTIONS,
                                              np.ones(NUM_ACTIONS)/NUM_ACTIONS)

    def act(self, obs):
        state_key = self.get_state_key(obs)
        self.init_state_if_needed(state_key)

        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        self.step_counter += 1

        if self.step_counter % self.update_every == 0:
            q = self.q_table[state_key]
            self.strategy_cache[state_key] = self.compute_nash(q, q.T)

        my_strategy, _ = self.strategy_cache[state_key]
        action_idx = np.random.choice(NUM_ACTIONS, p=my_strategy)
        return ACTIONS[action_idx]

    def observe(self, obs, actions, reward, next_obs, done):
        state_key = self.get_state_key(obs)
        next_state_key = self.get_state_key(next_obs)
        self.init_state_if_needed(state_key)
        self.init_state_if_needed(next_state_key)

        a_idx = ACTIONS.index(actions[self.name])
        o_idx = ACTIONS.index(actions[self.get_opponent_name()])

        next_q = self.q_table[next_state_key]

        if self.step_counter % self.update_every == 0:
            self.strategy_cache[next_state_key] = self.compute_nash(next_q, next_q.T)

        my_strategy, opp_strategy = self.strategy_cache[next_state_key]
        nash_value = self.expected_value(next_q, my_strategy, opp_strategy)

        target = reward[self.name] + self.gamma * (0 if done else nash_value)
        self.q_table[state_key][a_idx, o_idx] += self.alpha * (target - self.q_table[state_key][a_idx, o_idx])

    def set_opponent(self, opponent_name):
        self.opponent = opponent_name

    def get_opponent_name(self):
        return self.opponent

    def compute_nash(self, my_payoff, opp_payoff):
        game = nash.Game(my_payoff, opp_payoff)
        try:
            eqs = list(game.support_enumeration())
            if not eqs:
                raise ValueError("No Nash equilibrium found")
            return np.array(eqs[0][0]), np.array(eqs[0][1])
        except:
            return np.ones(NUM_ACTIONS) / NUM_ACTIONS, np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def expected_value(self, q_matrix, p1, p2):
        return np.sum(p1[:, None] * q_matrix * p2[None, :])

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump((self.q_table, self.strategy_cache), f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.q_table, self.strategy_cache = pickle.load(f)
