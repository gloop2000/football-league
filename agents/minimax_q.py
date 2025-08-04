import numpy as np
from scipy.optimize import linprog
import random
from agents.base_agent import BaseAgent
from utils.constants import ACTIONS, NUM_ACTIONS

class MinimaxQAgent(BaseAgent):
    def __init__(self, name, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(name)
        self.q_table = {}  # state → [NUM_ACTIONS x NUM_ACTIONS] Q-values
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, obs):
        pos_A = tuple(obs['positions'][self.name])
        pos_B = tuple(obs['positions'][self.get_opponent_name()])
        ball = obs['ball_owner']
        return (pos_A, pos_B, ball)

    def init_state_if_needed(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros((NUM_ACTIONS, NUM_ACTIONS))

    def act(self, obs):
        state_key = self.get_state_key(obs)
        self.init_state_if_needed(state_key)

        # Epsilon-greedy for exploration
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        q = self.q_table[state_key]

        # Get minimax strategy using linear programming
        strategy = self.solve_minimax(q)

        # Sample action from strategy
        action_idx = np.random.choice(NUM_ACTIONS, p=strategy)
        return ACTIONS[action_idx]

    def observe(self, obs, action, reward, next_obs, done):
        state_key = self.get_state_key(obs)
        next_state_key = self.get_state_key(next_obs)
        self.init_state_if_needed(state_key)
        self.init_state_if_needed(next_state_key)

        a_idx = ACTIONS.index(action[self.name])
        o_idx = ACTIONS.index(action[self.get_opponent_name()])

        # Estimate value of next state using minimax strategy
        next_q = self.q_table[next_state_key]
        next_value = self.get_minimax_value(next_q)

        # Q-learning update
        target = reward[self.name] + self.gamma * (0 if done else next_value)
        self.q_table[state_key][a_idx, o_idx] += self.alpha * (target - self.q_table[state_key][a_idx, o_idx])

    def get_minimax_value(self, q_matrix):
        strategy = self.solve_minimax(q_matrix)
        return np.dot(strategy, np.min(q_matrix, axis=1))

    def solve_minimax(self, q_matrix):
        """
        Solve min-max problem using linear programming:
        maximize v, subject to Q.T @ pi >= v, sum(pi)=1, pi>=0
        """
        c = [0] * NUM_ACTIONS + [-1]  # Objective: maximize v → minimize -v
        A_ub = []
        b_ub = []

        for j in range(NUM_ACTIONS):
            constraint = [-q_matrix[i][j] for i in range(NUM_ACTIONS)] + [1]
            A_ub.append(constraint)
            b_ub.append(0)

        A_eq = [[1] * NUM_ACTIONS + [0]]
        b_eq = [1]

        bounds = [(0, 1) for _ in range(NUM_ACTIONS)] + [(None, None)]

        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs')

        if res.success:
            return np.array(res.x[:NUM_ACTIONS])
        else:
            return np.ones(NUM_ACTIONS) / NUM_ACTIONS  # fallback uniform

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

    def set_opponent(self, opponent_name):
        self.opponent = opponent_name

    def get_opponent_name(self):
        return self.opponent