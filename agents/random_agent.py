import random
from agents.base_agent import BaseAgent

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']

class RandomAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)

    def act(self, observation):
        return random.choice(ACTIONS)

    def observe(self, obs, actions, reward, next_obs, done):
        # Random agent does not learn, so it ignores all inputs
        pass

    def set_opponent(self, opponent_name):
        # Random agent does not need to know the opponent
        pass

    def get_opponent_name(self):
        # Random agent does not need to know the opponent
        return None
