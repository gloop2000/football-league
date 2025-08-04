'''
Grid size: configurable (default 5x5)

Two agents: Player A and Player B

One ball: only one player has possession at a time

Goal zones: left and right edges of the grid

Discrete actions: ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'KICK']

Episode ends when a player scores or max steps reached

'''

import numpy as np
import random

class FootballEnv:
    def __init__(self, grid_size=5, max_steps=100, agent_names=None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.agent_names = agent_names if agent_names else ['A', 'B']
        self.reset()

    def reset(self):
        # Grid positions: [x, y]
        self.agent_positions = {
            self.agent_names[0]: [0, self.grid_size // 2],
            self.agent_names[1]: [self.grid_size - 1, self.grid_size // 2]
        }
        self.ball_owner = random.choice(self.agent_names)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            'positions': self.agent_positions.copy(),
            'ball_owner': self.ball_owner
        }

    def _is_goal(self):
        # Player A scores if they reach far right, B if far left
        a_x, _ = self.agent_positions[self.agent_names[0]]
        b_x, _ = self.agent_positions[self.agent_names[1]]

        if self.ball_owner == self.agent_names[0] and a_x == self.grid_size - 1:
            return self.agent_names[0]
        elif self.ball_owner == self.agent_names[1] and b_x == 0:
            return self.agent_names[1]
        return None

    def _move(self, pos, action):
        x, y = pos
        if action == 'UP': y = max(0, y - 1)
        elif action == 'DOWN': y = min(self.grid_size - 1, y + 1)
        elif action == 'LEFT': x = max(0, x - 1)
        elif action == 'RIGHT': x = min(self.grid_size - 1, x + 1)
        return [x, y]

    def step(self, action_A, action_B):
        self.steps += 1

        # Move agents
        self.agent_positions[self.agent_names[0]] = self._move(self.agent_positions[self.agent_names[0]], action_A)
        self.agent_positions[self.agent_names[1]] = self._move(self.agent_positions[self.agent_names[1]], action_B)

        # Handle ball possession switch if players collide
        if self.agent_positions[self.agent_names[0]] == self.agent_positions[self.agent_names[1]]:
            self.ball_owner = self.agent_names[1] if self.ball_owner == self.agent_names[0] else self.agent_names[0]

        # Check for goal
        scorer = self._is_goal()
        done = scorer is not None or self.steps >= self.max_steps

        reward = {name: 0 for name in self.agent_names}
        if scorer:
            reward[scorer] = 1
            for name in self.agent_names:
                if name != scorer:
                    reward[name] = -1

        return self._get_obs(), reward, done

    def render(self):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for name, pos in self.agent_positions.items():
            symbol = name + ('*' if self.ball_owner == name else '')
            x, y = pos
            grid[y][x] = symbol

        print("\n".join([" ".join(row) for row in grid]))
        print(f"Ball: {self.ball_owner}, Steps: {self.steps}")
        print()
