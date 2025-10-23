'''
Grid size: configurable (default 5x5)

Two agents: Player A and Player B

One ball: only one player has possession at a time

Goal zones: left and right edges of the grid

Discrete actions: ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'KICK']

Episode ends when a player scores or max steps reached

'''

import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import os

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

    def render(self, delay=0.3, saveDir=None):
        fig, ax = plt.subplots()
        # === Define grid boundaries ===
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')  # make cells square

        ax.set_xticks(np.arange(-.5, self.grid_size, 1))
        ax.set_yticks(np.arange(-.5, self.grid_size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(linestyle="-", color='black')
        ax.grid(True)

        for name, pos in self.agent_positions.items():
            print(name, pos, self.steps)
            x, y = pos
            color = 'blue' if name == self.agent_names[0] else 'red'
            ax.scatter(x, y, s=500, c=color, marker='o', label=name)

        # Ball indicator
        bx, by = self.agent_positions[self.ball_owner]
        ax.scatter(bx, by, s=200, c='gold', marker='*', label='Ball')
        

        ax.legend( loc ='upper right', bbox_to_anchor =(1.35, 1), markerscale=0.5)
        plt.title(f"Step {self.steps} | Ball: {self.ball_owner}")
        plt.savefig(os.path.join(saveDir, f"step_{self.steps}.png"))
        plt.pause(delay)
        plt.close(fig)
