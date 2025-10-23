import random
import csv
from collections import defaultdict
from football.football_env import FootballEnv
import os
from datetime import datetime

class LeagueManager:
    def __init__(self, agents, episodes_per_match=100, train_during_league=False, k_elo=32, should_render=False):
        self.agents = {agent.name: agent for agent in agents}
        self.episodes_per_match = episodes_per_match
        self.train_during_league = train_during_league
        self.k_elo = k_elo
        self.elo = {agent.name: 1000 for agent in agents}
        self.results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})

    def run_match(self, agent_1, agent_2):
        env = FootballEnv(agent_names=[agent_1.name, agent_2.name])
        agent_1.set_opponent(agent_2.name)
        agent_2.set_opponent(agent_1.name)
        name_1, name_2 = agent_1.name, agent_2.name
        wins_1 = wins_2 = draws = 0
        if self.should_render:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            savedir = f"./renders/{agent_1.name}_vs_{agent_2.name}_{timestamp_str}"
            os.makedirs(savedir, exist_ok=True)

        for _ in range(self.episodes_per_match):
            obs = env.reset()
            done = False

            while not done:
                action_1 = agent_1.act(obs)
                action_2 = agent_2.act(obs)
                actions = {name_1: action_1, name_2: action_2}
                next_obs, reward, done = env.step(action_1, action_2)

                if self.train_during_league:
                    agent_1.observe(obs, actions, reward, next_obs, done)
                    agent_2.observe(obs, actions, reward, next_obs, done)
                if self.should_render:
                    env.render(saveDir=savedir)
                obs = next_obs

            # Determine winner
            if reward[name_1] > reward[name_2]:
                wins_1 += 1
            elif reward[name_2] > reward[name_1]:
                wins_2 += 1
            else:
                draws += 1

        # Update raw stats
        self.results[name_1]['wins'] += wins_1
        self.results[name_1]['losses'] += wins_2
        self.results[name_1]['draws'] += draws

        self.results[name_2]['wins'] += wins_2
        self.results[name_2]['losses'] += wins_1
        self.results[name_2]['draws'] += draws

        # ELO Update
        self._update_elo(name_1, name_2, wins_1, wins_2, draws)

    def _update_elo(self, A, B, A_wins, B_wins, draws):
        total = A_wins + B_wins + draws
        score_A = (A_wins + 0.5 * draws) / total
        score_B = (B_wins + 0.5 * draws) / total

        expected_A = 1 / (1 + 10 ** ((self.elo[B] - self.elo[A]) / 400))
        expected_B = 1 - expected_A

        self.elo[A] += self.k_elo * (score_A - expected_A)
        self.elo[B] += self.k_elo * (score_B - expected_B)

    def run_round_robin(self):
        agent_names = list(self.agents.keys())
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                A = self.agents[agent_names[i]]
                B = self.agents[agent_names[j]]
                print(f"\nMatch: {A.name} vs {B.name}")
                self.run_match(A, B)

    def get_leaderboard(self):
        table = []
        for name in self.agents:
            stats = self.results[name]
            table.append({
                'Agent': name,
                'ELO': round(self.elo[name]),
                'Wins': stats['wins'],
                'Losses': stats['losses'],
                'Draws': stats['draws']
            })
        return sorted(table, key=lambda x: x['ELO'], reverse=True)

    def print_leaderboard(self):
        leaderboard = self.get_leaderboard()
        print("\nLeague Leaderboard")
        print("-" * 40)
        print(f"{'Agent':<12} {'ELO':<6} {'W':<5} {'L':<5} {'D':<5}")
        for row in leaderboard:
            print(f"{row['Agent']:<12} {row['ELO']:<6} {row['Wins']:<5} {row['Losses']:<5} {row['Draws']:<5}")

    def export_leaderboard(self, path="leaderboard.csv"):
        leaderboard = self.get_leaderboard()
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Agent", "ELO", "Wins", "Losses", "Draws"])
            writer.writeheader()
            for row in leaderboard:
                writer.writerow(row)
        print(f"Leaderboard exported to {path}")