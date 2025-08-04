import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from football.football_env import FootballEnv

def train_agents(agent_1, agent_2, episodes=1000, save_dir=None, verbose=True, log_to_csv=True, plot_results=True):
    env = FootballEnv(agent_names=[agent_1.name, agent_2.name])
    win_counts = defaultdict(int)
    history = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "training_log.csv")
    else:
        log_path = "training_log.csv"

    if log_to_csv:
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "winner", agent_1.name, agent_2.name])

    for episode in range(episodes):
        obs = env.reset()
        done = False

        while not done:
            action_1 = agent_1.act(obs)
            action_2 = agent_2.act(obs)
            actions = {agent_1.name: action_1, agent_2.name: action_2}
            next_obs, reward, done = env.step(action_1, action_2)

            agent_1.observe(obs, actions, reward, next_obs, done)
            agent_2.observe(obs, actions, reward, next_obs, done)

            obs = next_obs

        # Determine winner
        if reward[agent_1.name] > reward[agent_2.name]:
            winner = agent_1.name
        elif reward[agent_2.name] > reward[agent_1.name]:
            winner = agent_2.name
        else:
            winner = 'draw'
        win_counts[winner] += 1
        history.append(winner)

        if log_to_csv:
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode + 1, winner, reward[agent_1.name], reward[agent_2.name]])

        if verbose and (episode + 1) % 100 == 0:
            print(f"[{episode + 1}] {agent_1.name} : {win_counts[agent_1.name]}, {agent_2.name}: {win_counts[agent_2.name]}, Draws: {win_counts['draw']}")

    if save_dir:
        agent_1.save(os.path.join(save_dir, f"{agent_1.name}_{episodes}_model.pkl"))
        agent_2.save(os.path.join(save_dir, f"{agent_2.name}_{episodes}_model.pkl"))
        print(f"Models saved to {save_dir}")

    if plot_results:
        plot_win_rates(history, agent_1.name, agent_2.name, save_dir)

    return dict(win_counts)

def plot_win_rates(history, name_A, name_B, save_dir, smooth_window=50):
    A_win_rate, B_win_rate, draws = [], [], []
    for i in range(1, len(history) + 1):
        A_wins = sum(1 for h in history[:i] if h == name_A) / i
        B_wins = sum(1 for h in history[:i] if h == name_B) / i
        draw_rate = sum(1 for h in history[:i] if h == 'draw') / i
        A_win_rate.append(A_wins)
        B_win_rate.append(B_wins)
        draws.append(draw_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(A_win_rate, label=f"{name_A} Win Rate", color="blue")
    plt.plot(B_win_rate, label=f"{name_B} Win Rate", color="red")
    plt.plot(draws, label="Draw Rate", color="gray", linestyle='--')
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.title("Win Rate Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f"{name_A}_vs_{name_B}.png"))
