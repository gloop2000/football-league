import csv
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_win_rates_from_csv(csv_path, name_A, name_B, save_dir, log_scale=True):
    episodes, winners = [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["episode"])
            winner = row["winner"]
            episodes.append(ep)
            winners.append(winner)

    # Count outcomes
    A_win_rate, B_win_rate, draw_rate = [], [], []
    counts = {"A": 0, "B": 0, "draw": 0}

    for i, winner in enumerate(winners, start=1):
        if winner == name_A:
            counts["A"] += 1
        elif winner == name_B:
            counts["B"] += 1
        else:
            counts["draw"] += 1

        A_win_rate.append(counts["A"] / i)
        B_win_rate.append(counts["B"] / i)
        draw_rate.append(counts["draw"] / i)

    # --- Logarithmic sampling ---
    if log_scale:
        # Choose points at 1,2,4,8,16... episodes
        log_indices = np.unique(np.logspace(0, np.log10(len(episodes)), num=200, dtype=int))
        episodes = [episodes[i-1] for i in log_indices if i-1 < len(episodes)]
        A_win_rate = [A_win_rate[i-1] for i in log_indices if i-1 < len(A_win_rate)]
        B_win_rate = [B_win_rate[i-1] for i in log_indices if i-1 < len(B_win_rate)]
        draw_rate = [draw_rate[i-1] for i in log_indices if i-1 < len(draw_rate)]

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, A_win_rate, label=f"{name_A} Win Rate", color="blue")
    plt.plot(episodes, B_win_rate, label=f"{name_B} Win Rate", color="red")
    plt.plot(episodes, draw_rate, label="Draw Rate", color="gray", linestyle='--')

    plt.xscale('log' if log_scale else 'linear')
    plt.xlabel("Episodes (log scale)" if log_scale else "Episodes")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{name_A}_vs_{name_B}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")
