"""Plotting utilities for football training results."""
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size=500):
    """Compute rolling average with padding so curve starts at episode 0."""
    ma = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    # Pad start with first value so we keep same length as input
    pad = np.full(window_size-1, ma[0])
    return np.concatenate([pad, ma])

def plot_win_rates_from_csv(csv_path, name_A, name_B, save_dir, smooth=True):
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
    window_size = 500
    if smooth:
        # Choose points at 1,2,4,8,16... episodes
        A_win_rate = moving_average(A_win_rate, window_size)
        B_win_rate = moving_average(B_win_rate, window_size)
        draw_rate = moving_average(draw_rate, window_size)

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, A_win_rate, label=f"{name_A} Win Rate", color="blue")
    plt.plot(episodes, B_win_rate, label=f"{name_B} Win Rate", color="red")
    plt.plot(episodes, draw_rate, label="Draw Rate", color="gray", linestyle='--')

    plt.xscale('linear')
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Win Rate", fontsize=14)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{name_A}_vs_{name_B}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")
