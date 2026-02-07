import os
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tools.utils import ensure_dir
from matplotlib.ticker import MaxNLocator


def set_style():
    # seaborn theme & palette
    sns.set_theme(style="darkgrid", palette="muted")

    # default figure size for all plots
    plt.rcParams["figure.figsize"] = (12, 6)

    # preferred sans-serif fonts
    plt.rcParams["font.sans-serif"] = ["Inter", "Roboto", "Arial"]


def plot_training(log_file="logs/training_logs.json", output_dir="plots"):
    """
    Plot CF-MADRL training metrics: mean_reward and mean_queue over federated rounds.

    Input log_file sample:
    [
        {
            "round": 1,
            "agent": "J1",
            "status": "trained",
            "cluster": -1,
            "mean_reward": -2.38,
            "mean_queue": 0.73,
            "mean_wait": 16.5
        },
        ...
    ]

    Output:
        - Reward plot per agent
        - Queue plot per agent
        Both saved to output_dir
    """

    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        return

    output_dir = os.path.join(output_dir, "train")
    ensure_dir(output_dir)

    # Load JSON logs
    with open(log_file, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    if df.empty or "mean_reward" not in df.columns:
        print("No training data found.")
        return

    # Only include fully trained entries
    df = df[df["status"] == "trained"]

    # Make round column integer
    df["round"] = df["round"].astype(int)

    # Shorten agent names for plotting
    df["Agent Short"] = df["agent"].apply(
        lambda x: x[:12] + "..." if len(x) > 12 else x
    )

    # Reward over Rounds
    plt.figure(figsize=(12, 6))

    # Plot each agent separately
    for agent in df["Agent Short"].unique():
        subset = df[df["Agent Short"] == agent].sort_values("round")
        plt.plot(subset["round"], subset["mean_reward"], "o-", label=agent, alpha=0.8)

    plt.title(
        "CF-MADRL Training: Reward per Round (Higher is Better)", fontweight="bold"
    )
    plt.xlabel("Federated Round")
    plt.ylabel("Mean Reward")
    plt.legend(title="Agent", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_training_rewards.png"), dpi=200)

    # Queue over Rounds
    if "mean_queue" in df.columns:
        plt.figure(figsize=(12, 6))
        for agent in df["Agent Short"].unique():
            subset = df[df["Agent Short"] == agent].sort_values("round")
            plt.plot(
                subset["round"], subset["mean_queue"], "o-", label=agent, alpha=0.8
            )

        plt.title(
            "CF-MADRL Training: Average Queue per Round (Lower is Better)",
            fontweight="bold",
        )
        plt.xlabel("Federated Round")
        plt.ylabel("Mean Queue Length")
        plt.legend(title="Agent", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_training_queue.png"), dpi=200)

    # --- System-Wide Averages (Single Line) ---

    # Calculate averages per round
    # We filter only numeric columns that exist in the dataframe
    numeric_metrics = ["mean_reward", "mean_queue", "mean_wait"]
    valid_metrics = [m for m in numeric_metrics if m in df.columns]

    if not valid_metrics:
        return

    avg_df = df.groupby("round")[valid_metrics].mean().reset_index()

    # 1. Average Reward
    if "mean_reward" in avg_df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=avg_df,
            x="round",
            y="mean_reward",
            marker="o",
            linewidth=3,
            color="#2ecc71",
        )
        plt.title(
            "System-Wide Average Reward (Higher is Better)",
            fontweight="bold",
            fontsize=14,
        )
        plt.xlabel("Federated Round", fontsize=12)
        plt.ylabel("Average Mean Reward", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_training_avg_reward.png"), dpi=200)

    # 2. Average Queue
    if "mean_queue" in avg_df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=avg_df,
            x="round",
            y="mean_queue",
            marker="o",
            linewidth=3,
            color="#e74c3c",
        )
        plt.title(
            "System-Wide Average Queue Length (Lower is Better)",
            fontweight="bold",
            fontsize=14,
        )
        plt.xlabel("Federated Round", fontsize=12)
        plt.ylabel("Average Queue Length", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_training_avg_queue.png"), dpi=200)

    # 3. Average Wait Time
    if "mean_wait" in avg_df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=avg_df,
            x="round",
            y="mean_wait",
            marker="o",
            linewidth=3,
            color="#3498db",
        )
        plt.title(
            "System-Wide Average Wait Time (Lower is Better)",
            fontweight="bold",
            fontsize=14,
        )
        plt.xlabel("Federated Round", fontsize=12)
        plt.ylabel("Average Wait Time (s)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_training_avg_wait.png"), dpi=200)


def plot_evaluation(log_file="logs/evaluation_logs.json", output_dir="plots"):
    """
    Plot comparison of agent performance: CF-MADRL vs Fixed-Time.

    Inputs:
      log_file: JSON log containing evaluation results per agent
        Example format:
        {
          "madrl": {
              "junction_1": {"queues": [3, 5, 2], "waits": [12, 15, 10]},
              "junction_2": {"queues": [4, 6, 3], "waits": [10, 11, 12]}
          },
          "fixed_time": {
              "junction_1": {"queues": [5, 7, 6], "waits": [20, 18, 22]},
              "junction_2": {"queues": [6, 8, 5], "waits": [15, 17, 16]}
          }
        }
      output_dir: folder to save plots

    Output:
      Bar plot comparing Avg Queue per agent for the two methods
    """

    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        return

    output_dir = os.path.join(output_dir, "eval")
    ensure_dir(output_dir)

    with open(log_file, "r") as f:
        data = json.load(f)

    madrl, fixed = data.get("madrl", {}), data.get("fixed_time", {})

    if not madrl:
        print("No MADRL data found in evaluation log.")
        return

    records = []

    # Process MADRL agent results
    for aid, m in madrl.items():
        records.append(
            {
                "Agent": aid,
                "Method": "CF-MADRL",
                "Avg Queue": np.mean(m["queues"]),
                "Avg Wait": np.mean(m["waits"]),
                "Total Cost": -np.mean(m["rewards"]),
            }
        )

    # Process Fixed-Time agent results
    for aid, m in fixed.items():
        records.append(
            {
                "Agent": aid,
                "Method": "Fixed-Time",
                "Avg Queue": np.mean(m["queues"]),
                "Avg Wait": np.mean(m["waits"]),
                "Total Cost": -np.mean(m["rewards"]),
            }
        )

    df = pd.DataFrame(records)

    # Shorten agent names for readability
    df["Agent Short"] = df["Agent"].apply(lambda x: x[:12] + ".." if len(x) > 12 else x)

    # Plot Cost per agent
    plt.figure(figsize=(12, 6))
    # Example of how bars look:
    # junction_1: CF-MADRL ~ 3.33, Fixed-Time ~ 6.0
    # junction_2: CF-MADRL ~ 4.33, Fixed-Time ~ 6.33
    sns.barplot(
        data=df,
        x="Agent Short",
        y="Total Cost",
        hue="Method",
        palette=["#2ecc71", "#e74c3c"],  # green = MADRL, red = Fixed-Time
    )

    plt.title(
        "System Performance: Total Traffic Cost (Lower is Better)", fontweight="bold"
    )
    plt.xticks(rotation=45)
    plt.ylabel("Average Queue Length")
    plt.xlabel("Agent")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_eval_cost_comparison.png"), dpi=200)
    print(
        f"Cost comparison plot saved to: {os.path.join(output_dir, 'plot_eval_cost_comparison.png')}"
    )

    # Plot Queue Length Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="Agent Short",
        y="Avg Queue",
        hue="Method",
        palette=["#2ecc71", "#e74c3c"],
    )
    plt.title(
        "System Performance: Average Queue Length (Lower is Better)", fontweight="bold"
    )
    plt.xticks(rotation=45)
    plt.ylabel("Average Queue Length")
    plt.xlabel("Agent")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_eval_queue_comparison.png"), dpi=200)
    print(
        f"Queue comparison plot saved to: {os.path.join(output_dir, 'plot_eval_queue_comparison.png')}"
    )

    # Plot Waiting Time Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="Agent Short",
        y="Avg Wait",
        hue="Method",
        palette=["#2ecc71", "#e74c3c"],
    )
    plt.title(
        "System Performance: Average Waiting Time (Lower is Better)", fontweight="bold"
    )
    plt.xticks(rotation=45)
    plt.ylabel("Average Waiting Time (s)")
    plt.xlabel("Agent")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_eval_wait_comparison.png"), dpi=200)
    print(
        f"Wait time comparison plot saved to: {os.path.join(output_dir, 'plot_eval_wait_comparison.png')}"
    )


def plot_clusters(log_file="logs/training_logs.json", output_dir="plots"):
    """
    Plot cluster assignments of agents across federated rounds.

    Input log_file sample:
    [
        {
            "round": 1,
            "agent": "J1",
            "status": "trained",
            "cluster": 0,
            "mean_reward": -2.38,
            "mean_queue": 0.73
        },
        ...
    ]

    Output:
        - Scatter plot showing which agent belongs to which cluster in each round.
        - Saved as 'plot_clusters.png' in output_dir.
    """

    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        return

    output_dir = os.path.join(output_dir, "train")
    ensure_dir(output_dir)

    # Load JSON logs
    with open(log_file, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Failed to read JSON.")
            return

    df = pd.DataFrame(data)

    # Ensure there is cluster information
    if df.empty or "cluster" not in df.columns:
        print("No cluster info found.")
        return

    # Only include trained entries
    df = df[df["status"] == "trained"]

    # Make round column integer
    df["round"] = df["round"].astype(int)

    # Shorten agent names for plotting
    df["Agent Short"] = df["agent"].apply(
        lambda x: x[:12] + "..." if len(x) > 12 else x
    )

    plt.figure(figsize=(14, 8))
    plt.gca().set_facecolor("#f9f9f9")  # subtle background for clarity

    # Scatter plot: x = round, y = agent, color = cluster
    # Example: round 1: J1 in cluster 0, J2 in cluster 1
    sns.scatterplot(
        data=df,
        x="round",
        y="Agent Short",
        hue="cluster",
        palette="bright",
        s=150,
        marker="p",
        edgecolor="black",
        linewidth=1.5,
        style="cluster",  # marker style per cluster
        legend="full",
    )

    # Force integer ticks for rounds
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(
        "CF-MADRL: Agent Cluster Evolution",
        fontsize=16,
        fontweight="bold",
        color="#2c3e50",
    )
    plt.xlabel("Federated Round", fontsize=12)
    plt.ylabel("Agent / Network Junction", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.5)

    plt.legend(
        title="Cluster Group",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        shadow=True,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_clusters.png"), dpi=200)
    print("Cluster plot saved to:", os.path.join(output_dir, "plot_clusters.png"))


if __name__ == "__main__":
    # Set global plotting style (dark grid, muted colors, larger figure)
    set_style()

    # Argument parser allows controlling which plots to generate
    parser = argparse.ArgumentParser(description="CF-MADRL Plotting Utility")

    # --type : choose which plots to generate
    # "train" => training metrics (reward, queue)
    # "eval" => evaluation metrics (avg queue, wait time)
    # "all"   => both
    parser.add_argument(
        "--type",
        type=str,
        default="all",
        choices=["train", "eval", "all"],
        help="Type of plots to generate: 'train', 'eval', or 'all'",
    )

    # --output : directory to save generated plots
    parser.add_argument(
        "--output", type=str, default="plots", help="Directory to save plots"
    )

    args = parser.parse_args()

    # Example usage:
    # python visualize.py --type train --output my_plots

    # Generate training plots (reward and queue dynamics)
    if args.type in ["train", "all"]:
        plot_training(output_dir=args.output)
        plot_clusters(output_dir=args.output)

    # Generate evaluation plots (avg queue, wait time comparison)
    if args.type in ["eval", "all"]:
        plot_evaluation(output_dir=args.output)
