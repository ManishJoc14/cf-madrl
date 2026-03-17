import os
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tools.utils import ensure_dir
from matplotlib.ticker import MaxNLocator


# ==========================================================
# Global Style
# ==========================================================
def set_style():
    """
    Configure consistent, publication-quality plotting style.
    """

    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="colorblind",
    )

    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Roboto", "Arial"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "lines.linewidth": 2.5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ==========================================================
# Training Plots
# ==========================================================
def plot_training(
    log_file="logs/training_logs.json",
    output_dir="plots",
    agent_name=None,
    algo_name="Traffic RL",
    sma_window=5,
):

    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        return

    if agent_name:
        output_dir = os.path.join(output_dir, agent_name)
    output_dir = os.path.join(output_dir, "train")
    ensure_dir(output_dir)

    try:
        with open(log_file, "r") as f:
            data = json.load(f)
    except Exception:
        print("Failed to read training JSON.")
        return

    df = pd.DataFrame(data)

    if df.empty or "mean_reward" not in df.columns:
        print("No training data found.")
        return

    df = df[df["status"] == "trained"].copy()
    df["round"] = df["round"].astype(int)
    df["Agent Short"] = df["agent"].apply(lambda x: x[:12] + ".." if len(x) > 12 else x)

    # --------------------------------------------------
    # Agent-Level Plot
    # --------------------------------------------------
    def plot_per_agent(metric, ylabel, filename):

        if metric not in df.columns:
            return

        plt.figure()

        for agent in sorted(df["Agent Short"].unique()):
            subset = df[df["Agent Short"] == agent].sort_values("round")
            sma = subset[metric].rolling(window=sma_window, min_periods=1).mean()

            plt.plot(subset["round"], sma, label=agent)

        plt.title(f"{algo_name} Training: {ylabel} Trend (SMA={sma_window})")
        plt.xlabel("Training Round")
        plt.ylabel(ylabel)
        plt.legend(title="Agent", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    plot_per_agent("mean_reward", "Mean Reward", "plot_training_rewards.png")
    plot_per_agent("mean_queue", "Mean Queue Length", "plot_training_queue.png")

    # --------------------------------------------------
    # System-Wide Averages
    # --------------------------------------------------
    numeric_metrics = ["mean_reward", "mean_queue", "mean_wait"]
    valid_metrics = [m for m in numeric_metrics if m in df.columns]

    if not valid_metrics:
        return

    avg_df = df.groupby("round")[valid_metrics].mean().reset_index()

    def plot_system_trend(metric, ylabel, color, filename):

        if metric not in avg_df.columns:
            return

        plt.figure()

        sma = avg_df[metric].rolling(window=sma_window, min_periods=1).mean()

        plt.plot(avg_df["round"], sma, color=color, linewidth=3, label="System Trend")

        plt.title(f"{algo_name}: System {ylabel} Trend (SMA={sma_window})")
        plt.xlabel("Training Round")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    plot_system_trend(
        "mean_reward", "Mean Reward", "#2ecc71", "plot_training_avg_reward.png"
    )
    plot_system_trend(
        "mean_queue", "Mean Queue Length", "#e74c3c", "plot_training_avg_queue.png"
    )
    plot_system_trend(
        "mean_wait", "Mean Wait Time (s)", "#3498db", "plot_training_avg_wait.png"
    )

    print("\nTraining plots generated successfully.")


# ==========================================================
# Evaluation (Single Model vs Fixed-Time)
# ==========================================================
def plot_evaluation(
    log_file="logs/evaluation_logs.json",
    output_dir="plots",
    agent_name=None,
    algo_name="CF-MADRL",
    model_name="cfmadrl",
):

    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        return

    if agent_name:
        output_dir = os.path.join(output_dir, agent_name)
    output_dir = os.path.join(output_dir, "eval")
    ensure_dir(output_dir)

    try:
        with open(log_file, "r") as f:
            data = json.load(f)
    except Exception:
        print("Failed to read evaluation JSON.")
        return

    model = data.get(model_name, {})
    fixed = data.get("fixed_time", {})

    if not model:
        print(f"No {model_name} data found.")
        return

    records = []

    for aid, m in model.items():
        records.append(
            {
                "Agent": aid,
                "Method": algo_name,
                "Avg Queue": np.mean(m["queues"]),
                "Avg Wait": np.mean(m["waits"]),
                "Total Cost": -np.mean(m["rewards"]),
            }
        )

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
    df["Agent Short"] = df["Agent"].apply(lambda x: x[:12] + ".." if len(x) > 12 else x)

    method_order = [algo_name, "Fixed-Time"]
    df["Method"] = pd.Categorical(df["Method"], categories=method_order, ordered=True)

    palette = {algo_name: "#2ecc71", "Fixed-Time": "#e74c3c"}

    def create_bar_plot(metric_col, title, ylabel, filename):

        plt.figure()
        ax = sns.barplot(
            data=df,
            x="Agent Short",
            y=metric_col,
            hue="Method",
            hue_order=method_order,
            palette=palette,
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)

        plt.title(title)
        plt.xlabel("Agent")
        plt.ylabel(ylabel)
        plt.xticks(rotation=40)
        plt.tight_layout()

        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    create_bar_plot(
        "Total Cost",
        "Total Traffic Cost Comparison (Lower is Better)",
        "Average Weighted Cost",
        "plot_eval_cost_comparison.png",
    )

    create_bar_plot(
        "Avg Queue",
        "Average Queue Length Comparison (Lower is Better)",
        "Average Queue Length",
        "plot_eval_queue_comparison.png",
    )

    create_bar_plot(
        "Avg Wait",
        "Average Waiting Time Comparison (Lower is Better)",
        "Average Waiting Time (seconds)",
        "plot_eval_wait_comparison.png",
    )

    print("\nEvaluation plots generated successfully.")


# ==========================================================
# Cluster Evolution
# ==========================================================
def plot_clusters(log_file="logs/training_logs.json", output_dir="plots", agent_name=None):

    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        return

    if agent_name:
        output_dir = os.path.join(output_dir, agent_name)
    output_dir = os.path.join(output_dir, "train")
    ensure_dir(output_dir)

    try:
        with open(log_file, "r") as f:
            data = json.load(f)
    except Exception:
        print("Failed to read JSON.")
        return

    df = pd.DataFrame(data)

    if df.empty or "cluster" not in df.columns:
        print("No cluster info found.")
        return

    df = df[df["status"] == "trained"].copy()
    df["round"] = df["round"].astype(int)
    df["Agent Short"] = df["agent"].apply(
        lambda x: x[:12] + "..." if len(x) > 12 else x
    )

    plt.figure(figsize=(14, 8))

    sns.scatterplot(
        data=df,
        x="round",
        y="Agent Short",
        hue="cluster",
        palette="bright",
        s=150,
        edgecolor="black",
        linewidth=1.2,
    )

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title("CF-MADRL: Agent Cluster Evolution")
    plt.xlabel("Training Round")
    plt.ylabel("Agent / Network Junction")

    plt.legend(title="Cluster Group", bbox_to_anchor=(1, 0.5), loc="center left")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "plot_clusters.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Cluster plot saved to: {save_path}")


# ==========================================================
# Combined Evaluation (All Models)
# ==========================================================
def plot_all_models_evaluation(
    logs_dir="logs",
    output_dir="plots",
    model_files=None,
):

    if model_files is None:
        model_files = {
            "cfmadrl": ("CF-MADRL", "evaluation_logs.json"),
            "qtable": ("Q-Table", "evaluation_logs_qtable.json"),
            "dqn": ("DQN", "evaluation_logs_dqn.json"),
        }

    records = []
    fixed = None

    for model_key, (algo_name, filename) in model_files.items():
        log_path = os.path.join(logs_dir, filename)
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} not found. Skipping {algo_name}.")
            continue

        try:
            with open(log_path, "r") as f:
                data = json.load(f)
        except Exception:
            print(f"Failed to read {log_path}.")
            continue

        model = data.get(model_key, {})
        if not model:
            print(f"No {model_key} data found in {log_path}.")
            continue

        for aid, m in model.items():
            records.append(
                {
                    "Agent": aid,
                    "Method": algo_name,
                    "Avg Queue": np.mean(m["queues"]),
                    "Avg Wait": np.mean(m["waits"]),
                    "Total Cost": -np.mean(m["rewards"]),
                }
            )

        if fixed is None and "fixed_time" in data:
            fixed = data.get("fixed_time", {})

    if fixed:
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

    if not records:
        print("No combined evaluation data available.")
        return

    output_dir = os.path.join(output_dir, "eval")
    ensure_dir(output_dir)

    df = pd.DataFrame(records)
    df["Agent Short"] = df["Agent"].apply(lambda x: x[:12] + ".." if len(x) > 12 else x)

    method_order = [m[0] for m in model_files.values()] + ["Fixed-Time"]
    method_order = [m for m in method_order if m in df["Method"].unique()]
    df["Method"] = pd.Categorical(df["Method"], categories=method_order, ordered=True)

    palette = {
        "CF-MADRL": "#2ecc71",
        "Q-Table": "#9b59b6",
        "DQN": "#3498db",
        "Fixed-Time": "#e74c3c",
    }

    def create_bar_plot(metric_col, title, ylabel, filename):

        plt.figure()
        ax = sns.barplot(
            data=df,
            x="Agent Short",
            y=metric_col,
            hue="Method",
            hue_order=method_order,
            palette=palette,
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)

        plt.title(title)
        plt.xlabel("Agent")
        plt.ylabel(ylabel)
        plt.xticks(rotation=40)
        plt.tight_layout()

        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    create_bar_plot(
        "Total Cost",
        "Total Traffic Cost Comparison (Lower is Better)",
        "Average Weighted Cost",
        "plot_eval_cost_comparison.png",
    )

    create_bar_plot(
        "Avg Queue",
        "Average Queue Length Comparison (Lower is Better)",
        "Average Queue Length",
        "plot_eval_queue_comparison.png",
    )

    create_bar_plot(
        "Avg Wait",
        "Average Waiting Time Comparison (Lower is Better)",
        "Average Waiting Time (seconds)",
        "plot_eval_wait_comparison.png",
    )

    print("\nCombined evaluation plots generated successfully.")


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    set_style()

    parser = argparse.ArgumentParser(description="CF-MADRL Plotting Utility")

    parser.add_argument(
        "--type",
        type=str,
        default="all",
        choices=["train", "eval", "all"],
        help="Type of plots to generate",
    )

    parser.add_argument(
        "--output", type=str, default="plots", help="Directory to save plots"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent name to create a subfolder inside plots",
    )

    args = parser.parse_args()

    agents_all = ["cfmadrl", "qtable", "dqn"]
    agent_meta = {
        "cfmadrl": {
            "algo_name": "CF-MADRL",
            "model_name": "cfmadrl",
            "train_log": "logs/training_logs.json",
            "eval_log": "logs/evaluation_logs.json",
        },
        "qtable": {
            "algo_name": "Q-Table",
            "model_name": "qtable",
            "train_log": "logs/qtable/training_logs.json",
            "eval_log": "logs/evaluation_logs_qtable.json",
        },
        "dqn": {
            "algo_name": "DQN",
            "model_name": "dqn",
            "train_log": "logs/dqn/training_logs.json",
            "eval_log": "logs/evaluation_logs_dqn.json",
        },
    }
    if args.agent == "all":
        combined_dir = os.path.join(args.output, "combined")
        if args.type in ["train", "all"]:
            for agent in agents_all:
                meta = agent_meta[agent]
                plot_training(
                    log_file=meta["train_log"],
                    output_dir=args.output,
                    agent_name=agent,
                    algo_name=meta["algo_name"],
                )
                if agent == "cfmadrl":
                    plot_clusters(
                        log_file=meta["train_log"],
                        output_dir=args.output,
                        agent_name=agent,
                    )

        if args.type in ["eval", "all"]:
            plot_all_models_evaluation(logs_dir="logs", output_dir=combined_dir)
            for agent in agents_all:
                meta = agent_meta[agent]
                plot_evaluation(
                    log_file=meta["eval_log"],
                    output_dir=args.output,
                    agent_name=agent,
                    algo_name=meta["algo_name"],
                    model_name=meta["model_name"],
                )
    else:
        if args.type in ["train", "all"]:
            if args.agent in agent_meta:
                meta = agent_meta[args.agent]
                plot_training(
                    log_file=meta["train_log"],
                    output_dir=args.output,
                    agent_name=args.agent,
                    algo_name=meta["algo_name"],
                )
                if args.agent == "cfmadrl":
                    plot_clusters(
                        log_file=meta["train_log"],
                        output_dir=args.output,
                        agent_name=args.agent,
                    )
            else:
                plot_training(output_dir=args.output, agent_name=args.agent)
                if args.agent == "cfmadrl":
                    plot_clusters(output_dir=args.output, agent_name=args.agent)

        if args.type in ["eval", "all"]:
            if args.agent in agent_meta:
                meta = agent_meta[args.agent]
                plot_evaluation(
                    log_file=meta["eval_log"],
                    output_dir=args.output,
                    agent_name=args.agent,
                    algo_name=meta["algo_name"],
                    model_name=meta["model_name"],
                )
            else:
                plot_evaluation(output_dir=args.output, agent_name=args.agent)
