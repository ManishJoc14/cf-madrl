import os
import sys
import json
import yaml
import sumolib
import numpy as np
import traci
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme


class SilenceStdout:
    """Context manager to suppress stdout/stderr, including C-level output."""

    def __enter__(self):
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.saved_stderr_fd = os.dup(self.stderr_fd)
        self.devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull_fd, self.stdout_fd)
        os.dup2(self.devnull_fd, self.stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.saved_stdout_fd, self.stdout_fd)
        os.dup2(self.saved_stderr_fd, self.stderr_fd)
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)
        os.close(self.devnull_fd)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from a YAML file.
    Retruns a dictionary (key-value pairs) with all settings.
    """

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def ensure_dir(path: str):
    """
    Create a folder if it doesn't exist.
    """

    if not os.path.exists(path):
        os.makedirs(path)


def log_metrics(record: Dict[str, Any], log_file: str):
    """
    Append a single record (dictionary) to a JSON file.
    Keeps an array of metrics over time.
    """

    data = []

    # Load existing metrics
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []

    # Append new record
    data.append(record)

    # Save back to file
    with open(log_file, "w") as f:
        json.dump(data, f, indent=2)


class GlobalMetrics:
    """
    A global dictionary to store the latest metrics for each agent.
    """

    LATEST: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def update(agent_id: str, reward: float, queue: float, wait: float):
        GlobalMetrics.LATEST[agent_id] = {
            "mean_reward": float(reward),
            "mean_queue": float(queue),
            "mean_wait": float(wait),
        }


THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "yellow",
        "error": "bold red",
        "header": "bold magenta",
        "section": "bold blue",
    }
)

try:
    narrative_file = (
        open("CONOUT$", "w", encoding="utf-8")
        if os.name == "nt"
        else open("/dev/tty", "w", encoding="utf-8")
    )
except Exception:
    narrative_file = sys.stdout

console = Console(theme=THEME, file=narrative_file, force_terminal=True)


class PrevTrafficState:
    """
    Stores previous queue and wait per agent
    to compute reward as delta(before - after).
    """

    def __init__(self):
        self.prev = {}

    def reset(self):
        self.prev = {}

    def get(self, aid):
        return self.prev.get(aid, None)

    def set(self, aid, queue, wait):
        self.prev[aid] = {
            "queue": float(queue),
            "wait": float(wait),
        }


class RunningNorm:
    def __init__(self, shape=1, eps=1e-8):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = eps

    def update(self, x):
        x = np.array(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class Logger:
    @staticmethod
    def header(text: str):
        console.print(Panel(text.upper(), style="header"))

    @staticmethod
    def section(text: str):
        console.print(f"\n[section]─── {text} ───[/section]\n")

    @staticmethod
    def info(text: str):
        console.print(f"[info]•[/info] {text}")

    @staticmethod
    def success(text: str):
        console.print(f"[success]✓[/success] {text}")

    @staticmethod
    def warning(text: str):
        console.print(f"[warning]! WARNING:[/warning] {text}")

    @staticmethod
    def narrative(text: str):
        console.print(f"[dim]⏵[/dim] [italic]{text}[/italic]")

    @staticmethod
    def round_banner(round_id: int, total: int, reward: float = None):
        title = f"ROUND {round_id}/{total}"
        if reward is not None:
            title += f" | REWARD {reward:.2f}"

        console.print(Panel(title, border_style="cyan", title="Federated Phase"))


def scan_topology(config):
    """
    Scan the SUMO network to discover junctions, max lanes, and max signal phases.

    Input:
        config: dict loaded from config.yaml with SUMO paths and system settings
    Output:
        max_lanes: maximum lanes across all controlled junctions
        max_phases: maximum traffic light phases across all junctions

    Notes:
        - Auto-updates config.yaml if junctions in SUMO differ from config.
    """

    # Get SUMO binary path
    sumo_bin = sumolib.checkBinary("sumo")
    sumo_cmd = [
        sumo_bin,
        "-c",
        config["sumo"]["config_file"],
        "--no-step-log",
        "--no-warnings",
        "--duration-log.disable",
        "--error-log",
        "nul",
        "--message-log",
        "nul",
    ]

    # Random label for this SUMO instance (prevents port conflicts)
    label = f"scan_{np.random.randint(999999)}"

    # Silence SUMO output while scanning
    with SilenceStdout():
        traci.start(sumo_cmd, label=label)
        conn = traci.getConnection(label)

    # Discover junction IDs from SUMO
    discovered_junctions = sorted(conn.trafficlight.getIDList())
    config_junctions = config["system"].get("controlled_junctions") or []

    # If mismatch, update config.yaml
    if set(discovered_junctions) != set(config_junctions):
        Logger.warning(
            "Junction mismatch detected! Updating config.yaml with discovered junctions."
        )
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config.yaml")

        # Update in-memory config
        config["system"]["controlled_junctions"] = discovered_junctions

        # Update YAML file safely
        try:
            full_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    full_config = yaml.safe_load(f) or {}

            if "system" not in full_config:
                full_config["system"] = {}
            full_config["system"]["controlled_junctions"] = discovered_junctions

            with open(config_path, "w") as f:
                yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
            Logger.success(
                f"config.yaml updated with {len(discovered_junctions)} junctions: {discovered_junctions}"
            )
        except Exception as e:
            Logger.warning(f"Failed to update config.yaml: {e}")

    # Compute max lanes and max phases across junctions
    junctions = config["system"]["controlled_junctions"]
    max_lanes = 0
    max_phases = 0
    for j_id in junctions:
        lanes = sorted(list(set(conn.trafficlight.getControlledLanes(j_id))))
        logics = conn.trafficlight.getCompleteRedYellowGreenDefinition(j_id)

        if logics:
            current_program_id = conn.trafficlight.getProgram(j_id)
            active_logic = next(
                (l for l in logics if l.programID == current_program_id), logics[0]
            )
            num_phases = len(active_logic.phases)

            max_lanes = max(max_lanes, len(lanes))
            max_phases = max(max_phases, num_phases)

        # fallback max
        max_lanes = max(max_lanes, len(lanes))
        max_phases = max(max_phases, num_phases)

    # Close SUMO connection
    try:
        traci.close()
    except:
        pass

    return max_lanes, max_phases
