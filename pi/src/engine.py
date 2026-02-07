"""Traffic inference engine"""

import os
import time
import torch
import numpy as np
import json
import sys
import threading
from pathlib import Path
from utils.inference_utils import RunningNorm

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from .monitor import Config, LaneMonitor
    from ultralytics import YOLO

    YOLO_DEPLOY_AVAILABLE = True
except ImportError:
    YOLO_DEPLOY_AVAILABLE = False


class TrafficInference:
    """PPO-based traffic light control inference engine"""

    def __init__(
        self,
        model_filename="model.pt",
        max_lanes=6,
        max_phases=4,
        num_green_phases=2,
        green_phases_indices=None,
        use_yolo_deploy=True,
        yolo_config_file="configs/yolo_config.yaml",
    ):
        self.max_lanes = max_lanes
        self.max_phases = max_phases
        self.num_green_phases = num_green_phases
        self.green_phases_indices = green_phases_indices or [0, 2]
        self.durations = [10, 20, 30, 40, 50, 60]
        self.num_durations = len(self.durations)
        self.last_phase = None

        self.model_path = str((BASE_DIR / "models" / model_filename).resolve())

        self.queue_norm = RunningNorm(shape=max_lanes)
        self.wait_norm = RunningNorm(shape=max_lanes)
        self.policy = self._load_model()

        # Load norm stats
        stats_path = str((BASE_DIR / "norm_stats.json").resolve())
        if os.path.exists(stats_path):
            self._load_norm_stats(stats_path)

        # Initialize traffic monitor
        self.use_yolo_deploy = use_yolo_deploy and YOLO_DEPLOY_AVAILABLE
        self.lane_monitors = {}
        self.monitor_threads = []

        if self.use_yolo_deploy:
            self._init_yolo_deploy(yolo_config_file)

    def _load_norm_stats(self, path):
        """Load normalization statistics"""
        try:
            with open(path, "r") as f:
                stats = json.load(f)
            agent_id = "J5" if "J5" in stats else next(iter(stats.keys()))
            self.queue_norm.set_state(stats[agent_id]["queue"])
            self.wait_norm.set_state(stats[agent_id]["wait"])
            print(f"✓ Normalization stats loaded for {agent_id}")
        except Exception as e:
            print(f"Warning: Failed to load norm stats: {e}")

    def _init_yolo_deploy(self, yolo_config_file):
        """Initialize traffic monitoring system"""
        config_path = str((BASE_DIR / yolo_config_file).resolve())

        if not os.path.exists(config_path):
            print(f"Warning: Config not found: {config_path}")
            self.use_yolo_deploy = False
            return

        try:
            config = Config(config_path)

            # Load models
            primary_model = None
            fallback_model = None

            try:
                primary_model = YOLO(config.MODEL_PATH)
                print(f"✓ Primary model loaded: {config.MODEL_PATH}")
            except Exception as e:
                print(f"Warning: Primary model failed: {e}")

            try:
                fallback_model = YOLO(config.FALLBACK_MODEL)
                print(f"✓ Fallback model loaded: {config.FALLBACK_MODEL}")
            except Exception as e:
                print(f"Warning: Fallback model failed: {e}")

            if primary_model is None and fallback_model is None:
                print("Error: No models available!")
                self.use_yolo_deploy = False
                return

            # Initialize lanes
            for lane_cfg in config.LANES:
                try:
                    monitor = LaneMonitor(
                        lane_cfg, primary_model, fallback_model, config
                    )
                    self.lane_monitors[lane_cfg["id"]] = monitor

                    thread = threading.Thread(target=monitor.process_loop, daemon=True)
                    thread.start()
                    self.monitor_threads.append(thread)
                    print(f"✓ Monitor started for {lane_cfg['id']}")
                except Exception as e:
                    print(f"Warning: Lane {lane_cfg['id']} failed: {e}")

            print(f"✓ monitor initialized with {len(self.lane_monitors)} lane(s)")
            time.sleep(2)

        except Exception as e:
            print(f"Error initializing monitor: {e}")
            self.use_yolo_deploy = False

    def _load_model(self):
        """Load PPO model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            from utils.model import PPOPolicy

            state_dict = torch.load(self.model_path, weights_only=True)
            obs_dim = state_dict["network.0.weight"].shape[1]
            act_dim = state_dict["network.4.weight"].shape[0]

            policy = PPOPolicy(obs_dim, act_dim)
            policy.load_state_dict(state_dict)
            policy.eval()
            print(f"✓ Model loaded | Obs: {obs_dim}, Actions: {act_dim}")
            return policy
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def get_real_metrics(self):
        """Get real traffic metrics from cameras"""
        if not self.use_yolo_deploy or not self.lane_monitors:
            return None

        queues = []
        waits = []

        for lane_id in sorted(self.lane_monitors.keys()):
            monitor = self.lane_monitors[lane_id]
            metrics = monitor.latest_metrics
            queues.append(metrics.get("queue_length", 0))
            waits.append(metrics.get("avg_wait_time", 0.0))

        # Pad to max_lanes
        queues = queues + [0] * (self.max_lanes - len(queues))
        waits = waits + [0.0] * (self.max_lanes - len(waits))

        return queues[: self.max_lanes], waits[: self.max_lanes]

    def get_action(self, queues, waits, current_phase):
        """Get phase decision from PPO model"""
        q_padded = np.array(list(queues) + [0.0] * (self.max_lanes - len(queues)))
        w_padded = np.array(list(waits) + [0.0] * (self.max_lanes - len(waits)))

        q_norm = self.queue_norm.normalize(q_padded)
        w_norm = self.wait_norm.normalize(w_padded)

        self.queue_norm.update(q_padded)
        self.wait_norm.update(w_padded)

        state = np.concatenate(
            [
                np.clip(q_norm, -10.0, 10.0),
                np.clip(w_norm, -10.0, 10.0),
                [current_phase / self.max_phases],
            ]
        ).astype(np.float32)

        action_id = self.policy.predict(state)

        dur_idx = action_id % self.num_durations
        green_idx_local = (action_id // self.num_durations) % self.num_green_phases
        duration_val = self.durations[min(dur_idx, self.num_durations - 1)]
        target_phase = self.green_phases_indices[green_idx_local]

        print(
            f"[DECISION] Queues: {[int(q) for q in queues[:4]]} -> Phase {target_phase}, Duration {duration_val}s"
        )

        yellow_required = (
            self.last_phase is not None and target_phase != self.last_phase
        )
        self.last_phase = target_phase

        return target_phase, duration_val, yellow_required
