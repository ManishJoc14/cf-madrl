import os
import numpy as np
import logging
import warnings
from typing import Dict, Any

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import torch

from tools.utils import Logger, ensure_dir, GlobalMetrics

# SILENCE - to make training outputs clean
# Shhh. Sit down. Do your job.
os.environ["RAY_TRAIN_PROGRESS_REPORTING"] = "0"
os.environ["RAY_metrics_export_port"] = "0"
os.environ["RAY_METRICS_EXPORT_PORT"] = "0"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["RAY_DASHBOARD_STATS_EXPORT_INTERVAL_MS"] = "0"
os.environ["RAY_metrics_report_interval_ms"] = "0"
os.environ["RAY_EVENT_LOG_LEVEL"] = "FATAL"
os.environ["RAY_GCS_RPC_TIMEOUT_MS"] = "100"
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
os.environ["RAY_DISABLE_EVENT_LOGGING"] = "1"

logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("sumolib").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# It exists to satisfy RLlib and does nothing
class SumoMetricsCallbacks(DefaultCallbacks):
    """Placeholder for custom callbacks (logic moved to GlobalMetrics for robustness)."""

    pass


# Environment registration
def env_creator(env_config):
    from src.multi_agent_sumo_env import MultiAgentSumoEnv

    return MultiAgentSumoEnv(env_config)


# Register globally
register_env("sumo_env", env_creator)


class AgentManager:
    """
    Manages RLlib multi-agent PPO training for clustered federated learning.
    """

    def __init__(self, env, config: Dict[str, Any], junction_ids: list):
        # env => already-created SUMO env
        # config => global config
        # junction_ids => agent IDs (same as env agents)

        # Store references
        self.env = env
        self.config = config
        self.junction_ids = junction_ids
        self.model_save_path = config["system"]["model_save_path"]

        # Ray initialization
        if not ray.is_initialized():
            ray.init(
                logging_level=logging.ERROR,
                configure_logging=True,
                ignore_reinit_error=True,
                include_dashboard=False,
                num_cpus=os.cpu_count() or 1,
                log_to_driver=False,
            )

        # Each agent has its own policy
        # But all policies share the same space shape
        # Extract single-agent spaces
        first_aid = junction_ids[0]
        obs_space = env.observation_space[first_aid]
        act_space = env.action_space[first_aid]

        # Read RL configurations
        rl_cfg = config.get("rl", {})

        # PPOConfig
        self.algo_config = (
            PPOConfig()
            .environment(
                env="sumo_env",
                # env_config is passed to env_creator
                env_config={
                    "config": config,
                    "junction_ids": junction_ids,
                    "max_lanes": env.max_lanes,
                    "max_phases": env.max_phases,
                },
            )
            .api_stack(
                # Classic RLlib stack.
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .framework("torch")
            # Needed for weight extraction later
            .training(
                lr=rl_cfg.get("lr", 0.0001),
                gamma=rl_cfg.get("gamma", 0.95),
                train_batch_size=rl_cfg.get("train_batch_size", 512),
                num_epochs=rl_cfg.get("num_sgd_iter", 20),
                model={
                    "fcnet_hiddens": rl_cfg.get("model", {}).get(
                        "fcnet_hiddens", [128, 64]
                    ),
                    "fcnet_activation": rl_cfg.get("model", {}).get(
                        "fcnet_activation", "relu"
                    ),
                },
            )
            .multi_agent(
                policies={
                    # None => default PPO policy
                    # same observation space
                    # same action space
                    # empty config, No parameter sharing
                    aid: (None, obs_space, act_space, {})
                    for aid in junction_ids
                },
                # agent "J1" => policy "J1"
                # agent "J2" => policy "J2"
                policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            )
            .env_runners(
                num_env_runners=0,
                create_env_on_local_worker=True,  # env lives in main process
                rollout_fragment_length=128,
            )
            .resources(
                num_gpus=1 if torch.cuda.is_available() else 0
            )  # Use 1 GPU if available
            .callbacks(SumoMetricsCallbacks)  # Callbacks wired (even if empty).
        )

        # Set parameters that might not be in .training() as keyword arguments
        self.algo_config.sgd_minibatch_size = rl_cfg.get("sgd_minibatch_size", 64)
        self.algo_config.clip_param = rl_cfg.get("clip_param", 0.2)

        self.algo = None

    def build(self):
        """Build the RLlib algorithm."""

        if self.algo is None:
            Logger.info("Building RLlib PPO algorithm stack...")
            self.algo = self.algo_config.build()
            Logger.success("RLlib algorithm initialized successfully.")

    def train(self, num_iterations: int = 1):
        """Train our agents."""
        if self.algo is None:
            self.build()

        result = None
        for _ in range(num_iterations):
            result = self.algo.train()

        return result

    def get_weights(self) -> Dict[str, Dict]:
        """Get weights of all agents."""

        if self.algo is None:
            raise ValueError("Algorithm not built yet.")

        weights = {}
        for aid in self.junction_ids:
            policy = self.algo.get_policy(aid)
            w = policy.get_weights()

            # Convert to numpy if they are tensors.
            import torch

            w_numpy = {
                k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in w.items()
            }

            weights[aid] = w_numpy

        return weights

    def set_weights(self, weights: Dict[str, Dict]):
        """loads weights into the RLlib PPO agent's policy"""

        if self.algo is None:
            self.build()

        import torch

        for aid, w in weights.items():
            if aid in self.junction_ids:
                policy = self.algo.get_policy(aid)
                # Convert back to torch if necessary
                w_torch = {
                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for k, v in w.items()
                }
                policy.set_weights(w_torch)

    def save(self) -> str:
        """Save a checkpoint."""

        if self.algo is None:
            self.build()

        ensure_dir(self.model_save_path)
        checkpoint = self.algo.save(self.model_save_path)
        Logger.info(f"Checkpoint created: {checkpoint}")

        return checkpoint

    def load(self, checkpoint_path: str):
        """Restore agent state from saved checkpoint."""

        if self.algo is None:
            self.build()

        self.algo.restore(checkpoint_path)

    def get_metrics(self, result: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract per-agent metrics from GlobalMetrics tracker or RLlib."""

        metrics = {}

        for aid in self.junction_ids:
            if aid in GlobalMetrics.LATEST:
                metrics[aid] = GlobalMetrics.LATEST[aid]
            else:
                # Fallback to RLlib metrics
                policy_rewards_mean = result.get("policy_reward_mean", {})
                custom_metrics = result.get("custom_metrics", {})

                reward = float(policy_rewards_mean.get(aid, 0.0))
                queue = float(custom_metrics.get(f"{aid}_queue_mean", 0.0))
                wait = float(custom_metrics.get(f"{aid}_wait_mean", 0.0))

                metrics[aid] = {
                    "mean_reward": reward,
                    "mean_queue": queue,
                    "mean_wait": wait,
                }

            # {
            #     "agent_1_id": {"mean_reward": 2.3, "mean_queue": 1.2, "mean_wait": 0.5},
            #     "agent_2_id": {"mean_reward": 1.8, "mean_queue": 0.9, "mean_wait": 0.3},
            # }

        return metrics

    def close(self):
        """Cleanly stop the RL algorithm (releases memory, stops workers)."""

        if self.algo is not None:
            self.algo.stop()
            self.algo = None
