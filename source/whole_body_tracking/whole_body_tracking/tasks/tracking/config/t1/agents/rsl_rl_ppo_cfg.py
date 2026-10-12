# Copyright (c) 2026 TARS
# SPDX-License-Identifier: MIT

from isaaclab.utils import configclass

# NOTE: T1 uses the vanilla ``rsl_rl`` library (rsl_rl_lib >= 5.5.0), not the
# MOE/history-observation fork used by the other robots
# (isaaclab_rl.rsl_rl.RslRlOnPolicyRunnerCfg/RslRlPpoActorCriticCfg are shaped for that old fork and are
# intentionally NOT used here). This config's ``to_dict()`` output must match what
# ``rsl_rl.runners.OnPolicyRunner`` / ``rsl_rl.algorithms.PPO.construct_algorithm`` expect directly:
# top-level "actor", "critic", "obs_groups", "algorithm" keys. See rsl_rl/tests/runners/test_on_policy_runner.py
# in the vanilla clone for the canonical schema this mirrors.
#
# Per-model observation normalization (``obs_normalization``) replaces the old top-level
# ``empirical_normalization`` flag, and there is no separate history/state-estimation observation group -
# just the plain "policy"/"critic" observation groups from ObservationsCfg (flat, non-history observations).


@configclass
class T1PPOAlgorithmCfg:
    """PPO algorithm hyperparameters (``rsl_rl.algorithms.PPO`` constructor kwargs)."""

    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@configclass
class T1PPOActorCfg:
    """Actor MLP model config (``rsl_rl.models.MLPModel`` constructor kwargs)."""

    class_name: str = "MLPModel"
    hidden_dims: list = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = True
    distribution_cfg: dict = {"class_name": "GaussianDistribution", "init_std": 1.0}


@configclass
class T1PPOCriticCfg:
    """Critic (value) MLP model config (``rsl_rl.models.MLPModel`` constructor kwargs)."""

    class_name: str = "MLPModel"
    hidden_dims: list = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = True


@configclass
class T1FlatPPORunnerCfg:
    """Runner configuration for T1, shaped for the vanilla ``rsl_rl_lib`` (>=5.5.0) API."""

    seed: int = 42
    device: str = "cuda:0"

    num_steps_per_env: int = 24
    max_iterations: int = 30000
    save_interval: int = 1000
    experiment_name: str = "t1_flat"
    run_name: str = ""

    # -- checkpoint resume --
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    # -- logging --
    logger: str = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"

    clip_actions: float | None = None

    # -- observation groups: which env observation groups feed the actor/critic --
    obs_groups: dict = {"actor": ["policy"], "critic": ["critic"]}

    algorithm: T1PPOAlgorithmCfg = T1PPOAlgorithmCfg()
    actor: T1PPOActorCfg = T1PPOActorCfg()
    critic: T1PPOCriticCfg = T1PPOCriticCfg()
