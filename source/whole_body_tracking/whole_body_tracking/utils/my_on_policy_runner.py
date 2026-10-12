# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils.wandb_log_writer import WandbLogWriter

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

# NOTE: this file targets the vanilla ``rsl_rl`` API (rsl_rl_lib >= 5.5.0). The base ``OnPolicyRunner`` no
# longer exposes ``self.logger_type``/``self.obs_normalizer``/``self.alg.policy`` - instead it exposes
# ``self.logger`` (a ``Logger`` wrapping a ``LogWriter`` such as ``WandbLogWriter``) and
# ``self.alg.get_policy()`` (the raw actor ``MLPModel``, which normalizes observations internally).


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if isinstance(self.logger.writer, WandbLogWriter):
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            self.export_policy_to_onnx(path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information, and export the policy to ONNX locally."""
        super().save(path, infos)
        # Export the policy to ONNX next to the .pt checkpoint. This is local-only: no wandb upload.
        policy_path = path.split("model")[0]
        filename = os.path.splitext(os.path.basename(path))[0] + ".onnx"
        export_motion_policy_as_onnx(
            self.env.unwrapped, self.alg.get_policy(), path=policy_path, filename=filename
        )
        attach_onnx_metadata(
            self.env.unwrapped,
            os.path.basename(os.path.normpath(policy_path)),
            path=policy_path,
            filename=filename,
        )
