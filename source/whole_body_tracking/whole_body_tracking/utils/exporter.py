# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
import torch.nn as nn

import onnx

from isaaclab.envs import ManagerBasedRLEnv

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    policy: object,
    path: str,
    filename="policy.onnx",
    verbose=False,
):
    """Export a vanilla ``rsl_rl`` ``MLPModel`` actor as a motion-conditioned ONNX file.

    Unlike ``isaaclab_rl.rsl_rl.exporter.export_policy_as_onnx`` (built for the old "ActorCritic"-wrapper
    style policy with a separate external normalizer), ``policy`` here is expected to be an
    ``rsl_rl.models.MLPModel`` instance directly (e.g. ``runner.alg.get_policy()``), which already carries its
    own observation normalizer internally - so no separate ``normalizer`` argument is needed.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, policy, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(nn.Module):
    """ONNX export wrapper around an ``rsl_rl`` ``MLPModel`` actor, augmented with motion-replay outputs."""

    is_recurrent: bool = False

    def __init__(self, env: ManagerBasedRLEnv, policy, verbose=False):
        super().__init__()
        self.verbose = verbose
        # Copy the actor's own normalizer/MLP/output-head - normalization now lives inside the model itself.
        self.obs_normalizer = copy.deepcopy(policy.obs_normalizer)
        self.mlp = copy.deepcopy(policy.mlp)
        if policy.distribution is not None:
            self.deterministic_output = policy.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = policy.obs_dim

        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        actions = self.deterministic_output(self.mlp(self.obs_normalizer(x)))
        return (
            actions,
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        obs = torch.zeros(1, self.input_size)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    # ``history_length`` is an observation-group attribute only present on some IsaacLab
    # versions; fall back to per-term defaults when it is missing.
    group_history_length = getattr(env.observation_manager.cfg.policy, "history_length", None)
    if group_history_length:
        observation_history_lengths = [group_history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg.get("history_length", 0)
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
