# Copyright (c) 2026 TARS
# SPDX-License-Identifier: MIT

"""Deploy and evaluate a trained whole-body tracking policy in MuJoCo (sim2sim).

Point it at a deployment bundle directory (the folders under ``logs/upload/``) and it
auto-discovers the policy, robot model, env config and motion clips, then rolls the
policy out in MuJoCo. It can open an interactive viewer or render an offline mp4.

Usage::

    # original per-file usage
    python sim2sim.py --motion_file <motion.npz> --onnx_load_path <policy.onnx> \
        --xml_path <robot.xml> --env_path <env.yaml> --visualize

    # interactive viewer (needs a display)
    python sim2sim.py logs/upload/<experiment> --visualize

    # offline video per motion (headless, MuJoCo EGL backend)
    python sim2sim.py logs/upload/<experiment> --record_video

    # replay the reference motion only (no policy)
    python sim2sim.py logs/upload/<experiment> --record_video --play_reference

A bundle directory is expected to contain::

    model_*.onnx                  policy (newest ``model_<iter>.onnx`` wins)
    env.yaml                      IsaacLab env config (PD gains / obs scales)
    configs/mjcf/<robot>.xml      MuJoCo model (+meshes/)
    *.npz                         motion clip(s); every one is evaluated

The motion npz must carry ``body_names``/``joint_names`` metadata in IsaacLab body
order (the output of the retargeting pipeline). The legacy "APT"
export is not supported — see ``MotionLoader``.
"""
import os
import sys
import argparse
import pathlib
import re
import time
from typing import Dict, List, Optional

# MuJoCo picks its GL backend at import time, so select EGL *before* importing mujoco
# when we are going to render an offline video headlessly. The interactive --visualize
# path keeps the default (GLFW) backend.
if "--record_video" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import onnxruntime as ort
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from common.isaaclab_utils_math import matrix_from_quat, subtract_frame_transforms
from common.parse_config import parse_args as parse_env_config


# --------------------------------------------------------------------------- #
# Motion data
# --------------------------------------------------------------------------- #
class MotionLoader:
    """Loads a motion ``.npz`` (retargeted, with body/joint-name metadata) for tracking."""

    def __init__(self, motion_file: str, device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self.motion_file = motion_file
        data = np.load(motion_file)

        missing = [k for k in ("body_names", "joint_names") if k not in data.files]
        if missing:
            raise ValueError(
                f"{motion_file} is missing {', '.join(missing)}. It looks like the "
                "legacy 'APT' export (31 body rows in URDF/chain-DFS order), whose "
                "body order does NOT match the deployment. Regenerate the motion with a "
                "retargeting pipeline that writes these two arrays."
            )

        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.joint_names = data["joint_names"].tolist()
        self.body_names = data["body_names"].tolist()
        self.time_step_total = self.joint_pos.shape[0]

    def get_future_command(self, current_step: int, future_steps: List[int]) -> torch.Tensor:
        future_data = []
        for item in future_steps:
            ti = min(current_step + item, self.time_step_total - 1)
            future_data.append(self.joint_pos[ti])
        return torch.cat(future_data)

    @property
    def body_pos_w(self):
        return self._body_pos_w

    @property
    def body_quat_w(self):
        return self._body_quat_w

    @property
    def body_lin_vel_w(self):
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self):
        return self._body_ang_vel_w


# --------------------------------------------------------------------------- #
# Observation helpers (mirror the IsaacLab tracking environment)
# --------------------------------------------------------------------------- #
def get_projected_gravity(quat: np.ndarray) -> np.ndarray:
    r = R.from_quat(quat)  # expects xyzw
    return r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.float64)


def cal_motion_anchor_pos_b(robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w):
    pos, _ = subtract_frame_transforms(robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w)
    return pos.view(-1)


def cal_motion_anchor_ori_b(robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w):
    _, ori = subtract_frame_transforms(robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w)
    return matrix_from_quat(ori)[..., :2].reshape(-1)


def cal_body_pos(robot_anchor_pos_w, robot_anchor_quat_w, robot_body_pos_w, robot_body_quat_w):
    num_bodies = robot_body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        robot_body_pos_w,
        robot_body_quat_w,
    )
    return pos_b.view(-1)


def cal_body_ori(robot_anchor_pos_w, robot_anchor_quat_w, robot_body_pos_w, robot_body_quat_w):
    num_bodies = robot_body_pos_w.shape[1]
    _, ori_b = subtract_frame_transforms(
        robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        robot_body_pos_w,
        robot_body_quat_w,
    )
    return matrix_from_quat(ori_b)[..., :2].reshape(-1)


def pd_control(target_q, q, kp, target_dq, dq, kd) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


# --------------------------------------------------------------------------- #
# MuJoCo helpers
# --------------------------------------------------------------------------- #
def get_mujoco_joint_names(model: mujoco.MjModel) -> List[str]:
    names = []
    for i in range(model.njnt):
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
    return names[1:]  # drop the free "root" joint


def get_body_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> Optional[np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        return None
    return data.xpos[body_id].copy()


def get_body_quat(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> Optional[np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        return None
    return data.xquat[body_id].copy()


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Add a visualization capsule to a ``mjvScene`` (viewer or offscreen renderer)."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3), np.zeros(3), np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
        np.asarray(point1, dtype=np.float64),
        np.asarray(point2, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# Bundle discovery
# --------------------------------------------------------------------------- #
def discover_bundle(bundle_dir: str) -> Dict[str, object]:
    """Resolve the policy/env/model/motion files inside a deployment bundle."""
    bundle = pathlib.Path(bundle_dir).absolute()
    if not bundle.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {bundle}")

    def latest_iter(paths: List[pathlib.Path]) -> pathlib.Path:
        return max(paths, key=lambda p: ([int(s) for s in re.findall(r"\d+", p.stem)] + [0])[-1])

    onnx_files = sorted(bundle.glob("*.onnx"))
    model_files = [p for p in onnx_files if p.name.startswith("model_")]
    if model_files:
        onnx_path = latest_iter(model_files)
    elif len(onnx_files) == 1:
        onnx_path = onnx_files[0]
    else:
        raise FileNotFoundError(f"expected a single *.onnx in {bundle}, found {len(onnx_files)}")

    env_path = bundle / "env.yaml"
    if not env_path.is_file():
        raise FileNotFoundError(f"no env.yaml in {bundle}")

    xml_files = sorted((bundle / "configs" / "mjcf").glob("*.xml"))
    if not xml_files:
        xml_files = sorted(bundle.glob("**/*.xml"))
    if len(xml_files) != 1:
        raise FileNotFoundError(f"expected exactly one robot *.xml in {bundle}/configs/mjcf, found {len(xml_files)}")

    motion_files = sorted(bundle.glob("*.npz"))
    if not motion_files:
        raise FileNotFoundError(f"no motion *.npz in {bundle}")

    return {
        "onnx": str(onnx_path),
        "env": str(env_path),
        "xml": str(xml_files[0]),
        "motions": [str(p) for p in motion_files],
    }


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
def draw_keypoints(scene, model, data, motion_loader, keypoint_body_names, inference_counter, play_reference):
    """Draw reference (red) and simulated (green) keypoint capsules into ``scene``."""
    for body_name in keypoint_body_names:
        for j, ref_name in enumerate(motion_loader.body_names):
            if ref_name == body_name and inference_counter < motion_loader.time_step_total:
                ref_pos = motion_loader.body_pos_w[inference_counter, j].numpy()
                add_visual_capsule(scene, ref_pos, ref_pos + np.array([0.001, 0, 0]), 0.03, np.array([1, 0, 0, 1]))
                break

    if not play_reference:
        for body_name in keypoint_body_names:
            sim_pos = get_body_pos(model, data, body_name)
            if sim_pos is not None:
                add_visual_capsule(scene, sim_pos, sim_pos + np.array([0.001, 0, 0]), 0.03, np.array([0, 1, 0, 1]))


def run_rollout(
    model,
    policy,
    motion_loader,
    config,
    *,
    play_reference: bool = False,
    visualize: bool = False,
    video_path: Optional[str] = None,
    video_fps: float = 50.0,
    video_height: int = 480,
    video_width: int = 640,
    max_steps_override: Optional[int] = None,
):
    data = mujoco.MjData(model)
    model.opt.timestep = config["simulation_dt"]

    # --- config / PD mapping -------------------------------------------------
    joint_names = config["joint_names"]
    joint_stiffness = np.array([float(i) for i in config["joint_stiffness"]], dtype=np.float32)
    joint_damping = np.array([float(i) for i in config["joint_damping"]], dtype=np.float32)
    default_joint_pos = np.array(config["default_joint_pos"], dtype=np.float32)
    action_scale_cfg = np.array([float(i) for i in config["action_scale"]], dtype=np.float32)

    stiffness_dict = dict(zip(joint_names, joint_stiffness))
    damping_dict = dict(zip(joint_names, joint_damping))
    default_pos_dict = dict(zip(joint_names, default_joint_pos))
    action_scale_dict = dict(zip(joint_names, action_scale_cfg))

    keypoint_body_names = config["body_names"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]

    mujoco_joint_names = get_mujoco_joint_names(model)
    assert len(mujoco_joint_names) == len(joint_names), f"{len(mujoco_joint_names)} vs {len(joint_names)}"
    assert set(mujoco_joint_names) == set(joint_names), "mujoco joint names != policy joint names"

    kps = np.array([stiffness_dict[k] for k in mujoco_joint_names])
    kds = np.array([damping_dict[k] for k in mujoco_joint_names])
    default_angles = np.array([default_pos_dict[k] for k in mujoco_joint_names])
    action_scale = np.array([action_scale_dict[k] for k in mujoco_joint_names])

    policy_to_robot_indices = [joint_names.index(name) for name in mujoco_joint_names]
    robot_to_policy_indices = [mujoco_joint_names.index(name) for name in joint_names]

    observation_names = config["observation_names"]
    print("Observation names:", observation_names)

    def build_observation(current_obs: Dict[str, np.ndarray]) -> np.ndarray:
        parts = [np.asarray(current_obs[name], dtype=np.float32).reshape(-1) for name in observation_names]
        return np.concatenate(parts).astype(np.float32)

    USE_DELTA_JOINT_POS_W_REF = config["use_delta_joint_pos_w_ref"]
    USE_DELTA_JOINT_VEL_W_REF = config["use_delta_joint_vel_w_ref"]
    USE_DELTA_ACTION_W_REF = config["use_delta_action_w_ref"]

    input_names = [inp.name for inp in policy.get_inputs()]
    input_name_obs = next(n for n in input_names if n != "time_step")

    # --- init ---------------------------------------------------------------
    action = np.zeros(len(joint_names), dtype=np.float32)
    target_dof_pos = default_angles.copy()
    inference_counter = 0

    data.qpos[:3] = motion_loader.body_pos_w[inference_counter, 0].numpy()
    data.qpos[3:7] = motion_loader.body_quat_w[inference_counter, 0].numpy()
    data.qpos[7:] = motion_loader.joint_pos[inference_counter].numpy()[policy_to_robot_indices]
    mujoco.mj_step(model, data)
    target_dof_pos = data.qpos[7:].copy()

    # --- rendering backend ---------------------------------------------------
    viewer = None
    renderer = None
    cam = None
    writer = None

    if visualize:
        viewer = mujoco.viewer.launch_passive(model, data)
        _, _, y = R.from_quat(motion_loader.body_quat_w[inference_counter, 0].numpy()).as_euler("zyx", degrees=False)
        viewer.cam.distance = 10.0
        viewer.cam.azimuth = -y * 180.0 / np.pi
        viewer.cam.elevation = -20.0
        viewer.cam.lookat[:] = [data.qpos[0] + 2.0, data.qpos[1], data.qpos[2]]

    if video_path:
        import imageio.v2 as iio

        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        renderer = mujoco.Renderer(model, video_height, video_width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        cam.distance = 3.0
        cam.azimuth = 90.0
        cam.elevation = -15.0
        writer = iio.get_writer(video_path, fps=video_fps, codec="libx264", quality=8)

    max_steps = (motion_loader.time_step_total + 100) * control_decimation
    if max_steps_override is not None:
        max_steps = min(max_steps, max_steps_override * control_decimation)

    try:
        counter = 0
        for _ in tqdm(range(max_steps), desc=os.path.basename(motion_loader.motion_file)):
            step_start = time.time()

            tau = pd_control(target_dof_pos, data.qpos[7:], kps, np.zeros_like(kds), data.qvel[6:], kds)
            if play_reference:
                data.qpos[:3] = motion_loader.body_pos_w[inference_counter, 0].numpy()
                data.qpos[3:7] = motion_loader.body_quat_w[inference_counter, 0].numpy()
                data.qpos[7:] = motion_loader.joint_pos[inference_counter].numpy()[policy_to_robot_indices]
                data.qvel[0:3] = motion_loader.body_lin_vel_w[inference_counter, 0].numpy()
                data.qvel[3:6] = motion_loader.body_ang_vel_w[inference_counter, 0].numpy()
                data.qvel[6:] = motion_loader.joint_vel[inference_counter].numpy()[policy_to_robot_indices]
            else:
                data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            counter += 1

            if counter % control_decimation == 0:
                if inference_counter >= motion_loader.time_step_total:
                    target_dof_pos = default_angles.copy()
                else:
                    # advance the reference frame first: observation/inference use the
                    # post-increment frame (matches the training env's "track next").
                    if inference_counter < motion_loader.time_step_total - 1:
                        inference_counter += 1

                    sim_body_pos = {n: get_body_pos(model, data, n) for n in keypoint_body_names}
                    sim_body_quat = {n: get_body_quat(model, data, n) for n in keypoint_body_names}

                    qj = data.qpos[7:]
                    dqj = data.qvel[6:]
                    ref_joint_pos = motion_loader.joint_pos[inference_counter].numpy()[policy_to_robot_indices]
                    ref_joint_vel = motion_loader.joint_vel[inference_counter].numpy()[policy_to_robot_indices]

                    if USE_DELTA_JOINT_POS_W_REF:
                        qj = (qj - ref_joint_pos) * dof_pos_scale
                    else:
                        qj = (qj - default_angles) * dof_pos_scale
                    if USE_DELTA_JOINT_VEL_W_REF:
                        dqj = (dqj - ref_joint_vel) * dof_vel_scale
                    else:
                        dqj = dqj * dof_vel_scale

                    current_obs: Dict[str, np.ndarray] = {}
                    if "command" in observation_names:
                        current_obs["command"] = torch.cat([
                            motion_loader.joint_pos[inference_counter],
                            motion_loader.joint_vel[inference_counter],
                        ]).numpy()
                    if "future_command" in observation_names:
                        current_obs["future_command"] = motion_loader.get_future_command(
                            inference_counter, config.get("future_commands_step")
                        ).numpy()
                    if "command_joint_pos" in observation_names:
                        current_obs["command_joint_pos"] = motion_loader.joint_pos[inference_counter].numpy()
                    if "command_joint_vel" in observation_names:
                        current_obs["command_joint_vel"] = motion_loader.joint_vel[inference_counter].numpy()

                    anchor_pos_w, anchor_quat_w = (
                        motion_loader.body_pos_w[inference_counter, motion_loader.body_names.index("torso_link")].unsqueeze(0),
                        motion_loader.body_quat_w[inference_counter, motion_loader.body_names.index("torso_link")].unsqueeze(0),
                    )
                    sim_anchor_pos = torch.tensor(sim_body_pos["torso_link"], dtype=torch.float32).unsqueeze(0)
                    sim_anchor_quat = torch.tensor(sim_body_quat["torso_link"], dtype=torch.float32).unsqueeze(0)

                    if "motion_anchor_pos_b" in observation_names:
                        current_obs["motion_anchor_pos_b"] = cal_motion_anchor_pos_b(
                            sim_anchor_pos, sim_anchor_quat, anchor_pos_w, anchor_quat_w
                        ).numpy()
                    if "motion_anchor_ori_b" in observation_names:
                        current_obs["motion_anchor_ori_b"] = cal_motion_anchor_ori_b(
                            sim_anchor_pos, sim_anchor_quat, anchor_pos_w, anchor_quat_w
                        ).numpy()

                    if "body_pos" in observation_names:
                        current_obs["body_pos"] = cal_body_pos(
                            sim_anchor_pos, sim_anchor_quat,
                            torch.tensor(np.array([sim_body_pos[n] for n in keypoint_body_names])).unsqueeze(0).float(),
                            torch.tensor(np.array([sim_body_quat[n] for n in keypoint_body_names])).unsqueeze(0).float(),
                        ).numpy()
                    if "body_ori" in observation_names:
                        current_obs["body_ori"] = cal_body_ori(
                            sim_anchor_pos, sim_anchor_quat,
                            torch.tensor(np.array([sim_body_pos[n] for n in keypoint_body_names])).unsqueeze(0).float(),
                            torch.tensor(np.array([sim_body_quat[n] for n in keypoint_body_names])).unsqueeze(0).float(),
                        ).numpy()

                    if "projected_gravity" in observation_names:
                        quat = data.qpos[3:7][[1, 2, 3, 0]]
                        current_obs["projected_gravity"] = get_projected_gravity(quat)
                    if "base_lin_vel" in observation_names:
                        current_obs["base_lin_vel"] = data.qvel[:3]
                    if "base_ang_vel" in observation_names:
                        current_obs["base_ang_vel"] = data.qvel[3:6] * ang_vel_scale
                    if "anchor_projected_gravity" in observation_names:
                        quat_torso = data.sensordata[10:14][[1, 2, 3, 0]]
                        current_obs["anchor_projected_gravity"] = get_projected_gravity(quat_torso)
                    if "anchor_ang_vel" in observation_names:
                        current_obs["anchor_ang_vel"] = data.sensordata[14:17] * ang_vel_scale

                    if "joint_pos" in observation_names:
                        current_obs["joint_pos"] = qj[robot_to_policy_indices]
                    if "joint_vel" in observation_names:
                        current_obs["joint_vel"] = dqj[robot_to_policy_indices]
                    if "actions" in observation_names:
                        current_obs["actions"] = action[robot_to_policy_indices]
                    if "action_kp_kd" in observation_names:
                        current_obs["action_kp_kd"] = np.concatenate([
                            kps[robot_to_policy_indices], kds[robot_to_policy_indices]
                        ]) * 0.01

                    # run the policy; feed time_step when the model exposes it
                    model_inputs = {input_name_obs: build_observation(current_obs).reshape(1, -1)}
                    if "time_step" in input_names:
                        model_inputs["time_step"] = np.array([[inference_counter]], dtype=np.float32)
                    outputs = policy.run(None, model_inputs)

                    action = outputs[0].squeeze()[policy_to_robot_indices]
                    scaled_action = action * action_scale
                    if USE_DELTA_ACTION_W_REF:
                        target_dof_pos = scaled_action + motion_loader.joint_pos[inference_counter].numpy()[policy_to_robot_indices]
                    else:
                        target_dof_pos = scaled_action + default_angles

                    # render one video frame per control step
                    if renderer is not None:
                        mujoco.mj_forward(model, data)
                        cam.lookat[:] = data.body("pelvis").xpos
                        renderer.update_scene(data, camera=cam)
                        draw_keypoints(renderer.scene, model, data, motion_loader, keypoint_body_names, inference_counter, play_reference)
                        writer.append_data(renderer.render())

            # live viewer: render every sim step
            if viewer is not None:
                viewer.user_scn.ngeom = 0
                draw_keypoints(viewer.user_scn, model, data, motion_loader, keypoint_body_names, inference_counter, play_reference)
                viewer.sync()
                time_until_next_step = simulation_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        if writer is not None:
            writer.close()
        if viewer is not None:
            viewer.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Deploy a tracking policy in MuJoCo from a bundle directory or per-file arguments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bundle", nargs="?", default=None,
                        help="deployment bundle directory (logs/upload/<experiment>)")
    # Original per-file usage (--motion_file --onnx_load_path --xml_path --env_path).
    parser.add_argument("--motion_file", default=None, help="motion *.npz (per-file usage)")
    parser.add_argument("--onnx_load_path", default=None, help="policy *.onnx (per-file usage)")
    parser.add_argument("--xml_path", default=None, help="robot *.xml (per-file usage)")
    parser.add_argument("--env_path", default=None, help="IsaacLab env.yaml (per-file usage)")
    parser.add_argument("--visualize", action="store_true", help="open the interactive viewer")
    parser.add_argument("--record_video", action="store_true", help="render an offline mp4 per motion (headless)")
    parser.add_argument("--video_dir", default=None, help="where to write mp4s (default: next to each motion npz)")
    parser.add_argument("--fps", type=float, default=50.0, help="output video fps")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max_steps", type=int, default=None, help="cap control steps (default: full motion)")
    parser.add_argument("--play_reference", action="store_true", help="replay the reference motion only (no policy)")
    args = parser.parse_args()

    per_file = any(getattr(args, k) is not None
                   for k in ("motion_file", "onnx_load_path", "xml_path", "env_path"))
    if per_file:
        # Original per-file usage: resolve the bundle from the explicit paths.
        missing = [f"--{k}" for k in ("motion_file", "onnx_load_path", "xml_path", "env_path")
                   if getattr(args, k) is None]
        if missing:
            parser.error("per-file usage requires all four args; missing: " + ", ".join(missing))
        bundle = {
            "onnx": args.onnx_load_path,
            "env": args.env_path,
            "xml": args.xml_path,
            "motions": [args.motion_file],
        }
    else:
        if args.bundle is None:
            parser.error("provide a bundle directory, or the per-file "
                         "--motion_file/--onnx_load_path/--xml_path/--env_path args")
        bundle = discover_bundle(args.bundle)
    config = parse_env_config(bundle["env"], bundle["onnx"])
    policy = ort.InferenceSession(bundle["onnx"], providers=["CPUExecutionProvider"])
    model = mujoco.MjModel.from_xml_path(bundle["xml"])

    for motion_file in bundle["motions"]:
        print(f"==> {motion_file}")
        motion_loader = MotionLoader(motion_file)

        video_path = None
        if args.record_video:
            stem = pathlib.Path(motion_file).stem
            out_dir = args.video_dir or str(pathlib.Path(motion_file).parent)
            video_path = os.path.join(out_dir, f"{stem}.mp4")

        run_rollout(
            model, policy, motion_loader, config,
            play_reference=args.play_reference,
            visualize=args.visualize,
            video_path=video_path,
            video_fps=args.fps,
            video_height=args.height,
            video_width=args.width,
            max_steps_override=args.max_steps,
        )


if __name__ == "__main__":
    main()