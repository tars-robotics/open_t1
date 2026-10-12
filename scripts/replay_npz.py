# Copyright (c) 2026 TARS
# SPDX-License-Identifier: MIT

"""This script replays a motion from a local .npz file.

.. code-block:: bash

    # Usage
    python replay_npz.py --motion_file /path/to/motion.npz

    # Headless, record the replay to an mp4
    python replay_npz.py --motion_file /path/to/motion.npz --headless --video --video_path replay.mp4
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay a motion from a local .npz file.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to a local motion .npz file.")
parser.add_argument("--video", action="store_true", default=False, help="Record the replay to an mp4 video.")
parser.add_argument("--video_path", type=str, default="replay.mp4", help="Output mp4 path (default: replay.mp4).")
parser.add_argument("--video_fps", type=int, default=50, help="Frames per second of the output video (default: 50).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Recording a video needs the camera/rendering pipeline to be enabled; turn it on
# automatically so callers don't have to pass --enable_cameras by hand.
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.t1 import T1_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # camera sensor (only used when recording a video; requires --enable_cameras)
    camera: CameraCfg | None = None


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    motion = MotionLoader(
        args_cli.motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Video recording (optional): record exactly one full motion pass, then exit.
    writer = None
    camera = None
    if args_cli.video:
        import imageio as iio

        out_dir = os.path.dirname(os.path.abspath(args_cli.video_path))
        os.makedirs(out_dir, exist_ok=True)
        writer = iio.get_writer(args_cli.video_path, fps=args_cli.video_fps, codec="libx264", quality=8)
        camera = scene["camera"]
        print(f"[INFO]: Recording video to {args_cli.video_path}")

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        if bool(reset_ids.any().item()):
            if writer is not None:
                break  # one full motion pass recorded
            time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])

        # Aim the recording camera at the robot base (world frame), reusing the same
        # look-at as the viewport camera.
        if camera is not None:
            base_pos = root_states[0, :3]
            eye = (base_pos + torch.tensor([2.0, 2.0, 0.5], device=sim.device)).unsqueeze(0)
            camera.set_world_poses_from_view(eye, base_pos.unsqueeze(0))

        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if writer is not None:
            writer.append_data(camera.data.output["rgb"][0].cpu().numpy())

    if writer is not None:
        writer.close()
        print(f"[INFO]: Video saved to {args_cli.video_path}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    if args_cli.video:
        scene_cfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
        )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()