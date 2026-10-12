# Copyright (c) 2026 TARS
# SPDX-License-Identifier: MIT

from isaaclab.utils import configclass

from whole_body_tracking.robots.t1 import T1_ACTION_SCALE, T1_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class T1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_roll_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_roll_link",
        ]

        # T1's tracked wrist end-effector link is "*_wrist_roll_link".
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_roll_link$)(?!right_wrist_roll_link$).+$"
        ]
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_roll_link",
            "right_wrist_roll_link",
        ]
