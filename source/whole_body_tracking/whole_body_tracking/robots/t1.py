# Copyright (c) 2026 TARS
# SPDX-License-Identifier: MIT

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# Armature values (kg*m^2), adapted from the T1 reference actuator tuning.
ARMATURE_HIP = 0.0149
ARMATURE_KNEE = 0.0081
ARMATURE_ANKLE = 0.0038

# PD gains adapted from the reference repo's tuned "robot_cfg_PD" values.
STIFFNESS_HIP_PITCH = 60.0
STIFFNESS_HIP_ROLL = 80.0
STIFFNESS_HIP_YAW = 100.0
STIFFNESS_KNEE = 100.0
STIFFNESS_ANKLE = 45.0
STIFFNESS_WAIST_YAW = 45.0
STIFFNESS_WRIST = 40.0

DAMPING_HIP_PITCH = 10.0
DAMPING_HIP_ROLL = 10.0
DAMPING_HIP_YAW = 4.0
DAMPING_KNEE = 6.0
DAMPING_ANKLE = 3.0
DAMPING_WAIST_YAW = 3.0
DAMPING_WRIST = 2.0

T1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/t1/t1.urdf",
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.90),
        joint_pos={
            ".*_hip_pitch_joint": -0.15,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.15,
            ".*_elbow_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 180.0,
                ".*_hip_pitch_joint": 180.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 17.0,
                ".*_hip_roll_joint": 16.0,
                ".*_hip_pitch_joint": 16.0,
                ".*_knee_joint": 17.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_HIP_PITCH,
                ".*_hip_roll_joint": STIFFNESS_HIP_ROLL,
                ".*_hip_yaw_joint": STIFFNESS_HIP_YAW,
                ".*_knee_joint": STIFFNESS_KNEE,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_HIP_PITCH,
                ".*_hip_roll_joint": DAMPING_HIP_ROLL,
                ".*_hip_yaw_joint": DAMPING_HIP_YAW,
                ".*_knee_joint": DAMPING_KNEE,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_HIP,
                ".*_hip_roll_joint": ARMATURE_HIP,
                ".*_hip_yaw_joint": ARMATURE_KNEE,
                ".*_knee_joint": ARMATURE_KNEE,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=45.0,
            velocity_limit_sim=9.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=STIFFNESS_ANKLE,
            damping=DAMPING_ANKLE,
            armature=2.0 * ARMATURE_ANKLE,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=120.0,
            velocity_limit_sim=17.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_WAIST_YAW,
            damping=DAMPING_WAIST_YAW,
            armature=ARMATURE_KNEE,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 45.0,
                ".*_shoulder_roll_joint": 45.0,
                ".*_shoulder_yaw_joint": 45.0,
                ".*_elbow_joint": 45.0,
                ".*_wrist_roll_joint": 10.0,
                ".*_wrist_pitch_joint": 10.0,
                ".*_wrist_yaw_joint": 10.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 9.0,
                ".*_shoulder_roll_joint": 9.0,
                ".*_shoulder_yaw_joint": 9.0,
                ".*_elbow_joint": 9.0,
                ".*_wrist_roll_joint": 22.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_ANKLE,
                ".*_shoulder_roll_joint": STIFFNESS_ANKLE,
                ".*_shoulder_yaw_joint": STIFFNESS_ANKLE,
                ".*_elbow_joint": STIFFNESS_ANKLE,
                ".*_wrist_roll_joint": STIFFNESS_WRIST,
                ".*_wrist_pitch_joint": STIFFNESS_WRIST,
                ".*_wrist_yaw_joint": STIFFNESS_WRIST,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_ANKLE,
                ".*_shoulder_roll_joint": DAMPING_ANKLE,
                ".*_shoulder_yaw_joint": DAMPING_ANKLE,
                ".*_elbow_joint": DAMPING_ANKLE,
                ".*_wrist_roll_joint": DAMPING_WRIST,
                ".*_wrist_pitch_joint": DAMPING_WRIST,
                ".*_wrist_yaw_joint": DAMPING_WRIST,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_ANKLE,
                ".*_shoulder_roll_joint": ARMATURE_ANKLE,
                ".*_shoulder_yaw_joint": ARMATURE_ANKLE,
                ".*_elbow_joint": ARMATURE_ANKLE,
                ".*_wrist_roll_joint": ARMATURE_ANKLE,
                ".*_wrist_pitch_joint": ARMATURE_ANKLE,
                ".*_wrist_yaw_joint": ARMATURE_ANKLE,
            },
        ),
    },
)

T1_ACTION_SCALE = {}
for a in T1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            T1_ACTION_SCALE[n] = 0.25  #* e[n] / s[n]
