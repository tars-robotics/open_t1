# Copyright (c) 2026 TARS
# SPDX-License-Identifier: MIT

"""Parse ONNX metadata and env.yaml into the config dict used by deployment.

Extracts joint names/defaults, PD stiffness/damping, action scales, and the
observation/control settings consumed by run_rollout.
"""
from typing import Sequence, Dict, List, Any, Tuple, Optional
import ast,json,yaml
import onnx
import os
import re

def parse_metadata_value(value: str) -> Any:
    """Parse a metadata value, attempting to convert it to an appropriate data type."""
    # If it is a string in list format
    if value.startswith('[') and value.endswith(']'):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    
    # If it is a CSV-formatted string
    if ',' in value and not value.startswith('{'):
        return value.split(',')
    
    # Try to parse as JSON
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    
    # Keep the original string
    return value
def load_onnx_metadata(onnx_path: str) -> Dict[str, Any]:
    """Load the metadata of an ONNX model."""
    model = onnx.load(onnx_path)
    return {prop.key: parse_metadata_value(prop.value) for prop in model.metadata_props}

def get_data_from_info(info: Dict[str, Any], joint_names: List[str]) -> List[float]:
    data = []
    for joint_name in joint_names:
        if joint_name in info:
            data.append(info[joint_name])
            continue
        matched = False
        for pattern, value in info.items():
             if pattern.startswith('.*'):
                if re.match(pattern, joint_name):
                    data.append(value)
                    matched = True
                    break
        if not matched:
            data.append(0.0)
    return data

def get_policy_kpkd(info: Dict[str, Any], joint_names: List[str]):
    default_stiffness_policy = []
    default_damping_policy = []
    for joint_name in joint_names:
        stiffness_found = False
        damping_found = False
        stiffness_value = 0.0  # default value
        damping_value = 0.0    # default value   
        # Check each body part
        for body_part, config in info.items():
            # Handle stiffness
            if not stiffness_found:
                if isinstance(config['stiffness'], dict):
                    # stiffness is a dict, use the inner expressions
                    for pattern, value in config['stiffness'].items():
                        if re.match(pattern, joint_name):
                            stiffness_value = value
                            stiffness_found = True
                            break
                else:
                    # stiffness is a scalar, use joint_names_expr
                    for expr in config['joint_names_expr']:
                        if re.match(expr, joint_name):
                            stiffness_value = config['stiffness']
                            stiffness_found = True
                            break
            
            # Handle damping
            if not damping_found:
                if isinstance(config['damping'], dict):
                    # damping is a dict, use the inner expressions
                    for pattern, value in config['damping'].items():
                        if re.match(pattern, joint_name):
                            damping_value = value
                            damping_found = True
                            break
                else:
                    # damping is a scalar, use joint_names_expr
                    for expr in config['joint_names_expr']:
                        if re.match(expr, joint_name):
                            damping_value = config['damping']
                            damping_found = True
                            break
            
            # If both stiffness and damping are found, break the loop early
            if stiffness_found and damping_found:
                break
        # Append to the result lists
        default_stiffness_policy.append(stiffness_value)
        default_damping_policy.append(damping_value)
    return default_stiffness_policy, default_damping_policy

def parse_args(env_yaml, onnx_path) -> Dict[str, Any]:
    new_config = {}
    onnx_file_path   = onnx_path
    env_config_path  = env_yaml

     
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX config file {onnx_file_path} does not exist.")

    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f"ONNX config file {env_config_path} does not exist.")


    metadata = load_onnx_metadata(onnx_file_path)
    with open(env_config_path, 'r', encoding='utf-8') as file:
        env_config = yaml.unsafe_load(file)

    new_config = metadata

    info = env_config.get("scene").get("robot").get("init_state").get("joint_pos")
    default_dof_pos_policy = get_data_from_info(info, metadata['joint_names'])

    info = env_config.get("actions").get("joint_pos").get("scale")
    action_scale = get_data_from_info(info, metadata['joint_names'])
    
    info = env_config.get("scene").get("robot").get("actuators")
    kps, kds = get_policy_kpkd(info, metadata['joint_names'])

    new_config["onnx_file_path"]   = onnx_file_path
    new_config["obs_buffer_length"] = env_config.get("observations").get("policy").get("history_length")
    new_config["keypoints"] = env_config.get("commands").get("motion").get("body_names")
    # new_config["policy_to_robot_indices"] = policy_to_robot_indices
    # new_config["robot_to_policy_indices"] = robot_to_policy_indices
    new_config["default_angles_policy"] = default_dof_pos_policy
    new_config["use_delta_action_w_ref"] = env_config.get("events").get("add_joint_default_pos").get("params").get("use_delta_action_w_ref", False)
    new_config["use_delta_joint_pos_w_ref"] = env_config.get("events").get("add_joint_default_pos").get("params").get("use_delta_joint_pos_w_ref", False)
    new_config["use_delta_joint_vel_w_ref"] = env_config.get("events").get("add_joint_default_pos").get("params").get("use_delta_joint_vel_w_ref", False)
    new_config["joint_stiffness"] = kps
    new_config["joint_damping"] = kds
    new_config["action_scale"] = action_scale
    new_config["simulation_dt"] = env_config.get("sim").get("dt")
    new_config["control_decimation"] = env_config.get("sim").get("render_interval")
    
    new_config["ang_vel_scale"] = env_config.get("observations").get("policy").get("base_ang_vel").get("scale") 
    new_config["dof_pos_scale"] = env_config.get("observations").get("policy").get("joint_pos").get("scale") 
    new_config["dof_vel_scale"] = env_config.get("observations").get("policy").get("joint_vel").get("scale") 

    try:
        future_commands_step = env_config.get("observations").get("policy").get("future_command").get("params").get("future_steps", [1,2,3])
    except:
        future_commands_step = None
        pass  # default value
    
    if future_commands_step is None:
        new_config["future_commands_step"] = [1,2,3]
    elif type(future_commands_step) is int:
        new_config["future_commands_step"] = [i*3+1 for i in range(future_commands_step)]
    else:
        new_config["future_commands_step"] = future_commands_step

    if new_config["ang_vel_scale"] is None:
        new_config["ang_vel_scale"] = 1.0
    if new_config["dof_pos_scale"] is None:
        new_config["dof_pos_scale"] = 1.0
    if new_config["dof_vel_scale"] is None:
        new_config["dof_vel_scale"] = 1.0

    return new_config
    

