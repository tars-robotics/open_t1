# Copyright (c) 2026 TARS
# SPDX-License-Identifier: MIT

"""Build the upload/experiment directory for deployment.

Copies the ONNX model, motion file, robot MJCF + meshes, and env.yaml into
logs/upload/<experiment_name>/ using the experiment metadata defined below.
"""
import os
from pathlib import Path
import shutil
from lxml import etree


## Paths of the following simulation files must be specified before uploading 
feature = "t1_apt" # experiment behavior
data1 = ""
data2 = "1000"    # must be unique
## motion npz must carry body_names/joint_names metadata (30 bodies) -- the
## legacy APT export (31 bodies, no names) is not readable by sim2sim.py.
upload_motion_file = "data/APT_27dof_train_example.npz"

task = "t1"
robot_xml_path = "source/whole_body_tracking/whole_body_tracking/assets/t1/t1.xml"  
mesh_ori_path = "source/whole_body_tracking/whole_body_tracking/assets/t1/meshes" 
robot = "T1"


#=======================The following content does not need to be modified====================================================
upload_model_file = 'logs/rsl_rl/t1_flat/' + data1 + "/" + 'model_' + data2 + '.onnx'
experiment_name = feature.strip() + "_" + robot.strip() + "_" + data1 + "-" + data2
upload_path = Path("logs/upload") / experiment_name
env_yaml_path = Path(upload_model_file).parent / "params/env.yaml"
config_path = upload_path / "configs" 
mjcf_config_path = config_path/"mjcf"
tree = etree.parse(robot_xml_path)
compiler_element = tree.find('.//compiler')  # find via XPath
if compiler_element is not None:
    meshdir = compiler_element.get('meshdir')
mesh_config_path = mjcf_config_path/meshdir


os.makedirs(upload_path, exist_ok=True)

os.makedirs(config_path, exist_ok=True)
os.makedirs(mjcf_config_path, exist_ok=True)
os.makedirs(mesh_config_path, exist_ok=True)
all_items = os.listdir(mesh_ori_path) 
for item in all_items:
    shutil.copy2(mesh_ori_path+"/"+item, mesh_config_path)
shutil.copy2(robot_xml_path, mjcf_config_path)

shutil.copy2(env_yaml_path, upload_path)
shutil.copy2(upload_model_file, upload_path)
shutil.copy2(upload_motion_file, upload_path)

print(f"Files prepared at {upload_path}, experiment name: {experiment_name}")
