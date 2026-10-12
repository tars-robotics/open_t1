#!/usr/bin/env bash
# Train the T1 motion tracking policy using a local motion .npz file
# (bypasses the WandB registry download).
#
# This is a thin convenience wrapper around the generic scripts/rsl_rl/train.py,
# which supports --motion_file as a local-file override for --registry_name.
# It pins --task=Tracking-Flat-T1-v0 and --motion_file to the local motion data,
# and forwards all other arguments.
#
# Usage:
#   scripts/train_t1.sh [extra args...]
#
# The motion .npz must store its 30 body rows in the same order as
# robot.find_bodies() (IsaacLab/PhysX BFS-by-joint-level), and must carry explicit
# body_names/joint_names metadata. A file whose body rows are in URDF/chain-DFS
# order will silently compare the wrong bodies against each other (tracking error
# pinned near a constant ~4.5 m, every episode terminating after one step).
#

python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-T1-v0 \
    --motion_file="data/APT_27dof_train_example.npz" \
    --headless \
    --num_envs 4096
