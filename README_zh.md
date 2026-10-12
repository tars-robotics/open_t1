# T1 全身运动跟踪

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Isaac Sim](https://img.shields.io/badge/IsaacSim-5.x-silver.svg)](https://www.isaacsim.omniverse.nvidia.com/)
[![rsl_rl](https://img.shields.io/badge/rsl__rl-%E2%89%A55.5.0-orange)](https://github.com/leggedrobotics/rsl_rl)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

**T1 全身运动跟踪** 是一套基于 [BeyondMimic](https://beyondmimic.github.io/) 的人形机器人全身运动跟踪框架。本仓库是开源项目 [`HybridRobotics/whole_body_tracking`](https://github.com/HybridRobotics/whole_body_tracking) 的衍生版本，针对 **T1** 人形机器人做了改造。它能把任意参考动作以可部署的质量复现到 **T1** 上，并提供一条端到端、面向 Sim2Real 的完整链路：

- **训练** 基于 Isaac Lab，但后端使用原生 [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl)（而非 Isaac Lab 自带的 `isaaclab_rl.rsl_rl` fork）；
- **导出** 单个自包含的 ONNX 策略（内置参考动作、附带部署所需的 metadata）；
- **验证** 在一个**不依赖 Isaac Lab / Isaac Sim** 的独立 MuJoCo 仿真器里验证导出策略的真实行为。

## 亮点

- **T1 机器人任务** —— 完整的 `Tracking-Flat-T1-v0` 训练任务：机器人 PD/动作缩放配置、14 个关键点 anchor-relative 观测、DeepMimic 风格奖励、终止条件与域随机化。
- **原生 `rsl_rl`（≥ 5.5.0）后端** —— 替换了 `isaaclab_rl` 使用的旧 MOE / history-observation fork，采用模型内部观测归一化、`obs_groups` actor/critic 选择，以及一个基于 `TensorDict` 的轻量 `VecEnv` wrapper。
- **自动 ONNX 导出** —— 每个 save interval 自动导出一个确定性 ONNX 策略，其 metadata（`joint_names`、PD 增益、动作缩放、观测布局、body/anchor 名称、参考动作）是下游部署的**唯一配置来源**。
- **独立 MuJoCo Sim2Sim** —— 一个依赖极轻（`mujoco` + `onnxruntime`）的评估器，能回放参考动作与导出策略，并渲染关键点叠加的视频（交互查看器或离线 mp4）。

## 环境要求

| 组件 | 版本 / 说明 |
|---|---|
| Python | 3.11 |
| Isaac Sim | 5.x |
| Isaac Lab | 2.3.2（`isaaclab` 0.54.3） |
| [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl) | **≥ 5.5.0**（原生，也是 T1 唯一使用的 RL 后端） |
| `tensordict` | `rsl_rl` ≥ 5.x 的依赖，必须有 |
| PyTorch | 2.7（CUDA） |
| GPU | 带较新驱动的 NVIDIA GPU |

## 安装

### 1. 安装 Isaac Lab

按[安装教程](https://docs.robotsfan.com/isaaclab/source/setup/installation/binaries_installation.html)安装 Isaac Lab 2.3.2 及其 conda 环境。

### 2. 安装原生 `rsl_rl`（≥ 5.5.0）—— 关键步骤

Isaac Lab 自带的 `isaaclab_rl[rsl-rl]` 只会装到 `rsl-rl-lib==5.0.1`，而本仓库需要原生 `rsl_rl` **≥ 5.5.0** 的 API（`construct_algorithm`、`get_policy()`、`WandbLogWriter` 等），因此显式安装：

```bash
# 从 PyPI
python -m pip install -U "rsl-rl-lib>=5.5.0"

# 或从你自己的 clone（内网/离线时更稳）
python -m pip install -e /path/to/rsl_rl --no-deps --no-build-isolation
```

> ⚠️ 若环境里残留旧版 `rsl_rl`（2.x）的可编辑安装，会遮蔽 5.x——因为两者顶层包都叫 `rsl_rl`。

### 3. 安装本仓库

```bash
git clone https://github.com/TARS-AI/whole_body_tracking.git
cd whole_body_tracking
python -m pip install -e source/whole_body_tracking
```

> `--no-deps --no-build-isolation` 仅供离线/内网安装时跳过依赖解析与构建隔离；此时需自行装好 `psutil`、`onnxscript`、`wandb>=0.19`。

### 4. 安装自检

```bash
python - <<'PY'
import importlib.util, importlib.metadata as m
import rsl_rl

assert "whole_body_tracking" in importlib.util.find_spec("whole_body_tracking").origin
assert m.version("rsl-rl-lib") == "5.5.0"          # 或 >= 5.5.0

from rsl_rl.utils.wandb_log_writer import WandbLogWriter   # 仅 >= 5.x 存在
from rsl_rl.env import VecEnv
from tensordict import TensorDict
print("OK")
PY
```

## 快速开始

### 1. 准备参考动作

参考动作是 `.npz` 文件，存有重定向后的关节位置/速度，以及 body 位置/四元数/速度。

- 直接用本地 `.npz` 配合 `--motion_file`。

> `.npz` 内的 body/joint 顺序**必须**与 `robot.body_names` / `robot.joint_names` 一致。顺序不一致会导致跟踪误差爆炸、每个 episode 一步即终止。可参考示例：data/APT_27dof_train_example.npz

### 2. 回放参考动作（可视化检查）

在 Isaac Sim 里把本地 `.npz` 动作回放（replay）到 T1 机器人上做快速可视化检查，或 headless 录制为 mp4：

```bash
# 交互窗口（需要显示器）
python scripts/replay_npz.py --motion_file data/APT_27dof_train_example.npz

# headless，录制回放为 mp4
python scripts/replay_npz.py --motion_file data/APT_27dof_train_example.npz \
    --headless --video --video_path data/tmp.mp4 --video_fps 50
```

`--video` 会启用离线相机传感器，录制完一整趟动作后自动退出；`--video_path` / `--video_fps` 用于指定输出路径与帧率。

### 3. 训练

```bash
# 本地动作文件
python scripts/rsl_rl/train.py --task=Tracking-Flat-T1-v0 \
    --motion_file /path/to/motion.npz \
    --headless --logger wandb --log_project_name my_project --run_name my_run

# 或走 WandB 动作 registry
python scripts/rsl_rl/train.py --task=Tracking-Flat-T1-v0 \
    --registry_name {org}-org/wandb-registry-motions/my_motion \
    --headless --logger wandb --log_project_name my_project
```

常用参数：`--num_envs`、`--max_iterations`、`--seed`、`--experiment_name`、`--resume`/`--load_run`/`--checkpoint`。
使用 `--logger wandb` 时，每次保存都会在 `.pt` checkpoint 同目录（`logs/rsl_rl/t1_flat/{run}/`）导出 ONNX 策略。

### 4. 评估 / 导出 ONNX

```bash
# 从 WandB run 播放
python scripts/rsl_rl/play.py --task=Tracking-Flat-T1-v0 --num_envs=2 --wandb_path={org}/{project}/{run_id}

# 本地 checkpoint + 本地动作
python scripts/rsl_rl/play.py --task=Tracking-Flat-T1-v0 \
    --load_path logs/rsl_rl/t1_flat/{run}/model_xxx.pt --motion_file /path/to/motion.npz

# 只导出 ONNX
python scripts/rsl_rl/play.py --task=Tracking-Flat-T1-v0 --load_path ... --export_only
```

### 5. Sim2Sim（MuJoCo，独立运行）

`sim2sim.py` 有两种调用方式：

**原始用法（逐个指定文件）**：

```bash
python sim2sim/t-deploy_mujoco/sim2sim.py \
    --motion_file /path/to/motion.npz \
    --onnx_load_path /path/to/policy.onnx \
    --xml_path /path/to/t1.xml \
    --env_path logs/rsl_rl/t1_flat/{run}/params/env.yaml \
    --visualize           # 或 --record_video / --play_reference
```

**bundle 用法（索引目录）** —— 直接指向一个部署 bundle 目录（由 `scripts/build_bundle.py` 构建，包含 `model_*.onnx`、`env.yaml`、`configs/mjcf/<robot>.xml` 和动作 `*.npz`），脚本会自动解析这些文件：

```bash
# 交互可视化（需要显示器）
python sim2sim/t-deploy_mujoco/sim2sim.py logs/upload/<experiment> --visualize

# 为每条 motion 离线渲染 mp4（无头，MuJoCo EGL 后端）
python sim2sim/t-deploy_mujoco/sim2sim.py logs/upload/<experiment> --record_video

# 只回放参考动作（不加载策略）
python sim2sim/t-deploy_mujoco/sim2sim.py logs/upload/<experiment> --record_video --play_reference
```

只需 `numpy`、`torch`、`onnxruntime`、`mujoco`、`scipy`、`tqdm`（`--record_video` 时额外需要 `imageio`），不依赖 Isaac Lab / Isaac Sim。动作 `*.npz` 必须携带 `body_names`/`joint_names` 元数据（30 个 body）；旧的 APT 导出（31 个 body、无 names）不受支持。详见 [`sim2sim/t-deploy_mujoco/`](sim2sim/t-deploy_mujoco/)。

> **用 `build_bundle.py` 构建 bundle 前**，先修改 [`scripts/build_bundle.py`](scripts/build_bundle.py) 顶部 4 个硬编码变量——`feature`、`data1`、`data2`、`upload_motion_file`——仓库默认值指向过期的 run 与不存在的 `data/motion.npz`；该脚本还依赖 `lxml`。

## 项目结构

```
scripts/rsl_rl/                训练 / 评估 / 导出入口脚本
scripts/replay_npz.py          回放本地动作（交互或 headless 录制 mp4）
scripts/build_bundle.py        构建 logs/upload/ 下的部署 bundle
scripts/train_t1.sh            T1 训练便捷启动脚本
source/whole_body_tracking/whole_body_tracking/
├── robots/t1.py               T1 PD 增益 / armature / 动作缩放配置
├── tasks/tracking/
│   ├── tracking_env_cfg.py    机器人无关的 MDP 基类配置
│   ├── mdp/                   commands / observations / rewards / events / terminations
│   └── config/t1/             T1 环境配置 + PPO 配置 + gym 注册
└── utils/                     runner / vec-env wrapper / ONNX 导出器
sim2sim/t-deploy_mujoco/       MuJoCo sim2sim 评估工具
```

## 引用

本工作基于 **BeyondMimic**，如用于研究请引用：

```bibtex
@misc{liao2025beyondmimic,
  title         = {BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion},
  author        = {Qiayuan Liao and Takara E. Truong and Xiaoyu Huang and Yuman Gao and Guy Tevet and Koushil Sreenath and C. Karen Liu},
  year          = {2025},
  eprint        = {2508.08241},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2508.08241},
}
```

## 致谢

- [BeyondMimic](https://beyondmimic.github.io/)（HybridRobotics）—— 本仓库基于的运动跟踪框架。
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab) 与 [Isaac Sim](https://www.isaacsim.omniverse.nvidia.com/)。
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)（ETH Zürich / NVIDIA）—— RL 后端。

## 许可证与第三方声明

本项目采用 [MIT 许可证](LICENSE)，与其上游 [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) 仓库保持一致。

本代码库还包含源自以下开源项目的组件，其原始版权与许可证声明均保留在对应源文件中：

| 组件 | 许可证 | 版权 / 来源 |
|---|---|---|
| [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) | MIT | HybridRobotics |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab) | BSD-3-Clause | Isaac Lab Project Developers |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) | BSD-3-Clause | ETH Zürich / NVIDIA Corporation |