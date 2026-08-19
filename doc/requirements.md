# 环境要求 —— S10 巡逻赛题（当前主链路）

> 当前主链路（SMppi/TMppi Cruise + RL-Stair + TK1/TK2）为**纯 Python + MuJoCo**，
> **无需 ROS / colcon**。开发机环境：WSL2 + Ubuntu 24.04 + NVIDIA GPU。

## 1. 基础环境

| 项目 | 要求 |
| --- | --- |
| 系统 | Ubuntu 24.04（开发机为 WSL2，x86_64） |
| Python | 3.12（项目 venv `~/DR_competition/.venv`） |
| GPU | NVIDIA 推荐（RL 训练必须；纯巡航 CPU 可跑但慢） |
| 内存 / 磁盘 | ≥16 GB RAM（训练推荐 32 GB）/ ≥20 GB 磁盘 |

## 2. 系统依赖

```bash
sudo apt update
sudo apt install -y python3.12-venv build-essential git   libgl1 libglfw3 libglew-dev
```

> MuJoCo viewer 需要 OpenGL/GLFW 运行库；主链路默认无头（`S10_USE_VIEWER=0`），
> 无 GUI 环境可省略 GL 相关包。

## 3. Python 虚拟环境与依赖

```bash
cd ~/DR_competition
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip setuptools wheel

# 主链路运行（numpy / mujoco / torch）
./.venv/bin/pip install "numpy>=2.0" "mujoco>=3.11" "torch>=2.0" "matplotlib"

# RL-Stair 训练（jax + mjx；CUDA 12 由 jax[cuda12] 自带，系统仅需 NVIDIA 驱动）
./.venv/bin/pip install "jax[cuda12]==0.4.38" "mujoco-mjx>=3.11" "tqdm"
```

## 4. 环境验证

```bash
cd ~/DR_competition
./.venv/bin/python - <<'PY'
import numpy, mujoco, torch
print("numpy", numpy.__version__, "| mujoco", mujoco.__version__, "| torch", torch.__version__)
import jax
print("jax backend:", jax.default_backend())   # 期望 gpu（无 GPU 则 cpu）
PY

# 主链路模型加载验证
./.venv/bin/python - <<'PY'
import mujoco
m = mujoco.MjModel.from_xml_path(
    "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10_track.xml")
print("S10_track.xml OK, nq =", m.nq, ", nu =", m.nu)
PY
```

## 5. 运行

```bash
cd ~/DR_competition/0810new/deeprobot_competition
bash SMppi_TMppi_Cruise_RL-Stair_TK1_TK2/run_smppi_tmppi_cruise_rlstair_tk12.sh
```

首次启动 JAX/JIT 编译约几秒~几十秒（缓存目录 `~/.cache/s10_dial_mpc`，可整体备份热启动）。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| `jax.default_backend()` 为 cpu | 检查 `nvidia-smi` 驱动；确认安装的是 `jax[cuda12]` |
| MuJoCo 模型加载失败 | 确认 `src/S10_sdk_deploy/S10_description/s10_mjcf/` 完整（xml+meshes） |
| `ModuleNotFoundError: s10_mpc` | 主循环启动脚本已自动把 `src/S10_sdk_deploy` 加入 sys.path，请勿直接 `python cruise_main.py` 之外的目录运行 |
| 训练显存不足 | 调小 `--num_envs`（如 256）并设 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6` |
