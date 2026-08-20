# 环境要求 —— S10 巡逻赛题主链路（SMppi/TMppi Cruise + RL-Stair + TK1/TK2）

> 主链路目录：`SMppi_TMppi_Cruise_RL-Stair_TK1_TK2/`，纯 Python + MuJoCo，
> **无需 ROS / colcon / C++ 构建**。在全新 WSL2 + Ubuntu 24.04 上按本文档即可运行。
> 版本依据：官方比赛模板仓库（`DeepRoboticsLab/goai_embodied_future_material`，
> 即本仓库 initial commit 的来源）与官方比赛环境 `comp_env` 实测 freeze，见 §2。

## 1. 官方比赛安装（原始口径，git initial commit 的 README）

官方模板仓库安装要求原文：

```bash
# Use Ubuntu 24.04 with ROS 2 Jazzy. Source ROS before building:
pip install "numpy < 2.0" mujoco
git clone https://github.com/DeepRoboticsLab/goai_embodied_future_material.git
cd goai_embodied_future_material
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to s10_sdk_deploy --cmake-args -DBUILD_PLATFORM=x86
```

即官方口径：**Ubuntu 24.04 + ROS 2 Jazzy、numpy < 2.0、mujoco（当时 latest → 3.11.0）**。
ROS + colcon 只用于官方 ROS 仿真接口（rl_deploy ↔ mujoco_simulation 两个节点）；
**当前主链路不经过 ROS，这部分可不装**。

## 2. 官方比赛环境 comp_env 实测版本（本机 freeze）

`comp_env`（官方比赛环境专用 venv）实测版本：

| 包 | 版本 | 说明 |
| --- | --- | --- |
| Python | 3.12.3 | 系统 python3.12 |
| **numpy** | **1.26.4**（<2.0） | 官方 `pip install "numpy < 2.0"` 的结果 |
| **mujoco** | **3.11.0** | 官方 `pip install mujoco` 当时 latest |
| torch | 2.13.0+cpu | CPU 版 |
| matplotlib | 3.11.1 | |
| scipy | 1.13.1 | |
| glfw / PyOpenGL | 2.10.2 / 3.1.10 | viewer 用 |

已实测：**主链路在 comp_env（numpy 1.26.4）下整链导入通过**（`import cruise_main` OK）。
开发机 `.venv` 为 numpy 2.2.6 + torch 2.7.0+cu126，同样跑通——两个版本均可，
官方比赛口径以 **numpy 1.26.4 / mujoco 3.11.0** 为准。

## 3. 主链路运行时依赖清单

逐模块核对（第三方 import 就这些，无隐藏依赖）：

| 模块 | 第三方依赖 | 用途 |
| --- | --- | --- |
| `cruise_main.py` | numpy, mujoco | 200Hz 主循环、MuJoCo 仿真 |
| `nav_waypoint.py` | numpy, mujoco | 航点提取 / 直线段 |
| `smppi.py` + `s10_mpc/body_mppi.py` | numpy | SMppi 走线采样（N=1024, H=40） |
| `tmppi.py` | numpy | TMppi 原地转向 |
| `carvmc.py` + `s10_mpc/vmc_legs.py` | numpy | CarVMC 轮足执行 |
| `stair_mode.py` + `s10_mpc/auto_nav.py` | numpy | CRUISE/STAIR 模式判定与楼梯交接 |
| `perception_lidar.py` + `s10_mpc/lidar_terrain_v2.py` + `rl_stair/deploy/elev_tile.py` | numpy | lidar 高程图 / riser 检测 |
| `rlstair_ctrl.py` / `rlstair_obs.py` | numpy, torch | RL 策略推理（`torch.jit.load`，CPU 即可） |
| `plot_traj_speed.py`（可选，画图用） | numpy, matplotlib | 轨迹 xy-速度图 |

**不依赖**：jax / mujoco-mjx（仅 RL 训练需要）、scipy、mujoco-lidar、opencv、
taichi、ROS 2、drdds、onnxruntime、C++ 编译链。

**需要仓库自带文件**（clone 即包含，无需额外下载）：
- `src/S10_sdk_deploy/S10_description/s10_mjcf/`（S10_track.xml + meshes）
- `src/S10_sdk_deploy/s10_mpc/`（auto_nav / body_mppi / lidar_terrain_v2 / vmc_legs）
- `rl_stair/deploy/elev_tile.py`、主链路目录内的 `policy.pt`

## 4. 全新 WSL 安装步骤（按官方版本口径）

```bash
# 0) Windows 侧安装 WSL（管理员 PowerShell）
wsl --install -d Ubuntu-24.04

# 1) 系统依赖（headless 运行只需 python 与 git；MuJoCo viewer 需要 GL 库）
sudo apt update
sudo apt install -y git python3.12-venv python3-pip
# 可选（要用 viewer 时）：
sudo apt install -y libgl1 libglfw3 libglew-dev
```

```bash
# 2) 拉取仓库并创建 venv
mkdir -p ~/DR_competition && cd ~/DR_competition
git clone https://github.com/wangfanxin/deeprobot_competition.git
cd deeprobot_competition
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip setuptools wheel

# 3) 官方口径依赖：numpy<2.0（1.26.4）+ mujoco 3.11.0
./.venv/bin/pip install "numpy<2.0" "mujoco==3.11.0" "matplotlib"
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> 版本对照：官方 `comp_env` = numpy 1.26.4 / mujoco 3.11.0 / torch 2.13.0+cpu；
> 开发机 `.venv` = numpy 2.2.6 / mujoco 3.11.0 / torch 2.7.0+cu126。
> 主链路两者都验证过；如需与官方比赛环境完全一致，用上面第 3 步的版本。
> GPU 对主链路**非必需**：SMppi 为 numpy 采样、policy.pt 为 CPU torch 推理。

## 5. 环境验证

```bash
cd ~/DR_competition/deeprobot_competition
./.venv/bin/python - <<'PY'
import numpy, mujoco, torch
print("numpy", numpy.__version__, "| mujoco", mujoco.__version__, "| torch", torch.__version__)
m = mujoco.MjModel.from_xml_path(
    "src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10_track.xml")
print("S10_track.xml OK, nq =", m.nq, ", nu =", m.nu)
PY

# 主链路整链导入（应在主链路目录内执行）
cd SMppi_TMppi_Cruise_RL-Stair_TK1_TK2
../.venv/bin/python -c "import cruise_main; print('cruise_main OK')"
```

## 6. 运行

```bash
cd ~/DR_competition/deeprobot_competition
bash SMppi_TMppi_Cruise_RL-Stair_TK1_TK2/run_smppi_tmppi_cruise_rlstair_tk12.sh
# 等价于：
#   cd SMppi_TMppi_Cruise_RL-Stair_TK1_TK2
#   ~/DR_competition/deeprobot_competition/.venv/bin/python cruise_main.py
```

首次运行无 JIT 编译（主链路为 numpy 实现）；轨迹输出
`SMppi_TMppi_Cruise_RL-Stair_TK1_TK2/tmp/tmp_cruise_traj.npy`。

## 7. RL-Stair 训练附加依赖（可选，主链路运行不需要）

```bash
./.venv/bin/pip install "jax[cuda12]==0.4.38" "mujoco-mjx>=3.11" "tqdm"
```

训练需要 NVIDIA GPU + CUDA 12 驱动；命令见 README.md「RL-Stair 训练 / 评估 /
导出 / sim2sim」一节。

## 8. 常见问题

| 现象 | 处理 |
| --- | --- |
| `ModuleNotFoundError: s10_mpc` | 通过启动脚本运行（脚本会 `cd` 到主链路目录，模块内已自动把 `src/S10_sdk_deploy` 加入 sys.path） |
| `ModuleNotFoundError: No module named 'torch'` | 未装 torch；按 §4 用 CPU wheel 安装 |
| `S10_track.xml` 加载失败 / mesh 缺失 | 确认 `src/S10_sdk_deploy/S10_description/s10_mjcf/` 完整（clone 时未被忽略） |
| 找不到 policy.pt | 主链路目录自带 `policy.pt`；也可用 `S10_RL_POLICY` 指向其他导出策略 |
| numpy 装到 2.x | 官方口径用 `"numpy<2.0"`；主链路 2.x 也能跑，但官方环境为 1.26.4 |
| 想看 viewer 画面 | 安装 §4 的 GL 库并把启动脚本中 `S10_USE_VIEWER` 改为 `1`（需 WSLg/X11） |
| torch 下载太慢 | 用 CPU 源 `https://download.pytorch.org/whl/cpu`，或国内镜像 |
