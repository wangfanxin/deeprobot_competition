# 环境要求 —— S10 巡逻赛题主链路（SMppi/TMppi Cruise + RL-Stair + TK1/TK2）

> 主链路目录：`SMppi_TMppi_Cruise_RL-Stair_TK1_TK2/`，纯 Python + MuJoCo，
> **无需 ROS / colcon / C++ 构建**。在全新 WSL2 + Ubuntu 24.04 上按本文档即可运行。
> 开发机实测环境：WSL2 + Ubuntu 24.04 + Python 3.12.3。

## 1. 主链路运行时依赖清单

逐模块核对（第三方 import 就这些，无隐藏依赖）：

| 模块 | 第三方依赖 | 用途 |
| --- | --- | --- |
| `cruise_main.py` | numpy, mujoco | 200Hz 主循环、MuJoCo 仿真 |
| `nav_waypoint.py` | numpy, mujoco | 航点提取 / 直线段 |
| `smppi.py` + `s10_mpc/body_mppi.py` | numpy | SMppi 走线采样（N=1024, H=40） |
| `tmppi.py` | numpy | TMppi 原地转向 |
| `carvmc.py` + `s10_mpc/vmc_legs.py` | numpy | CarVMC 轮足执行 |
| `stair_mode.py` + `s10_mpc/auto_nav.py` | numpy | CRUISE/STAIR/DROP 门控 |
| `perception_lidar.py` + `s10_mpc/lidar_terrain_v2.py` + `rl_stair/deploy/elev_tile.py` | numpy | lidar 高程图 / riser 检测 |
| `rlstair_ctrl.py` / `rlstair_obs.py` | numpy, torch | RL 策略推理（`torch.jit.load`，CPU 即可） |
| `plot_traj_speed.py`（可选，画图用） | numpy, matplotlib | 轨迹 xy-速度图 |

**不依赖**：jax / mujoco-mjx（仅 RL 训练需要）、scipy、mujoco-lidar、opencv、
taichi、ROS 2、drdds、onnxruntime、C++ 编译链。

**需要仓库自带文件**（clone 即包含，无需额外下载）：
- `src/S10_sdk_deploy/S10_description/s10_mjcf/`（S10_track.xml + meshes）
- `src/S10_sdk_deploy/s10_mpc/`（auto_nav / body_mppi / lidar_terrain_v2 / vmc_legs）
- `rl_stair/deploy/elev_tile.py`、主链路目录内的 `policy.pt`

## 2. 全新 WSL 安装步骤（Ubuntu 24.04）

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

# 3) 主链路依赖（torch 用 CPU 版 wheel 即可，约 200MB；
#    如需 CUDA 版去掉 --index-url 并改用 PyPI 默认 wheel）
./.venv/bin/pip install "numpy>=2.0,<2.5" "mujoco>=3.11" "matplotlib"
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> 实测版本（开发机）：numpy 2.2.6 / mujoco 3.11.0 / torch 2.7.0 / matplotlib 3.11.1。
> GPU 对主链路**非必需**：SMppi 为 numpy 采样、policy.pt 推理为 CPU torch，
> 无 GPU 也能跑，只是 40Hz 控制拍要按总方案 §3 核对实时性。

## 3. 环境验证

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

## 4. 运行

```bash
cd ~/DR_competition/deeprobot_competition
bash SMppi_TMppi_Cruise_RL-Stair_TK1_TK2/run_smppi_tmppi_cruise_rlstair_tk12.sh
# 等价于：
#   cd SMppi_TMppi_Cruise_RL-Stair_TK1_TK2
#   ~/DR_competition/deeprobot_competition/.venv/bin/python cruise_main.py
```

首次运行无 JIT 编译（主链路为 numpy 实现）；轨迹输出
`SMppi_TMppi_Cruise_RL-Stair_TK1_TK2/tmp/tmp_cruise_traj.npy`。

## 5. RL-Stair 训练附加依赖（可选，主链路运行不需要）

```bash
./.venv/bin/pip install "jax[cuda12]==0.4.38" "mujoco-mjx>=3.11" "tqdm"
```

训练需要 NVIDIA GPU + CUDA 12 驱动；命令见 README.md「RL-Stair 训练 / 评估 /
导出 / sim2sim」一节。

## 6. 常见问题

| 现象 | 处理 |
| --- | --- |
| `ModuleNotFoundError: s10_mpc` | 通过启动脚本运行（脚本会 `cd` 到主链路目录，模块内已自动把 `src/S10_sdk_deploy` 加入 sys.path） |
| `ModuleNotFoundError: No module named 'torch'` | 未装 torch；按 §2 用 CPU wheel 安装 |
| `S10_track.xml` 加载失败 / mesh 缺失 | 确认 `src/S10_sdk_deploy/S10_description/s10_mjcf/` 完整（clone 时未被忽略） |
| 找不到 policy.pt | 主链路目录自带 `policy.pt`；也可用 `S10_RL_POLICY` 指向其他导出策略 |
| 想看 viewer 画面 | 安装 §1 的 GL 库并把启动脚本中 `S10_USE_VIEWER` 改为 `1`（需 WSLg/X11） |
| torch 下载太慢 | 用 CPU 源 `https://download.pytorch.org/whl/cpu`，或国内镜像 |
