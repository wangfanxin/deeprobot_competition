# Ubuntu 22.04（非 WSL）安装要求 —— S10 巡逻赛题

> 本文档面向 **Ubuntu 22.04 LTS 原生系统（裸机/非 WSL）**，用于复现
> `deeprobot_competition` 仓库的 S10 轮足机器狗仿真比赛环境。
> 当前开发环境为 WSL2 + Ubuntu 24.04 + ROS 2 Jazzy，本文件是移植到
> Ubuntu 22.04 + ROS 2 Humble（22.04 官方 LTS 发行）的安装要求。

## 1. 目标环境

| 项目 | 要求 |
|---|---|
| 系统 | Ubuntu 22.04 LTS（x86_64，原生安装，非 WSL） |
| ROS 2 | Humble Hawksbill（`/opt/ros/humble`） |
| Python | 3.10（系统默认） |
| GPU | NVIDIA（推荐；CPU 可运行但 MPC 频率大幅下降） |
| 内存 / 磁盘 | ≥16 GB RAM（推荐 32 GB）/ ≥30 GB 磁盘 |

## 2. 前置检查

```bash
lsb_release -a        # 应为 Ubuntu 22.04
python3 --version     # 应为 3.10.x
nvidia-smi            # GPU 驱动可见；推荐驱动 ≥535（JAX CUDA12 运行时）
```

- 无 GPU 时仍可运行：`jax_platform` 用 `cpu`，MPC 频率约 2~8 Hz（仅调试用）。
- GUI 仿真需要 X11/Wayland 显示环境；无头运行见 §10。

## 3. 目录布局与代码获取

仓库是 ROS 2 工作空间，建议保持如下布局（与开发机一致）：

```
~/DR_competition/
├── .venv/                       # Python 虚拟环境（本机创建）
└── deeprobot_competition/       # 本仓库（git clone）
    ├── dial-mpc/                # dial-mpc 采样 MPC 库（已内置 S10 补丁，clone 即用）
    ├── src/S10_sdk_deploy/      # 仿真节点/感知/导航/控制器/模型
    ├── doc/                     # 0808.md + 部署 yaml + 官方材料
    └── tmp/                     # 核心测试入口与结果分析脚本
```

```bash
mkdir -p ~/DR_competition && cd ~/DR_competition
git clone https://github.com/wangfanxin/deeprobot_competition.git
```

> **dial-mpc 说明**：本项目的 `mpc_controller.py` 依赖 `dial_mpc` 包，
> 且当前 MPC 行为依赖 S10 定制（新增 `dial_mpc/envs/s10_env.py`，并修改
> `dial_core.py` / `dial_config.py`）。**S10 版 dial-mpc 已作为普通目录
> 内置在本仓库 `deeprobot_competition/dial-mpc/`**（剔除了 unitree 模型
> 资产等非必需文件），克隆仓库后无需再单独获取上游代码或打补丁。

## 4. 系统依赖（apt）

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  build-essential cmake git curl ca-certificates \
  python3-pip python3-venv python3-dev \
  libgl1 libglfw3 libglew-dev \
  libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev
```

> MuJoCo 原生 viewer 需要 OpenGL/GLFW 运行库；无头运行可省略 GUI 相关包。

## 5. ROS 2 Humble 安装

按官方步骤安装（ros2.org 精简版）：

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop ros-dev-tools
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
sudo apt install -y python3-colcon-common-extensions python3-rosdep
sudo rosdep init && rosdep update
```

本仓库两个 ROS 包 `drdds`（消息定义）与 `s10_sdk_deploy` 仅使用标准
依赖（rclcpp / sensor_msgs / geometry_msgs / tf2_ros /
rosidl_default_generators），Humble 桌面版已全部包含。

## 6. Python 虚拟环境与依赖

```bash
cd ~/DR_competition
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip setuptools wheel
```

核心依赖（实测版本来自 Ubuntu 24.04 开发机；Ubuntu 22.04 上 pip 会自动
解析兼容版本）：

```bash
./.venv/bin/pip install \
  "numpy>=2.0,<2.5" \
  "scipy>=1.11" \
  "jax[cuda12]==0.4.38" \
  "mujoco>=3.11" "mujoco-mjx>=3.11" \
  "mujoco-lidar>=0.3" \
  "PyYAML>=6.0" \
  "opencv-python" "matplotlib" "pynput" "tqdm"
```

可选（参考/备用与可视化）：

```bash
./.venv/bin/pip install "torch>=2.0" "brax>=0.10" "taichi>=1.7" \
  "tyro" "art" "emoji" "SciencePlots" "jax-cosmo"
```

> **版本注意**：
> - Ubuntu 22.04 系统 Python 为 3.10，**无法安装 numpy 2.5+**（要求
>   Python ≥3.11）；开发机当前 numpy 2.5.1 为 Ubuntu 24.04 实测值，
>   Ubuntu 22.04 使用 `numpy>=2.0,<2.5`（pip 将解析到 2.2.x）。
> - jax 0.4.38 支持 Python 3.9~3.12，与 Python 3.10 兼容；CUDA 运行时
>   由 `jax[cuda12]` 自带，系统仅需 NVIDIA 驱动。
> - 若安装 dial-mpc 时 pip 强制 `numpy<2.0.0`（上游 setup.py 约束），
>   可改用 `--no-deps` 安装并手动装齐依赖（见 §7）。

## 7. dial-mpc 安装

S10 版 dial-mpc 已内置在仓库内（`deeprobot_competition/dial-mpc/`），
直接 editable 安装：

```bash
cd ~/DR_competition/deeprobot_competition/dial-mpc
~/DR_competition/.venv/bin/pip install -e . --no-deps
```

`--no-deps` 避免上游 `numpy<2.0.0` 约束与 §6 版本冲突；dial-mpc 所需的
numpy / jax / mujoco / brax / matplotlib / tyro 等已在 §6 安装。

## 8. ROS 2 工作空间构建

```bash
cd ~/DR_competition/deeprobot_competition
source /opt/ros/humble/setup.bash
colcon build --packages-up-to s10_sdk_deploy --cmake-args -DBUILD_PLATFORM=x86
source install/setup.bash
```

- x86 原生环境用 `-DBUILD_PLATFORM=x86`；真机（ARM）迁移时用 `arm`。
- 若首次构建报缺少依赖，可用 rosdep：
  ```bash
  sudo rosdep init && rosdep update   # 已初始化可跳过
  rosdep install --from-paths src --ignore-src -r -y
  ```

## 9. 环境验证

```bash
cd ~/DR_competition
./.venv/bin/python - <<'PY'
import jax, mujoco, mujoco_lidar
print("jax backend:", jax.default_backend())   # 期望 gpu（无 GPU 则 cpu）
print("mujoco:", mujoco.__version__)
PY
```

```bash
# dial-mpc S10 导入验证
cd ~/DR_competition
./.venv/bin/python - <<'PY'
from dial_mpc.core.dial_core import DialConfig, MBDPI
from dial_mpc.envs.s10_env import S10WheeledEnv, S10WheeledEnvConfig
print("dial_mpc S10 import OK")
PY
```

```bash
# ROS 消息可用性
cd ~/DR_competition/deeprobot_competition
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 -c "from drdds.msg import ImuData, JointsData; print('drdds OK')"
```

## 10. 运行命令

```bash
cd ~/DR_competition/deeprobot_competition
source /opt/ros/humble/setup.bash
source install/setup.bash
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/s10_dial_mpc

# 模式 B 遥控（GUI：z 站起 / c 进 MPC / wasd 移动 / qe 转向）
S10_MPC_ENABLE=1 ~/DR_competition/.venv/bin/python \
  src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py

# 模式 A 自动导航（无头）
S10_MPC_ENABLE=1 S10_MODE=auto_nav S10_USE_VIEWER=0 \
  ~/DR_competition/.venv/bin/python \
  src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py
```

首次启动会触发 JAX JIT 编译（约 4~30 s，取决于缓存）；建议把
`~/.cache/s10_dial_mpc` 一并备份部署，热启动可压到 ~5 s。

## 11. 常见问题

| 现象 | 处理 |
|---|---|
| `jax.default_backend()` 为 cpu | 检查 `nvidia-smi` 驱动；重装 `jax[cuda12]==0.4.38` |
| LiDAR core dump / 花屏 | 保持 `S10_LIDAR_BACKEND=cpu`（taichi GPU 后端在部分环境不稳定） |
| 找不到 `drdds` 模块 | 未 `source install/setup.bash`，或未执行 §8 构建 |
| 找不到 `dial_mpc` 模块 | 未执行 §7 的 `pip install -e .`，或 PYTHONPATH 未包含 dial-mpc |
| `numpy` 装不上 | Ubuntu 22.04 用 `numpy>=2.0,<2.5`（3.10 无法装 2.5+） |
| pip 把 numpy 降到 1.x | dial-mpc 用 `--no-deps` 安装（§7） |
| 无头运行 | `S10_USE_VIEWER=0`；如需远程看画面，用 X11 转发或录制轨迹分析 |
| MPC 频率明显低于 13 Hz | 检查是否有其他进程占用 GPU；确认 Nsample/Ndiffuse 未调大 |

## 12. 版本对照表

| 包 | Ubuntu 24.04 实测 | Ubuntu 22.04 建议 |
|---|---|---|
| ROS 2 | Jazzy | Humble |
| Python | 3.12.3 | 3.10 |
| jax / jaxlib | 0.4.38 | 0.4.38（py3.10 兼容） |
| mujoco / mujoco-mjx | 3.11.0 | ≥3.11 |
| mujoco-lidar | 0.3.3 | ≥0.3 |
| numpy | 2.5.1 | >=2.0,<2.5（pip 自动解析） |
| scipy | 1.14.1 | ≥1.11 |
| PyYAML | 6.0.3 | ≥6.0 |
| taichi（可选） | 1.7.4 | ≥1.7 |
| torch / brax（可选） | 2.7.0 / 0.14.2 | ≥2.0 / ≥0.10 |

> 详细架构、参数与运行说明见 [doc/0808.md](0808.md)；本文件只负责环境安装。