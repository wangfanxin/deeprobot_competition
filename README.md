# S10 巡逻赛题 · 感知-MPC 工程仓库

基于山猫 S10 四足轮式机器人的**巡逻赛题**参赛方案：LiDAR 感知 + 高程地形建模 +
dial-mpc 采样 MPC 控制，在 MuJoCo 仿真中完成 33 航点全程巡检。

> 完整工程文档见 [doc/0808.md](doc/0808.md)；参数唯一来源为
> [doc/s10_mpc_deploy.yaml](doc/s10_mpc_deploy.yaml)。

## 赛题与计分

- 仿真环境（官方提供）：`S10_track.xml` 场景 + `track_overlay.xml` 33 个航点
  （000_start ~ 032_end），base 进入 wp0 的 0.2 m 水平半径开始计时，逐点推进，
  到达终点停止计时并打印耗时。
- 计分：总成绩 = 完成时间 ÷ 模式系数，得分越低排名越靠前，30 个定位点须全部
  完成。模式系数以官方 PDF 为准：**遥控 ÷1.0 / 自主跟随 ÷1.3 / 自主导航 ÷1.4**
  （本仓库早期 README 中"导航 ÷1.2"为旧表述，已废弃）。
- 申报模式：**自主跟随（÷1.3）**——航点跟随 + 感知地形限速/爬坡 + roll 安全，
  不强制全局 A\*。

## 已实现（2026-08-08）

```mermaid
graph LR
    A["/rl_deploy"] -->|"/JOINTS_CMD"| B["/mujoco_simulation"]
    B -->|"/IMU_DATA"| A
    B -->|"/JOINTS_DATA"| A
```

- **感知**：Airy-96 LiDAR 仿真（120×48 线 @10Hz，前下 45° 安装）→ 世界对齐
  高程瓦片 `perception/local_map.py`（8×8m 锚定 / 输出 60×60@0.1m，跨帧累积 +
  空洞填补 + 运动学地面注入）→ 纯 jnp 查图 `elevation_lookup.py`（零 retrace）。
- **导航**：Catmull-Rom 平滑路径（过 33 航点 0.013m、全长 234m）+ 曲率/横脊/
  高架限速速度剖面 + 单调弧长游标/切线投影 + CRUISE/STAIR 双模式仲裁。
- **控制**：dial-mpc MBDPI（CRUISE H25/N768 ≈13.7~14.3Hz，STAIR H20/N512 ≈15Hz，
  solver_it 4），CRUISE/STAIR 双模式 cost 权重 + w_prog 进度奖励；模式 B 遥控
  （4.5 m/s 竞速档）+ 模式 A 自动导航。
- **爬坡**：foot_place 抬轮 + 前瞻抬轮 + 抬腿伸展引导（r_ext）+ 锁轮推身
  （lockpush）+ 机身逐级抬升（w_clear），wp7 连续台阶可爬 3~4 级。
- **工程**：JAX 编译缓存（冷启动 ≈4~5s）、文档化环境变量覆盖、`doc/0808.md`
  完整记录赛程/规则/参数/实验链。

## 目录结构

```
DR_competition/
├── .venv/                       # 项目虚拟环境（Python 3.12.3）
├── dial-mpc/                    # dial-mpc 采样 MPC 库（含 S10 补丁）
├── deeprobot_competition/       # 本仓库：ROS2 工作空间
│   ├── src/S10_sdk_deploy/      # 仿真节点/感知/导航/控制器/模型
│   ├── doc/                     # 0808.md + 部署 yaml + 官方材料
│   └── tmp/                     # 核心测试入口与结果分析脚本
└── refs/                        # 参考仓库（go2w_rl_gym、unitree_mujoco）
```

## 环境与快速开始

要求：**Ubuntu 24.04 + ROS 2 Jazzy + NVIDIA GPU（可选，CPU 可跑但慢）**。
完整环境配置（含包版本）见 `doc/0808.md` §3。

```bash
# 1) 虚拟环境（仓库根目录，注意在 deeprobot_competition 上一级）
cd ~/DR_competition
/usr/bin/python3 -m venv .venv
./.venv/bin/pip install "numpy<2" mujoco mujoco-lidar jax[cuda12]==0.4.38

# 2) 构建 ROS2 工作空间
cd ~/DR_competition/deeprobot_competition
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to s10_sdk_deploy --cmake-args -DBUILD_PLATFORM=x86
source install/setup.bash

# 3) 模式 B 遥控（窗口内 z 站起 / c 进入 MPC / wasd 移动 / qe 转向）
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/s10_dial_mpc
S10_MPC_ENABLE=1 ~/DR_competition/.venv/bin/python \
  src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py

# 4) 模式 A 自动导航（无头加 S10_USE_VIEWER=0）
S10_MPC_ENABLE=1 S10_MODE=auto_nav ~/DR_competition/.venv/bin/python \
  src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py
```

加载自定义 MJCF（如加装传感器）：

```bash
S10_MUJOCO_XML=/absolute/path/to/model.xml \
  ~/DR_competition/.venv/bin/python \
  src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py
```

## 仿真器常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `S10_USE_VIEWER` | `1` | 0 = 无头运行 |
| `S10_MODE` | `remote` | `auto_nav` = 模式 A 自动导航 |
| `S10_MPC_ENABLE` | `0` | `1` 启用 MPC 控制 |
| `S10_MUJOCO_SCENE` | `track` | 场景（S10_track.xml） |
| `S10_LIDAR_BACKEND` | `cpu` | WSL 下勿用 taichi（core dump） |
| `S10_LIDAR_FREQ` | `10` | LiDAR 频率 (Hz) |

完整环境变量表见 `doc/0808.md` §6；手动键盘控制见下文。

## 手动控制（仿真窗口）

- `z`：默认位置 / `c`：RL 控制默认位置
- `w/a/s/d`：前后左右平移 / `q/e`：逆/顺时针旋转
- `Ctrl` + 右键双击 body：跟踪该 body；`Esc`：停止跟踪
- 仿真窗口失焦时可右键选择 "always on top"

## 附加场景

- **new_wp30（原赛道平面版）**：原赛道全部 33 航点 XY 一致、z 全部清零的
  全平地赛道，用于隔离纯弯道动力学测试（替代旧蛇形版）。
  文件 `src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/new_wp30.xml`，
  路径图 [new_wp30_path.png](doc/new_wp30_path.png)，说明见 doc/0808.md §9.1。

## 当前进度与待办

- 模式 B（遥控）：稳定，4.5 m/s 竞速档已调优。
- 模式 A（自动导航）：
  - **巡航突破**：wp0→wp5（含 S 弯）**3.50 m/s 干净完赛**（v196a_r2，
    run_t 10.5s），复测可靠性 ~40%（S 弯 R=0.84m 为翻车主因）。
  - **new_wp30 平面版 33 航点全通**：224.2m 无翻车，avg 2.04 m/s（v193）。
  - **阻塞点 = wp7 连续台阶区**（4~5 级 0.13m 台阶）：可爬 3~4 级，
    末级维持与稳定性收敛中。
- 待办：S 弯可靠性（3.4+ 成功率）、wp7 台阶区收敛、33 航点真赛道全程、
  真机迁移（vel_scale 回退 50、IMU 闭环、Orin 实测）、A\* 全局规划（可选）。

详细实验记录与参数演进见 `doc/0808.md`（§9）与归档 `_archive_20260808/doc/0806.md`
（v1~v197 全记录）。

## 相关文档

- [doc/0808.md](doc/0808.md) —— 工程总文档（环境配置/架构/参数/进度/待办）
- [doc/s10_mpc_deploy.yaml](doc/s10_mpc_deploy.yaml) —— 部署配置
- [doc/requirements.md](doc/requirements.md) —— Ubuntu 22.04（非 WSL）安装要求
- `doc/比赛规则_赛道四_具身未来.md`、`doc/赛道四_具身未来.pdf` —— 官方规则
- `doc/Airy雷达用户手册.pdf`、`doc/hardware spec.pdf` —— 真机硬件资料
- `doc/cruise_3.50_xy_speed.png`、`doc/new_wp30_full_xy_speed.png` —— 最新结果图