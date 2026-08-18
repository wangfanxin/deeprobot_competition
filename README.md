# S10 巡逻赛题 · 感知-控制工程仓库

基于山猫 S10 四足轮式机器人的**巡逻赛题**参赛方案。当前主线（2026-08-18，代码对齐版）：
直线航点路径 + **SMppi/TMppi 巡航** + **CarVMC 轮足执行** + **RL-Stair 爬梯** +
**TK1/TK2 楼梯交接**，在 MuJoCo 官方 `S10_track.xml` 中完成 33 航点全程巡检。

> 当前主线方案见 [doc/SMppi_TMppi_CarVMC_方案_20260818.md](doc/SMppi_TMppi_CarVMC_方案_20260818.md)。
> 旧双管线文档、v890 MPPI 配置与 `dial-mpc/` 属历史/遗留，不参与当前主链路。

## 赛题与计分

- 官方仿真环境与路线：
  - 场景：`src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10_track.xml`
  - 航点：`.../mjcf/track_overlay.xml`，33 个 `track_waypoint_000_start` ~
    `track_waypoint_032_end`；base 进入 wp0 的 0.2 m 水平半径开始计时，逐点推进，
    到 wp32 停止计时。
- 计分：总成绩 = 完成时间 ÷ 模式系数，得分越低排名越靠前；须完成全部定位点。
  模式系数以官方 PDF 为准：**遥控 ÷1.0 / 自主跟随 ÷1.3 / 自主导航 ÷1.4**。
- 力矩硬约束：运行中不得超过模型允许的最大力矩；短时峰值允许，但**连续超限
  >0.5 s 判定不合格**。仓库内验收口径：腿 |τ|≤48 Nm、轮 |τ|≤13.5 Nm（官方阈值
  以组委会技术说明为准）。
- 申报模式：**自主导航（÷1.4）**——AutoNav 航点路径 + SMppi/TMppi 巡航 +
  CarVMC + RL-Stair + TK1/TK2。

## 当前主链路（2026-08-18，代码对齐）

```mermaid
graph LR
    M["MuJoCo 200Hz<br/>S10_track.xml + track_overlay.xml"] --> P["LidarTerrainV2<br/>地形/墙栅格 4Hz 累计"]
    P -->|"local tile / wall"| N
    P -->|"local tile"| SM["stair_mode 20Hz<br/>CRUISE/STAIR 判定"]["nav_waypoint 20Hz<br/>只输出原始航点直线段"]
    N -->|"[vx, vyaw]"| K["控制器选择 20Hz<br/>TK1/TK2 修正 + TMppi / SMppi"]
    K -->|"CRUISE [vx,omega]"| C["CarVMC 200Hz<br/>16 维力矩"]
    K -->|"STAIR 交接"| R["RLStairCtrl<br/>policy 50Hz + 腿PD/轮速 200Hz"]
    C --> M
    R --> M
```

| | 巡航链路 | 楼梯链路 |
|---|---|---|
| 用途 | 航点间直线巡航、横脊、平台 | wp6→7 六级楼梯 + 交接 |
| 控制 | 直线航点 → TK1/TK2 → TMppi / SMppi → CarVMC | stair_mode/TK1 → PRETRANS → RL policy → 腿 PD + 轮速；TK2 后恢复 |
| 频率 | nav/MPPI 20Hz，CarVMC 200Hz | lidar 4Hz，nav 20Hz，policy 50Hz，PD 200Hz |
| 关键配置 | `VMC_MPPI_N=512/H=20`（1.0s 视界），`S10_VMC_OM_CAP=2.0` | `S10_STAIR_WHEEL_CLEAR=0.05`，PRETRANS 位姿预交接 |
| 状态 | SMppi/TMppi、TK1/TK2 已落地；待验证 wp1→wp2 | RL 官方环境独立验收 96.7%；全链 wp0-33 待跑 |

## 完整主链路状态机

```text
nav_waypoint.line -> 当前直线 heading/dist_to_wp
stair_mode.update -> CRUISE / STAIR

CRUISE:
  TK1 检测到楼梯 -> 减速+对准 -> 满足门控后切 STAIR
  TK2 刚从 STAIR 回来 -> 对准下一航点 -> |yaw_err|<=0.15 后释放
  TMppi: 距 wp<0.2m 且 速度<0.2m/s 且 下一段方向误差>10°
  SMppi: 其余所有 CRUISE

STAIR:
  RL-Stair
  四轮 >= 最高 riser top - 0.05 -> CRUISE + TK2
```

## 方案要点

### 1. 路径规划与导航（`nav_waypoint.py`）

- `S10_GLOBAL_FILLET_R=0`：相邻航点直线段，不做 biarc 圆角。
- nav 层只保留 `wp` 数组并输出 `start/end/heading/length/dist_to_wp`。
- nav 层只输出原始航点直线段（`start/end/heading/dist_to_wp`），不做
  `[vx,vyaw]`、曲率 vlim、CTE、CRUISE/STAIR 判定。
- 主循环由直线 heading 误差和到 wp 距离生成简单 `[vx,vyaw]`，交给 SMppi/TMppi。

### 2. 巡航控制器选择（`scripts/cruise_vmc_noros.py`）

按优先级实际生效的链路是：

```text
TK1 / TK2 对 nav [vx, vyaw] 的修正
  → 近 wp 且实际速度 <0.2：TMppi（yaw>10° 时）
  → 其余：SMppi（BodyMPPI）
  → omega 终限 S10_VMC_OM_CAP=2.0
```

### 3. SMppi（`s10_mpc/body_mppi.py`）

- 6 状态模型 `[x,y,yaw,vx,vy,ω]`；`N=512, H=20, dt=0.05` → 1.0s 视界。
- 采样中心 `[v_ref, guide_om=nav vyaw]`；rollout 含摩擦锥 `|vx·ω|≤μg/v`、
  CarVMC 能力表和 `S10_MPPI_A_MAX=2.0` 加速度约束。
- 成本：`2.0·路径距离 + 0.8·速度误差 + 0.5·guide误差 + 0.05·平滑`
  （`S10_MPPI_W_HEAD=0`）。
- 输出：vx 受 `v_ref` 与加速度 slew 约束；ω 受能力表与 slew 约束。

### 4. TMppi（航点低速转向）

- 触发：距当前 wp `<0.2m` 且世界速度范数 `<0.2m/s`，且 `S10_TURN_SPLIT=1`。
- 动作：`vx≤0.2`，`ω=clip(3.0·yaw_err, ±2.0)`；`|yaw_err|≤10°` 交回 SMppi。
- 已知风险：判点半径与 TMppi 触发半径同为 0.2m，快速进点可能来不及触发
  （见主线方案 §12）。

### 5. CarVMC（`s10_mpc/vmc_legs.py`）

- 巡航执行器：轮驱动 + 差速 yaw 反馈 + 腿主动悬架（半蹲
  `S10_CAR_SQUAT=1`，hipy∓1.10 / knee1.90）。
- 腿：`mg/4 + roll/pitch 姿态分配 + 地形阻抗`；地形阻抗
  `S10_VMC_KPH=300 / KDH=60`。
- 轮：速度 PID（`wheel_k=4.0 / d=0.08`）+ yaw 反馈 60 + 摩擦前馈；
  直线轮矩上限 13.5Nm，弯道/近脊收敛到 μN·r。
- 压弯：`roll_tar=-0.06·ω·|vx|`（clip ±0.06）。
- 最终 omega 上限 `S10_VMC_OM_CAP=2.0`；`S10_AUTO_LAT_MAX=5.0` 侧向包线。

### 6. 楼梯感知 / TK1 / RL-Stair / TK2

```text
LidarTerrainV2（96×48 地形射线 + 61×13 wall 射线，mount 抬高 0.6m，4Hz 累计）
  → elev_tile 局部栅格（step_flag）
  → update_mode：前方 ≥2 级 riser、跨度 ≤3m、总爬升 ≥0.4m → STAIR
  → TK1：riser 在 5m 内减速到 2.0m/s，yaw 对准爬升方向
  → 距首级 riser <2m 且 |yaw_err|≤0.20 且 vx<2.0 → 交付 RL
  → RL：policy.pt 55 维观测 → 16 动作；腿 PD 50/1/clip48，轮速 kp2/vel24/clip13.5
  → 退出：四轮高于最高 riser - 0.05 → CRUISE
  → TK2：四轮全上最后一级台阶后立即对准下一航点，然后交回 SMppi/TMppi
```

- TK1 heading 用双检测器：terrain 梯度（宽单级台阶）与 on-path wall 垂直面
  （六级楼梯）。
- RL 预交接 `S10_PRETRANS=1`：楼梯前按 lidar 检测的首级 riser 距离
  3.0m→2.0m 把 CarVMC 半蹲姿态过渡到 RL 高站姿；楼梯后按 handback 点起的前进
  距离 2.0m 平滑交还 CarVMC；`S10_RL_WARMUP=0`。
- 55 维观测：`angvel*0.25 | gravity | cmd | leg_err | leg_vel*0.05 | last_action |
  heading[cos,sin] | terrain_ctx(4) | rough`；heading 目标由 TK1 设为 riser 方向。
- 注意：riser 表完全来自 lidar 在线检测，无已知地图硬编码表；检测精度列为待验证项。

### 7. 避障与已知地图

- 避障 costmap / `S10_MPPI_OBSTACLE` 已完全删除。
- lidar 高程图保留，只给 TK1/TK2/RL-Stair 使用。
- god-view ray、mj_ray 预扫描、硬编码 `STAIR_RISERS/TOPS` 已删除。

### RL 训练 / 评估 / 导出 / sim2sim

```bash
# 仓库根目录
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 ~/DR_competition/.venv/bin/python \
  rl_stair/train.py --num_envs 1024 --max_iters 3000 --logdir rl_stair/logs
~/DR_competition/.venv/bin/python rl_stair/eval.py \
  --ckpt rl_stair/logs/model_latest.pt --stages T3_stairs6,T6_handoff
~/DR_competition/.venv/bin/python rl_stair/export.py \
  --ckpt rl_stair/logs/model_latest.pt --out rl_stair/deploy/policy.pt
~/DR_competition/.venv/bin/python rl_stair/sim2sim_exact.py \
  --ckpt rl_stair/logs/model_latest.pt --seeds 20
```

## 目录结构

```
DR_competition/
├── .venv/                        # 项目虚拟环境（开发机，Python 3.12）
├── comp_env/                     # 官方比赛环境专用 venv（numpy<2 + mujoco）
├── deeprobot_competition/        # 本仓库
│   ├── run_dialmpc_stair_wp033.sh # ★ 当前主链路入口（无 ROS，wp0→33）
│   ├── src/S10_sdk_deploy/
│   │   ├── interface/robot/simulation/   # ROS2 仿真节点（模式 A/B 入口，备用）
│   │   ├── perception/                   # 感知：local_map / elevation_lookup / points_to_heightmap
│   │   ├── s10_mpc/
│   │   │   ├── auto_nav.py               # AutoNavFollower：路径/速度剖面/CRUISE-STAIR 模式
│   │   │   ├── body_mppi.py              # BodyMPPI（SMppi 层，20Hz）
│   │   │   ├── vmc_legs.py               # CarVMC / FootPlaceVMC / VMCController（执行层，200Hz）
│   │   │   ├── lidar_terrain_v2.py       # 高程图 + 墙通道 + riser 检测
│   │   │   └── stair_*.py / mpc_controller.py / mppi_controller.py  # 历史遗留
│   │   ├── scripts/
│   │   │   ├── cruise_vmc_noros.py       # ★ 当前主循环（SMppi/TMppi + CarVMC + RL-Stair）
│   │   │   └── stair_dial_noros.py / cruise_test.py  # 历史遗留
│   │   └── S10_description/s10_mjcf/mjcf/ # S10_track.xml / track_overlay.xml / new_wp30.xml
│   ├── rl_stair/                 # ★ RL 爬梯训练与部署
│   │   ├── train.py / ppo.py / eval.py / export.py
│   │   ├── configs/rl_stair_config.py    # T0-T6 课程 + PPO 配置
│   │   ├── envs/s10_env.py terrain.py    # MJX 环境与地形
│   │   ├── deploy/rlstair_ctrl.py        # 部署控制器（policy→腿 PD + 轮速）
│   │   ├── deploy/obs_np.py              # 55 维观测编码（与训练一致）
│   │   └── sim2sim.py / sim2sim_exact.py
│   ├── dial-mpc/                 # 遗留采样 MPC 库（不参与当前主链路）
│   ├── doc/                      # 方案 / 规则 / RL 文档
│   └── tmp/                      # 测试入口与结果分析脚本
```

## 环境与快速开始

要求：**Ubuntu 24.04 + ROS 2 Jazzy + NVIDIA GPU**。完整环境配置见
[doc/requirements.md](doc/requirements.md)。

```bash
# 1) 当前主链路：直接运行（无 ROS）
cd ~/DR_competition/0810new/deeprobot_competition
bash run_dialmpc_stair_wp033.sh
# 等价于：
#   export ...（见脚本全部 env）
#   /home/wfx/DR_competition/.venv/bin/python src/S10_sdk_deploy/scripts/cruise_vmc_noros.py

# 2) ROS2 构建（仅备用接口需要）
cd ~/DR_competition
source /opt/ros/jazzy/setup.bash
cd deeprobot_competition
colcon build --packages-up-to s10_sdk_deploy --cmake-args -DBUILD_PLATFORM=x86
source install/setup.bash

# 3) 模式 B 遥控（备用；z 站起 / c 进 MPC / wasd 移动 / qe 转向）
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/s10_dial_mpc
S10_MPC_ENABLE=1 ~/DR_competition/.venv/bin/python \
  src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py

# 4) 模式 A 自动导航（备用 ROS 入口；无头加 S10_USE_VIEWER=0）
S10_MPC_ENABLE=1 S10_MODE=auto_nav ~/DR_competition/.venv/bin/python \
  src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py
```

> 当前 run 脚本使用开发 `.venv`（numpy 2.2 + mujoco 3.11）。官方比赛环境
> `comp_env` 为 numpy 1.26 + mujoco 3.11，如需切换请修改 run 脚本的 Python 路径。

## 关键参数（当前 run_dialmpc_stair_wp033.sh）

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `S10_VMC_MODE` | `rlstair` | CRUISE=CarVMC，STAIR=RLStairCtrl |
| `S10_RL_ELEV` | `1` | 启用 lidar 高程图 STAIR 入口/出口 |
| `S10_NAV_HZ` | `20` | AutoNav / SMppi / TMppi 同拍频率 |
| `S10_GLOBAL_FILLET_R` | `0` | 航点直线段（关闭 biarc） |
| `S10_WP_ARRIVE_R` / `S10_WP_ADVANCE_DIST` | `0.2 / 0.2` | TMppi 触发半径 / 判点半径 |
| `S10_TURN_SPLIT / K / OM_MAX / ERR_DEG` | `1 / 3.0 / 2.0 / 10` | TMppi 转向 |
| `VMC_MPPI_N / H` | `512 / 20` | SMppi 样本数 / 视界步数（1.0s） |
| `S10_MPPI_ADA / A_MAX / OMAX` | `1 / 2.0 / 2.5` | 自适应 sigma / 加速度 / 转向上限 |
| `S10_VMC_OM_CAP` | `2.0` | 最终 omega 上限（2026-08-18 修正） |
| `S10_TK1 / S10_TK2` | `1 / 1` | 楼梯前减速对准 / 楼梯后恢复对准 |
| `S10_TK1_LOOKAHEAD / VX / YAW_DB` | `5.0 / 2.0 / 0.20` | TK1 检测距离 / 速度门控 / 航向门控 |
| `S10_STAIR_ENTER_DIST` / `S10_STAIR_WHEEL_CLEAR` | `2.0 / 0.05` | RL 交付距离 / 四轮越顶容差 |
| `S10_PRETRANS_ENTER_DIST / BLEND_LEN / HOLD_DIST / EXIT_LEN` | `2.0 / 1.0 / 2.0 / 2.0` | 按 riser 距离 / handback 距离切换站姿 |
| `S10_RL_POLICY` | `policy.pt`（新文件夹） | RL 策略路径覆盖 |

## 当前进度与待办（2026-08-18）

- **已落地**：SMppi/TMppi 分离；TK1/TK2 门控；`S10_VMC_OM_CAP=2.0` 修正；
  `S10_WP_ADVANCE_DIST=0.2`；RL 退出改为四轮越顶；PRETRANS 预交接；
  避障 costmap 与已知地图硬编码已删除。
- **独立能力**：RL-Stair 官方环境 96.7%（29/30）；巡航 v890 历史成绩
  wp0→4≈13.5s / wp0→6≈30.5s（旧 MPPI 配置，勿作为当前主链结果）。
- **当前卡点/待验证**：wp1→wp2（快速进点可能使 TMppi 来不及触发）；wp0→33
  全程；lidar riser 表实测校验；真机迁移；初赛材料（8.20）。
- 当前实现：全局 lidar 地形已生效；`VMC_TRAJ` 用于轨迹导出；新主链路模块在
  `SMppi_TMppi_Cruise_RL-Stair_TK1_TK2/`，运行前先由用户确认。

## 相关文档

- **[doc/SMppi_TMppi_CarVMC_方案_20260818.md](doc/SMppi_TMppi_CarVMC_方案_20260818.md) —— 当前主线方案（2026-08-18）**
- [doc/路径规划与巡航楼梯方案_20260818.md](doc/路径规划与巡航楼梯方案_20260818.md) —— wp0-33 直线航点/航点转向设计
- [doc/RL_stair_方案_20260814.md](doc/RL_stair_方案_20260814.md)、[最终验收](doc/RL_stair_最终验收_20260815.md)、[迁移达标](doc/RL_stair_迁移达标方案_95percent.md) —— RL-Stair
- [doc/requirements.md](doc/requirements.md) —— 环境安装；[doc/s10_mpc_deploy.yaml](doc/s10_mpc_deploy.yaml) —— 部署配置
- `doc/比赛规则_赛道四_具身未来.md` / PDF —— 官方规则与计分；`doc/Airy雷达用户手册.pdf` / `hardware spec.pdf` —— 硬件资料
- 历史（未删除，仅备查）：`doc/双数据管线_autonav_20260816.md`、`doc/carvmc_方案与数据管线_20260810.md`、`doc/组合技能_交付总结_20260816.md`
