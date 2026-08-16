# S10 巡逻赛题 · 感知-控制工程仓库

基于山猫 S10 四足轮式机器人的**巡逻赛题**参赛方案：LiDAR 感知 + 高程地形建模 +
导航（平滑路径/速度剖面）→ **CarVMC 车化控制**（轮驱动/差速 + 腿=主动悬架）的
层级方案，其中巡航与爬梯由 **CarVMC 车化巡航 + RL-Stair 爬梯**双技能实现，
在 MuJoCo 仿真中完成 33 航点全程巡检（历史 dial-mpc 方案已归档）。

## 赛题与计分

- 仿真环境（官方提供）：`S10_track.xml` 场景 + `track_overlay.xml` 33 个航点
  （000_start ~ 032_end），base 进入 wp0 的 0.2 m 水平半径开始计时，逐点推进，
  到达终点停止计时并打印耗时。
- 计分：总成绩 = 完成时间 ÷ 模式系数，得分越低排名越靠前，30 个定位点须全部
  完成。模式系数以官方 PDF 为准：**遥控 ÷1.0 / 自主跟随 ÷1.3 / 自主导航 ÷1.4**。
- 申报模式：**自主跟随（÷1.3）**——航点跟随 + 感知地形限速/爬坡 + roll 安全，
  不强制全局 A\*。

## 系统架构（2026-08-16）

```mermaid
graph LR
    S["mujoco (S10_track.xml)"] -->|"200Hz"| P["mujoco-lidar → LidarTerrain 高程图 (10Hz)"]
    P -->|"高程/riser"| N["AutoNavFollower (20Hz): 路径+速度剖面+判点<br/>CRUISE⇄STAIR 切换"]
    N -->|"[vx,ω]"| C["CRUISE: CarVMC (200Hz)"]
    N -->|"STAIR 交接"| R["RL-Stair: rlstair_ctrl (200Hz)"]
    T["rl_stair/ MJX PPO 训练 T1-T6"] -->|"policy.pt"| R
    C -->|"tau"| S
    R -->|"tau"| S
```

- **感知**：mujoco-lidar 扇形射线（前下 45°）→ `LidarTerrain` 世界栅格累积
  高程图（10Hz，瓦片 res 0.05、riser 检测、运动学 fallback）；楼梯区以
  已知 riser 表提前触发 STAIR。
- **巡航**：CarVMC（200Hz）轮驱动/差速 + 腿=主动悬架，连续地形响应；
  平滑路径 + 曲率/横脊限速速度剖面（20Hz）。
- **爬梯（RL）**：`rl_stair/` MJX 并行 PPO（T1-T6 课程 + 域随机化 DR），
  策略导出 `deploy/policy.pt` → `rlstair_ctrl.py`（腿 PD + 轮速）部署到
  C++ MuJoCo 真实赛道；交接流程：cruise 带速接近 → carvmc+PD 抬身 →
  RL 接管爬梯 → 完成后平滑回巡航。

## 目录结构

```
DR_competition/
├── .venv/                       # 项目虚拟环境（开发机）
├── comp_env/                    # 官方比赛环境专用 venv（numpy<2 + mujoco）
├── deeprobot_competition/       # 本仓库：ROS2 工作空间
│   ├── src/S10_sdk_deploy/      # 仿真节点/感知/导航/控制器/模型
│   ├── rl_stair/                # RL 爬梯训练（MJX PPO + sim2sim 部署）
│   ├── doc/                     # v890 cruise + RL_stair 文档 + 官方材料
│   └── tmp/                     # 测试入口与结果分析脚本
```

## 环境与快速开始

要求：**Ubuntu 24.04 + ROS 2 Jazzy + NVIDIA GPU**。完整环境配置见
[doc/requirements.md](doc/requirements.md)（Ubuntu 22.04 安装）。

```bash
# 1) 虚拟环境 + ROS2 构建
cd ~/DR_competition
/usr/bin/python3 -m venv .venv
./.venv/bin/pip install "numpy<2" mujoco mujoco-lidar jax[cuda12]==0.4.38
cd deeprobot_competition && source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to s10_sdk_deploy --cmake-args -DBUILD_PLATFORM=x86
source install/setup.bash

# 2) 模式 B 遥控（z 站起 / c 进 MPC / wasd 移动 / qe 转向）
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/s10_dial_mpc
S10_MPC_ENABLE=1 ~/DR_competition/.venv/bin/python \
  src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py

# 3) 模式 A 自动导航（无头加 S10_USE_VIEWER=0）
S10_MPC_ENABLE=1 S10_MODE=auto_nav ~/DR_competition/.venv/bin/python \
  src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py
```

RL 训练 / 评估 / 导出 / sim2sim：

```bash
cd ~/DR_competition/deeprobot_competition
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 ~/DR_competition/.venv/bin/python \
  rl_stair/train.py --num_envs 256 --max_iters 30000 --logdir rl_stair/logs
~/DR_competition/.venv/bin/python rl_stair/eval.py \
  --ckpt rl_stair/logs/model_latest.pt --stages T3_stairs6,T6_handoff
~/DR_competition/.venv/bin/python rl_stair/export.py \
  --ckpt rl_stair/logs/model_latest.pt --out rl_stair/deploy/policy.pt
~/DR_competition/.venv/bin/python rl_stair/sim2sim.py --ckpt rl_stair/deploy/policy.pt
```

> 官方比赛环境（真实 mesh S10_track.xml）使用专用 venv `comp_env`；
> 部署验收走 `rl_stair/sim2sim_exact.py` 与集成流 `cruise_vmc_noros.py`
> （`S10_VMC_MODE=rlstair` + `S10_RL_ELEV=1`）。

## 关键参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `S10_USE_VIEWER` | `1` | 0 = 无头运行 |
| `S10_MODE` | `remote` | `auto_nav` = 模式 A 自动导航 |
| `S10_MPC_ENABLE` | `0` | `1` 启用 MPC 控制 |
| `S10_LIDAR_BACKEND` | `cpu` | WSL 下勿用 taichi（core dump） |
| `S10_VMC_MODE` | `wbc` | `rlstair` = RL 爬梯模式 |
| `S10_RL_POLICY` | `deploy/policy.pt` | RL 策略路径覆盖（A/B 评估） |
| `S10_PRETRANS_Y0 / EXIT_Y0` | `32.0 / 40.5` | RL 交接进入/退出位置 |

## 手动控制（仿真窗口）

- `z`：默认位置 / `c`：RL 控制默认位置
- `w/a/s/d`：前后左右平移 / `q/e`：逆/顺时针旋转
- `Ctrl` + 右键双击 body：跟踪该 body；`Esc`：停止跟踪

## 当前进度与待办（2026-08-16）

- **CarVMC 巡航（v890 稳定）**：wp0→4 ≈13.5s、wp0→6 30.5s；
  wp0→33 分段通过 18 点，卡点 = 坡底脊区 / wp17 大弯 / wp4→5 发卡+横脊。
- **RL-Stair**：MJX PPO T1-T6 训练中；box 地形 12/12 直立爬完 6 级；
  交接根因链已修复（riser 表 / PD 腿覆盖过渡 / STAIR 提前触发）。
  **核心瓶颈 = sim2sim 迁移（MJX box → 真实 mesh）**：真实 mesh 下策略
  趴地爬行（z 0.27~0.44 vs 直立 0.84~0.98），早期"96.7% 官方验收"为
  平地/趴地假象已更正（图 `doc/figures/box_vs_real_z.png`）。
- 待办：① 真实 mesh 直立爬迁移（C++ 训练 env 重构 或 更强 DR + real-mesh
  eval 闭环）；② wp6→7 全流程连续成功；③ 33 航点全程；④ 真机迁移；
  ⑤ 初赛材料（8.20 技术方案 PDF + Demo + GitHub 链接）。

## 相关文档

- **[doc/carvmc_方案与数据管线_20260810.md](doc/carvmc_方案与数据管线_20260810.md) —— CarVMC 巡航（v890）方案与数据管线**
- [doc/RL_stair_方案_20260814.md](doc/RL_stair_方案_20260814.md) —— RL-Stair 技能方案（定稿 v3）
- [doc/RL_stair_最终验收_20260815.md](doc/RL_stair_最终验收_20260815.md) —— RL-Stair 最终验收（含假象更正）
- [doc/RL_stair_迁移达标方案_95percent.md](doc/RL_stair_迁移达标方案_95percent.md) —— 迁移达标方案（阶段0-3）
- [doc/RL_stair_go2w_s10_参数审计_20260815.md](doc/RL_stair_go2w_s10_参数审计_20260815.md) —— go2w/S10 参数审计
- [doc/RL_stair_奖励增强_4项_20260815.md](doc/RL_stair_奖励增强_4项_20260815.md) —— 奖励增强 4 项
- [doc/requirements.md](doc/requirements.md) —— 环境安装要求
- [doc/s10_mpc_deploy.yaml](doc/s10_mpc_deploy.yaml) —— 部署配置
- `doc/比赛规则_赛道四_具身未来.md` / `赛道四_具身未来.pdf` —— 官方规则
- `doc/Airy雷达用户手册.pdf` / `Airy User Guide.pdf` / `hardware spec.pdf` —— 真机硬件资料