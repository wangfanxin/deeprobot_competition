# nr60b 最佳策略备份（2026-08-08）

当前 cruise 模式稳定最优配置，作为后续 MPCC/转弯/频率优化的基准。
git tag: nr60b-best（对应 commit 13678e7 及其后已提交改动）

## 完整参数（环境变量，与真实一致）

```bash
# 导航层
export S10_AUTO_VMAX=5.0          # 导航最大速度指令
export S10_AUTO_VYAW_MAX=3.0      # yaw 角速度上限 (rad/s)
export S10_AUTO_YAW_GAIN=2.5      # pursuit 转向增益
export S10_AUTO_LOOKAHEAD=1.5     # pursuit 前视 (m)
export S10_AUTO_BIGERR_VX=1.8     # err>1.0 时限速
export S10_AUTO_TURN_VX=2.5       # err 0.3~1.0 时限速
export S10_AUTO_ERR_GATE=0.30     # err 触发减速阈值
export S10_AUTO_WP_AIM=0.8        # 接近航点瞄准距离
export S10_CURVE_DECEL_AHEAD=5.0  # 弯道减速前瞻 (m)
export S10_CURVE_SWING_WINDOW=6.0 # S 弯检测窗口 (m)
export S10_CURVE_SWING_VX=4.0     # S 弯组合限速
export S10_GLOBAL_TANGENT_K=0.7   # Catmull-Rom 切线因子（核心：弯道半径 1.36->2.1m）
export S10_AUTO_MAX_ACCEL=6.0
export S10_AUTO_VYAW_SLEW=0.8
export S10_ZONE_MARGIN=0.3

# MPC 层（构建期）
export S10_MPC_VEL_SCALE=56       # 轮速上限 56*0.081=4.54 m/s
export S10_MPC_H_CRUISE=20        # 巡航视界 20 步 * dt0.02 = 0.4s
export S10_MPC_ANG_W=60           # yaw 跟踪奖励（yaml 默认 20）
export S10_MPC_POSE_ROLL_GAIN=0.10  # 压弯 roll 增益
export S10_MPC_POSE_ROLL_MAX=0.35   # 压弯 roll 上限 (rad)
# 采样（与真实一致，不缩减）
# Nsample=2048, Ndiffuse=1, Hsample=20, dt=0.02
```

## 统计结果（11 次干净 MPC 状态）

- 中位 2.76 m/s，均值 ~2.76，峰值 3.16，零卡死/崩溃
- 样本：3.16/3.10/3.07/2.99/2.93/2.59/2.59/2.49/2.45/2.27/2.68
- wp 时间典型：wp1 3.9~4.2s, wp2 7~8.7s, wp3 8.7~11.5s, wp4 10.7~13.5s

## 复现命令

```bash
cd /home/wfx/DR_competition
export JAX_COMPILATION_CACHE_DIR=/home/wfx/.cache/s10_dial_mpc
export S10_MPC_VEL_SCALE=56 S10_MPC_H_CRUISE=20 S10_MPC_ANG_W=60
export S10_MPC_POSE_ROLL_GAIN=0.10 S10_MPC_POSE_ROLL_MAX=0.35
export S10_AUTO_VMAX=5.0 S10_AUTO_VYAW_MAX=3.0 S10_AUTO_YAW_GAIN=2.5   S10_AUTO_LOOKAHEAD=1.5 S10_CURVE_DECEL_AHEAD=5.0   S10_CURVE_SWING_WINDOW=6.0 S10_CURVE_SWING_VX=4.0   S10_AUTO_BIGERR_VX=1.8 S10_AUTO_TURN_VX=2.5   S10_MPC_ANG_W=60 S10_GLOBAL_TANGENT_K=0.7
timeout 300 ./.venv/bin/python deeprobot_competition/src/S10_sdk_deploy/scripts/cruise_noros.py
```

## 理论参考

- 切线因子 0.7 后保守理论均速 3.43 m/s（含 wp5 后横脊限速），理想版 3.89
- 实际 2.76 = 理论 80% 执行率；主要损失：起步段 err 门控、弯道采样质量
