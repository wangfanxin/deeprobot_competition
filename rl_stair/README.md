# rl_stair — S10 轮足狗 RL-Stair 技能（MJX 训练栈，定稿）

> 状态：已定稿——训练、导出、部署控制器、官方环境验收（96.7%）、wp0→33 主链路集成全部完成。
> 完整方案见 `doc/RL_stair_方案_20260814.md`。

## 快速命令（仓库根目录，WSL）
```bash
# 训练（256 env，课程 T0→T6）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 python rl_stair/train.py --num_envs 256 \
    --max_iters 30000 --steps_per_env 24 --logdir rl_stair/logs
# 断点续训
python rl_stair/train.py ... --resume rl_stair/logs/model_latest.pt
# 评估 checkpoint（只读）
python rl_stair/eval.py --ckpt rl_stair/logs/model_latest.pt \
    --stages T3_stairs6,T6_handoff --num_envs 128 --episodes 5
# 导出 actor → TorchScript(.pt)
python rl_stair/export.py --ckpt rl_stair/logs/model_latest.pt --out rl_stair/deploy/policy.pt
# 比赛赛道 wp6→7 验证（CPU mujoco）
python rl_stair/sim2sim.py --ckpt rl_stair/deploy/policy.pt
# watchdog（每 2 分钟巡检资源与进程健康）
python rl_stair/watchdog.py
```

## 结构
| 文件 | 说明 |
| --- | --- |
| `envs/s10_env.py` | 函数式 MJX 环境（55 obs / 72 priv / 16 act；PD；交接态热启动；自动重置） |
| `envs/terrain.py` | 地形生成（比赛几何 0.061+0.125×5、tread 0.4） |
| `configs/rl_stair_config.py` | T0-T6 课程 + PPO 配置 |
| `ppo.py` | PPO 非对称 actor-critic（rsl_rl 风格） |
| `train.py / eval.py / export.py / sim2sim.py / watchdog.py` | 训练/评估/导出/赛道验证/巡检 |
| `deploy/obs_np.py` | 部署侧 55 维观测编码（与 MJX 逐位一致，diff 3.7e-9） |
| `deploy/rlstair_ctrl.py` | 部署控制器（policy.pt + 腿 PD + 轮速闭环） |

## 设计要点
- 奖励：velocity-tracking 主奖励（lin 4.0 / ang 2.0 / σ0.25，only_positive_rewards）+ 位置制 +
  4 项爬梯专项 shaping（见 `doc/RL_stair_奖励增强_4项_20260815.md`）。
- 非对称 AC：critic 特权含真实接触力（×0.01、clip ±5）；actor 纯 proprio + terrain_ctx。
- 观测布局（55 维，部署端逐位复刻）：
  `angvel*0.25(3) | gravity(3) | cmd vx/yaw(2) | leg_err(12) | leg_vel*0.05(12) | last_action(16) |
  heading[cos,sin](2) | terrain_ctx(4) | rough(1)`
- heading 目标 = `S10_RL_HEADING`，TK1 交接时由 `RLStairCtrl.set_heading()` 设为楼梯爬升方向。