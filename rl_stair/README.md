# rl_stair — S10 轮足狗 RL-Stair/Ridge 技能（MJX 训练栈）

完整方案见 `doc/RL_stair_方案_20260814.md`（v3）。本目录与 DiAL 代码**严格隔离**，不修改任何现有源文件/比赛 XML。

## 快速命令（仓库根目录，WSL）
```bash
# 训练（256 env，课程 T0→T6，24h 目标）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 python rl_stair/train.py --num_envs 256 \
    --max_iters 30000 --steps_per_env 24 --logdir rl_stair/logs
# 断点续训
... --resume rl_stair/logs/model_latest.pt
# 评估 checkpoint（各关卡 succ/progress/fall 率，只读）
python rl_stair/eval.py --ckpt rl_stair/logs/model_latest.pt \
    --stages T3_stairs6,T5_mixed,T6_handoff --num_envs 128 --episodes 5
# 导出 actor → TorchScript(.pt) / 可选 ONNX（trace forward，勿 trace act()）
python rl_stair/export.py --ckpt rl_stair/logs/model_latest.pt --out rl_stair/deploy/policy.pt
# 比赛赛道 wp6→7 验证（CPU mujoco，不占训练 GPU）
python rl_stair/sim2sim.py --ckpt rl_stair/deploy/policy.pt
# watchdog（每 2 分钟巡检：卡死/GPU 尖峰/磁盘/内存；异常写 logs/watchdog.log）
python rl_stair/watchdog.py
```

## 结构
| 文件 | 说明 |
|---|---|
| `envs/s10_env.py` | 函数式 MJX 环境（55 obs / 72 priv / 16 act；PD 控制；交接态热启动；自动重置；only_positive_rewards + velocity-tracking 奖励） |
| `envs/terrain.py` | 地形生成（flat/single_step/stairs/ridge/mixed；比赛几何 0.061+0.125×5、tread 0.4） |
| `configs/rl_stair_config.py` | T0-T6 课程 + PPO 配置 |
| `ppo.py` | PPO 非对称 actor-critic（rsl_rl 风格） |
| `train.py` | rollout / 课程晋级 / 断点续训 / 资源日志 |
| `eval.py` | checkpoint 评估（只读） |
| `export.py` | TorchScript/ONNX 导出（forward 路径） |
| `deploy/obs_np.py` | 部署侧 53 维观测编码（已验证与 MJX 一致，diff 3.7e-9） |
| `sim2sim.py` | 比赛赛道验证 harness |
| `watchdog.py` | 资源/卡死 watchdog |

## 关键修复（均有文献/源码背书，见方案 §3.7-3.9，可回退）
1. **std 冻结**：`log_std` 被 detach → 探索噪声永不学习（rsl_rl 中 std 是 `nn.Parameter`，可学习）
2. **only_positive_rewards=True**（go2w_rl_gym/legged_gym）：密集奖励 clip≥0，终止惩罚在 clip 后加
3. **velocity-tracking 主奖励**（legged_gym：`exp(-err/tracking_sigma)`，4.0/2.0/σ0.25）

## 观测布局（55 维，部署端必须逐位复刻）
`angvel*0.25(3) | gravity(3) | cmd vx/yaw(2) | leg_err(12) | leg_vel*0.05(12) | last_action(16) | heading[cos,sin](2) | terrain_ctx(4) | rough(1)`

heading 目标 = `S10_RL_HEADING`（默认 pi/2），TK1 交接时由 `RLStairCtrl.set_heading()` 设为楼梯爬升方向。

## 待批准改进（有源码背书，未落地，等用户确认）
1. priv 真接触力（legged_gym net_contact_forces）
2. 3 个速度惩罚项（go2w_rl_gym scales：lin_vel_z/ang_vel_xy/dof_vel）
3. push 扰动（legged_gym push_robots：interval 15s、max 1.0 m/s）
