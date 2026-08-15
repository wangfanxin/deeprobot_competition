# RL-Stair 迁移达标方案（真实赛道部署成功率 ≥95%）

> 交接文档：供主 session 直接执行。目标：比赛仿真（C++ MuJoCo `S10_track.xml`）wp6→7 部署成功率 ≥95%（多 seed 实测）。

## 0. 现状基线
- MJX 训练 T3（比赛六级）0.72 可靠；sim2sim 旧策略（0.30）真实赛道只爬 **1/6 级**。
- 迁移差根因：训练=胶囊轮/box 台阶（MJX），部署=圆柱轮/mesh 台阶（C++）→ 接触/摩擦差异。

## 1. 阶段 0：建立真实基线（0.5h）
1. 导出当前 `model_latest.pt` → `rl_stair/deploy/policy.pt`。
2. `sim2sim.py` 跑 **≥10 个随机 seed**（`--steps 1500`），统计 `risers_crossed=6/6` 比例 → 当前真实部署成功率。
3. 记录为基线（预期 <50%）。

## 2. 阶段 1：域随机化 DR（核心，低成本，4-8h）
在训练 env 加随机化（`rl_stair/envs/s10_env.py` 的 cfg + `_pd`/reset）：

| 参数 | 随机范围 | 位置 |
|---|---|---|
| 地面/台阶摩擦 | 0.5–1.2 | `terrain.py` 每 episode 采样 |
| 机身质量 | ±12% | env 构造时改 base body mass |
| COM 偏移 | ±2cm（x/y/z） | body_ipos |
| kp_leg / kd_leg | 50±20% / 1±20% | `_pd` 每 episode 采样 |
| kp_wheel | 2.0±20% | `_pd` |
| 轮扭矩限 | 13.5±20% | `_pd` clip |
| 观测噪声 | angvel ±0.05→0.1、joint ±0.01→0.02 | `train.py` obs_noise |
| 随机推扰 | 每 750 步 1.0 m/s 脉冲 | env.step（legged_gym `push_robots` 原文） |

实现要点：
- 用 `jax.random` 每 episode 采样一组物理参数，`data.replace` 生效。
- **每轮 DR 后重跑 sim2sim 回归**（阶段 0 方法），看迁移是否提升。

## 3. 阶段 2：MJX 圆柱-mesh 支持核查（0.5-1h）
- 在 MJX 里试 `S10_track.xml` 的轮（cylinder geom）直接跑一段，确认 MJX 是否支持圆柱-mesh 接触。
- **若支持**：训练地形轮改回圆柱（去掉胶囊/接触 hack），差距大减。
- 若不支持：保持胶囊，靠 DR 覆盖。

## 4. 阶段 3：物理标定对齐（1-2h）
- 用同一场景（站立/轮驱/单级 12.5cm 台阶）在 MJX 与 CPU 各跑，量测：轮速-位移曲线、爬阶时间/成功率。
- 调 MJX 摩擦/接触阻尼（`mujoco.mjx` options）直到行为一致。
- 标定数据写进 `doc/` 作记录。

## 5. 阶段 4（条件触发）：C++ MuJoCo 直接训练（1-2 天）
**若阶段 1-3 后 sim2sim 成功率仍 <60%**，转此路径：
- 训练后端 MJX → C++ MuJoCo 批处理（S10 仅 16 dof，256 env CPU 可行）。
- 直接用 `S10_track.xml` 的几何/接触训练（迁移差归零，这是唯一能保证 95% 的路径）。
- `rl_stair/envs/s10_env.py` 的 step 改 C++ 批处理，obs/reward/课程逻辑复用。
- 重新跑完整课程（T0→T6，门槛 0.6+）。

## 6. 阶段 5：95% 验收（每阶段都做）
- **验收标准**：sim2sim ≥10 seed，`risers_crossed=6/6` 且 `fell=False` 比例 **≥95%**。
- 达标后再做 wp0-33 集成（`S10_VMC_MODE=rlstair` + lidar ctx 转换，见文档 §3.30）。
- 每轮改动 → sim2sim 回归 → 记录结果。

## 7. 决策门槛（避免无限循环）
| 节点 | 判定 |
|---|---|
| 阶段 1 后 | sim2sim ≥60% → 继续 DR 细化至 95%；<60% → 转阶段 4 |
| 阶段 4 后 | C++ 训练直接以 95% 为门槛 |

## 8. 时间
- 阶段 0-3：约 1 天。
- 阶段 4（如需）：+1-2 天。
- **关键提示**：只有阶段 4（C++ 直接训练）能"保证"95%，DR 只能"提高概率"。
