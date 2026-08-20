# RL-Stair 迁移达标方案（定稿）

> 目标：比赛仿真（C++ MuJoCo S10_track.xml）wp6→7 部署成功率 ≥95%（多 seed）。**已达标：官方环境 29/30（96.7%）**。

## 执行路径（已全部完成）
1. 基线：sim2sim 多 seed 统计。
2. DR 域随机化：地面/台阶摩擦 0.5–1.2、质量 ±12%、COM ±2cm、kp_leg/kd_leg ±20%、kp_wheel ±20%、
   轮扭矩限 ±20%、观测噪声加倍、每 750 步 1.0 m/s 推扰。
3. MJX 圆柱-mesh 支持核查。
4. 物理标定对齐：同一场景 MJX↔CPU 轮速-位移、爬阶时间/成功率比对
   （平地轮驱 MJX 12.16 vs C++ 11.90 rad/s，差 2.2%）。
5. 验收（每阶段）：sim2sim ≥10 seed，risers_crossed=6/6 且 fell=False 比例 ≥95%。
6. 达标后接入 wp0→33 集成（S10_VMC_MODE=rlstair + lidar ctx 转换）。

## 验收结果
- 官方比赛环境（30 seed，spawn_y=34.0、vx=1.5）：**reach=29/30（96.7%）**，wp6→7 平均 7.96s。
- 轨迹图：`doc/figures/wp67_traj_officialenv_96pct.png`。