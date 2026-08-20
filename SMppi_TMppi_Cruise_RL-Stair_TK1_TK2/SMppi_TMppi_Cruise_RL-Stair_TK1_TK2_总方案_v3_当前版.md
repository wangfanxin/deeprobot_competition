# SMppi/TMppi Cruise + RL-Stair + TK1/TK2 总体方案（定稿 2026-08-20）

> 目标：wp0→33 全程通关。验收标准：每两航点间用时 <5s；该段直线距离 >5m 放宽至 <8s；
> 整场 ≤120s；力矩合规（腿 ≤48/50 Nm、轮 ≤13.5/14 Nm、连续超限 ≤0.5s）；全程无侧翻。
> 设计原则：全部策略全局统一——SMppi / TMppi / TK1 / TK2 / RL-Stair 同一套参数覆盖整条赛道，
> 不设任何局部特判。方案已定稿，代码、参数与文档对齐。

## 1. 系统组成

| 模块 | 职责 | 关键点 |
| --- | --- | --- |
| SMppi | 直线巡航 | 采样规划 N=1024、H=40（2s 视界）；终点代价；到点减速剖面 |
| TMppi | 近点转向 | 航点附近原地对准；om=K·err−KD·ω（终端阻尼，停稳不甩） |
| TK1 | 楼梯前交接 | 对准爬升轴、限速交付 RL-Stair |
| TK2 | 楼梯后交接 | 四轮登顶→对准下一航点→交回 SMppi/TMppi |
| RL-Stair | 台阶爬升 | 车载 lidar 在线 riser 表 + 预训练 policy（50Hz 推理） |
| CarVMC | 巡航执行（200Hz） | 轮速 PID+差速偏航+半蹲腿 PD+姿态环+地形跟踪 |

## 2. 控制频率

| 层 | 频率 |
| --- | --- |
| MuJoCo | 200Hz（DT=0.005） |
| 执行 CarVMC / RL-Stair | 200Hz |
| 规划 / 模式 tick | 40Hz |
| RL policy | 50Hz（decimation=4） |
| lidar 高程图 | 4Hz 增量 |
| MPPI 视界 | 2s（H=40×dt=0.05） |

## 3. 数据管线

33 航点 → 直线路径 → 线控制（段航向+横向纠偏；近点直瞄航点；sqrt 刹车剖面）
→ 规划二选一：SMppi（巡航）/ TMppi（近点转向）
→ 楼梯前后由 TK1 / TK2 交接，台阶段由 RL-Stair 接管（riser 表 → policy → 腿 PD + 轮速）
→ cmd{速度,转向,姿态} → CarVMC / RL-Stair → 16 维力矩 → 仿真。
航点推进按半径判点；模式切换时速度、航向、姿态连续（decel 插值 + PRETRANS 站姿预过渡）。

## 4. 感知管线（仅用车载 lidar）

LidarTerrainV2（4Hz）高程图 → 多级 / 单级 riser 检测 → 在线 riser 表 → RL-Stair 实时更新爬升参考。
无世界系 ray_cast、无避障 costmap——感知只由车载 lidar 提供。

## 5. 关键参数（定稿值，与启动脚本一致）

    SMppi:  N=1024 H=40 dt=0.05 CTRL_DT=0.025 A_MAX=8.0 OMAX=2.5 W_GUIDE=2.5 W_DIST=2.0
            W_HEAD=2.0 STOP_DX=3.5 W_TPOS=10 W_TV=10 LAT_MAX=3.6 MU=0.36
    TMppi:  K=3.0 KD=1.5 OM_MAX=1.5 V_MAX=4.5 TURN_VX=0.2
    TK1:    VX=1.0 YAW_K=2.5 YAW_MAX=1.5 LOOKAHEAD=8.0
    TK2:    VX=1.2 YAW_K=2.5 YAW_MAX=1.0
    RL-Stair: RL_VX=2.0 WARMUP=200 ENTER_DIST=2.0 EXIT_VX=1.0 PRETRANS 3.0/1.5/3.0/2.0
    CarVMC: KPH=300 KDH=60 WHEEL_K=12 WHEEL_D=0.02 TERRAIN_LP=0.4 TERRAIN_LOOKAHEAD=0.35
            KP_ROLL=150 KD_ROLL=20 ROLL_K=0.05 ROLL_AMP=0.10 Z_DES_OFFSET=0.26
            YAW_K_WHEEL=80 OM_ABS_MAX=1.6 OM_CAP=1.0 WHEEL_TMAX=13.5 MU=0.8 SQUAT=1
    感知:   ELEV_HZ=4 LOOKAHEAD=8.0 CLIMB_TH=0.2
    判点:   WP_ADVANCE_DIST=0.6

## 6. 各航段与验收标准

| 段 | 直线距离 | 验收 |
| --- | --- | --- |
| wp0 -> wp1 | 13.40 m | < 8s |
| wp1 -> wp2 | 8.02 m | < 8s |
| wp2 -> wp3 | 4.84 m | < 5s |
| wp3 -> wp4 | 4.60 m | < 5s |
| wp4 -> wp5 | 5.94 m | < 8s |
| wp5 -> wp6 | 8.59 m | < 8s |
| wp6 -> wp7 | 9.35 m | < 8s |
| wp7 -> wp8 | 5.85 m | < 8s |
| wp8 -> wp9 | 4.68 m | < 5s |
| wp9 -> wp10 | 16.61 m | < 8s |
| wp10 -> wp11 | 5.43 m | < 8s |
| wp11 -> wp12 | 9.31 m | < 8s |
| wp12 -> wp13 | 6.95 m | < 8s |
| wp13 -> wp14 | 15.65 m | < 8s |
| wp14 -> wp15 | 8.54 m | < 8s |
| wp15 -> wp16 | 5.43 m | < 8s |
| wp16 -> wp17 | 2.49 m | < 5s |
| wp17 -> wp18 | 8.14 m | < 8s |
| wp18 -> wp19 | 8.88 m | < 8s |
| wp19 -> wp20 | 5.57 m | < 8s |
| wp20 -> wp21 | 16.03 m | < 8s |
| wp21 -> wp22 | 7.90 m | < 8s |
| wp22 -> wp23 | 7.45 m | < 8s |
| wp23 -> wp24 | 5.43 m | < 8s |
| wp24 -> wp25 | 1.56 m | < 5s |
| wp25 -> wp26 | 6.21 m | < 8s |
| wp26 -> wp27 | 1.28 m | < 5s |
| wp27 -> wp28 | 5.72 m | < 8s |
| wp28 -> wp29 | 4.24 m | < 5s |
| wp29 -> wp30 | 2.14 m | < 5s |
| wp30 -> wp31 | 4.31 m | < 5s |
| wp31 -> wp32 | 3.71 m | < 5s |

## 7. 验证流程

- 120s 全链连跑（S10_TEST_MAX_SIM=120、S10_AUTO_MAX_WP=33）；
- 逐腿计时对照 §6 验收标准，整场 ≤120s；
- 力矩合规与无侧翻自动检查；
- 轨迹 xy 回放（颜色=速度）存档。
