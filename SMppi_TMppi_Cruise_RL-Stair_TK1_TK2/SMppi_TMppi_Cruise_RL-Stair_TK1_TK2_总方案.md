# SMppi/TMppi Cruise + RL-Stair + TK1/TK2 总方案（2026-08-18，终版）

> 本文档只描述主链路允许的技能：
> **SMppi、TMppi、CarVMC（Cruise）、RL-Stair、TK1、TK2**。
> nav_waypoint、stair_mode 与 Lidar 高程图是支撑模块，不属于额外技能。
> direct-nav、CRUISE_TK、POST-STAIR AIM、STEP_HOMING、W45_PULL、避障 costmap、
> god-view ray / 已知地图预扫描均已从主链路删除或禁用。

主链路脚本：`src/S10_sdk_deploy/scripts/cruise_vmc_noros.py`
启动脚本：`run_dialmpc_stair_wp033.sh`

## 1. 技能划分

- SMppi：直线段走线保持（BodyMPPI，输出 `[vx, omega]`）。
- TMppi：航点附近低速转向（yaw 误差 > 10° 时接管）。
- CarVMC：巡航轮足执行器（200Hz，16 维力矩）。
- TK1：楼梯前 lidar 检测、减速、yaw 对准、交付 RL。
- RL-Stair：爬楼梯（policy 50Hz + 腿 PD / 轮速 200Hz）。
- TK2：四轮全部站上最后一级台阶后，立即低速对准下一航点，随后交回 SMppi/TMppi。

## 2. 数据管线

```text
33 个 track_waypoint_*（原始航点折线）
  → nav_waypoint（20Hz）：只输出当前航点直线段
  → 主循环直线控制：由直线 heading / 到 wp 距离生成 [vx, vyaw]
  → TK1 / TK2 修正 [vx, vyaw]
  → TMppi（近点低速转向） / SMppi（BodyMPPI）
  → CarVMC（CRUISE，200Hz） / RLStairCtrl（STAIR，policy 50Hz + PD 200Hz）
  → 16 joint torque → MuJoCo
```

感知管线（全局 lidar，不使用 god-view ray）：

```text
LidarTerrainV2（96×48 地形射线 + 61×13 wall 射线，mount 抬高 0.6m，4Hz 累计）
  → h / hmax / wall 栅格
  → 轮下 terrain_at（CarVMC 腿控）
  → elev_tile + update_mode（TK1/TK2/RL 的 STAIR 判定）
  → lidar 在线 riser 表（RL policy terrain_ctx）
```



## 2.1 完整主链路状态机（20Hz 一拍内顺序执行）

```text
1. nav_waypoint.line(next_idx, robot_xy)
     -> 当前直线段 heading / dist_to_wp

2. stair_mode.update(local_map, ...)
     -> mode = CRUISE 或 STAIR

3. CRUISE 时的修正层
     TK1 检测到前方楼梯：
        vx <= 2.0，对准楼梯方向，按距离减速；
        满足距首级 riser<2m 且 |yaw_err|<=0.20 且 vx<2.0 时
        stair_mode 切到 STAIR。

     TK2 刚从 STAIR 切回 CRUISE：
        vx <= 1.5，对准下一航点；
        |yaw_err| <= 0.15 后释放，进入第 4 步。

4. CRUISE 规划器二选一
     TMppi：
        距当前 wp < 0.2m
        且 实际速度 < 0.2m/s
        且 当前 yaw 与下一航段方向误差 > 10°
        -> vx<=0.2，omega=clip(3.0*err, ±2.0)

     SMppi：
        除 TMppi 触发以外的所有 CRUISE 时间
        -> BodyMPPI 规划 [vx, omega]

5. 执行器
     mode == CRUISE -> CarVMC 200Hz
     mode == STAIR  -> RL policy 50Hz + 腿PD/轮速 200Hz

6. STAIR 退出（TK2 的起点）
     四轮轮心高度 >= max(lidar riser top) - 0.05
     -> mode 切回 CRUISE
     -> 同一拍置位 TK2
```

TMppi / SMppi 分工判定表：

| 控制器 | 使用场景 | 触发/退出判定 | 输出 |
|---|---|---|---|
| TMppi | 只在当前航点附近低速转向 | 距 wp<0.2m 且 实际速度<0.2m/s 且 下一段 yaw 误差>10° | vx<=0.2，omega=P*yaw_err，限幅±2.0 |
| SMppi | 除 TMppi 以外的全部巡航 | 上述条件不满足 | BodyMPPI 输出 [vx,omega] |

TK1 / RL-Stair / TK2 分工判定表：

| 模块 | 阶段 | 进入条件 | 退出条件 |
|---|---|---|---|
| TK1 | 楼梯前 CRUISE | lidar 检测到路径前方 riser | 距首级 riser<2m、yaw 已对准、vx<2.0，切 STAIR |
| RL-Stair | 楼梯中 STAIR | TK1 交付 | 四轮全部站上最后一级台阶（轮心>=最高台面-0.05） |
| TK2 | 楼梯后 CRUISE 的第一段 | STAIR→CRUISE 的同一拍 | 对准下一航点（|yaw_err|<=0.15），交回 TMppi/SMppi |

## 3. 频率

| 模块 | 频率 | 实现 |
|---|---|---|
| 仿真 | 200Hz | DT=0.005 |
| nav_waypoint / stair_mode | 20Hz | `S10_NAV_HZ=20`，每 round(200/HZ) 步一拍 |
| SMppi / TMppi | 20Hz | 与 nav_waypoint / stair_mode 同拍 |
| CarVMC | 200Hz | 每控制步 |
| lidar 高程图更新 | 4Hz | `S10_ELEV_HZ=4`，全模块共用同一栅格 |
| RL policy | 50Hz | `DECIMATION=4`，动作零阶保持 |

## 4. 路径规划：原始航点折线

- 几何路径 = `wp[:, :2]` 的原始折线，**不做 biarc 圆角、不做走廊平移、
  不做 diagonal bump**。
- 实现：`nav_waypoint.py` 只保留 `wp` 数组，`line()` 输出当前航点直线段；
  CRUISE/STAIR 判定所需的路径几何由 `stair_mode.py` 单独构建。
- nav 层不做 `vx/vyaw` 控制、不做曲率 vlim、不做 CTE、不做 CRUISE/STAIR 判定。
- 主循环只根据直线 heading 误差与到下一航点距离做简单直线控制。
- 航点推进只按水平距离：`S10_WP_ADVANCE_DIST=0.2`，不使用弧长兜底、
  不使用 wp7 走廊平移判点。
- `WaypointLineNav.line(next_idx, robot_xy)` 输出：
  `start / end / heading / length / dist_to_wp`。
- CRUISE/STAIR 判定移到独立模块 `stair_mode.py`，不再属于 nav 层。

参数：

```bash
S10_GLOBAL_FILLET_R=0
S10_WP_ARRIVE_R=0.2
S10_WP_ADVANCE_DIST=0.2
S10_AUTO_VMAX=3.0
S10_LINE_VMAX=3.0
S10_LINE_YAW_GAIN=2.5
S10_LINE_YAW_MAX=2.0
S10_LINE_BRAKE_DIST=1.5
```

## 5. SMppi 直线保持（BodyMPPI）

- 状态输入：`[x, y, yaw, body_vx, body_vy, omega]`；`body_vx/body_vy` 由世界
  速度经 `xmat` 旋转到机体系。
- 采样配置：`N=512, H=20, dt=0.05` → 视界 1.0s；`S10_MPPI_ADA=1`。
- 采样中心：`[v_ref, guide_om]`，其中
  `v_ref = 主循环直线控制 vx`（TK 限速与 decel 也会进入 v_ref），
  `guide_om = nav vyaw`。
- 参考轨迹：从 `s_cur` 起 0~12m、步长 0.5m，截止到当前航点弧长 +1.5m。
- rollout 约束：
  - 摩擦锥 `|vx·omega| <= mu·g / |vx|`；
  - CarVMC 能力表 `car_omega_limit(vx)`；
  - 纵向加速度 `|dvx| <= S10_MPPI_A_MAX * dt`。
- 成本：
  `2.0 * dist + 0.8 * v_err^2 + 0.5 * guide_err^2 + 0.05 * smooth`
  （`S10_MPPI_W_HEAD=0`）。
- 输出约束：vx 不超过 `v_ref`，并按 `S10_MPPI_A_MAX` 做速率限幅；omega 按
  摩擦锥/能力表/slew 限幅。
- 最终 omega 上限：`S10_VMC_OM_CAP=2.0`。

参数：

```bash
VMC_MPPI_N=512
VMC_MPPI_H=20
S10_MPPI_ADA=1
S10_MPPI_A_MAX=2.0
S10_MPPI_OMAX=2.5
S10_MPPI_W_GUIDE=0.5
S10_MPPI_W_DIST=2.0
S10_MPPI_W_HEAD=0.0
S10_VMC_OM_CAP=2.0
```

## 6. TMppi 航点转向

- 触发条件（三者同时满足）：
  `S10_TURN_SPLIT=1`、距当前 wp `< S10_WP_ARRIVE_R (0.2m)`、
  世界速度范数 `< S10_TURN_V_MAX (0.2m/s)`。
- 动作：
  `vx <= S10_WP_TURN_VX (0.2)`，
  `omega = clip(S10_TURN_K * yaw_err, +-S10_TURN_OM_MAX)`，
  其中 `yaw_err` 是当前 yaw 与下一航段方向的误差。
- 交回 SMppi：`|yaw_err| <= S10_TURN_ERR_DEG (10°)`。
- 风险：判点半径与 TMppi 触发半径都是 0.2m。若快速进点来不及减速触发，
  需要把减速半径与判点半径分离（下一步验证项）。

参数：

```bash
S10_TURN_SPLIT=1
S10_TURN_K=3.0
S10_TURN_OM_MAX=2.0
S10_TURN_ERR_DEG=10
S10_TURN_V_MAX=0.2
S10_WP_TURN_VX=0.2
```

## 7. 航点推进

```bash
S10_WP_ADVANCE_DIST=0.2
```

- 只按当前 wp 水平距离判点：`dist(base_xy, wp_xy) <= 0.2`。
- 不使用 `S10_WP_ADVANCE_BY_S` 弧长兜底，不偏移 wp7 坐标。
- 判点半径 `S10_WP_ARRIVE_R=0.2` 同时用于 TMppi 触发与主循环直线刹车区。

## 8. CarVMC 基线

- 半蹲站姿：`S10_CAR_SQUAT=1`，关节目标
  `hipx=+-0.05, hipy=∓1.10, knee=±1.90`。
- 腿控制：每腿垂直力 = `mg/4 + roll/pitch 姿态分配 + 地形阻抗`；
  地形阻抗增益 `S10_VMC_KPH=300 / KDH=60`。
- 轮控制：速度 PID `wheel_k=4.0 / d=0.08` + 差速 yaw 反馈
  `S10_VMC_YAW_K_WHEEL=60` + 摩擦前馈；直线轮矩上限 13.5Nm，
  弯道/近脊收敛到 μN·r。
- 压弯：`roll_tar = clip(-S10_CAR_ROLL_K * omega * |vx|, +-S10_CAR_ROLL_AMP)`，
  默认 `ROLL_K=0.06 / ROLL_AMP=0.06`。
- 最终 omega 上限：`S10_VMC_OM_CAP=2.0`；侧向包线 `S10_AUTO_LAT_MAX=5.0`。
- 力矩钳制：腿 ±48Nm、轮 ±13.5Nm；连续超限 >0.5s 判不合格。

```bash
S10_CAR_SQUAT=1
S10_VMC_KPH=300
S10_VMC_KDH=60
S10_VMC_WHEEL_K=4.0
S10_VMC_WHEEL_D=0.08
S10_VMC_YAW_K_WHEEL=60
S10_VMC_OM_ABS_MAX=2.0
S10_VMC_OM_CAP=2.0
S10_VMC_WHEEL_TMAX=13.5
S10_VMC_MU=0.8
```

## 9. TK1（楼梯前接管）

触发：`S10_TK1=1`、lidar 高程图已启用、当前为 CRUISE，且路径前方
`[s+0.5, s+S10_TK1_LOOKAHEAD]` 内检测到 riser。

检测使用双检测器：
1. terrain hmax 梯度：`rise=0.05, max_dh=0.16`，适合宽单级台阶；
2. on-path wall 垂直面：距路径 `< S10_OBST_LAT_MIN(0.5m)` 的 wall 格子
   `>= S10_TK1_MIN_CELLS(8)`，适合六级楼梯。

动作：
- 速度上限：`vx <= S10_TK1_VX (2.0)`。
- yaw 对准：`|yaw_err| > S10_TK1_YAW_DB(0.20)` 时
  `vyaw = clip(S10_TK1_YAW_K * yaw_err, +-S10_TK1_YAW_MAX)`。
- 减速：`decel_request` 随到首级 riser 的距离从 0（5m）连续升到 1（2m），
  `vx` 向 `S10_ELEV_DECEL_VX=2.0` 混合，混合结果进入 SMppi 的 `v_ref`。
  公式：`decel_request = clip((LOOKAHEAD - dist) / (LOOKAHEAD - ENTER), 0, 1)`，
  当前 `LOOKAHEAD=5.0, ENTER=2.0`。

交付 RL 门控（`stair_mode.StairGate.update`）：
- 距首级 riser `< S10_STAIR_ENTER_DIST (2.0m)`；
- `|yaw_err| <= S10_TK1_YAW_DB (0.20)`；
- `body_vx < S10_TK1_VX (2.0m/s)`。

防重入：STAIR→CRUISE 后，距离 handback 点 `< S10_STAIR_REENTRY_GUARD (3.0m)`
时不允许再次进入 STAIR。无 `S10_FORCE_MODE` 调试强制模式。

```bash
S10_TK1=1
S10_TK1_LOOKAHEAD=5.0
S10_ELEV_ENTER=2.0
S10_ELEV_DECEL_VX=2.0
S10_TK1_VX=2.0
S10_TK1_YAW_DB=0.20
S10_TK1_YAW_K=2.5
S10_TK1_YAW_MAX=1.5
S10_STAIR_ENTER_DIST=2.0
S10_STAIR_REENTRY_GUARD=3.0
S10_TK1_MIN_CELLS=8
```

## 10. RL-Stair

- 执行器切换：`S10_VMC_MODE=rlstair`；CRUISE 用 CarVMC，STAIR 用
  `rl_stair/deploy/rlstair_ctrl.py`。
- riser 表：只允许 lidar 在线检测注入。`_lidar_riser_table()` 沿原始折线
  调用 `LidarTerrainV2.detect_risers(rise=0.05, max_dh=0.16)`，把每个 riser
  的世界 xy 与 top 高度注入 `RLStairCtrl.set_risers()`。
  禁止 `STAIR_RISERS/STAIR_TOPS` 硬编码表，禁止预扫描表。
- 坐标处理：`obs_np.py` 把 riser 世界坐标投影到楼梯爬升方向
  `[cos(heading), sin(heading)]`，前/后轴位置用
  `cos(yaw - target_heading)` 投影。训练环境（+y 楼梯、target=pi/2）是该
  公式的特例，部署不再依赖世界 y 轴。
- PRETRANS（按距离，不使用 y 坐标）：
  - 楼梯前：距首级 riser `3.0m → 2.0m` 内把 CarVMC 半蹲姿态插值到 RL
    高站姿；`<=2.0m` 后腿部用 stand PD 锁定。
  - 楼梯后：从 STAIR→CRUISE handback 点起，前进 `S10_PRETRANS_EXIT_LEN=2.0m`
    内把腿部控制平滑交还 CarVMC。
- 策略：TorchScript `policy.pt`（55 维观测 → 16 动作 tanh）。
  观测布局：
  `angvel*0.25 | gravity | cmd | leg_err | leg_vel*0.05 | last_action |
  heading[cos,sin] | terrain_ctx(4) | rough`。
  `terrain_ctx` 为前/后轴到下一 riser 的距离与高差；`rough=1` 表示前方有 riser。
- 执行：腿 PD `Kp=50/Kd=1/clip±48`，动作缩放 0.7；轮速
  `Kp=2 * (action*24 - qd)/clip±13.5`；policy 50Hz（`DECIMATION=4`）。
- 航向目标：`S10_RL_HEADING` 默认 pi/2；TK1 交接时设为 lidar 检测到的
  riser 爬升方向。
- 退出：四轮轮心高度均 `>= max(lidar riser top) - S10_STAIR_WHEEL_CLEAR(0.05)`，
  即四轮全部站上最后一级台阶后立即 CRUISE。fallback：沿爬升方向前进超过
  `S10_STAIR_MIN_CLIMB_S=2.5m`（仅 lidar top 缺失时使用）。
- RL→CRUISE：`CarVMC.reset_state()` 复位滤波器。

```bash
S10_RL_ELEV=1
S10_RL_POLICY=rl_stair/deploy/policy.pt
S10_RL_VX=1.5
S10_RL_WARMUP=0
S10_PRETRANS=1
S10_PRETRANS_ENTER_DIST=2.0
S10_PRETRANS_BLEND_LEN=1.0
S10_PRETRANS_HOLD_DIST=2.0
S10_PRETRANS_EXIT_LEN=2.0
S10_STAIR_WHEEL_CLEAR=0.05
S10_STAIR_MIN_CLIMB_S=2.5
```

## 11. TK2（楼梯后立即交接）

- 触发：`update_mode` 检测到四轮全部站上最后一级台阶并切回 CRUISE 的同一拍，
  `_tk2` 置位。
- 动作：只做一次“对准下一航点”的低速转向：
  `vx <= S10_TK2_VX (1.5)`，若 `|yaw_err| > S10_TK2_YAW_DB(0.15)` 则
  `vyaw = clip(S10_TK2_YAW_K * yaw_err, +-S10_TK2_YAW_MAX)`。
- 释放：`|yaw_err| <= S10_TK2_YAW_DB` 后立即释放，交回 **SMppi/TMppi**。
- 删除内容：楼梯后慢速区、POST-STAIR AIM、平台 step-turn、transition pause、
  平台专用 CarVMC 参数切换全部不使用。

```bash
S10_TK2=1
S10_TK2_VX=1.5
S10_TK2_YAW_DB=0.15
S10_TK2_YAW_K=2.5
S10_TK2_YAW_MAX=1.5
```

## 12. 全局 lidar 地形

- `terrain_at()` 只使用 `LidarTerrainV2` 的 `height(x,y)`；数据缺失时用
  运动学兜底，不再调用 `mj_ray` god-view ray。
- 站起阶段、CarVMC 轮下地形、TK1/TK2/RL 高程图共用同一个 lidar 栅格，
  更新频率 `S10_ELEV_HZ=4`。
- `S10_VMC_TERRAIN=lidar` 现在真实生效。

```bash
S10_VMC_TERRAIN=lidar
S10_ELEV_HZ=4
```

## 13. 已删除 / 禁用内容

| 删除项 | 处理 |
|---|---|
| 避障 costmap / `S10_MPPI_OBSTACLE` | 完全删除；lidar 高程图保留给 TK1/TK2/RL |
| god-view ray / mj_ray 预扫描 | 完全删除 |
| 硬编码 `STAIR_RISERS/TOPS`、STAIR_ZONE | 删除为默认空表；只能 lidar 或 env 注入 |
| `S10_FORCE_MODE` 强制模式 | 删除 |
| direct-nav / `S10_SWITCH_*` / LINE_TURN | 删除，巡航只有 SMppi/TMppi |
| CRUISE_TK、POST-STAIR AIM、STEP_HOMING、W45_PULL | 删除 |
| 楼梯走廊平移 `S10_STAIR_CORRIDOR_X` | 默认 0，路径用原始折线 |
| 楼梯后 8m 慢速区、平台 step-turn/pause | 删除，TK2 对齐后交回 SMppi/TMppi |

## 14. 当前状态与验证点（2026-08-18）

- 已落地：原始航点折线；全局 lidar；SMppi/TMppi；TK1 距离减速；RL 使用
  lidar riser 表与距离式 PRETRANS；TK2 立即交接；避障与已知地图删除。
- 待验证：
  1. lidar 在线 riser 表在真实 mesh 上的检出率/top 精度（原硬编码表删除后，
     需要实测确认 6 级楼梯每级 top 是否正确；若个别 tread 被遮挡，需要调
     `S10_LIDAR_RAISE_Z`、`S10_LIDAR_NZ_MIN` 或 `detect_risers` 参数）。
  2. TK1 `LOOKAHEAD=5 / ENTER=2` 的减速效果。
  3. TK2 在四轮登顶后转向下一航点的稳定性。
  4. wp0→33 全程。
  5. 判点半径与 TMppi 触发半径同为 0.2m 的进点风险。

## 15. 遗留代码（不参与本方案）

- `dial-mpc/`
- `src/S10_sdk_deploy/s10_mpc/mpc_controller.py`、`mppi_controller.py`
- `src/S10_sdk_deploy/scripts/stair_dial_noros.py`、`cruise_test.py`
- `src/S10_sdk_deploy/s10_mpc/stair_*.py`（历史楼梯控制器）

以上仅为历史遗留，主链路不依赖，可归档/删除。

---

## 附：新文件夹模块清单

| 文件 | 作用 |
|---|---|
| `run_smppi_tmppi_cruise_rlstair_tk12.sh` | 新启动脚本，所有参数显式写出 |
| `cruise_main.py` | 新主循环：nav → TK1/TK2 → TMppi/SMppi → CarVMC/RL |
| `nav_waypoint.py` | 导航模块：只输出原始航点直线段 |
| `stair_mode.py` | CRUISE/STAIR 判定模块（独立于 nav） |
| `smppi.py` | SMppi：BodyMPPI 封装，无避障 |
| `tmppi.py` | TMppi：近点低速转向，独立实现 |
| `carvmc.py` | CarVMC：半蹲站姿执行封装 |
| `perception_lidar.py` | 感知：全局 lidar 高程图 + stair heading + riser 表 |
| `rlstair_ctrl.py` | RL-Stair 部署控制器（policy→腿 PD/轮速） |
| `rlstair_obs.py` | 55 维观测编码（含 lidar 世界坐标投影） |
| `policy.pt` | RL 策略权重 |

运行方式（等确认后再运行）：

```bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition/SMppi_TMppi_Cruise_RL-Stair_TK1_TK2
bash run_smppi_tmppi_cruise_rlstair_tk12.sh
```
