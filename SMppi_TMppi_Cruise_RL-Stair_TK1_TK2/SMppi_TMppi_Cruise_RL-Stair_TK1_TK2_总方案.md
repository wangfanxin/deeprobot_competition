# SMppi/TMppi Cruise + RL-Stair + TK1/TK2 总方案（2026-08-19 重构版，代码对齐）

> 本文档与 2026-08-19 代码完全对齐。模块目录：`SMppi_TMppi_Cruise_RL-Stair_TK1_TK2/`，
> 启动脚本：`run_smppi_tmppi_cruise_rlstair_tk12.sh`，主循环：`cruise_main.py`。
> 旧单体链路（`src/S10_sdk_deploy/scripts/cruise_vmc_noros.py` 等）为遗留代码，不参与本方案。

## 1. 目标与技能分工

- 在 MuJoCo 官方 `S10_track.xml` 中完成 33 个 `track_waypoint_*` 全程巡检。
- 技能分工（用户口径）：
  - **SMppi**：直线段走线保持（BodyMPPI 采样规划）。只做三件事：航向保持、快加速、到点减速。**不过弯**。
  - **TMppi**：航点原地转向（四轮差速）。到点后尽快转到下一 wp 方向，转完交回 SMppi。
  - **CarVMC**：巡航执行器（200Hz，16 维力矩）。
  - **STAIR（RL-Stair）**：接管**一切 riser**（多级楼梯 + ≥8cm 单级台阶/台面沿），policy.pt 50Hz。
  - **TK1**：楼梯前交接。**减速由 SMppi 终点代价 + decel 负责；TK1 只做对准 <1s**（总预算 <2s），交付速度 1.5 m/s，对准目标 = lidar riser 航向。
  - **TK2**：四轮上顶后立即转出对准下一 wp（<1s），随后交回 SMppi/TMppi。
  - **DROP**：下行落差（≥8cm 跌落沿）不交 RL，强制 0.3 m/s 低速爬行兜底。

## 2. 模块清单

| 文件 | 职责 |
|---|---|
| cruise_main.py | 200Hz 主循环：状态机、线控、TK1/TK2/DROP/roll 门控、航点推进、计时、traj |
| nav_waypoint.py | 航点提取、直线段输出（start/end/heading/dist_to_wp）、判点 |
| smppi.py + s10_mpc/body_mppi.py | SMppi 采样规划（终点代价、40Hz slew、耗时统计） |
| tmppi.py | TMppi 近点原地转向 |
| carvmc.py + s10_mpc/vmc_legs.py | CarVMC 轮足执行（防打滑仅 vx>0.5 启用） |
| perception_lidar.py + s10_mpc/lidar_terrain_v2.py | lidar 高程图、局部 tile、riser/heading 检测 |
| stair_mode.py + s10_mpc/auto_nav.py | CRUISE/STAIR 门控（update_mode）、decel_request、单级/下行检测 |
| rlstair_ctrl.py / rlstair_obs.py / policy.pt | RL 楼梯控制器（55 维观测→16 动作） |
| plot_traj_speed.py | 轨迹 xy-速度图（traj 第 6 列 = 速度） |

## 3. 控制频率与实时性

| 层 | 频率 | 说明 |
|---|---|---|
| MuJoCo 仿真 | 200Hz | DT=0.005 |
| 执行层 CarVMC / RLStairCtrl | 200Hz | 每仿真步一次 |
| 规划/模式 tick | **40Hz** | S10_NAV_HZ=40，每 5 步一拍（SMppi/TMppi/TK/判点/模式） |
| lidar 高程图 | 4Hz 增量 | S10_ELEV_HZ=4，tile 每拍重建 |
| MPPI 内部 | 2s 视界 | N=1024、H=40、rollout dt=0.05（H×dt=2.0s） |

**实时性保证（名义=实际）**：
- 40Hz 预算 25ms。实测（WSL/竞赛 .venv，本轮基准）：plan 平均 **9.9~10.3ms**、最大 **19.2ms** < 25ms；实测控制拍 **40.0Hz**。
- 输出 slew 与 rollout dt 解耦：rollout 用 S10_MPPI_DT=0.05，输出加速度/角加速度限幅用 S10_MPPI_CTRL_DT=0.025（=1/40s），保证限幅名义值在 40Hz 下不变形。
- 参考路径间距 1.0m（12m 视界内 ≤13 点），把 _cost 的距离矩阵从 (1024,41,25) 压到 (1024,41,13)，plan 由 21.9ms 档降到 13.4ms 档。
- 结束打印：规划器: plan=N次 avg=..ms max=..ms | 实际控制拍=..Hz，每轮测试必须核对。

## 4. 数据管线

### 4.1 运动控制管线（40Hz tick）

    33 track_waypoint_*
      -> nav_waypoint.line(next_idx, pos)  [直线段 start/end/heading/dist_to_wp]
      -> 线控制器: vyaw = clip(2.5*(line_head+CTE-yaw), ±1.0)
                   vx   = LINE_VMAX(4.0) * brake(dist_wp)   # 到点线性刹车
      -> 修正层: TK1 对准(lidar heading, 交付圈内 vx<=1.5)
                 TK2 对准下一 wp (vx<=1.2)
                 DROP 前方跌落沿 -> vx<=0.3, |vyaw|<=0.5
                 post-stair hold 0.6s / 慢速瞄准
                 decel_request -> vx 向 2.0 插值
      -> v_ref = min(...)
      -> 规划二选一:
           TMppi: dist<0.5m 且 speed<0.3 且 |yaw_err|>10deg -> vx<=0.2, om=clip(3*err,±3.0)
           SMppi: ref_path(末端精确=wp, 1m间距) + 终点代价(dx=0/ref_v=0) -> [vx,om]
      -> om 上限: TMppi=min(TURN_OM_MAX=3.0, latmax/|vx|); SMppi=min(VMC_OM_CAP=1.0, 1.8/|vx|)
      -> roll 门控(0.30/0.15 滞回): 触发 -> om=0, vx<=0.4
      -> TK 阶段直接对准: om=clip(vyaw, ±2.0)   # 绕过 MPPI 的 1.0 om 上限
      -> post-stair / 高台(z>1: vx<=0.8, om<=0.3, 压弯关)
      -> cmd{vx,omega,roll_tar,pitch_tar}
      -> STAIR ? RLStairCtrl.compute_tau : CarVMC.compute_tau
      -> 16 joint torque -> mujoco(200Hz)

### 4.2 感知管线（lidar，无 god-view）

    LidarTerrainV2 (4Hz 增量, 96 地形射线 + wall 通道, mount+0.6m)
      -> 高程栅格 h/hmax
      -> perc.local_tile(pos)  [step_flag 梯度]
      -> StairGate.update(update_mode):
           _elev_rises_on_path: 沿路径窗口 0.5~5.0m 剖面
             - 多级楼梯: 步跳>=0.10 且 0.5m 内确认, >=2 级, 跨度<=3m, 总爬升>=0.4
             - 单级 riser: 跳变>=0.08 且 0.2~0.6m 内平台持续  -> 也进 STAIR
             - 下行跌落: 剖面下降>=0.08  -> drop_ahead_dist
           -> stair_rises_s / stair_rises_tops / stair_ahead_dist /
              decel_request / drop_ahead_dist / stair_first_heading
      -> perc.riser_table(fol) -> RLStairCtrl.set_risers(xy, tops, heading)

## 5. 状态机与门控

    CRUISE ------------ STAIR ------------ CRUISE
      |  SMppi走线         |  RL policy         |  post-stair hold
      |  TMppi原地转        |  PRETRANS 腿锁     |  TK2 对准 -> 交回
      |  TK1对准(不动模式)   |                    |
      |  DROP 慢爬          |                    |

    CRUISE -> STAIR: 前方 riser(多级或单级) 且 stair_ahead_dist<=ENTER(2.0)
                     且 TK1 门控通过: |yaw-riser航向|<=0.20 且 body_vx<=1.5
    STAIR -> CRUISE: 四轮 z >= max(riser tops)-0.05 (S10_STAIR_WHEEL_CLEAR)
                     兜底: 沿爬升方向前进 >2.5m
                     重入保护: 退出后 3m 内禁止重进 STAIR

**航点推进（防抢跑）**：dist<0.3 且 |yaw-下一段heading|<=0.25 且 |ω_body|<=0.3 才推点。
**roll 安全网**：|roll|>0.30 触发（om=0、vx<=0.4），<0.15 释放；实际施加指令回同步进 MPPI slew 基准（sync_applied），释放后从真实指令按 3.5 m/s² 慢升，禁止 0.4→4.0 阶跃。
**终止条件**：|roll|>0.9 或 body z<0.12 侧翻；卡死超时 90s；到达 MAX_WP。

## 6. 时序预算（验收口径）

| 阶段 | 定义（起→止） | 预算 | 日志字段 |
|---|---|---|---|
| TK1 | 首次检测到前方 riser（decel/TK1 激活）→ STAIR 交付 | <2.0s（含减速） | [TK1] 减速+对准 X.XXs / 对准 X.XXs |
| TK1 对准 | 交付圈内 |ey|>DB 首次出现 → 交付 | **<1.0s** | 同上第二字段 |
| 交付状态 | 交付瞬间 | body_vx≈1.5 m/s、|ey|<=0.20 | [RL-DIAG] takeover... |
| TK2 | 四轮过顶（STAIR→CRUISE）→ 对准下一 wp | **<1.0s** | [TK2] 上顶->对准 X.XXs |
| TMppi | 触发（dist<0.5, speed<0.3）→ |err|<=10° 释放 | 尽量 <1.0s | plan=TMppi 段 |
| post-stair hold | STAIR→CRUISE 后 | 0.6s 直线 | corr 字段 |

## 7. 模块细节

### 7.1 线控制器（只走线，不过弯）
- 方向：航段 heading + CTE 修正（K=1.0，|cte| 限 1.0）；dist<0.5m 直接瞄 wp。
- vyaw = clip(2.5·err, ±1.0)，作为 SMppi 的 guide_om。
- vx = 4.0 × brake，brake=(dist-0.2)/2.5 线性（终点代价二次兜底）。
- 已删除：head-err 降速、下一段锐角预刹、MIN_VX 地板。

### 7.2 SMppi（BodyMPPI）
- 状态 [x,y,yaw,vx,vy,ω]；dt=0.05；N=1024、H=40（2s 视界）；ADA=1。
- 采样中心 [v_ref, vyaw]；约束：摩擦锥 |vx·ω|≤μg（μ 标定 0.36 档）、car_omega_limit 表、加速度钳制 3.5 m/s²。
- 成本：2.0·路径距离 + 0.8·速度偏差 + 0.5·guide 偏差 + 0.05·控制平滑 + **终点代价**。
- **终点代价（dx=0 / ref_v=0）**：参考路径末端精确=wp；STOP_DX=4.0 内生效；
  cost += 10·dist(rollout终点, wp) + 10·(v_end − 4.0·dx/4.0)² → 到点速度自动归零。
- 输出：vx 按 ctrl_dt(0.025) 限幅 3.5 m/s²；ω slew 6.0 rad/s²；ω 上限 min(OMAX=2.5, VMC_OM_CAP=1.0)。

### 7.3 TMppi（原地转向）
- 触发：dist<0.5（S10_TURN_ARRIVE_R，独立于判点半径）且世界速度<0.3 且 |yaw_err|>10°。
- 动作：vx<=0.2，om=clip(3·err, ±3.0)；om 上限独立于 VMC_OM_CAP（min(3.0, 1.8/|vx|)）。
- 释放：|err|<=10° 交回 SMppi。

### 7.4 CarVMC（巡航执行）
- 半蹲站姿 S10_CAR_SQUAT=1（hipy∓1.10 / knee±1.90）。
- 轮：速度 PID（K=12, D=0.02）+ 差速 yaw 反馈（K=80）+ 摩擦前馈；直线轮矩限 13.5 Nm。
- **防打滑仅实际 vx>0.5 启用**（S10_CAR_SLIP_VX_GATE=0.5）：打滑回缩、v855 反向硬刹都不在原地转（vx≈0）时触发。
- 压弯 roll_tar=clip(-0.06·ω·|vx|, ±0.06)；高台(z>1)关闭压弯。
- 抬轮前馈/地形前瞻已删除（riser 全交 STAIR）。

### 7.5 感知与 riser 检测
- 高程图 4Hz 增量；tile 半宽 8m；step_flag=|hmax 梯度|。
- 沿路径窗口 ±1.2m 取最高剖面：
  - 多级：≥2 级、跨度≤3m、总爬升≥0.4（6 级楼梯）；
  - **单级：跳变≥0.08 且 0.2~0.6m 内平台持续** → 进 STAIR；
  - **下行：剖面下降≥0.08** → drop_ahead_dist → DROP 慢爬。
- TK1 对准目标 = lidar 检测 riser 的路径航向（perc.stair_heading）。

### 7.6 STAIR / TK1 / RL / PRETRANS
- 交接条件：dist<=2.0 + |ey|<=0.20 + body_vx<=1.5（TK1 只负责对准；减速归 SMppi）。
- PRETRANS：距离式（enter 3.0 / blend 1.5 / hold 3.0 / exit 2.0），半蹲→RL 高站姿，y≥32 后 stand PD 锁腿。
- RL：policy.pt（55→16，tanh），腿 PD(Kp50/Kd1/clip48，动作×0.7) + 轮速 PD(Kp2, 24m/s, clip13.5)，50Hz 零阶保持；WARMUP=200；riser 表 = lidar 在线。
- 交付诊断：[RL-DIAG] takeover pos/yaw/max_leg_err。

### 7.7 TK2 / post-stair
- 四轮 z ≥ max(tops)-0.05 → CRUISE：0.6s 直线 hold（vx<=0.5，vyaw=0，防平台边缘 yaw 反冲）。
- TK2：|yaw_err|>0.25 时 vyaw=clip(2.5·err,±1.5)、vx<=1.2；<=0.25 释放（<1s 预算），直接执行对准（om 上限 2.0）。
- post-stair 慢速瞄准：距下一 wp >1.5m 时 vx<=0.6、om=clip(0.5·err,±0.2)；<=1.5m 释放回 SMppi/TMppi。

### 7.8 航点推进
- 判点：水平距离 <=0.3 或过点兜底（proj>len-0.5 且 lat<0.8）。
- 追加门控：对准下一段（DB 0.25）且 |ω_body|<=0.3，防转向扫过目标航向时抢跑。

## 8. 关键参数（run_smppi_tmppi_cruise_rlstair_tk12.sh 实际值）

    S10_NAV_HZ=40 S10_WP_ARRIVE_R=0.2 S10_WP_ADVANCE_DIST=0.3 S10_WP_ALIGN_DB=0.25 S10_WP_ALIGN_OM=0.3
    S10_AUTO_VMAX=4.0 S10_LINE_VMAX=4.0 S10_LINE_YAW_GAIN=2.5 S10_LINE_YAW_MAX=1.0 S10_LINE_BRAKE_DIST=2.5 S10_LINE_CTE_K=1.0
    VMC_MPPI_N=1024 VMC_MPPI_H=40 S10_MPPI_DT=0.05 S10_MPPI_CTRL_DT=0.025 S10_MPPI_ADA=1
    S10_MPPI_A_MAX=3.5 S10_MPPI_OMAX=2.5 S10_MPPI_W_GUIDE=0.5 S10_MPPI_W_DIST=2.0 S10_MPPI_W_HEAD=0.0
    S10_SMppi_STOP_DX=4.0 S10_MPPI_W_TPOS=10.0 S10_MPPI_W_TV=10.0
    S10_TURN_SPLIT=1 S10_TURN_ERR_DEG=10 S10_TURN_K=3.0 S10_TURN_OM_MAX=3.0 S10_TURN_V_MAX=0.3 S10_WP_TURN_VX=0.2 S10_TURN_ARRIVE_R=0.5
    S10_CAR_SLIP_VX_GATE=0.5
    S10_ELEV_HZ=4 S10_LIDAR_WALL=1 S10_STAIR_SINGLE_RISE=0.08 S10_ELEV_DROP_TH=0.08 S10_DROP_LOOKAHEAD=2.0 S10_DROP_VX=0.3
    S10_TK1=1 S10_TK1_LOOKAHEAD=5.0 S10_ELEV_ENTER=2.0 S10_ELEV_DECEL_VX=2.0 S10_STAIR_ENTER_DIST=2.0
    S10_TK1_VX=1.5 S10_TK1_YAW_DB=0.20 S10_TK1_YAW_K=2.5 S10_TK1_YAW_MAX=1.5 S10_TK_OM_MAX=2.0 S10_TK_VX=1.5
    S10_RL_ELEV=1 S10_RL_WARMUP=200 S10_PRETRANS=1 S10_PRETRANS_ENTER_DIST=3.0 S10_PRETRANS_BLEND_LEN=1.5 S10_PRETRANS_HOLD_DIST=3.0 S10_PRETRANS_EXIT_LEN=2.0
    S10_POSTSTAIR_HOLD_DIST=1.5 S10_TK2=1 S10_TK2_YAW_DB=0.25 S10_TK2_YAW_K=2.5 S10_TK2_YAW_MAX=1.5 S10_TK2_VX=1.2 S10_STAIR_WHEEL_CLEAR=0.05
    S10_CAR_SQUAT=1 S10_VMC_KPH=300 S10_VMC_KDH=60 S10_VMC_WHEEL_K=12.0 S10_VMC_WHEEL_D=0.02
    S10_VMC_YAW_K_WHEEL=80 S10_VMC_OM_ABS_MAX=2.0 S10_VMC_OM_CAP=1.0 S10_VMC_WHEEL_TMAX=13.5 S10_VMC_MU=0.8 S10_AUTO_LAT_MAX=1.8

## 9. 已删除 / 禁用清单

避障 costmap（整体删除）、抬轮前馈与地形前瞻（删除）、god-view mj_ray 预扫描（删除）、
硬编码楼梯表 STAIR_RISERS/TOPS（清空，由 lidar 在线检测替代）、CRUISE_TK wp4→5 特殊段、
head-err 降速、锐角预刹、MIN_VX 地板、S10_AUTO_STAIR_VX 死参数。

## 10. 测试矩阵（执行顺序）

| # | 测试 | 配置 | 判据 |
|---|---|---|---|
| T1 | 修复后首跑 smoke | 40s, MAX_WP=5 | 无侧翻；过 wp1~2；实际拍=40Hz；plan max<25ms |
| T2 | 台面专项 wp4→5 | 80s, MAX_WP=5 | 上沿触发 STAIR(单级) 且 RL 上 12.5cm；台面 SMppi 巡航；下沿 DROP 慢爬不栽头；推进 wp5 |
| T3 | 单级 wp5→6 | 60s, MAX_WP=6 | 同 T2 单级路径 |
| T4 | 六级楼梯 wp6→7 | 100s, MAX_WP=8 | [TK1] 总<2s/对准<1s；RL 上 6 级；[TK2]<1s；交接日志正常 |
| T5 | 平台段 wp8→12 | 120s, MAX_WP=12 | 高台限速段通过，无侧翻 |
| T6 | 全量回归 | 600s, MAX_WP=33 | 33 点全过；力矩合规（腿48/轮13.5，连续超限<0.5s）；无侧翻；40Hz 保持 |
| T7 | 退化回归 | 重放 tune4 配置 | 确认 wp4 侧翻不回潮 |
| T8 | RL 单级 12.5cm eval（T2 前置） | sim2sim 20 seeds | 上步成功率>=80%；不达标则规划下行课程微调 |

traj 列（8）：t,x,y,z,yaw,next_idx,speed,mode(STAIR=1)。

## 11. 当前状态（2026-08-19）

- 代码全部修改完成、编译通过；**尚未跑通**。
- 两轮 smoke（修复前）失败记录：
  1. wp1 附近绕圈 6s + 高速过点侧翻 → 终点代价按 STOP_DX 门控、权重 10、TMppi 提前 0.5m 触发、判点加角速度门。
  2. 坡面 roll 门控 0.4→4.0 阶跃正反馈侧翻 → roll 滞回 0.30/0.15 + sync_applied 回同步。
- 以上两修复已落码，T1 是第一次验证。

## 12. 待确认事项

1. 模块命名口径：SMppi=走线 / TMppi=原地转（文档与实现一致；用户此前消息中两名称写反，按此口径执行）。
2. RL 单级 12.5cm 上步未验证（课程 T1d 曾删除）——T8 先行。
3. 下行 riser：当前 DROP 慢爬兜底（RL 未练下行）；若要 RL 接管下行需补课程微调。
