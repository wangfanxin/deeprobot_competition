# SMppi/TMppi Cruise + RL-Stair + TK1/TK2 总方案（2026-08-19 v2，与工作区代码逐行对齐）

> 本版对 cruise_main.py（1384 行）及各模块逐行核对，以代码为唯一事实源；
> 旧版与代码不一致的描述已修正（见 §8 参数对照与 §12 待确认）。
> 模块目录：SMppi_TMppi_Cruise_RL-Stair_TK1_TK2/，启动脚本：
> run_smppi_tmppi_cruise_rlstair_tk12.sh，主循环：cruise_main.py。

## 1. 目标与技能分工

在 MuJoCo 官方 S10_track.xml 完成 33 个 track_waypoint_* 全程巡检。

| 技能 | 职责 | 关键点（代码口径） |
| --- | --- | --- |
| **SMppi** | 直线段走线保持 | 只做航向保持/快加速/到点减速，不过弯；40Hz、N=1024、H=40（2s 视界）、终点代价 |
| **TMppi** | 航点原地转向（四轮差速） | dist<0.5m 且速度<0.8 且 |yaw_err|>10° 且无楼梯 ahead；转完交回 SMppi |
| **CarVMC** | 巡航执行器（200Hz） | 轮速 PID+差速 yaw+半蹲腿；防打滑仅实际 vx>0.5 启用 |
| **STAIR（RL-Stair）** | 接管一切 riser（多级+≥8cm 单级） | policy.pt 55→16 tanh；腿 PD 50Hz+轮速 200Hz；riser 表 = lidar 在线检测 |
| **TK1** | 楼梯前交接 | 减速归 SMppi 终点代价+decel；TK1 只做对准（<1s，总预算<2s）；交付 vx≤1.5、|ey|≤0.20 |
| **TK2** | 楼梯后交接 | 四轮上顶→对准下一 wp（<1s）→交回 |
| **DROP** | 下行落差（≥8cm） | 不交 RL，分档低速爬行（0.3/0.6/1.0/1.2）兜底 |

## 2. 模块清单

| 文件 | 职责 |
| --- | --- |
| cruise_main.py | 200Hz 主循环：状态机、线控、TK1/TK2/DROP/EDGE/LIP/roll 门控、航点推进、计时、traj |
| nav_waypoint.py | 航点提取、直线段输出（start/end/heading/length/dist_to_wp）、判点（含过点兜底） |
| smppi.py + s10_mpc/body_mppi.py | SMppi 采样规划（终点代价、40Hz slew、sync_applied、耗时统计） |
| tmppi.py | TMppi 近点原地转向（触发半径独立于判点半径） |
| carvmc.py + s10_mpc/vmc_legs.py | CarVMC 轮足执行（含 car_omega_limit 能力表） |
| perception_lidar.py + s10_mpc/lidar_terrain_v2.py | lidar 高程图（h/hmax 双栅格+wall 通道）、tile、riser/heading/riser_table |
| stair_mode.py + s10_mpc/auto_nav.py | CRUISE/STAIR/DROP 门控（update_mode）、decel_request、riser/drop 检测、爬升航向合成 |
| rlstair_ctrl.py / rlstair_obs.py / policy.pt | RL 控制器（55 维观测→16 动作，torch.jit CPU） |
| plot_traj_speed.py | 轨迹 xy-速度图 |

## 3. 控制频率与实时性

| 层 | 频率 | 说明 |
| --- | --- | --- |
| MuJoCo 仿真 | 200Hz | DT=0.005 |
| 执行层 CarVMC / RLStairCtrl | 200Hz | 每仿真步一次 |
| 规划/模式 tick | 40Hz | S10_NAV_HZ=40，每 5 步一拍（SMppi/TMppi/TK/DROP/EDGE/LIP/判点/模式） |
| RL policy | 50Hz | decimation=4，动作零阶保持到 200Hz PD |
| lidar 高程图 | 4Hz 增量 | S10_ELEV_HZ=4（96×48 地形射线）；wall 通道 61×13 半速 ≈2.5Hz |
| MPPI 内部 | 2s 视界 | N=1024、H=40、rollout dt=0.05 |

- 40Hz 预算 25ms；实测 plan avg 9.9~10.3ms / max 19.2ms <25ms，实际控制拍 40.0Hz。
- slew 与 rollout 解耦：rollout dt=S10_MPPI_DT=0.05，输出限幅用 S10_MPPI_CTRL_DT=0.025（=1/40s）。
- 参考路径间距 1.0m（12m 视界 ≤13 点），距离矩阵 (1024,41,13)。
- 结束打印：规划器: plan=N次 avg=..ms max=..ms | 实际控制拍=..Hz，每轮必查。

## 4. 数据管线

### 4.1 运动控制管线（40Hz tick 内按序合成，后写覆盖先写）

    33 航点 -> nav.line [start/end/heading/dist_to_wp]
     ① 线控制器: line_head=段heading−1.0·clip(cte,±1)；段起点后方/平顶段首横偏→瞄段起点；
                 dist<0.5→直瞄 wp；vyaw=LP(clip(2.5·head_err,±1.0),0.4)；
                 vx=4.0·√clip((dist−0.2)/3.5,0,1)   # sqrt 刹车剖面
     ② 航线夹角门: |riser爬升轴−线段heading|≤0.45 才放行台阶类门控（路径外障碍不骑）
     ③ TK1: 交付圈(ad≤2.0)内 vx≤1.5；|yaw−riser航向|>0.20 → vyaw=clip(2.5·err,±1.5)
     ④ TK2: 瞄下一 wp；|err|>0.25 → vyaw=clip(2.5·err,±1.5)、vx≤1.2；≤0.25 释放
     ⑤ post-stair hold 0.6s(vx≤0.2,vyaw=0)/慢速瞄准(vx≤0.6,om=clip(0.5·err,±0.2))
     ⑥ LIP 骑坎锁存: 触发条件见 §5.3；锁存期 vx=1.2 冲量+正对航向
     ⑦ decel_request: vx=vx·(1−d)+dv·d（圈外 2.0；圈内航线对齐 1.2）
     ⑧ DROP 四档 + EDGE 探针 + STOP_DX 硬刹车 -> vx/vyaw/v_ref 修正
     ⑨ 参考路径: 航线投影点起、1m 间距、≤12m、末端精确=wp；wp_dx=dist_to_wp
     ⑩ 规划二选一:
          TMppi: dist<0.5 且 speed<0.8 且 |yaw_err|>10° 且无楼梯 -> vx=0.2, om=clip(3·err,±3.0)
          SMppi: BodyMPPI(v_ref, vyaw, wp_dx) -> [vx,om]
     ⑪ omcap: TMppi=min(3.0,1.8/max|vx|)；SMppi=min(1.0,1.8/|vx|)；高台 z>1.2→0.6、z>2.0→0.3
     ⑫ roll 安全网 0.34/0.28（窄脊 0.22/0.18）: om≤±0.3、vx≤0.3；>2s 低台倒车 −0.4
     ⑬ post-stair 1.0s 硬停 / 高台分档 / SEG0 / 锁存转向 / 大偏航纠偏 / TK 直接对准
     ⑭ cmd{vx,omega,roll_tar,pitch_tar,...} -> STAIR?RLStairCtrl:CarVMC -> tau(16) -> mj_step(200Hz)

### 4.2 感知管线（lidar，无 god-view）

    LidarTerrainV2（4Hz 增量；site 抬高 0.6m；FOV±90°；96×48 地形射线；cutoff 20m）
      -> h=min-z（轮下地形）/ hmax=max-z（台面/riser）；0.05m 栅格 x∈[-25,40] y∈[-5,55]；
         新命中横向左右各填 1 格；法向 |nz|≥0.6 滤竖直面
      -> wall 通道（|nz|<0.4 竖直面，61×13 近平射线，2.5Hz）
      -> build_local_tile: 16×16m hmax 瓦片 + step_flag=|梯度|>0.08
      -> _elev_rises_on_path（扫描窗口 v2：s_cur+0.1 → min(s_cur+8.0, 下一wp弧长)，下限 s_cur+1.2；横向 ±1.2m 最高剖面）:
           多级: 跳变≥0.10 且 0.5m 确认、≥2 级、跨度≤3m、总爬升≥0.2
           单级: 跳变≥0.10 且 0.2~0.6m 台面持续（含台面沿）
           下行: 下降≥0.10 且 0.2~0.6m 低位持续 -> drop_ahead_dist/_elev_drops/_elev_drop_ds
         -> stair_rises_s / stair_ahead_dist / decel_request / stair_first_heading
      -> perc.riser_table: detect_risers（hmax 跳变 0.05~0.16m，台面顶=跳变后 0.30m 窗内 max）
         单级台面：远侧跌落沿补成虚拟第二级 riser（补 RL 观测分布）
      -> RLStairCtrl.set_risers(xy, tops, heading)；climb_axis=riser 航向

### 4.3 RL 观测/执行管线

    obs(55)=angvel·0.25(3)|gravity(3)|cmd[vx,yaw](2)|leg_err(12)|leg_vel·0.05(12)
            |last_action(16)|heading[cos,sin](2)|terrain_ctx 前后轴距下一级 riser(4)|rough(1)
    policy(torch.jit, CPU) -> a(16) tanh
    腿: tau=clip(50·(a·0.7 − leg_err) − 1·leg_vel, ±48)
    轮: tau=clip(2·(a·24 − qd), ±13.5)
    前 S10_RL_WARMUP=200 步站姿 PD 锁腿（轮自由）；PRETRANS 距离式切换站姿（§7.6）

## 5. 状态机与门控总表（代码口径）

### 5.1 主状态机

    CRUISE ----------- STAIR ----------- CRUISE
      | SMppi走线        | RL policy        | post-stair hold 0.6s
      | TMppi原地转       | PRETRANS 腿锁     | TK2 对准 -> 交回
      | TK1对准(不改模式)  |                    |
      | DROP/EDGE 慢爬    |                    |

### 5.2 STAIR 入口 / 出口（全部条件）

- CRUISE→STAIR：stair_ahead_dist≤1.2 且（riser≥2 级，或单级 riser 且 drop 可见）
  且距航线横向 ≤1.0 且最低轮心 z≤1.2 且前轮未已上台（前轮心 < top0+0.02）
  且重入保护通过（s_cur > 出口 s + 2.0）且 TK1 门（|yaw−riser航向|≤0.20、
  body_vx≤1.5、|riser航向−路径航向|≤0.45）。
- STAIR→CRUISE：四轮 z ≥ max(riser tops)+0.02 且 s_cur > 末级 riser s + 0.8
  且 body_vx≤1.0、|pitch|≤0.3、|roll|≤0.25、|vy|≤0.8。
  兜底：沿爬升轴前进 >1.2m（单级）/ span+1.0m（多级）且姿态收敛。
- drift-abort：STAIR 中 |cte|>1.2 且进入 >1.0s → 强退 CRUISE 自救（西漂 6m 实测修复）。

### 5.3 按 z 高度门控（body_pos[2]，代码硬编码阈值总表）

| z 条件 | 门控 |
| --- | --- |
| z < 0.12 | 摔倒终止 |
| z ≤ 1.1 | TK1 激活；decel 减速生效；LIP 骑坎锁存触发 |
| z ≤ 1.15（否则延迟 2.0s） | TK2 出楼梯立即对准；锁存强制转向让位 |
| 最低轮心 z ≤ 1.2 | STAIR 入口允许（挡平顶假入口） |
| z > 1.0 | roll 门控期 om=0；压弯 roll_tar=0；terr 钳制 min(terr, z−0.25)；roll 死锁不倒车 |
| z > 1.2 | 判点半径放大 2.5m；平台限速 S10_PLAT_VX + omcap≤0.6；SEG0 段首纠偏；判点跳过对准门；TMppi omcap≤0.6；TK 转向 omcap≤0.4；大偏航 cte 阈值 0.8；卡死脱困启用；rock_kill=1 |
| z > 1.3 | DROPW：drop<1.5m → vx≤1.0 |
| z ≤ 1.3 | DDE（近沿落差≥0.25 → vx≤0.6）；DNW（Σ落差≥0.3 且每级≤0.08 → vx≤1.2） |
| z ≤ 1.5 | 下沿 s 投影保护生效（>1.5 仅剩轮下跨骑兜底） |
| z > 2.0 | 弱抓地高台：vx≤0.8、omcap≤0.3 |

平台限速（z>1.2）：S10_PLAT_VX=2.5（5.0 实测走廊撞墙 round297 回退），
近楼梯（ad≤5.0m）或下沿（drop<2.5m）回退 1.8。

### 5.4 按地形 / 几何 / 姿态 / 速度门控

| 类别 | 条件 | 门控 |
| --- | --- | --- |
| 地形 | 前轮−最低轮 ≥0.08（z≤1.1 且 next_idx≥4） | LIP 骑坎锁存 |
| 地形 | 后轮−最低轮 ≥0.08 | 跨骑 → DROP 兜底 |
| 地形 | 锁存期前−最低 ≥0.04 且无 drop | vx=1.2 冲量把后轮拉过立面 |
| 地形 | raw terr 中位 >1.2（对应 wp7→8 的 1.235 窄脊） | roll 门 0.22/0.18、vx≤1.0 |
| 地形 | 前方 1.5m 升 0.08~0.25、平顶、1.5m−0.5m 差 ≥0.08 | EDGE：vx≤0.6；≥0.10 加锁存+前轮抬轮 |
| 地形 | 单轮前探升 0.06~0.25（仅锁存期、前轮） | 抬轮前馈 |
| 地形 | 轮间地形差 ≥0.04 | v595 骑坎找平（全抬到 max−0.02） |
| 几何 | 距 wp≤0.5（平顶 2.5）；投影>len−0.5 且 lat<0.8；投影>len+0.8 | 判点/过点兜底推点 |
| 几何 | dist<0.6 且 speed<2.2 且 |yaw_err|>10° | TMppi |
| 几何 | dist_wp≤4.0（STOP_DX） | v_ref 线性归零硬刹车 |
| 几何 | 段首投影<−1 或段首 1m 内 |cte|>0.8（z>1.2） | SEG0 瞄段起点，om≤±0.6、vx≤1.0 |
| 几何 | |cte|>1.2（平顶 0.8） | 大偏航直瞄 wp/航线投影点，om≤±1.2、vx≤1.0 |
| 几何 | TK1：dist_wp≤2.5 且 |cte|≤0.8 且 ad≤2.0 | TK1 对准/交付速度 1.5 |
| 几何 | s_cur > wp_s(next_idx)+0.2 | TK2 改瞄 next_idx+1 |
| 几何 | 过点甩头：投影>len+0.2 且 dist<1.5 且无楼梯 | 瞄下一 wp，vx≤1.2 |
| 几何 | 4s 内 s 前进 <0.8（幻影 drop） | 下沿保护放行 2s |
| 几何 | 距 wp≤0.8 且 2s 距离变化 <0.03 | 悬停死锁跳过对准门 |
| 几何 | 5s 位移 <0.3m（z>1.2） | 卡死脱困：倒车 0.5（1.2s）→正东 0.8 前插（2.8s） |
| 姿态 | roll 0.34 触发 / 0.28 释放（窄脊 0.22/0.18） | om≤±0.3、vx≤0.3；>2s 低台倒车 −0.4；release 经 sync_applied 慢升 |
| 姿态 | |roll|>0.9 | 侧翻终止 |
| 姿态 | 出楼梯 |pitch|≤0.3、|roll|≤0.25、|vy|≤0.8、vx≤1.6 | STAIR 交还条件 |
| 姿态 | 判点对准门 |yaw−下一段|≤0.25 且 |ω|≤0.3 | 防抢跑（楼梯顶/平顶/悬停跳过） |
| 速度 | 出楼梯 0.6s | vx≤0.2、vyaw=0；超 5.0s 清除 |
| 速度 | 楼梯 2m 内且未跨骑 | 逼近全局 ≤1.5（STAIR_APPROACH_VX） |
| 速度 | DROP：ad<0.8 或跨骑 | vx≤0.3、|vyaw|≤0.5、cmd om≤±0.5 |

无绝对 xy 坐标门控：代码不含任何 x>某值 / y>某值 分支；注释中的具体坐标
（x=−4.79 柱、平台东角、wp14→15 缓坡等）均为历史失败记录，修复已写成通用规则。
仅剩的方向硬编码：卡死脱困前插阶段固定朝正东（yaw=0），以及 RL 观测编码
TARGET_HEADING=1.5708 默认值（STAIR 入口由 set_heading 覆盖为 lidar riser 航向）。

## 6. 时序预算（验收口径）

| 阶段 | 定义 | 预算 | 日志 |
| --- | --- | --- | --- |
| TK1 | 首次检测前方 riser（decel/TK1 激活）→ STAIR 交付 | <2.0s（含减速） | [TK1] 减速+对准 X.XXs / 对准 X.XXs |
| TK1 对准 | 交付圈内 |ey|>DB 首现 → 交付 | <1.0s | 同上 |
| 交付状态 | 交付瞬间 | body_vx≈1.5、|ey|≤0.20 | [RL-DIAG] takeover pos/yaw/max_leg_err |
| TK2 | 四轮过顶 → 对准下一 wp | <1.0s | [TK2] 上顶->对准 X.XXs |
| TMppi | 触发（dist<0.5, speed<0.8）→ |err|≤10° | 尽量 <1.0s | plan=TMppi 段 |
| post-stair hold | STAIR→CRUISE 后 | 0.6s 直线（vx≤0.2） | corr 字段 |

## 7. 模块细节

### 7.1 线控制器（只走线，不过弯）
- line_head = 段 heading − 1.0·clip(cte,±1)；段起点后方投影<−1m、或段首 1m 内
  横偏>0.8 且 z>1.2 → 直瞄段起点；dist_wp<0.5 → 直瞄 wp。
- vyaw = 一阶低通(clip(2.5·head_err, ±1.0), α=0.4) → SMppi 的 guide_om。
- vx = 4.0·√clip((dist_wp−0.2)/3.5, 0, 1)——sqrt 物理一致刹车剖面（线性剖面短段刹不住）。
- 已删除：head-err 降速、锐角预刹、MIN_VX 地板。

### 7.2 SMppi（BodyMPPI）
- 状态 [x,y,yaw,vx,vy,ω]；dt=0.05；N=1024、H=40（2s）；ADA=1。
- 采样中心 [v_ref, guide_om]；σvx=0.45 / σom=0.55。
- rollout 约束：摩擦锥 |vx·ω|≤μ·g（μ=代码默认 0.75，run 脚本未设 S10_MPPI_MU）、
  car_omega_limit 表（vx≤2→3.0；3→1.5；4→1.2；5→1.0）、加速度钳 3.5 m/s²、vx≥0。
- 成本：2.0·路径距离 + 0.8·速度偏差 + 0.5·guide 偏差 + 0.05·控制平滑 + 终点代价
  （wp_dx≤4.0 时 10·dist(末端,wp) + 10·(v_end−vref_t)²，vref_t 随 dx 线性归零）。
- 输出：vx slew ≤3.5·0.025；om slew 6.0·0.025；om≤min(2.5, μg/|v|, car_omega_limit, 1.0)；
  vx≤min(vx_max, guide_vx)；sync_applied 使 slew 基准=真实施加指令（门控释放不阶跃）。

### 7.3 TMppi（原地转向）
- 触发：dist<0.5（S10_TURN_ARRIVE_R，独立于判点半径）且世界速度<0.8（S10_TURN_V_MAX）
  且 |yaw_err|>10° 且 stair_ahead_dist 为空。
- 动作：vx=0.2，om=clip(3.0·err, ±3.0)；omcap=min(3.0, 1.8/max|vx|)，高台 z>1.2 再压 0.6。
- 释放：|err|≤10° 交回 SMppi。

### 7.4 CarVMC（巡航执行）
- 半蹲 S10_CAR_SQUAT=1（hipy∓1.10 / knee1.90，PRETRANS 期间切换高站姿）。
- 轮：v_ref=vx±ω·0.24；速度跟踪（τv=0.6）+ 差速 yaw 反馈（K=80）+ 摩擦前馈
  （S10_CAR_WHEEL_GF，抬轮期 0.5）；轮矩限 13.5 Nm；腾空 >0.02m 时 yaw 反馈衰减到 0。
- 腿：F=mg/4 + roll 分配(kp150/kd20) + pitch 分配(kp250/kd20) + 地形阻抗(kp_h300/kd_h60)；
  hipx 位置 PD+侧倾。
- 压弯 roll_tar=clip(−0.06·ω·|vx|, ±0.06)；z>1.0 关闭；roll 门控期 −2·roll 反向扶正（±0.25）。
- 地形处理：低通 0.4；高台钳制 min(terr, z−0.25)；坡顶前瞻 0.35m（升 0.05~0.25 预伸）；
  v595 骑坎找平；前轮抬轮前馈（仅锁存期、0.06~0.25m 台阶，后轴不抬）。

### 7.5 感知与 riser 检测
见 §4.2。补充要点：
- 多级/单级/下行三通道阈值：0.10 / 0.08 / 0.08（可 env 覆盖）。
- TK1 对准目标 = lidar 检测 riser 的路径航向（perc.stair_heading：≥4 级连续楼梯
  用地形 riser，否则 wall 通道 on-path 垂直面 ≥8 cells）。
- 单级台面爬升航向合成（stair_mode._climb_heading）：多级=riser1→末级轴；
  单级=riser→台面远沿（max drop > riser+0.8）；无远沿时按段末距离回退路径航向/下一 wp。

### 7.6 STAIR / TK1 / RL / PRETRANS
- 入口/出口/重入保护/drift-abort 全条件见 §5.2。
- TK1 只做对准（减速归 decel + 终点代价）：交付圈内 vx≤1.5；|ey|>0.20 时
  vyaw=clip(2.5·ey,±1.5)；圈外 decel 目标 2.0。
- PRETRANS（非 STAIR 时）：进楼梯前 ad≤3.0+1.5 blend 把 CarVMC 半蹲线性过渡到
  RL 高站姿；3.0m hold 内腿 PD 锁 default_dof（kp60/kd4/clip48）；出楼梯按 handback
  距离 2.0m 平滑交还。
- RL：前 200 步（≈1s）站姿 PD 锁腿、轮自由；policy 50Hz（decimation=4）；
  六级楼梯（riser≥3 级）set_cmd(1.2)，其余保持 1.5；每次 STAIR 入口用
  stair_first_heading 更新 heading 目标；riser 表 = lidar 在线（单级补虚拟第二级）。
- 交还时 CarVMC reset_state(vx,ω,roll,pitch) 清旧滤波器（防台顶打转/倒滑）。

### 7.7 TK2 / post-stair
- 出楼梯 0.6s：vx≤0.2、vyaw=0（防平台边缘 yaw 反冲）；>5.0s 清除慢速态。
- TK2：|yaw_err|>0.25 → vyaw=clip(2.5·err,±1.5)、vx≤1.2；≤0.25 打印耗时并释放；
  直接执行（om 上限 2.0，守 1.8/|vx|，高台 0.4）。
- post-stair 慢速瞄准：出楼梯后 1.0s 硬停（vx=om=0），之后距下一 wp>1.2m 或 2.5s 内
  vx≤0.6、om=clip(0.5·err,±0.2)；roll 门控期让位。

### 7.8 航点推进
- 判点：水平距 ≤0.3（S10_WP_ADVANCE_DIST；平顶 z>1.2 放大 2.5m）。
- 过点兜底：投影>len−0.5 且横向<0.8；投影>len+0.8 无条件推点（RL 斜爬出口）。
- 防抢跑对准门：到点且未明显越过时要求 |yaw−下一段|≤0.25 且 |ω_body|≤0.3；
  近点越过（距 wp≤0.8）同样要求；楼梯顶交还后、平顶、悬停死锁跳过。
- s 弧长判点兜底已删除（路径跟随器 s_cur 会跑飞，round218 实测）。

## 8. 关键参数（run_smppi_tmppi_cruise_rlstair_tk12.sh 实际值）

    S10_NAV_HZ=40 S10_WP_ARRIVE_R=0.2 S10_WP_ADVANCE_DIST=0.3 S10_WP_ALIGN_DB=0.25 S10_WP_ALIGN_OM=0.3
    S10_AUTO_VMAX=4.0 S10_LINE_VMAX=4.0 S10_LINE_YAW_GAIN=2.5 S10_LINE_YAW_MAX=1.0
    S10_LINE_BRAKE_DIST=3.5 S10_LINE_CTE_K=1.0
    VMC_MPPI_N=1024 VMC_MPPI_H=40 S10_MPPI_DT=0.05 S10_MPPI_CTRL_DT=0.025 S10_MPPI_ADA=1
    S10_MPPI_A_MAX=3.5 S10_MPPI_OMAX=2.5 S10_MPPI_W_GUIDE=0.5 S10_MPPI_W_DIST=2.0 S10_MPPI_W_HEAD=0.0
    S10_SMppi_STOP_DX=4.0 S10_MPPI_W_TPOS=10.0 S10_MPPI_W_TV=10.0
    S10_TURN_SPLIT=1 S10_TURN_ERR_DEG=10 S10_TURN_K=3.0 S10_TURN_OM_MAX=3.0
    S10_TURN_V_MAX=0.8 S10_WP_TURN_VX=0.2 S10_TURN_ARRIVE_R=0.5
    S10_CAR_SLIP_VX_GATE=0.5
    S10_ELEV_HZ=4 S10_LIDAR_WALL=1 S10_STAIR_SINGLE_RISE=0.08 S10_ELEV_DROP_TH=0.08
    S10_DROP_LOOKAHEAD=2.0 S10_DROP_VX=0.3 S10_TK1_MIN_CELLS=40
    S10_TK1=1 S10_TK1_LOOKAHEAD=5.0 S10_ELEV_ENTER=2.0 S10_ELEV_DECEL_VX=2.0 S10_STAIR_ENTER_DIST=2.0
    S10_TK1_VX=1.5 S10_TK1_YAW_DB=0.20 S10_TK1_YAW_K=2.5 S10_TK1_YAW_MAX=1.5
    S10_TK_OM_MAX=2.0 S10_TK_VX=1.5
    S10_RL_ELEV=1 S10_RL_POLICY=policy.pt S10_RL_VX=1.5 S10_RL_WARMUP=200
    S10_PRETRANS=1 S10_PRETRANS_ENTER_DIST=3.0 S10_PRETRANS_BLEND_LEN=1.5
    S10_PRETRANS_HOLD_DIST=3.0 S10_PRETRANS_EXIT_LEN=2.0
    S10_POSTSTAIR_HOLD_DIST=1.5 S10_TK2=1 S10_TK2_YAW_DB=0.25 S10_TK2_YAW_K=2.5
    S10_TK2_YAW_MAX=1.5 S10_TK2_VX=1.2 S10_STAIR_WHEEL_CLEAR=0.05
    S10_CAR_SQUAT=1 S10_CAR_WHEEL_GF=1.0 S10_VMC_KPH=300 S10_VMC_KDH=60
    S10_VMC_WHEEL_K=12.0 S10_VMC_WHEEL_D=0.02 S10_VMC_TERRAIN_LP=0.4
    S10_VMC_TERRAIN_LOOKAHEAD=0.35 S10_VMC_TERRAIN_AHEAD_W=0.6
    S10_CAR_KP_ROLL=150 S10_CAR_KD_ROLL=20
    S10_VMC_YAW_K_WHEEL=80 S10_VMC_OM_ABS_MAX=2.0 S10_VMC_OM_CAP=1.0
    S10_VMC_WHEEL_TMAX=13.5 S10_VMC_MU=0.8
    S10_TRAJ_DENSE=1 VMC_TRAJ=tmp_cruise_traj.npy

代码默认值（脚本未设、生效中的常用项）：
S10_AUTO_LAT_MAX=1.8、S10_TK1_ROUTE_ANGLE=0.45、S10_TK1_WP_MAX=2.5、
S10_TK1_CTE_MAX=0.8、S10_STAIR_ENTRY_LAT_MAX=1.0、S10_STAIR_EXIT_VX=1.6、
S10_STAIR_MIN_CLIMB_S=2.5、S10_STAIR_REENTRY_GUARD=1.0（代码内用 +2.0 弧长条件）、
S10_ELEV_STEP_TH=0.10、S10_ELEV_SEQ_SPAN=3.0、S10_ELEV_CLIMB_TH=0.4、
S10_ELEV_LAT_WIN=1.2、S10_ROLL_GATE_HI/LO=0.34/0.28、S10_DROP_OM_LOOKAHEAD=0.8、
S10_LIP_LATCH=1、S10_LIP_BURST_VX=1.2、S10_EDGE_LOOKAHEAD=1.2、S10_EDGE_VX=0.6、
S10_EDGE_CTE_MAX=0.8、S10_LINE_VYAW_LP=0.4、S10_PLAT_VX=1.8（工作区未提交改 5.0）、
S10_PLAT_OM=0.6、S10_POST_STAIR_HOLD_T=0.6、S10_POST_STAIR_MAX_T=5.0、
S10_POSTSTAIR_HOLD_T=2.5、S10_POSTSTAIR_HOLD_VX=0.6、S10_LIFT_HOLD_T=1.0、
S10_STUCK_TIMEOUT=90、S10_LIDAR_RAISE_Z=0.6、S10_LIDAR_NZ_MIN=0.6。

与旧版文档的三处修正（已按代码改）：
1. S10_TURN_V_MAX=0.8（旧写 0.3）。
2. S10_LINE_BRAKE_DIST=3.5 且刹车为 sqrt 剖面（旧写 2.5 线性）。
3. S10_MPPI_MU 未在启动脚本设置 → 生效 μ=0.75（旧写标定 0.36）——如需
   0.36 档需在脚本显式加 S10_MPPI_MU=0.36，见 §12。

## 9. 已删除 / 禁用清单

避障 costmap（整体删除）、god-view mj_ray 预扫描（删除）、硬编码楼梯表
STAIR_RISERS/TOPS（清空，lidar 在线检测替代）、航点 z 先验 step/stair 区域
（_precompute 内删除）、s 弧长判点兜底（删除）、CRUISE_TK wp4→5 特殊段、
head-err 降速、锐角预刹、MIN_VX 地板、S10_AUTO_STAIR_VX 死参数、
走廊偏移/对角 bump（S10_STAIR_CORRIDOR_X=0 / S10_START_CORRIDOR_X=0 /
S10_STAIR_DIAG_AMP=0）、RL 轮制动暖机（回退，非对称制动力矩放大 yaw 误差）。

## 10. 测试矩阵（执行顺序）

| # | 测试 | 配置 | 判据 |
| --- | --- | --- | --- |
| T1 | 修复后首跑 smoke | 40s, MAX_WP=5 | 无侧翻；过 wp1~2；实际拍=40Hz；plan max<25ms |
| T2 | 台面专项 wp4→5 | 80s, MAX_WP=5 | 上沿触发 STAIR(单级) 且 RL 上 12.5cm；台面 SMppi 巡航；下沿 DROP 慢爬不栽头；推进 wp5 |
| T3 | 单级 wp5→6 | 60s, MAX_WP=6 | 同 T2 单级路径 |
| T4 | 六级楼梯 wp6→7 | 100s, MAX_WP=8 | [TK1] 总<2s/对准<1s；RL 上 6 级；[TK2]<1s；交接日志正常 |
| T5 | 平台段 wp8→12 | 120s, MAX_WP=12 | 高台限速段通过，无侧翻 |
| T6 | 全量回归 | 600s, MAX_WP=33 | 33 点全过；力矩合规（腿48/轮13.5，连续超限<0.5s）；无侧翻；40Hz 保持 |
| T7 | 退化回归 | 重放 tune4 配置 | 确认 wp4 侧翻不回潮 |
| T8 | RL 单级 12.5cm eval（T2 前置） | sim2sim 20 seeds | 上步成功率>=80%；不达标则规划下行课程微调 |

traj（S10_TRAJ_DENSE=1，22 列）：t,x,y,z,yaw,next_idx,speed,mode(STAIR=1),roll,
rollrate,om_c,vx_c,terr中位,roll_tar,四轮 z(4),前后轮 qvel(2),前后轮 tau(2),vx_c,vx_body。

## 11. 当前状态（2026-08-19）

- 代码全部修改完成；py_compile、整链 import（cruise_main）、S10_track.xml 加载
  均已实测通过；尚未跑通全程，T1 是第一次验证。
- 修复前两轮 smoke 失败（wp1 绕圈 6s+高速过点侧翻 → 终点代价/STOP_DX 门控/TMppi
  提前 0.5m/判点角速度门；坡面 roll 0.4→4.0 阶跃侧翻 → roll 滞回+sync_applied）已落码。
- 工作区唯一未提交修改：cruise_main.py 平台限速 5.0（楼梯前 5m/跌落前 2.5m 回退
  1.8），已随本版文档一并提交。

## 12. 待确认事项

1. 模块命名口径：SMppi=走线 / TMppi=原地转（文档与实现一致）。
2. RL 单级 12.5cm 上步未验证（课程 T1d 曾删除）——T8 先行。
3. 下行 riser：当前 DROP 慢爬兜底（RL 未练下行）；若要 RL 接管下行需补课程微调。
4. S10_MPPI_MU：代码默认 0.75 生效中，旧文档称 μ 标定 0.36——需确认是否在
   启动脚本显式加 S10_MPPI_MU=0.36（影响摩擦锥转向上限）。
5. 平台限速 5.0（S10_PLAT_VX）为未提交实验值：T5/T6 若平台段不稳可回 1.8。