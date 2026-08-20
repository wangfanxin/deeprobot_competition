# SMppi/TMppi Cruise + RL-Stair + TK1/TK2 当前方案 v3（2026-08-20，与工作区代码对齐）

> 用户目标：微调 SMppi/TMppi/TK1/TK2，重点调试 CarVMC；
> 硬约束：①不得用航点 xyz 坐标做特殊干预（无 per-wp 分支，无世界系方向硬编码）；
> ②不得用世界 ray_cast 扫描高程图（感知只允许车载 lidar）。

## 1. 技能分工

| 技能 | 职责 | 关键点 |
| --- | --- | --- |
| SMppi | 直线走线/加速/到点减速 | BodyMPPI：N=1024、H=40（2s 视界）、终点代价、STOP_DX 线性归零 |
| TMppi | 近点原地转向（PD） | dist<0.6 且 speed<2.2 且 |err|>10°；om=k·err−kd·ω（停稳不甩） |
| CarVMC | 巡航执行器（200Hz） | 轮速 PID+差速 yaw+半蹲腿 PD+roll/pitch 环+地形跟踪 |
| STAIR(RL) | 台阶爬升 | policy.pt；riser 表=lidar 在线；PRETRANS 站姿切换 |
| TK1 | 楼梯前交接 | 只对准，不减速（减速归 SMppi 终点代价+decel） |
| TK2 | 楼梯后交接 | 四轮上顶→对准下一 wp→交回 |

## 2. 控制频率

| 层 | 频率 |
| --- | --- |
| MuJoCo | 200Hz（DT=0.005） |
| 执行 CarVMC/RL | 200Hz |
| 规划/模式 tick | 40Hz |
| RL policy | 50Hz（decimation=4） |
| lidar 高程图 | 4Hz 增量 |
| MPPI 视界 | 2s（H=40×dt=0.05） |

## 3. 数据管线

### 3.1 控制管线（40Hz tick 内按序合成，后写覆盖先写）

    33航点 -> nav.line [start/end/heading/dist_to_wp]
     ① 线控制器: line_head=段heading−CTE_K·clip(cte,±1)；段首后方/平顶段首横偏→瞄段起点；
                 dist<0.5→直瞄wp；vyaw=LP(clip(YAW_GAIN·head_err,±1.0),0.4)；
                 vx=VMAX·√clip((dist−0.2)/BRAKE_DIST,0,1)  # sqrt 刹车剖面
     ② 过点甩头: 过wp 0.2m 且 dist<1.5 且无楼梯 → 瞄下一wp, vx≤1.2
     ③ 航线夹角门: |riser爬升轴−线段heading|≤0.45 才放行台阶类门控
     ④ TK1: 楼梯可见+dist_wp≤2.5+|cte|≤0.8+z≤1.1 → 对准爬升轴，交付圈 vx≤1.5
     ⑤ TK2: 交还后(>2s 或 z≤1.15) → 瞄下一wp, vx≤1.2
     ⑥ 楼梯逼近限速: 楼梯≤2.0 → vx≤1.5（STAIR 接管不超速）
     ⑦ decel: vx=vx·(1−d)+dv·d（圈外2.0/圈内对齐1.2）
     ⑧ EDGE: 1.5m探针 rise 0.08~0.25 且平顶 → vx≤0.6+抬轮
     ⑨ STOP_DX=5.0: 距wp≤5m 内 v_ref 线性归零（SMppi减速长度，用户指示强化）
     ⑩ 参考路径: 航线投影点起、1m 间距、≤12m、末端精确=wp；wp_dx=dist_to_wp
     ⑪ 规划二选一:
          TMppi: dist<0.6 且 speed<2.2 且 |yaw_err|>10° → vx=0.2, om=clip(3·err−1.5·ω,±1.5)
          SMppi: BodyMPPI(v_ref, vyaw, wp_dx) → [vx,om]；A_MAX=4.5 减速能力
     ⑫ omcap: TMppi=min(1.5,1.8/max|vx|)；SMppi=min(1.0,1.8/|vx|)
     ⑬ post-stair hold 1.0s 硬停 / SEG0 / 锁存转向 / 大偏航 / TK 直接对准
     ⑭ cmd{vx,omega,roll_tar,...} → STAIR?RL:CarVMC → tau(16) → mj_step

### 3.2 感知管线（只用车载 lidar，无世界 ray_cast）

    LidarTerrainV2（4Hz；site 抬高0.6m；FOV±90°；96×48 地形射线；cutoff 20m）
      -> h=min-z / hmax=max-z；0.05m 栅格；法向|nz|≥0.6 滤竖直面；wall 通道(|nz|<0.4)
      -> build_local_tile: 16×16m hmax 瓦片 + step_flag
      -> _elev_rises_on_path（沿路径 0.1m 步距、横向 ±1.2m 最高剖面）:
           多级 riser: 跳变≥0.10+0.5m确认+≥2级+跨度≤3m+总爬升≥0.4
           单级 riser: 跳变≥0.08+0.2~0.6m 台面持续
           跌落沿: 下降≥0.08+低位持续（仅用于 STAIR 单级入口确认与日志）
      -> 航点裁剪 v2（用户指示）: 裁界=min(s_cur+8m, 下一航点s)，下限前方1.2m
      -> stair_rises_s/stair_ahead_dist/decel_request/stair_first_heading
      -> perc.riser_table → RLStairCtrl.set_risers(xy,tops,heading)

## 4. 门控总表（当前生效）

| 机制 | 条件 | 动作 |
| --- | --- | --- |
| TMppi(PD) | dist<0.6 且 speed<2.2 且 err>10° | vx=0.2, om=3·err−1.5·ω(±1.5) |
| SMppi | 其余 CRUISE | 采样规划+终点代价+STOP_DX=5.0 |
| 过点甩头 | 过wp 0.2m 且 dist<1.5 且无楼梯 | 瞄下一wp, vx≤1.2 |
| TK1 | 楼梯+dist_wp≤2.5+cte≤0.8+z≤1.1 | 对准爬升轴, 交付vx≤1.5 |
| TK2 | 交还后(>2s 或 z≤1.15) | 瞄下一wp, vx≤1.2 |
| 楼梯逼近限速 | 楼梯≤2.0 | vx≤1.5 |
| EDGE | 1.5m rise 0.08~0.25 平顶 | vx≤0.6+抬轮 |
| LIP锁存 | 楼梯存在 且 rise≥0.10 或跨骑≥0.08 | 瞄wp(2·err±1.0)+交还1s稳向 |
| SEG0 | 段首横偏>0.8 | 瞄段首, om±0.6, vx≤1.0 |
| 大偏航 | |cte|>1.0 | 直瞄投影点, om±1.2 |
| post-stair hold | 交还后 1.0s | vx=om=0；之后 1.2m/2.5s 内 vx≤0.6+慢瞄 |
| 卡死脱困 | 5s 位移<0.3 且 vx≥0.8 | 倒车0.5(1.2s)→沿当前航向前插0.8(2.8s) |
| 航点推进 | 判点圆0.5+近点对齐(航向<0.25且|ω|≤0.3)+悬停跳过 | next_idx+1 |
| 摔倒检测 | |roll|>0.9 或 z<0.12 | 终止 |

已删除（用户指示）: DROP 全部保护（爬行/中窗/深沿/om锁/ghost/卡死释放）、
roll 门控（0.34/0.28/窄脊0.22/0.18/反向压弯/死锁倒车）、z 分层硬编码、
next_idx≥4 门、世界系固定朝东逃逸、避障 costmap、god-view mj_ray、s 弧长判点兜底。

## 5. 关键参数（run 脚本当前值）

    SMppi: N=1024 H=40 dt=0.05 CTRL_DT=0.025 ADA=1 A_MAX=4.5 OMAX=2.5
           W_GUIDE=1.5 W_DIST=2.0 W_HEAD=2.0 STOP_DX=5.0 W_TPOS=10 W_TV=10 LAT_MAX=1.8
    TMppi: ERR_DEG=10 K=3.0 KD=1.5 OM_MAX=1.5 V_MAX=2.2 TURN_VX=0.2 ARRIVE_R=0.6
    线控:  VMAX=4.0 YAW_GAIN=2.0 YAW_MAX=1.0 CTE_K=0.4 BRAKE_DIST=3.5
    TK1:   LOOKAHEAD=8.0 ENTER=1.2 VX=1.5 YAW_DB=0.30 YAW_K=2.5 YAW_MAX=1.5 WP_MAX=2.5
    TK2:   YAW_DB=0.25 YAW_K=2.5 YAW_MAX=1.5 VX=1.2
    STAIR: SINGLE_RISE=0.10 ENTER_DIST=1.2 WHEEL_CLEAR=−0.02 REENTRY_GUARD=1.0
    post-stair: HOLD_DIST=0.9 HOLD_VX=1.0 HOLD_T=2.0
    CarVMC: KPH=300 KDH=60 WHEEL_K=12 WHEEL_D=0.02 TERRAIN_LP=0.4
            TERRAIN_LOOKAHEAD=0.35 TERRAIN_AHEAD_W=0.6 KP_ROLL=150 KD_ROLL=20 ROLL_K=0
            Z_DES_OFFSET=0.26 YAW_K_WHEEL=80 OM_ABS_MAX=1.6 OM_CAP=1.0
            WHEEL_TMAX=13.5 MU=0.8 SQUAT=1（hipy∓1.10/knee1.90）
    RL:    RL_VX=1.5 WARMUP=200 PRETRANS 3.0/1.5/3.0/2.0

## 6. 重点调试区：CarVMC 边界（用户指定方向）

已探索到但尚未展开的 CarVMC 旋钮：
1. 腿地形阻抗 KPH=300/KDH=60 —— 落地/过沿的腿吸收能力（前轮下落后腿伸长的跟踪带宽）；
2. 地形前瞻 TERRAIN_LOOKAHEAD=0.35 —— 前轮对台阶/落沿的预伸提前量（可加大到 1.0+ 预判落地）；
3. roll 环 KP_ROLL=150/KD_ROLL=20 与 pitch 环 KP_PITCH=250/KD_PITCH=20 —— 高速落地姿态吸收；
4. 轮力矩限 13.5Nm / 摩擦前馈 GF / 差速 yaw K=80 —— 高速转向不打滑的轮层边界；
5. hipx 位置 PD（KP_POSE/KD_POSE）与外摆站姿 —— 用户提示的左右髋外翻（±0.25 两方向均断链，幅度/组合待重新探索）；
6. 腾空检测 ground_f 的 onset/zero —— 落地瞬间差速/偏航反馈的保持。

## 7. 当前链状态与断点

wp0-7 通过（66.4s）。断点：wp7→8 平顶交还后 3.3m/s 横摆过冲侧翻（SMppi 高速转向稳定性）。
已按用户指示落码未测：SMppi 减速强化（A_MAX=4.5/STOP_DX=5.0）、DROP 删除、roll 门控删除、
TMppi 终端角速度阻尼、航点裁剪 v2、世界系固定朝东逃逸移除。
