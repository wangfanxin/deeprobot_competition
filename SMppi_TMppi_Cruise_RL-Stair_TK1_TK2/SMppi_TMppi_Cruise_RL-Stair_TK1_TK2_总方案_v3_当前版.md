# SMppi/TMppi Cruise + RL-Stair + TK1/TK2 当前方案 v4（2026-08-20，与工作区代码对齐）

> 验收（2026-08-20 用户更新）：每两航点间用时 <5s；该段直线距离 >5m 的放宽至 <8s。
> 整场超时 120s（S10_TEST_MAX_SIM=120），超过即判不合格。目标 wp0→33 全程通关。

## 0. 铁律（用户 2026-08-20）

1. 所有策略算法全局统一：禁止为局部路段/航点设置门控或特判，哪里过不去就调全局参数，不给局部开洞。
2. 不擅自增加/改变高维参数和门控，除非论证过"特别特别特别必要"。
3. 不得用航点 xyz 坐标做特殊干预（无 per-wp 分支、无世界系方向硬编码）。
4. 不得用世界 ray_cast 扫描高程图（感知只允许车载 lidar）。
5. 微调优先级：TMppi / SMppi / TK1 / TK2，重点调试 CarVMC 全局参数。

## 1. 技能分工

| 技能 | 职责 | 关键点 |
| --- | --- | --- |
| SMppi | 直线走线/加速/到点减速 | BodyMPPI：N=1024、H=40（2s 视界）、终点代价、STOP_DX 内 v_ref 线性归零 |
| TMppi | 近点原地转向（PD） | dist<0.6 且 speed<2.2 且 \|err\|>10°；om=3·err−1.5·ω（终端角速度阻尼，停稳不甩） |
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

## 3. 数据管线（40Hz tick 内按序合成，后写覆盖先写）

     33航点 -> nav.line [start/end/heading/dist_to_wp]
      ① 线控制: line_head=段heading−CTE_K·clip(cte,±1)；段首后方/平顶段首横偏→瞄段起点；
                dist<0.5→直瞄wp；vyaw=LP(clip(YAW_GAIN·head_err,±1.0),0.4)；
                vx=VMAX·√clip((dist−0.2)/BRAKE_DIST,0,1)  # sqrt 刹车剖面
      ② 过点甩头: 过wp 0.2m 且 dist<1.5 且无楼梯 → 瞄下一wp, vx≤1.2
      ③ 航线夹角门: |riser爬升轴−线段heading|≤0.45 才放行台阶类门控
      ④ TK1: 楼梯可见+dist_wp≤2.5+|cte|≤0.8+z≤1.1 → 对准爬升轴，交付圈 vx≤1.5
      ⑤ TK2: 楼梯交还后 >2s → 瞄下一wp, vx≤1.2（平顶等蹲姿过渡完成）
      ⑥ 楼梯逼近限速: 楼梯≤2.0 → vx≤1.5（STAIR 接管不超速）
      ⑦ decel: vx=vx·(1−d)+dv·d（圈外2.0/圈内对齐1.2）
      ⑧ EDGE: 1.5m探针 rise 0.08~0.25 且平顶 → vx≤0.6 + ≥10cm 前轮抬轮前馈
      ⑨ STOP_DX=5.0: 距wp≤5m 内 v_ref 线性归零（SMppi减速长度，用户指示强化）
      ⑩ 参考路径: 航线投影点起、1m 间距、≤12m、末端精确=wp；wp_dx=dist_to_wp
      ⑪ 规划二选一:
           TMppi: dist<0.6 且 speed<2.2 且 |yaw_err|>10° → vx=0.2, om=clip(3·err−1.5·ω,±1.5)
           SMppi: BodyMPPI(v_ref, vyaw, wp_dx) → [vx,om]；A_MAX=4.5 减速能力
      ⑫ omcap: TMppi=min(1.5,1.8/max|vx|)；SMppi=min(1.0,1.8/|vx|)
      ⑬ cmd{vx,omega,roll_tar,step_lift=0,...} → STAIR?RL:CarVMC → tau(16) → mj_step
           （post-stair hold/SEG0/LIP锁存/大偏航/卡死脱困等 cmd 级特判已全部删除，
             cmd 级转向只保留 TK1/TK2 对准）
      ⑭ 航点推进: 纯半径判点 nav.reached ≤ S10_WP_ADVANCE_DIST=0.45m，next_idx+1
           （无对准门、无过点兜底、无悬停豁免、无 post-stair 豁免——推进由 SMppi/TMppi 到点能力达成）
      ⑮ 摔倒检测: |roll|>0.9 或 z<0.12 → 终止；卡死超时 90s（距上次推点）→ 终止

## 3.2 感知管线（只用车载 lidar，无世界 ray_cast）

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
| 航线夹角门 | riser爬升轴与线段heading夹角>0.45 | 该台阶类门控不放行（路径外障碍） |
| TK1 | 楼梯+dist_wp≤2.5+cte≤0.8+z≤1.1 | 对准爬升轴, 交付vx≤1.5 |
| TK2 | 楼梯交还后>2s | 瞄下一wp, vx≤1.2 |
| 楼梯逼近限速 | 楼梯≤2.0 | vx≤1.5 |
| decel | decel_request>0 且 z≤1.1 | vx 混向 2.0（圈内对齐 1.2） |
| EDGE | 1.5m rise 0.08~0.25 平顶 | vx≤0.6 + ≥10cm 抬轮前馈 |
| STOP_DX | dist_wp≤5.0 | v_ref 线性归零 |
| 航点推进 | dist_wp≤0.45 | next_idx+1 |
| 摔倒检测 | \|roll\|>0.9 或 z<0.12 | 终止 |
| 卡死超时 | 距上次推点>90s | 终止 |

### 已删除（用户指示）

LIP 锁存（跨骑+decel 双触发/1.2m/s 冲量/cmd 锁存转向/释放保持/台沿抬轮）、
SEG0 平顶段首纠偏、大偏航 cmd 纠偏、post-stair hold 两级（0.2m/s 直行/零指令硬停）、
卡死脱困（倒车+前插）、判点对准门/过点兜底/悬停豁免/post-stair 豁免、
DROP 全部保护（爬行/中窗/深沿/om锁/ghost/卡死释放）、roll 门控（0.34/0.28/窄脊
0.22/0.18/反向压弯/死锁倒车）、z 分层硬编码、next_idx≥4 门、世界系固定朝东逃逸、
避障 costmap、god-view mj_ray、s 弧长判点兜底。

### 保留但非门控

楼梯交还状态 5s 超时清理（S10_POST_STAIR_MAX_T=5.0）：仅状态生命周期，
供 TK1 门/TK2 延迟/PRETRANS 退出混合读取，不做任何 hold 动作。
逐轮抬轮 step_lift 恒 0（原唯一写入者=已删除的 LIP 门控块），ground_f 降载支路随之惰化。

## 5. 关键参数（run 脚本当前值）

    SMppi: N=1024 H=40 dt=0.05 CTRL_DT=0.025 ADA=1 A_MAX=4.5 OMAX=2.5
           W_GUIDE=1.5 W_DIST=2.0 W_HEAD=2.0 STOP_DX=5.0 W_TPOS=10 W_TV=10 LAT_MAX=1.8
    TMppi: ERR_DEG=10 K=3.0 KD=1.5 OM_MAX=1.5 V_MAX=2.2 TURN_VX=0.2 ARRIVE_R=0.6
    线控:  VMAX=4.0 YAW_GAIN=2.0 YAW_MAX=1.0 CTE_K=0.4 BRAKE_DIST=3.5
    判点:  WP_ADVANCE_DIST=0.45（纯半径；0.2 实测漏点甩头侧翻，t=25s wp1）
    TK1:   LOOKAHEAD=8.0 ENTER=1.2 VX=1.5 YAW_DB=0.30 YAW_K=2.5 YAW_MAX=1.5 WP_MAX=2.5
    TK2:   YAW_DB=0.25 YAW_K=2.5 YAW_MAX=1.5 VX=1.2
    STAIR: SINGLE_RISE=0.10 ENTER_DIST=1.2 WHEEL_CLEAR=−0.02 REENTRY_GUARD=1.0 EXIT_VX=1.0
    交还:  POST_STAIR_MAX_T=5.0（状态清理）
    CarVMC: KPH=300 KDH=60 WHEEL_K=12 WHEEL_D=0.02 TERRAIN_LP=0.4
            TERRAIN_LOOKAHEAD=0.35 TERRAIN_AHEAD_W=0.6 KP_ROLL=150 KD_ROLL=20 ROLL_K=0
            Z_DES_OFFSET=0.26 YAW_K_WHEEL=80 OM_ABS_MAX=1.6 OM_CAP=1.0
            WHEEL_TMAX=13.5 MU=0.8 SQUAT=1（hipy∓1.10/knee1.90） PLAT_VX=2.5
    RL:    RL_VX=1.5 WARMUP=200 PRETRANS 3.0/1.5/3.0/2.0

## 6. 各段长度与验收表（wp 坐标来自 track XML，2026-08-20 生成）

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

## 7. 重点调试区：CarVMC 边界（用户指定方向）

1. 腿地形阻抗 KPH=300/KDH=60 —— 落地/过沿的腿吸收能力（前轮下落后腿伸长的跟踪带宽）；
2. 地形前瞻 TERRAIN_LOOKAHEAD=0.35 —— 前轮对台阶/落沿的预伸提前量（可加大到 1.0+ 预判落地）；
3. roll 环 KP_ROLL=150/KD_ROLL=20 与 pitch 环 —— 高速落地姿态吸收；
4. 轮力矩限 13.5Nm / 摩擦前馈 GF / 差速 yaw K=80 —— 高速转向不打滑的轮层边界；
5. hipx 位置 PD 与外摆站姿 —— 左右髋外翻（±0.25 两方向均断链，幅度/组合待重新探索）；
6. 腾空检测 ground_f 的 onset/zero —— 落地瞬间差速/偏航反馈的保持。

## 8. 测试流程（不停机模式）

每轮：跑一次 120s 全链（S10_TEST_MAX_SIM=120）→ 分析断点与 [T]/[ADV] 日志
→ 只动一个全局参数（优先级 SMppi/TMppi/TK1/2，重点 CarVMC）→ 立即重跑；
每小时向用户汇报；轨迹 xy 图（颜色=速度）随轮次传 git。

## 9. 当前链状态与断点

- 大删除后首测（判点 0.2m，提交 35803da）：t=25.0s wp1→2 坡道侧翻
  （roll −1.29，cmd 3.5m/s）。根因：0.2m 判点圆漏 wp1（最近 0.34m），
  机器人持续瞄身后 wp1、yaw 1.36↔2.59 摆振侧翻。
- 修复：推进半径 0.45（提交 0cebc37），r2 测试进行中。
- 历史：删除前链状态 wp0-7 通过 66.4s（wp7→8 平顶 3.3m/s 横摆侧翻），
  大删除后链状态未知，需从头重建。
