# SMppi/TMppi Cruise + RL-Stair + TK1/TK2 当前方案 v5（2026-08-20，与 r74 工作区对齐；非停机测试循环启用）

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
      ⑨ STOP_DX=3.5: 距wp≤3.5m 内 v_ref 线性归零（SMppi减速长度，r42 实测链值）
      ⑩ 参考路径: 航线投影点起、1m 间距、≤12m、末端精确=wp；wp_dx=dist_to_wp
      ⑪ 规划二选一:
           TMppi: dist<0.6 且 speed<2.2 且 |yaw_err|>10° → vx=0.2, om=clip(3·err−1.5·ω,±1.5)
           SMppi: BodyMPPI(v_ref, vyaw, wp_dx) → [vx,om]；A_MAX=4.5 减速能力
      ⑫ omcap: TMppi=min(1.5,1.8/max|vx|)；SMppi=min(1.0,1.8/|vx|)
      ⑬ cmd{vx,omega,roll_tar,step_lift=0,...} → STAIR?RL:CarVMC → tau(16) → mj_step
           （post-stair hold/SEG0/LIP锁存/大偏航/卡死脱困等 cmd 级特判已全部删除，
             cmd 级转向只保留 TK1/TK2 对准）
      ⑭ 航点推进: 纯半径判点 nav.reached ≤ S10_WP_ADVANCE_DIST=0.6m，next_idx+1
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
| TMppi(PD) | 距当前wp<1.2 或 距下一wp<S10_TURN_NEXT_R(默认0关) 且 speed<4.5 且 err>15° | vx=0.2, om=3·err−1.5·ω(±1.5) |
| SMppi | 其余 CRUISE | 采样规划+终点代价+STOP_DX=5.0 |
| 过点甩头 | 过wp 0.2m 且 dist<1.5 且无楼梯 | 瞄下一wp, vx≤1.2 |
| 航线夹角门 | riser爬升轴与线段heading夹角>0.45 | 该台阶类门控不放行（路径外障碍） |
| TK1 | 楼梯+dist_wp≤2.5+cte≤0.8+z≤1.1 | 对准爬升轴, 交付vx≤1.5 |
| TK2 | 楼梯交还后>2s | 瞄下一wp, vx≤1.2 |
| 楼梯逼近限速 | 楼梯≤2.0 | vx≤1.5 |
| decel | decel_request>0 且 z≤1.1 | vx 混向 2.0（圈内对齐 1.2） |
| EDGE | 1.5m rise 0.08~0.25 平顶 | vx≤0.6 + ≥10cm 抬轮前馈 |
| STOP_DX | dist_wp≤3.5 | v_ref 线性归零 |
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

## 5. 关键参数（run 脚本当前值，2026-08-20 r42 实测链）

    判点:  WP_ADVANCE_DIST=0.6（纯半径；0.62 差 0.02 漏点曾致 wp6 回找侧翻）
    线控:  VMAX=4.0 YAW_GAIN=2.0 YAW_MAX=1.0 CTE_K=0.4 BRAKE_DIST=3.0 VYAW_LP=0.4
    TMppi: ERR_DEG=15 K=3.0 KD=1.5 OM_MAX=1.5 V_MAX=4.5 TURN_VX=0.2 ARRIVE_R=1.2
           omcap 用实际速度（vx_c 帽致 3.5m/s 硬转打滑的根因）
    TK1:   WP_MAX=3.5 YAW_DB=0.45 YAW_K=2.5 YAW_MAX=1.5 VX=1.0 LOOKAHEAD=8.0
           方向门：爬升轴 vs 航线 <=0.45 rad；仅 SMppi 模式进入（_tmppi_will 门）
    TK2:   交还延迟 0.5s YAW_DB=0.25 YAW_K=2.5 YAW_MAX=1.2 VX=1.2
           omcap 用实际速度；仅 SMppi 模式进入
    STAIR: ENTER_DIST=2.0（瓦片退化前交 RL）EXIT_VX=1.0 SINGLE_RISE=0.10
           WHEEL_CLEAR=-0.02 REENTRY_GUARD=1.0；多级兜底退出 span+2.0
           虚拟 riser 合成：任意级数+远沿 drop 补末级
    SMppi: N=1024 H=40 dt=0.05 CTRL_DT=0.025 ADA=1 A_MAX=8.0 OMAX=2.5
           W_GUIDE=2.5 W_DIST=2.0 W_HEAD=2.0 STOP_DX=3.5 W_TPOS=10 W_TV=10
           LAT_MAX=3.6 MU=0.36（规划物理=标定摩擦）
    CarVMC: KPH=300 KDH=60 WHEEL_K=12 WHEEL_D=0.02 TERRAIN_LP=0.4
            TERRAIN_LOOKAHEAD=0.35 TERRAIN_AHEAD_W=0.6 KP_ROLL=150 KD_ROLL=20
            ROLL_K=0.05 ROLL_AMP=0.10 Z_DES_OFFSET=0.26 YAW_K_WHEEL=80
            OM_ABS_MAX=1.6 OM_CAP=1.0 WHEEL_TMAX=13.5 MU=0.8 SQUAT=1
    RL:    RL_VX=1.5 WARMUP=200 PRETRANS 3.0/1.5/3.0/2.0（六级台阶默认速度爬升）
    感知:  ELEV_HZ=4 LOOKAHEAD=8.0 CLIMB_TH=0.2（sh）裁剪=min(s+8,下一航点s)

## 5b. 参数演进关键记录（r 轮次归档）

    r4  MU 0.36 + LAT_MAX 3.6：wp1 角部打通（4m/s 时 omcap 0.45 是宽弧根因）
    r8  判点 0.6：wp1/wp2 推进
    r11 TK1 方向门 + ROLL_K 0.05：wp3 平台东角 2.85s 攻破
    r13 TK omcap 实际速度：RL 交还 3.55m/s 硬转打滑修复
    r15 TK1_WP_MAX 2.5 + YAW_DB 0.45：wp5-6 两级台阶通过
    r16 TMppi omcap 实际速度：wp6 前 0.7m 硬转打滑修复
    r20 判点 0.62 尝试：wp2 走廊恰在 0.60-0.62，提前推点翻转角部→回 0.6
    r27 TK2 交还延迟 2.0->0.5：wp6 首次推进
    r29-31 ELEVPROF 取证：六级台阶瓦片只读前 2 踏面（上方读废值）
    r33 ENTER_DIST 2.0：六级台阶首次交 RL（1.2 时检测已死）
    r37 TK1_VX 1.0：wp0-6 全腿时达标（27.5s），六级台阶登顶 z1.26，
         RL 顶台 7m/s 飞跃翻滚（历史断点）
     r54 感知组合（桅杆1.0+判点0.7+TURN_V_MAX3.0+W_GUIDE3.0）：wp0-5 全腿达标 21.6s
     r58 平顶 ground_f 取证：lift 0.0029~0.088 振荡→gf 触 0→yaw 反馈全灭=wp7-8 滑旋根因；
          腾空灵敏度参数化 onset 0.01→0.03 / zero 0.03→0.10（r69 回退默认，保留钩子）
     r66-67 riser 表稠密重采样修复六级台阶：wp7 首次推进；r67=TMppi 交还期转体生效，
          wp0-7 33.8s 全腿达标，新断点=wp7-8 平顶 3.8m/s 偏航滑转（yaw 2.63→0.53 roll-1.34）
     r70 平顶 gf 方向1 PRETRANS_EXIT_LEN 2.0→1.0：wp2-3 两级台阶出梯混合翻转→回退
     r71 平顶 gf 测试1 VMC_TERRAIN_LP 0.4→0.8：lift 尖峰削平、gf 不触 0，
          但腿地形全局耦合 wp1-2 翻转→回退
     r73 平顶 gf 候选2 接触力次级信号（cfmax>0 锁 gf=1.0）：机制生效但 wp0-1 重掷 6.1→9.8s、
          wp3 拐角轮跳极限环卡死→回退
     r74 候选2 细化（四轮全接触 cfmin>0 才锁）：wp0-1 9.36s、wp3 同 r70/r72 模式侧翻；
          GF 取证 cfmin=0 占腿段绝大多数（巡航轮子频繁卸载/跳跃）→接触门改早期腿动力学→拐角连锁；
          平顶修复无法经全链验证。代码保留 HEAD，测试起点=S10_GF_CF_THR=1e9（等效 r67 基线）

## 5c. 原参数表（历史基线）


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
6. 腾空检测 ground_f 的 onset/zero —— 落地瞬间差速/偏航反馈的保持（r69-r74 三路线已探明结论，见 §9）。

## 8. 测试流程（不停机模式）

每轮：跑一次 120s 全链（S10_TEST_MAX_SIM=120）→ 分析断点与 [T]/[ADV] 日志
→ 只动一个全局参数（优先级 SMppi/TMppi/TK1/2，重点 CarVMC）→ 立即重跑；
每小时向用户汇报；轨迹 xy 图（颜色=速度）随轮次传 git。

## 9. 当前链状态与断点（r75 实测，2026-08-20）

- **交付基线 = r67 链**：wp0→7 全通（wp0-7 33.8s；wp0-6 全腿达标
  6.1/3.2/4.4/3.6/4.8/5.4s，wp6-7 台阶段+TMppi 交还期转体 @37.02s 推点）。
  判点 0.6m；验收 5s（>5m 段 8s）。
- **唯一历史断点：wp7→8 平顶**：出梯 3.8m/s，yaw 2.63→0.53 反向滑旋 roll-1.34
  （距 wp8 仅 1.6m）。根因（r58 取证）：平顶 terr 参考抖动+站姿偏差使
  _lift_amt 0.003~0.088 振荡 → ground_f 触 0 → 差速+yaw 反馈全灭 →
  转弯转不动/直线漂移滑旋。
- **ground_f 修复三条路线已全部探明（r69-r74，均回退或保留待判）**：
  1. 站姿偏差源（PRETRANS_EXIT_LEN 2.0→1.0）：wp2-3 两级台阶出梯混合动力学翻转（r70）。
  2. 轮地参考平滑（TERRAIN_LP 0.4→0.8）：lift 尖峰削平、gf 不触 0，
     但腿地形全局耦合 wp1-2 翻转（r71）。
  3. 接触力次级信号锁 gf：任一接触锁（r73）/四轮全接触锁（r74）两版都会
     改变早期腿动力学（GF 取证 cfmin=0 占腿段绝大多数——巡航轮子本就频繁
     卸载/跳跃）→ wp0-1 重掷 + wp3 拐角连锁。
- **r75 实测（S10_GF_CF_THR=1e9=禁接触门，即 r67 等效配置）**：wp1@10.47s、
  wp2@14.12s、wp3 拐角 t=17s 侧翻 roll-2.57（(-12.7,16.6)，lift 0.028→0.08→0.305
  时 gf 0.095→0→0，yaw 反馈全灭后侧滑翻转）。r75b 逐位复现（wp0 1.11/wp1 10.47/
  wp2 14.12/同点翻转）——仿真确定论。**说明 r67 的成功是一次性轨迹，同配置重跑
  不再复现；当前首要断点实际是 wp3 拐角抬轮 gf=0 连锁**。
- **下一步（非停机测试循环，§10）**：wp3 拐角优先——ground_f onset/zero 全局微调
  （小步进；r69 的 0.03/0.10 曾毁 wp1-2，须避开）；随后回 wp7-8 平顶。

- **r143-r182 门控与参数扫描**：TURN_NEXT_R/TURN_HOLD/HEAD_VX_K 三个全局门控（默认关）
  全部实测负收益；**r163（TERRAIN_AHEAD_W 0.6→0.4）= 当前最远 wp9@61.25**，
  wp8-9 下沿翻已修（2.28s），新断点=wp9-10 平顶直道 74° 大误差+2.5m/s 转弯 roll1.49 翻。
  剩余未达标腿：wp0-1 8.12s、wp3-4 10.76s、wp6-7 8.83s、wp7-8 15.7s。

- **r76-r124 扫描结论（进行中）**：
  - gf onset=0.03 必毁 wp1-2；onset 0.025/zero 0.06 + TK2_YAW_MAX 1.0（r100）= 当前最远，
    首过 wp7-8 平顶至 wp8@58.28（平顶 gf 恒 1.0，历史 gf=0 根因已灭）。
  - 剩余三大断点：① wp3-4 ±π 航向自旋卡 10.8s（多配置复现，仅 ROLL_K 0.03 与 gf 0.02/0.06 避开，
    但两者毁 wp5-6 转弯——压弯入弯在 wp5-6 承重）；② wp7-8 平顶 15s 慢爬（roll 摇摆吃速度）；
    ③ wp8-9 下沿 roll-1.33 翻。
  - run 脚本已全量 export 改 K:-V 可覆盖（默认不变，r94 sanity 逐位一致）。

## 9b. 历史断点与攻克记录（r42，2026-08-20）

- **wp0→6 全通且全腿达标（27.5s）**：wp0-1 6.1s / wp1-2 3.2s / wp2-3 4.4s /
  wp3-4 3.6s / wp4-5 4.8s / wp5-6 5.4s（验收 5s，>5m 段 8s）。
- **断点：wp6→7 六级台阶顶台**：RL 登顶 z1.26 后 7m/s 飞跃翻滚（roll −2.05）。
  根因链：lidar 瓦片远单元欠填充（掠射余量 1-3cm，上方踏面读废值）→ riser
  表退化 2/6 级 → RL 出分布，顶台步态暴力；post-stair hold 已删，无动量吸收。
  修复候选：(a) lidar 桅杆 0.6→1.0m（实测会使 wp2-3 角部翻转，需重调链）；
  (b) 最小化全局交还动量阻尼（待用户确认是否恢复，证据 r37/r41 7m/s 翻滚）。
- 历史断点已全部攻克：wp1 角宽弧（MU/LAT）、wp1/wp2 漏点（判点 0.6+TMppi
  高速触发）、平台东角（TK1 方向门+ROLL_K）、wp5-6 台阶（YAW_DB+ENTER）、
  wp6 漏点（TK2 延迟 0.5s）、六级台阶入口（ENTER 2.0）。

## 10. 非停机测试循环（2026-08-20 用户指示，目标 wp0→33）

- **循环**：每轮跑 120s 全链（S10_TEST_MAX_SIM=120、S10_AUTO_MAX_WP=33）→
  解析 [T] wp@t 逐腿用时与断点 → 只动一个全局参数 → 立即重跑；
  轨迹图 test_r*.png 随轮次入库。
- **参数纪律**：不新增/改变高维参数与门控，除非三重论证「特别特别特别必要」；
  所有策略算法全局统一，禁止 per-wp/per-段门控；只用已存在参数的数值微调。
- **微调优先级**：TMppi / SMppi / TK1 / TK2 微调；重点 CarVMC 全局参数（§7 清单）。
- **合理门控（用户 2026-08-20 更新授权）**：允许加入全局统一、条件连续的门控；禁止 per-wp/per-段特判。
  已加入：S10_TURN_NEXT_R 近下一航点低速转向门（tmppi.py，默认 0=关，对全部 wp 统一生效）。
- **验收**：每两航点间 <5s；段直线距离 >5m 放宽 <8s；整场 <120s；力矩合规；无侧翻。
- **汇报**：每半小时不停机汇报（断点、逐腿用时、本轮改动）；图随轮次传 git。
