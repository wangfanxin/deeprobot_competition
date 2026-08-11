> **已废弃（2026-08-11）**：本文为待审草案，已被终版取代。
> 终版见 [stair_StairWBC_终版_20260811.md](stair_StairWBC_终版_20260811.md)（无 MPPI、QP Checker 非分配、4 轮独立布尔相位）。
# Stair 技能修订方案：OCS2/WBC 系 · 位置基全身控制（2026-08-11）

> 状态：**方案待审**（用户审核通过后再开始改代码）
> 仓库：`0810new/deeprobot_competition`，写作时 HEAD = `dc4ec68`（v829）
> 本文定位：在既有 stair 实现（v822→v828 位置基地基）之上，对照 OCS2/WBC 系开源代码，
> 给出楼梯段执行层的**架构级修订方案**：位置基全身控制（IK 放轮 + body 姿态闭环 + QP 力分配）
> + 布尔几何相位硬切换 + 轮矩开环限幅。
>
> 配套文档：
> - 现状总结：[stair_技能方案与数据管线_20260811.md](stair_技能方案与数据管线_20260811.md)
> - 机制分析：[stair_机制分析_20260811.md](stair_机制分析_20260811.md)
> - 文献调研：[stair_文献调研_跑酷与步态_20260811.md](stair_文献调研_跑酷与步态_20260811.md)

## 一、为什么必须修订（问题回顾）

### 1.1 已落地成果（v822→v828）

- **布尔几何相位状态机**（`S10_STAIR_POSMODE=1`）：前/后轴 |d|<0.05 硬切换抬放，替代 sin² CPG 连续窗；
- **FootPlaceVMC 位置模式**（`S10_FP_POSMODE=1`）：全增益 PD + 支撑腿静压（`S10_FP_PRESS`）+ 抬升腿轮矩 0；
- **body 姿态闭环**（v826/v827）：期望 body z/pitch 由 4 轮目标解算 + 误差修正 + body-z 速率阻尼；
- **v828 里程碑**：支撑腿改单侧垂直阻抗（力控只做阻抗），消除 body 弹射（bz 稳定 0.60–0.65、pitch≈0、fn 峰值 110N）。

### 1.2 死锁根因：用 WBC 力控做"接触切换"

1. **接触切换本质是离散事件**：前轮离地瞬间，接触点从地面变空中，摩擦锥约束消失，6D wrench 分配的数学基础崩坏；
2. **连续 climb_mask 软切换失效**：`climb_mask` 是 0~1 连续量，力控闭环用软权重模拟硬切换——前轮离地瞬间 pinv/QP 优先满足"腿力最小"，直接把前轮抬力削掉；
3. **抬轮用"力控+补偿"自动收腿**：轮悬空后地形阻抗项变负（轮高>目标），力控自动收腿——力控永远打不过"轮悬空→目标高于实际"这个反馈；
4. **后轮推力没有静压保证**：靠力控下压，body 俯仰变化后实际下压力 ≠ 目标，N 无法恒定。

### 1.3 文献共识（ETH Bjelonic / IIT Centauro / 哈工大 2025）

- 接触切换是**硬相位**（离散 mode schedule），WBC 按 mode 分配力，不是力控涌现；
- 抬轮是**位置控制**（目标 z 由几何相位给定，PD 增益拉满），力控只做阻抗（±2cm 柔顺）；
- 后轮推力来自**轮地接触**（轮驱 + 腿压地），腿压地是位置控制下的静压，不是力控闭环；
- 轮矩**开环限幅**（μN·R 或 T_max），不做力控钳制。

**结论**：楼梯段执行层从"WBC 力控"切换到"位置基全身控制"，属架构级改动，是唯一能过楼梯的路。

## 二、参考仓库审阅结论（抄什么、差什么）

| 仓库 | 派系 | 可借鉴点 | 与本项目差距 |
|---|---|---|---|
| ADVRHumanoids/wb_mpc_centauro | OCS2 系，轮足原生 NMPC+WBC | `ModeSequenceTemplate.h`（切换时刻+mode 序列）、`SwitchedModelReferenceManager.h`（mode 驱动参考切换）、`CoordinateVelocityConstraintCppAd.h`（Pfaffian 轮约束：法向速度=0、切向滚动 v=ωR）、`MotionPhaseDefinition.h`（MODE 位掩码 FLY/STANCE/各腿组合） | URDF 是 CENTAURO 需换机体；ROS1 重 |
| leggedrobotics/ocs2（ocs2_legged_robot） | Bjelonic 轮足论文母库 | `GaitSchedule`/`ModeSequence`（楼梯相位=布尔开关序列）、ZeroVelocity/FrictionCone 约束改 Pfaffian 版、NMPC 100Hz + WBC 500Hz 双频接口 | ANYmal-W 专用脚本未全开，通用模块全有 |
| qiayuanl/legged_control | OCS2+Pinocchio+QP WBC，Go2 近亲 | `legged_wbc/src/WeightedWbc.cpp`：QP 加权任务（baseAccel/swingLeg/contactForce）+ 约束（EOM/力矩限幅/摩擦锥/noContactMotion）；NMPC 100Hz + WBC 500Hz 接线 | 原版无轮，轮足改造 1–2 人周 |
| ADVRHumanoids/casannis_walking | CasADi+IPOPT WBC | 轻量替代 OCS2 编译，轮足接触模型一致 | 需自行接线 |
| rl_sar / robot_lab（Go2W IK） | 非 RL IK 放腿 | FootPlaceVMC 已比它们贴近位置基全身控制 | 缺 body 6D IK 闭环 + QP 位置约束 |
| 哈工大 2025 J. Bionic Eng | 无公开代码，架构照抄 | 高程图→序列化节点生长落点（轮/腿模式自适应）→SRBD QP→NMPC+WBC，规划 10ms | stair_world 表已完成前半段，缺 SRBD QP |

**核心抄取**：接触时序用离散 ModeSequence（布尔），轮约束用 Pfaffian（v=ωR），
力分配用 QP（带摩擦锥/力矩限幅硬约束），抬轮用位置 PD 拉满。

## 三、目标架构

```
感知层（lidar 高程图 10Hz + stair_world 预扫描）
   ↓  riser 表 / 台面顶高 / 显式轮心 z 参考
规划层（AutoNavFollower 纯 xy，20Hz）
   ↓  vx/vyaw 指令 + MPPI 质心轨迹优化（20Hz）
执行层（StairWBC：位置基全身控制，200Hz）
   ├─ ModeSchedule：布尔几何相位硬切换（前/后轴抬放）
   ├─ 抬升腿：位置 PD（台面顶 + R），力控只做阻抗
   ├─ 支撑腿：位置 PD + 静压（台面顶 + R − PRESS）
   ├─ body：QP 解接触力 + 姿态闭环（z/pitch/roll）
   └─ 轮：开环限幅（支撑差速 PID 限 ±13.5Nm，抬升 0）
Mujoco 仿真
```

### 旧 vs 新执行层差异

| 维度 | 旧（力控 WBC，v828 前） | 新（位置基全身控制） |
|---|---|---|
| 接触切换 | 连续 climb_mask 软权重 | ModeSchedule 布尔硬切换 |
| 抬轮 | 力控目标 kp·Δz，离地即自动收腿 | 位置 PD 拉满，目标=台面顶+R，只留 ±2cm 阻抗 |
| 后轮保载 | 力控下压（`S10_VMC_WBC_PRESS`） | 位置静压（目标 −5mm，PD 拉满，硬顶台面） |
| 轮矩 | 力控钳制 | 开环限幅（支撑 ≤13.5Nm，抬升腿 0） |
| 姿态保持 | 力控 wrench 分配（离地瞬间失效） | QP 只分配接触力，姿态由位置环保证 |

## 四、数据管线与控制频率

| 环节 | 频率 | 内容 |
|---|---|---|
| lidar 高程图 | 10Hz | 局部栅格更新，提供台面顶高/棱位置 |
| stair_world 预扫描 | 启动一次 | riser 弧长→路径点+切线+台面顶高（含显式轮心 z 参考） |
| 导航 AutoNavFollower | 20Hz | 纯 xy 路径 + vx 参考（v829 架构铁律：禁止 z 先验） |
| MPPI | 20Hz | N=1024~2048、H=40，单次实测 ~12ms（峰值 47ms），在 50ms 周期内 |
| StairWBC | 200Hz | 每步：①几何相位布尔量 ②IK 轮目标 ③QP 解接触力 ④腿 PD + 轮矩 |
| QP 求解 | 200Hz 内 | osqp，12 决策变量 ≈1ms（预算 <3ms） |
| 本机实时比 | ≈0.53 | 瓶颈是 mujoco 单步+Python，不是 MPPI/QP；真机控制环按 200Hz 执行 |

## 五、StairWBC 具体设计（WBC QP 怎么实现）

### 5.1 决策变量与动力学

- 决策变量 **λ ∈ R¹²**：4 轮世界系接触力（F_x, F_y, F_z）× 4；
- 单刚体动力学（SRBD）：**a_base = A(q)·λ + b0(q, v)**，A 为接触支撑矩阵，b0 含重力/科氏项；
- a_des：body z/pitch/roll 由姿态闭环给出；vx/vyaw 由 MPPI/导航给出。

### 5.2 代价函数

```
min_λ  w1·‖Aλ + b0 − a_des‖² + w2·‖λ − λ_ref‖² + w3·‖轮矩增量‖²
```

- λ_ref：支撑腿 mg/4 均载（或按 mode 前/后轴分配）；
- 第三项平滑轮矩，防止抖振。

### 5.3 约束（硬约束，不是软权重）

- **支撑腿**：λ_z ≥ N_min（静压保载）；|λ_xy| ≤ μ·λ_z（摩擦锥，μ 取 0.5~0.8）；
- **抬升腿**：λ ≡ 0（mode 硬约束——对应 OCS2 ModeSequence 的 FLY 位）；
- **腿力矩**：τ_leg = Jᵀ·λ，|τ| ≤ 48Nm（合规）；
- **轮矩**：支撑轮差速 PID 输出限 ±13.5Nm；抬升轮 0。

### 5.4 腿层：位置控制 + 阻抗

- **抬升腿**：位置 PD（kp 拉满，`S10_FP_KP_POS`），参考 = `stair_wheel_ref(y)`（台面顶+R）；
  地形阻抗项只做 ±2cm 柔顺，不主导位置；
- **支撑腿**：位置 PD + 静压（目标 −5mm，`S10_FP_PRESS`），PD 拉满——位置控制下轮"硬顶"台面，
  接触力自然够，不需要力控闭环下压；
- **关键原则**：位置环主导，力控只做阻抗（柔顺），二者方向不再矛盾。

### 5.5 轮层：开环 + 限幅

- 支撑轮矩 = clip(差速 PID 输出, ±13.5Nm)；前轮离地期间前轮矩 = 0，全部推力给后轮；
- 不用 `S10_VMC_WBC_LIFT_TMAX` 做力控钳制（位置控制下腿不会顶 body，轮矩可开到上限）。

### 5.6 几何相位硬切换（ModeSchedule）

- **前轴抬**：前轴到最近棱 |d| < 0.05m → 前腿立即切位置控制，目标 z = 台面顶 + R；
- **前轴放**：|d| > 0.05m 且轮高 ≥ 台面顶 + R；
- **后轴抬**：前轴落地后 0.1m 弧长 → 后腿切位置控制，目标 z = 台面顶 + R；
- sin² 窗**只生成位置目标轨迹**，不控制切换逻辑（对应 OCS2 `ModeSequence`）；
- 切换为布尔量，可加 ±0.02m 滞回防抖振。

## 六、实现步骤（待批准后执行）

1. 读 S10 模型参数（质量/惯量/轮半径/轴距），新建 `s10_mpc/stair_wbc.py`：
   osqp QP（12 变量）+ 解析 IK + body 姿态闭环；
2. **平地验证**：4 轮位置跟踪 + 静压 + body z/pitch 保持；
3. **ModeSchedule 验证**：单级 riser（首级 0.063m）→ 逐级推进；
4. **接入主循环**：楼梯段 `S10_VMC_MODE` 切到 StairWBC，保留现有旋钮做退化开关；
5. **回归 wp6→8**：力矩合规（腿 ≤48Nm / 轮 ≤13.5Nm）、无弹射、推力连续。

## 七、风险与决策点

- **QP 对接触状态敏感**：布尔切换在传感器噪声下会抖振 → 相位加滞回（±0.02m）；
- **IK 模型误差**：mujoco/真机几何参数需标定；
- **计算预算**：osqp 12 变量 ~1ms，200Hz 可行；
- **未决问题 1**：当前卡点（wp7 前东漂、进梯线 yaw 漂移）是导航层问题，
  建议先修导航航向保持让机器人能进梯，再切 StairWBC 执行层（两者独立）；
- **未决问题 2**：楼梯段 MPPI 是否只优化质心轨迹、不优化接触力？（推荐：是，接触由 ModeSchedule 硬切）

## 八、当前测试命令（现状，供对照）

```bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition

S10_START_WP=6 S10_AUTO_MAX_WP=8 S10_TEST_MAX_SIM=90 S10_VMC_MODE=dual2 \
  S10_STAIR_POSMODE=1 S10_FP_POSMODE=1 S10_FP_PRESS=0.005 S10_FP_REACH=-0.36 \
  S10_FP_STAND_DROP=0.15 S10_FP_KP_POS=120 S10_FP_BODY_K=0.4 S10_FP_BODY_KD=0.06 \
  S10_STAIR_WIN_VX=1.2 bash tmp/run_v755_baseline.sh
```