# StairWBC：轮足狗楼梯爬升新方案（2026-08-11 终版）

> 状态：**已按用户终版定稿**，待实现
> 仓库：`0810new/deeprobot_competition`，写作时 HEAD = `dc4ec68`（v829）
> 旧草案（stair_修订方案_OCS2WBC_位置基全身控制_20260811.md）已按用户要求删除，历史保留在 git。
>
> **核心结论**：放弃"力控软切换"，转向"**位置基全身控制 + 几何相位硬切换**"。
> 楼梯段 WBC 不做力分配主环，仅作为接触合规校验；姿态由 IK + 位置 PD 闭环保证。
> 针对小型轮足，解决旧方案"前轮抬升 → 后轮悬空 → 推力归零"的死锁问题。

## 一、总体架构（三层，无 MPPI）

```
感知层（10Hz）
  └─ lidar 高程图 + stair_world 预扫描
      ↓ riser 表（弧长/高差）、台面顶高、显式轮心 z 参考（轮心 = 台面 + R）
AutoNavFollower（20Hz，纯导航，无轨迹优化层）
  └─ 纯 xy 路径跟踪，vx 连续插值（楼梯段不归零）
      ↓ vx 参考（连续）+ 航向修正
StairWBC（200Hz，位置基全身控制）
  ├─ ModeSchedule：4 轮独立布尔相位（硬切换）
  ├─ Body IK：姿态闭环 → 腿位置目标
  ├─ Leg Ctrl：位置 PD（增益拉满）+ 微阻抗（±2cm 柔顺）
  ├─ Wheel Ctrl：开环限幅（支撑轮 ±13.5Nm，抬升轮 0）
  └─ QP Checker：接触力合规校验（摩擦锥/法向下限），非力分配主环
Mujoco 仿真
```

## 二、感知与规划

### 2.1 感知（10Hz）

- **stair_world 预扫描**：启动时沿路径射线扫描，生成 riser 表（每级弧长和高度）和 stair_world 表（路径点、切线、台面顶高）。
- **显式轮心 z 参考**：`wheel_z_ref(y) = 台面顶高(y) + R`（R = 轮半径，模型实际 0.081m，见第六节参数核对）。棱前 RAMP_A 平滑过渡，棱后贴面。
- **关键**：所有几何量基于世界坐标，不依赖 `s_cur` 弧长投影（防漂移）。

### 2.2 AutoNavFollower（20Hz）

- **纯 xy 导航**：Catmull-Rom 平滑路径，位置式跟踪（纯跟随，无轨迹优化）。
- **vx 剖面**：楼梯段弧长窗内，vx 连续插值至 `STAIR_WIN_VX`（默认 1.8 m/s），不归零、不凹坑。
- **航向修正**：进梯前 1m 提高 yaw_gain，缩短 VLIM_LOOKAHEAD 至 1.5m，确保机身对齐楼梯切线。
- **禁止 z 先验**：导航层不读任何高程/台面信息，z 完全由执行层几何相位给出。

## 三、执行层：StairWBC（核心）

### 3.1 控制频率与预算

| 环节 | 频率 | 预算 |
|---|---|---|
| 主控制环 | 200Hz | DT = 0.005s |
| QP Checker | 200Hz | osqp 12 变量 <1ms |
| 仿真实时比 | — | ≈0.53x（瓶颈在 MuJoCo Python 步进） |

### 3.2 ModeSchedule：4 轮独立布尔相位（硬切换）

核心原则：**相位切换是布尔量，不是连续权重**。sin² 窗仅用于生成位置目标轨迹，不控制切换逻辑。

**四轮同态规则**：

- 前轴（FL+FR）共享 `d_front` = 前轴到最近棱距离
- 后轴（RL+RR）共享 `d_rear` = 后轴到最近棱距离
- `d_front` 和 `d_rear` 差 ≈ 轴距投影（0.456m × cos(楼梯倾角) ≈ 0.43m），前后轴天然错半阶

**单轮切换逻辑**：

- SWING（抬升）：`d < 0.05m`（带 ±0.02m 滞回防抖）
- STANCE（支撑）：`d > 0.05m`

**四轮状态（单阶通过，以"前轴遇棱"为 t0）**：

- `t0`：前轴 d_front<0.05 → FL/FR 切 SWING，目标 z = 台面顶 + R；后轴 d_rear 还远 → RL/RR 仍 STANCE（在当前地面硬压）
- `t0~t1`：FL/FR 被 PD 拉到台面顶+R，body 前抬（pitch 由几何给）；RL/RR 静压硬顶，后轮驱推力全开（前轮矩=0）
- `t1`：FL/FR 过棱 → 切 STANCE，落台面；后轴仍在下级地面 STANCE
- `t1~t2`：body 前移，后轮继续推
- `t2`：后轴 d_rear<0.05 → RL/RR 切 SWING，目标 z = 台面顶 + R（同前一级台面顶）
- `t2~t3`：RL/RR 被拉上台面，四轮渐全在台面
- `t3`：RL/RR 过棱 → 切 STANCE

→ 顺序是"前轴整轴 → 后轴整轴"，轴内左右同步，前后轴交叠 ≈0.1~0.15m 弧长，**永不双轴同时 SWING**。

### 3.3 Body IK：姿态闭环 → 腿位置目标

层级：**Body 姿态 PD（外环）→ IK（中环）→ 腿位置 PD（内环）**

每步计算：

1. 读 4 轮世界坐标 `wheel_xyz` + stair_world 台面顶高；
2. 算 4 个轮心 z 参考：`z_ref[i] = stair_terrain(wheel_xyz[i].y) + R`；
3. 算 body 姿态目标：
   - `z_base` = 前后轮心 z 线性插值 + 俯仰修正（随轮高变化，非固定模式）
   - `pitch` = 几何抬头角（前轴到棱距离 → tan⁻¹(阶高/阶距) 量级）
   - `roll ≈ 0`
4. Body 姿态 PD 出期望线加速度 `a_des`（6D：xy 跟 vx、z/pitch/roll 跟目标）；
5. 解析 IK 解 4 腿关节角目标（q_hip, q_thigh, q_calf × 4），含关节限位。

### 3.4 Leg Ctrl：位置 PD + 微阻抗

| 腿状态 | 位置目标 | PD 增益 | 阻抗 |
|---|---|---|---|
| SWING | z_ref = 台面顶 + R | kp 拉满（200~400 N·m/rad） | ±2cm 柔顺，仅高频吸收 |
| STANCE | z_ref − 0.005m（静压） | kp 拉满（同 SWING） | ±2cm 柔顺，仅高频吸收 |

- **位置环主导**，阻抗项不进闭环、仅做前馈微扰；
- **退出机制**：SWING 腿轮高 ≥ 台面顶 + R − 0.02m 且持续 0.05s → 切 STANCE（接触反馈释放，非时间窗）；
- **力矩限幅**：|τ_leg| ≤ 48 Nm（合规）。

### 3.5 Wheel Ctrl：开环限幅

| 轮状态 | 力矩 | 说明 |
|---|---|---|
| STANCE | clip(差速PID(vx), ±13.5Nm) | 全机推力来源 |
| SWING | 0 | 抬空轮不输出驱动力，微正转防卡沿 |

- **推力不中断保证**：轴距 0.456m > 阶距 0.4m → 前后轴永不同时 SWING → 至少一轴 STANCE → 推力连续；
- **禁用**：所有旧方案力控钳制逻辑（`S10_VMC_WBC_LIFT_TMAX`、`S10_VMC_WBC_PRESS` 等）移除。

### 3.6 QP Checker：接触力合规校验（非分配）

- **作用**：校验 Leg Ctrl 输出力矩是否导致接触力破环，破则微降增益或加阻尼，**不直接重算力分配**；
- **决策变量**：λ ∈ R¹²（4 轮世界系接触力 F_x, F_y, F_z）；
- **代价**：`min_λ w · ‖λ − J⁻ᵀτ_pd‖²`（让接触力贴近 PD 期望力）；
- **硬约束**：
  - 抬升轮：变量直接剔除（λ ≡ 0）
  - 支撑轮：λ_z ≥ N_min（静压保载）
  - 摩擦锥：|λ_xy| ≤ μ · λ_z（μ = 0.5~0.8）
- **求解器**：osqp，polish=True，超时 <3ms 沿用上一帧解。

## 四、200Hz 执行流（数据管线）

```
# 每 200Hz 控制步
1. 读 qpos, qvel, wheel_xyz, wheel_vel
2. ModeSchedule.update(wheel_xyz, stair_world):
     d_f = dist_to_nearest_riser(front_axle_xy, stair_world)
     d_r = dist_to_nearest_riser(rear_axle_xy, stair_world)
     mode[FL]=mode[FR] = SWING if d_f<0.05 else STANCE   # ±0.02m 滞回
     mode[RL]=mode[RR] = SWING if d_r<0.05 else STANCE   # ±0.02m 滞回
3. BodyIK.solve(qpos, wheel_xyz, mode):
     z_ref[4] = stair_terrain(wheel_xyz[y]) + R
     a_des = BodyPD(z_ref, pitch_geom, qpos)   # 6D 期望加速度
     q_leg_target[4] = IK(a_des, z_ref, qpos)  # 腿关节角目标
4. LegCtrl.update(q_leg_target, qpos, qvel, mode):
     for each leg i:
       target = z_ref[i] - (0.005 if mode[i]==STANCE else 0)
       τ_leg[i] = PD(q_target[i], qpos[i], qvel[i]) + τ_imp（微）
5. QPChecker.check(τ_leg, mode) -> τ_leg_cor
6. WheelCtrl.update(vx, mode) -> τ_wheel (clip ±13.5Nm / 0)
7. 输出 τ = [τ_leg_cor, τ_wheel] 到 MuJoCo
```

## 五、与旧方案对比（放弃清单）

**放弃（力控软切换路线）**：

- 连续 `climb_mask` 软权重、CPG sin² 窗控制切换逻辑；
- `S10_VMC_WBC_LIFT_TMAX`（力限抬轮钳制）、`S10_VMC_WBC_PRESS`（力控下压）、`S10_VMC_WBC_Z_LCOMP`（抬轮反作用补偿）等力控修补；
- 楼梯段 MPPI 轨迹优化（终版为纯导航，无轨迹优化层）。

**保留（可复用资产）**：

- `stair_world` / riser 表、`stair_wheel_ref` 显式轮心 z 参考（世界坐标，不依赖 s_cur）；
- v822 布尔几何相位雏形、v828 位置模式基础（FP 全增益 PD + 静压）；
- 力矩合规限幅（腿 ≤48Nm、轮 ≤13.5Nm）。

**新增**：

- 4 轮独立布尔相位（整轴同步抬放，前后轴交叠 0.1~0.15m）；
- Body IK 姿态闭环（外环姿态 PD → 中环 IK → 内环腿位置 PD）；
- QP Checker（接触合规校验，非力分配主环）。

## 六、参数核对（与当前代码/模型对齐）

| 参数 | 终版方案 | 代码/模型实测 | 结论 |
|---|---|---|---|
| 轮半径 R | 0.05m | **0.081m**（`vmc_legs.py` FK：L1=0.18、L2=0.18、r=0.081；XML 轮为 mesh 无显式 radius） | **差异待确认**：公式统一用模型实际值 R=0.081，若换轮径需同步 |
| 轴距 | 0.456m | 0.456m | ✓ |
| 阶距 | 0.4m | 0.4m（riser 间距） | ✓ |
| 首级 riser 高 | — | 0.063m | 首级试验对象 |
| STAIR_WIN_VX | 1.8 m/s | 默认 1.8（`cruise_vmc_noros.py`） | ✓ |
| 腿力矩限幅 | 48Nm | 合规要求 48Nm | ✓ |
| 轮力矩限幅 | 13.5Nm | 13.5Nm（`S10_VMC_WBC_WHEEL_TMAX`） | ✓ |
| 静压量 | 0.005m | `S10_FP_PRESS` 默认 0.005 | ✓ |
| 抬升腿 PD | 200~400 N·m/rad | `S10_FP_KP_POS`（120~400 可调） | ✓ 实现时拉满 |

## 七、实现步骤（待批准后执行）

1. 新建 `s10_mpc/stair_wbc.py`：ModeSchedule（布尔滞回）+ BodyIK（解析 IK + 姿态 PD）+ LegCtrl（位置 PD + 微阻抗）+ WheelCtrl（开环限幅）+ QPChecker（osqp）；
2. **平地验证**：4 轮位置跟踪 + 静压 + body z/pitch 保持，力矩合规；
3. **ModeSchedule 验证**：单级 riser（首级 0.063m）→ 逐级推进；
4. **接入主循环**：楼梯段 `S10_VMC_MODE` 切到 StairWBC（保留现有执行层做退化开关）；
5. **回归 wp6→8**：无弹射、推力连续、全程腿 ≤48Nm / 轮 ≤13.5Nm。

## 八、风险与未决

- **布尔切换抖振**：传感器噪声下 d 值抖动 → 方案已含 ±0.02m 滞回；
- **进梯线 yaw（wp7 东漂）**：属导航层问题，建议先修航向保持，再验证 StairWBC（两者独立）；
- **QP Checker 超时**：<3ms 未收敛沿用上一帧解，不阻塞 200Hz 主环；
- **R 值待确认**：0.05 vs 0.081，直接影响"台面顶 + R"抬升目标与 IK 解算。