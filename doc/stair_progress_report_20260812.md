# NmpcWBC 爬梯进度完整报告（107+ 次真原图实验）

> 生成时间：2026-08-12 · HEAD a08c915 · 目标：wp6→7 完整越过所有台阶

## 一、当前方案（三层 + 双技能门控）

感知层(10Hz) → 规划层(20Hz) → 执行层(20/200Hz)

- 感知层：lidar 高程图 + stair_world 预扫描 + riser 表(弧长/高差/台面顶)
- 规划层：AutoNavFollower（Catmull-Rom 平滑路径 + STAIR 模式航向锁）
- 执行层：NmpcWbc
  - NMPC 20Hz：SRBD（m·a=ΣF+mg、I·α=Σr×F），osqp [F(12),a(3)]，固定结构 <2ms
  - WBC 200Hz：支撑腿 J^T·(R^T·F_des) + 姿态正则 + 地形阻抗；摆腿纯位置 PD；轮速 PID + 差速

**技能门控**（唯一允许的离散切换）：CRUISE(CarVMC 巡航) ⇄ STAIR(NmpcWbc 爬梯)，前轴距首级 riser < 1.0m 切换。

## 二、数据管线

```
lidar ray 扫描 → terrain_h(4轮) + terr_ahead
stair_world 预扫描 → riser 表(弧长/高差/台面顶)
  ↓ 每 200Hz
ModeSchedule: 前/后轴 SWING 布尔 + HOVER
  ↓ 每 20Hz
NMPC: SRBD QP → F_des(4x3) + a_des
  ↓ 每 200Hz
WBC: 腿力矩(12) + 轮力矩(4) → MuJoCo
```

## 三、参数与门控清单

### 门控/状态机（6 个）
1. 技能切换（CRUISE⇄STAIR）——唯一外部离散门控
2. 前后轴 SWING 布尔（互斥）——几何相位硬切换
3. HOVER 子态——前轮过棱后悬停平移
4. 相位门（前轴等后轮上台阶）——防前轴 sprint 后轴 backlog
5. 相位门超时（2s 强制 SWING）——破前压-后滚死锁
6. 摆腿高度过滤（dh≤0.085 纯滚）——riser1 不抬轮

### NMPC 参数（40 个 S10_NMPC_*）
- 频率/结构：HZ=20、H=4
- 权重：WF=1e-3、WA=1.0、WM=0.3、WFR=0.02
- 跟踪增益：KP_Z=200/KD_Z=30、KP_VX=10、KP_PITCH=300/KD_PITCH=30、KP_YAW=6/KD_YAW=2
- 力界：FZ_MAX=180、SWING_FZ_MIN=46、FRONT_SWING_FZ_MIN=46、STANCE_FZ_MIN=95
- 摆腿：KP_SW=120、KP_SW_R=40、KD_SW=6、SWING_D=0.35、REACH=-0.34
- HOVER：HOVER_DROP=0.03、HOVER_LEN=0.10、HOVER_TMAX=0.5
- 限幅：AZ_MIN=-4、AZ_MAX=6、AL_LIM=30
- 姿态：KP_POSE=20/KD_POSE=6、KP_POSE_OPP=200/KD_POSE_OPP=20、KPH=300、PRESS=0.005
- 轮：WHEEL_K=12、YAW_DIFF=1.0、YAW_ERR_K=1.5、DRIVE_FLOOR=-2.0
- 杂项：Z_OFF=0.25、Z_RAMP=0.5、VX_TAU=0.10、PITCH_FF_SW=1.5、KP_OVR=300/KD_OVR=30

### 导航/楼梯参数
STAIR_WIN_VX=1.0、SWING_D=0.35、ENTER_DIST=2.0、HDG_K=1.0/HDG_D=3.0/HDG_KI=0.05/HDG_LAT=0.35、YAW_GAIN_STAIR=0.8、CTE_GAIN_STAIR=1.0、LOOKAHEAD_STAIR=3.5、YAW_DAMP=2.0

## 四、进度（107+ 次实验，9 个结构件）

| 里程碑 | 版本 | 结果 |
|---|---|---|
| v1075 基线 | — | 23s 原地弹跳，从未过 riser2 |
| mode 删列 | v1087 | 消除全轮悬空弹跳（模型==执行） |
| 纯位置摆腿 | v1141 | 前轴过冲消除（文献共识） |
| 折叠钳制 | v1146 | 后轴不折叠，y=38.44（最远） |
| 门控超时 | v1155 | 破前压-后滚死锁，相位交替 |
| 支撑力下限 | v1159 | body 0.86（可行位形） |
| 中抬坡 | v1165 | 推进+高度双达标 |
| yaw 积分削减 | v1174 | yaw 收敛 0.91-1.25 |
| 导航增益 0.8 | v1176 | 进梯对齐解决（yaw 1.35-1.40、横向 0.4m） |

当前最优配置：
- real114（tmp/run_nmpc_real114.sh）：进梯对齐最优
- real101：body 0.86、y=38.27（推进+高度）
- real112：22s 稳定、yaw 收敛

## 五、卡点（剩余 1 个）

**前轴 SWING 折叠墙**：身体 0.69-0.75 时轮目标与台面顶 margin 仅 3mm，轮撞面反弹（0.92 vs 目标 0.75）→ 折叠上折 → J^T 垂直权威归零 → 甩轮侧翻。Z_OFF 抬身(0.30)导致 pitch 过冲。折叠-高度 margin 耦合是纯执行层墙，15 个控制维度、107+ 次实验验证无参数稳定窗口。

## 六、建议

1. 短期：real114（进梯对齐已解决）基础上，把身体高度（v1159 的 0.86）与中抬坡（v1165）正确组合——两者曾各自达标但未同时成功
2. 中期（推荐）：OCS2 式全身 NMPC——身体轨迹（z/pitch）动态规划 + 接触力直接优化，替代位置 PD 摆腿（绕过折叠墙），107 次实验证明唯一未被穷尽的结构性方向
3. 执行层替代：摆腿改"腿力控爬升"（轮压面力与位置解耦，ETH/IIT 路线）——比完整 OCS2 轻量

## 七、实验记录索引

完整证据链见 doc/stair_方案与数据管线_当前版_20260812.md。