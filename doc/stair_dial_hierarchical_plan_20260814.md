# S10 轮足爬梯：DiAL 分层架构方案（2026-08-14）

## 0. 结论

DiAL-MPC 适合作为固定接触模式下的全阶扭矩执行器，不适合单独负责楼梯接触时序与落脚点规划。

采用三层架构：

1. 感知 10Hz：lidar 高程图 + riser 表 + 落脚点候选。
2. 接触规划 20Hz：确定 `mode[i] ∈ {STANCE, SWING}` 和 `foothold_i`。
3. DiAL 20Hz：在固定 mode 下优化全阶扭矩。
4. guard 200Hz：支撑约束、摆动轮锁轮、body roll/pitch 修正。

文献依据：

- DiAL-MPC：arXiv 2409.15610
- Layered Foothold MPC：arXiv 2506.09979
- 轮足台阶 MPC：JRM 35(1):160

---

## 1. 总体架构

```text
lidar 10Hz
   |
riser / 高程图 / 落脚点候选
   |
接触规划器 20Hz
   |  mode[4] + foothold[4]
   v
DiAL-MPC 20Hz（固定 mode）
   |  16D 力矩
   v
StairStanceGuard 200Hz
   |  支撑约束 / 锁轮 / 姿态修正
   v
MuJoCo 200Hz
```

CRUISE 段不变；STAIR 段使用上述链路。

---

## 2. 感知层

输入：模拟 lidar 点云或高程图。

输出：

```text
heightmap       世界对齐高程
valid           有效掩码
riser_y[]       各级台阶 y 坐标
tread_top_z[]   各级台面顶高
foothold_xy[i]  下一级踏面中心候选
foothold_z[i]   台面顶 + 轮半径
```

当前实现：`LidarTerrainV2` + `StairContactPlanner.update_risers()`。

---

## 3. 接触规划器

### 3.1 决策量

```text
mode[i] ∈ {0, 1}       # 0=STANCE, 1=SWING
foothold_xy[i] ∈ R2
foothold_z[i] ∈ R
```

### 3.2 已知楼梯的几何硬切换

```text
d_front = 前轴到下一 riser 的距离
d_rear  = 后轴到下一 riser 的距离

if d_front < 0.15m:
    mode[FL] = mode[FR] = 1
    foothold_z[FL] = foothold_z[FR] = 下一级台面顶 + R
elif 前轮已上台面:
    mode[RL] = mode[RR] = 1
    foothold_z[RL] = foothold_z[RR] = 当前台面顶 + R
else:
    mode = 0
```

不新增连续 `sin²` 窗口；`gait_swing` 降级为 mode 的执行软掩码。

### 3.3 未知地图时的低维采样

若比赛换地图，上层只采样：

```text
mode 序列 + 4 个落脚点
```

维度远小于 DiAL 全阶动作空间，采样效率可接受。

---

## 4. DiAL-MPC 固定 mode 执行

DiAL 不再搜索接触序列，代价按 mode 分支：

```text
if mode[i] == SWING:
    wheel_i 力矩目标 = 0 或微小正转
    leg_i   z 目标 = foothold_z[i]
    leg_i   xy 目标 = foothold_xy[i]
else:
    wheel_i 正常驱动
    leg_i  保持接地
```

DiAL 动作空间仍为 16 维：

```text
u = [leg_hipx, leg_hipy, leg_knee] * 4 + [wheel] * 4
```

接触 mode 在预测窗内固定，不进入采样搜索。

---

## 5. StairStanceGuard 200Hz

确定性执行层：

1. 用几何接触判断每轮是否接地。
2. 检查剩余支撑轮是否形成有效支撑多边形/线段。
3. 不满足则否决不安全 SWING。
4. 摆动轮锁轮，支撑轮按摩擦限幅。
5. 可选的显式 body roll/pitch 支撑控制。

---

## 6. 控制频率

| 层 | 频率 | 职责 |
|---|---|---|
| lidar | 10Hz | 高程图 / riser |
| 接触规划 | 20Hz | mode + foothold |
| DiAL | 20Hz | 固定 mode 全阶扭矩 |
| guard | 200Hz | 支撑 / 锁轮 / 姿态 |

---

## 7. 落地顺序

1. 接触规划器输出硬 `mode` + `foothold`。
2. DiAL 代价按 mode 分支。
3. guard 200Hz 执行支撑约束。
4. wp6→7 真原图验证。
5. 通过后扩展为低维采样，适应换地图。

---

## 8. 一句话总结

> 不要让 DiAL 暴力搜索“哪条腿该抬、落到哪里”；由 20Hz 接触规划器决定 mode 和 foothold，DiAL 在固定接触模式下优化全阶扭矩，200Hz guard 保证支撑稳定。
