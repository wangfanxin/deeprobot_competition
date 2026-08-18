# SMppi/TMppi Cruise + RL-Stair + TK1/TK2 总方案（2026-08-18）

## 1. 技能划分
- SMppi：直线段走线保持。
- TMppi：航点附近低速转向。
- CarVMC：轮足执行。
- TK1：楼梯前低速转向/减速/对准。
- RL-Stair：爬楼梯。
- TK2：楼梯后低速对准下一航点。

## 2. 数据管线
```text
33 waypoints
 → AutoNavFollower（直线段）
 → nav vx / vyaw
 → SMppi / TMppi
 → CarVMC
 → 16 joint torque
 → Mujoco
```

楼梯感知：
```text
lidar 96线
 → LidarTerrainV2
 → terrain / wall
 → riser 检测
 → TK1 / RL-Stair / TK2
```

## 3. 频率
| 模块 | 频率 |
|---|---|
| 仿真 | 200Hz 目标 |
| nav | 20Hz |
| SMppi/BodyMPPI | 20Hz |
| CarVMC | 200Hz |
| lidar | 4Hz |
| RL policy | 50Hz |

## 4. 路径规划
- 直线航点段：
```bash
S10_GLOBAL_FILLET_R=0
```

## 5. SMppi 直线保持
```bash
VMC_MPPI_N=512
VMC_MPPI_H=20
S10_MPPI_ADA=1
S10_AUTO_VMAX=2.5
S10_AUTO_LOOKAHEAD=3.5
S10_MPPI_A_MAX=1.5
S10_MPPI_OMAX=2.0
S10_MPPI_W_GUIDE=0.5
S10_MPPI_W_DIST=2.0
S10_MPPI_W_HEAD=0.0
```

## 6. TMppi 航点转向
### 触发
```text
距当前 wp < 0.2m
且 实际速度 < 0.2m/s
```
### 动作
```text
vx <= 0.2
omega = clip(S10_TURN_K * yaw_err, ±S10_TURN_OM_MAX)
```
### 参数
```bash
S10_TURN_SPLIT=1
S10_TURN_K=3.0
S10_TURN_OM_MAX=2.0
S10_TURN_ERR_DEG=10
S10_TURN_V_MAX=0.2
```
### 交回 SMppi
```text
yaw 误差 <= 10°
```

## 7. 航点推进
```bash
S10_WP_ARRIVE_R=0.2
S10_WP_TURN_VX=0.2
```
- 只按距离：
```text
距 wp < 0.2m → 下一航点
```

## 8. CarVMC 基线
```bash
S10_VMC_KPH=300
S10_VMC_KDH=60
S10_VMC_WHEEL_K=4.0
S10_VMC_WHEEL_D=0.08
S10_VMC_YAW_K_WHEEL=60
S10_VMC_OM_ABS_MAX=2.0
S10_VMC_WHEEL_TMAX=13.5
S10_VMC_MU=0.8
```

## 9. TK1
### 触发
```text
路径上 riser 距当前位置 < 5m
```
### 动作
```text
减速 + yaw 对准 riser 爬升方向
```
### 交付 RL
```text
距 riser < 2m
yaw 已对准
速度 < 2.0m/s
```
### 参数
```bash
S10_ELEV_ENTER=5.0
S10_ELEV_DECEL_VX=2.0
S10_STAIR_ENTER_DIST=2.0
S10_TK1_LOOKAHEAD=5.0
S10_TK1_VX=2.0
```

## 10. RL-Stair
- 进入 RL 后由 policy 控制。
- 退出条件：
```bash
S10_STAIR_WHEEL_CLEAR=0.05
```
- 四轮超过最高 riser。

## 11. TK2
### 触发
```text
STAIR → CRUISE 后
```
### 动作
```text
低速对准下一航点
```
### 参数
```bash
S10_TK2_YAW_DB=0.15
S10_TK2_YAW_K=2.5
S10_TK2_YAW_MAX=1.5
S10_TK2_VX=1.5
```

## 12. 当前状态
- SMppi/TMppi 与 TK1/TK2 门控已按最新要求修改。
- 仍在 wp1→wp2 附近卡住，继续调 SMppi/TMppi 参数。
