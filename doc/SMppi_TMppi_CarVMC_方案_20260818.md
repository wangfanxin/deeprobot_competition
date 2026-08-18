# SMppi / TMppi + CarVMC 巡航方案（2026-08-18）

## 总体架构
```text
nav(直线) -> SMppi(直线保持) -> CarVMC
nav(直线) -> TMppi(航点转向) -> CarVMC
楼梯 -> TK1 -> RL-stair -> TK2 -> SMppi/TMppi
```

## 技能分工
- nav：生成相邻航点直线，输出目标航向和速度。
- SMppi：直线段走线保持，BodyMPPI 采样。
- TMppi：航点处转向，yaw 误差 > 10° 时停止并转向。
- CarVMC：把 `vx / omega` 转成轮足扭矩。
- TK1：楼梯前减速、对准。
- RL-stair：爬楼梯。
- TK2：楼梯后回切、回线。

---

## 路径规划
- 相邻航点直线连线。
- `S10_GLOBAL_FILLET_R=0`
- 航点处：
  - `S10_WP_ARRIVE_R=0.2`
  - 转向前 `refv=0`
  - 转完前不推进航点

---

## SMppi（直线保持）
- BodyMPPI：
  ```bash
  VMC_MPPI_N=512
  VMC_MPPI_H=20
  S10_MPPI_ADA=1
  S10_MPPI_A_MAX=1.5
  S10_MPPI_OMAX=2.0
  S10_MPPI_W_GUIDE=0.5
  S10_MPPI_W_DIST=2.0
  S10_MPPI_W_HEAD=0.0
  ```
- 只跟踪直线，不跟踪曲线。

---

## TMppi（航点转向）
- 触发：
  - `S10_TURN_SPLIT=1`
  - 当前 yaw 与下一段航向误差 > `S10_TURN_ERR_DEG=10`
- 动作：
  - `vx = 0`
  - `omega = clip(S10_TURN_K * yaw_err, ±S10_TURN_OM_MAX)`
- 默认：
  ```bash
  S10_TURN_K=3.0
  S10_TURN_OM_MAX=2.0
  ```
- 转向完成后交回 SMppi。

---

## CarVMC 基线
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

---

## 楼梯
- 高程图检测路径上 `riser > 5cm`。
- 距离首级 riser < 4m 开始减速。
- `vx <= 1.8` 且距离 <= 3m 交给 RL-stair。
- 四轮越过最高 riser 后切回 SMppi/TMppi。
- 回线：
  - 横向偏差 < 0.5m
  - heading 偏差 < 10°

---

## 当前状态
- SMppi/TMppi 代码已加入。
- 仍卡在 `wp2`，需继续调 SMppi/TMppi 参数。

## git
- 最新提交：`ccbdbd6`
- 已 push `origin/main`
