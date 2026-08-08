# 交接给 stair 会话：ref_path 平滑化经验（2026-08-08）

## 一句话
MPC 的 ref_path 必须和导航 pursuit 用同一条曲线（平滑全局路径），
不能从航点折线采样——折线尖角是弯道振荡、速度方差和偶发侧翻的根因。

## 已完成的改动（cruise 侧，deeprobot_competition）
- `auto_nav.py ref_path_3d`（v183）：改从 `self.path_pts`（平滑全局路径，
  含圆角+切线因子）按弧长采样；窗口 = `_s_cur` 到当前航点后 n_wp 个航点的
  弧长（+2m 余量），spacing 0.3m；z 仍走高程图+站姿高。
- cruise 实测（H=25/0.5s 视界/2048 采样，wp0→5）：中位 3.08~3.22 m/s，
  成功率 3/4；卡弯从"系统性"降为"偶发"。

## 给 stair 侧的建议
1. 检查 stair 会话用的 ref_path 是否也走折线采样；若是，改成同一平滑路径。
2. 楼梯段（wp6→7）：xy 用平滑线，z 继续用 stair_wheel_ref 场（已实现），
  两套信息分工，不要混。
3. 台阶前的最后一个弯要保证 ref 窗口覆盖到台阶入口，否则 MPC 看不到台阶
  就进弯（视界 0.5s @3.2m/s 只有 ~1.6m）。
4. 如果 stair 侧有自己的 ref_path 实现，先对齐，避免双版本漂移。

## 相关代码位置
- ref_path_3d: src/S10_sdk_deploy/s10_mpc/auto_nav.py
- 平滑路径构造: 同文件 __init__（圆角折线 + S10_GLOBAL_TANGENT_K）
- stair z 参考场: AutoNavFollower.stair_wheel_ref_grid / mpc set_stair_ref