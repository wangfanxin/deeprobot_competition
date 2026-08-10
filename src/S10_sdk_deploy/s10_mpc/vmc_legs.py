"""vmc_legs.py — v218 方案执行层：VMC/阻抗腿层（200Hz+，纯 numpy）。

设计（选中对话方案第 3 条）：
- 身体层只给 [vx_cmd, omega_cmd]（+姿态参考）；
- 腿层把身体力/力矩映射到 4 条腿的轮端任务空间阻抗力，再经腿
  Jacobian 转关节力矩（virtual model control）；
- 轮子差速执行 yaw，腿阻抗执行高度/姿态，hipx 执行侧倾（压弯）。

S10 腿运动学（sagittal 2R 平面链，见 S10.xml）：
  L1=0.18 大腿(hipy→knee)，L2=0.18 小腿(knee→wheel)，轮半径 r=0.081。
  约定（body 系，x 前 / z 下，q 以"腿垂直向下=0"计）：
    px = L1 sin q1 + L2 sin(q1+q2)
    pz = L1 cos q1 + L2 cos(q1+q2)
  关节力矩 = J^T · 轮端力（同坐标系）。
qpos 腿关节索引（轮关节交错）：leg0=(0,1,2) leg1=(4,5,6)
leg2=(8,9,10) leg3=(12,13,14)，即 idx = [0,1,2,4,5,6,8,9,10,12,13,14]。
轮 body id：fl=5 fr=9 hl=13 hr=17。
"""
import os
import numpy as np

# ---- 腿安装参数（body 系，来自 S10.xml + FK 校准）----
LEG_ATTACH = np.array([
    [0.2277, 0.181191],   # fl (x, y)
    [0.2277, -0.181191],  # fr
    [-0.2277, 0.181191],  # hl
    [-0.2277, -0.181191], # hr
], dtype=np.float64)
LEG_Q_IDX = [7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21]
LEG_CTRL_IDX = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
LEG_QV_LEG = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]  # qvel[6:22] 内腿索引
WHEEL_BODY = [5, 9, 13, 17]


class S10LegFK:
    """正运动学（body 系 sagittal），带 mujoco 自检校准。"""

    def __init__(self, L1=0.18, L2=0.18, r=0.081):
        self.L1, self.L2, self.r = L1, L2, r

    def wheel_pos(self, q1, q2):
        """body 系轮心位置 (x, z_down)，z_down 向下为正。"""
        px = self.L1 * np.sin(q1) + self.L2 * np.sin(q1 + q2)
        pz = self.L1 * np.cos(q1) + self.L2 * np.cos(q1 + q2)
        return np.array([px, pz])

    def jac(self, q1, q2):
        c1, s1 = np.cos(q1), np.sin(q1)
        c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
        return np.array([
            [self.L1 * c1 + self.L2 * c12, self.L2 * c12],
            [-self.L1 * s1 - self.L2 * s12, -self.L2 * s12],
        ])

    def leg_pose(self, qpos):
        """返回每腿 (q1, q2, 轮 body 系位置)。"""
        out = []
        for leg in range(4):
            q1 = float(qpos[LEG_Q_IDX[leg * 3 + 1]])
            q2 = float(qpos[LEG_Q_IDX[leg * 3 + 2]])
            out.append((q1, q2))
        return out

    def verify(self, model, data):
        """mujoco 校准：返回每腿 (attach, s1, s2, err)。"""
        pos = data.xpos[1].copy()
        R = data.xmat[1].reshape(3, 3)
        out = []
        for leg in range(4):
            q1, q2 = self.leg_pose(data.qpos)[leg]
            rel = R.T @ (data.xpos[WHEEL_BODY[leg]].copy() - pos)
            best = None
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    p = self.wheel_pos(s1 * q1, s2 * q2)
                    est = np.array([LEG_ATTACH[leg, 0] + p[0],
                                    LEG_ATTACH[leg, 1], -p[1]])
                    err = float(np.linalg.norm(est - rel))
                    if best is None or err < best[3]:
                        best = (LEG_ATTACH[leg].copy(), s1, s2, err)
            out.append(best)
        return out


# ==================== VMC 控制器 ====================

# 轮 actuator/关节索引（16 维 ctrl/qpos 块）
WHEEL_Q_IDX = [3, 7, 11, 15]
WHEEL_QV_IDX = [9, 13, 17, 21]     # qvel[6:22] 内


class VMCController:
    """VMC/阻抗腿层：身体 [vx,omega] 指令 -> 16 维关节力矩。

    每 5ms（200Hz）调用，纯 numpy。任务空间阻抗作用在轮心（世界系），
    转 body 矢状面后经腿 Jacobian 映射 hipy/knee；hipx 管侧倾（压弯）；
    轮子差速管 yaw。terrain_h 支持横脊预抬（nav 层注入）。
    """

    def __init__(self, mass=19.0, g=9.81, L1=0.18, L2=0.18, r=0.081,
                 tau_v=0.60, kp_h=300.0, kd_h=60.0, kp_z=800.0, kd_z=80.0,
                 kp_roll=200.0, kd_roll=15.0,
                 kp_pitch=250.0, kd_pitch=20.0, pitch_ff=0.8,
                 kp_pose=80.0, kd_pose=6.0,
                 wheel_k=4.0, wheel_d=0.08,
                 track_half=0.24):
        self.m, self.g = mass, g
        self.fk = S10LegFK(L1, L2, r)
        self.tau_v = tau_v
        self.kp_h, self.kd_h = kp_h, kd_h
        # v219k: 地形阻抗可覆盖（软腿防轮子被压死/打滑）
        self.kp_h = float(os.environ.get("S10_VMC_KPH", str(self.kp_h)))
        self.kd_h = float(os.environ.get("S10_VMC_KDH", str(self.kd_h)))
        self.kp_z, self.kd_z = kp_z, kd_z
        self.kp_roll, self.kd_roll = kp_roll, kd_roll
        # v219l: roll/pitch 姿态增益可覆盖（硬增益压减载轮 → 轮推力崩）
        self.kp_roll = float(os.environ.get("S10_VMC_KP_ROLL", str(self.kp_roll)))
        self.kp_pitch, self.kd_pitch, self.pitch_ff = kp_pitch, kd_pitch, pitch_ff
        self.kp_pose, self.kd_pose = kp_pose, kd_pose
        self.pose_target = np.array([-0.05, -1.16, 2.30,
                                     0.05, -1.16, 2.30,
                                    -0.05,  1.16, -2.30,
                                     0.05,  1.16, -2.30], dtype=np.float64)
        self.wheel_k, self.wheel_d = wheel_k, wheel_d
        # v219j: 轮参数可环境覆盖（单测调整）
        self.wheel_k = float(os.environ.get("S10_VMC_WHEEL_K", str(self.wheel_k)))
        self.wheel_d = float(os.environ.get("S10_VMC_WHEEL_D", str(self.wheel_d)))
        self.yaw_k = 30.0
        # v218z: 轮差速 yaw 增益可覆盖（轮侧向摩擦 0.8 后转弯能力大幅提升，
        # 增益需降低避免过冲振荡）
        self.yaw_k_wheel = float(os.environ.get("S10_VMC_YAW_K_WHEEL", "60.0"))
        self.track_half = track_half
        self.wheelbase = 0.455
        self._vx_f, self._om_f = 0.0, 0.0
        self._roll_f = 0.0
        self._roll_prev = None
        self._pitch_prev = None

    def _body_state(self, qpos, qvel):
        q = qpos[3:7]
        w, x, y, z = q
        yaw = float(np.arctan2(2.0 * (w * z + x * y),
                               1.0 - 2.0 * (y * y + z * z)))
        roll = float(np.arctan2(2.0 * (w * x + y * z),
                                1.0 - 2.0 * (x * x + y * y)))
        pitch = float(np.arctan2(2.0 * (w * y - z * x),
                                 1.0 - 2.0 * (y * y + x * x)))
        R = np.asarray([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        vw = R.T @ np.asarray(qvel[0:3], dtype=np.float64)
        return dict(pos=qpos[0:3], yaw=yaw, roll=roll, pitch=pitch,
                    vx=float(vw[0]), vy=float(vw[1]),
                    omega=float(qvel[5]), R=R)

    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel,
                    cmd, terrain_h, dt=0.005):
        """-> tau(16)。wheel_xyz/wheel_vel: (4,3) 世界系轮心位姿/线速度。"""
        if os.environ.get("S10_VMC_MODE", "wbc") == "static":
            return self._static_pose_tau(qpos, qvel, cmd, dt)
        body = self._body_state(qpos, qvel)
        # v219w: 轮子腾空检测——过脊/跳跃时轮差速与腿 yaw 反馈会爆炸
        # （om 6.8rad/s 侧翻实测），离地>0.05m 衰减 yaw 反馈至 0
        if wheel_xyz is not None and terrain_h is not None:
            _lift_amt = float(np.mean(
                wheel_xyz[:, 2] - (np.asarray(terrain_h) + self.fk.r)))
            self._ground_f = float(np.clip(
                1.0 - max(0.0, _lift_amt - 0.02) / 0.05, 0.0, 1.0))
        else:
            self._ground_f = 1.0
        k = min(1.0, dt / 0.04)
        # v218c: 身体加速度温和化（轮腿猛推会抬头炸）——0.25s 斜坡
        k = min(1.0, dt / 0.25)
        self._vx_f += (float(cmd["vx"]) - self._vx_f) * k
        self._om_f += (float(cmd["omega"]) - self._om_f) * k
        self._roll_f += (float(cmd.get("roll_tar", 0.0)) - self._roll_f) * k
        pitch_tar = float(cmd.get("pitch_tar", 0.0))

        # 身体力：前进跟踪 + 坡度前馈
        Fx_body = self.m * (self._vx_f - body["vx"]) / self.tau_v
        Fx_body += self.m * self.g * float(np.sin(pitch_tar))

        tau = np.zeros(16, dtype=np.float64)
        # v218f: 完整 6D 身体 wrench -> 12 腿力（世界系，最小范数）——含水平分量，
        # 用实际轮位几何解耦俯仰/侧倾（此前仅垂直力 3 约束，带倾角腿强耦合）。
        z_des = float(np.mean(terrain_h)) + 0.205
        F_des_w = np.array([
            0.0, 0.0,
            self.m * self.g + self.kp_z * (z_des - body["pos"][2])
            - self.kd_z * float(qvel[2])])
        # v218k: 驱动 25% 由腿分担（轮为主），全轮在坡上推力不足
        _dsh = float(os.environ.get("S10_VMC_DRIVE_SHARE", "0.25"))
        fwd = body["R"] @ np.array([1.0, 0.0, 0.0])
        F_des_w += fwd * (_dsh * self.m
                          * (self._vx_f - body["vx"]) / self.tau_v)
        roll_rate = ((body["roll"] - self._roll_prev) / max(dt, 1e-4)
                     if self._roll_prev is not None else 0.0)
        self._roll_prev = body["roll"]
        pitch_rate = ((body["pitch"] - self._pitch_prev) / max(dt, 1e-4)
                      if self._pitch_prev is not None else 0.0)
        self._pitch_prev = body["pitch"]
        ax = (self._vx_f - body["vx"]) / self.tau_v
        T_roll_b = (self.kp_roll * (self._roll_f - body["roll"])
                    - self.kd_roll * roll_rate)
        T_pitch_b = (self.kp_pitch * (pitch_tar - body["pitch"])
                     - self.kd_pitch * pitch_rate
                     - self.pitch_ff * self.m * ax * 0.20)
        # v218h: 力矩钳制在支撑多边形可行域内（后轮无法上拉，|T|≤mg·lever/2）
        _tmax = float(os.environ.get("S10_VMC_TMAX", "25.0"))
        T_roll_b = float(np.clip(T_roll_b, -_tmax, _tmax))
        T_pitch_b = float(np.clip(T_pitch_b, -_tmax, _tmax))
        # v218j: yaw 力矩入 wrench（经腿侧向力执行，轮差速辅助）
        T_yaw_b = (self.yaw_k * (self._om_f - body["omega"])
                   - 0.5 * body["omega"])
        # v219x: cmd.yaw_scale（巡航层 RIDGE_LIFT 抬轮时置 0）+ 腾空衰减
        _ysc = float(cmd.get("yaw_scale", 1.0))
        T_yaw_b = (float(os.environ.get("S10_VMC_YAW_W", "0.0"))
                   * T_yaw_b * _ysc * getattr(self, "_ground_f", 1.0))
        T_des_w = body["R"] @ np.array([T_roll_b, T_pitch_b, T_yaw_b])
        W = np.concatenate([F_des_w, T_des_w])
        A6 = np.zeros((6, 12))
        for leg in range(4):
            rw = wheel_xyz[leg] - body["pos"]
            S = np.array([
                [0.0, -rw[2], rw[1]],
                [rw[2], 0.0, -rw[0]],
                [-rw[1], rw[0], 0.0]])
            # v218f: 身体 wrench = 腿对轮力 f 的反作用（-f）：A6 整体取反
            A6[0:3, leg * 3:leg * 3 + 3] = -np.eye(3)
            A6[3:6, leg * 3:leg * 3 + 3] = -S
        try:
            f_legs = np.linalg.pinv(A6) @ W
            _qpm = os.environ.get("S10_VMC_QP", "0")
            if _qpm == "1":
                f_legs = self._solve_wbc_qp(A6, W, f_legs)
            elif _qpm == "2":
                f_legs = self._project_cone(f_legs)
        except Exception:
            f_legs = np.zeros(12)
            f_legs[2::3] = self.m * self.g / 4.0
        for leg in range(4):
            b = leg * 3
            hipx_i, hipy_i, knee_i = (
                LEG_CTRL_IDX[b], LEG_CTRL_IDX[b + 1], LEG_CTRL_IDX[b + 2])
            q1 = float(qpos[LEG_Q_IDX[b + 1]])
            q2 = float(qpos[LEG_Q_IDX[b + 2]])
            J = self.fk.jac(q1, q2)
            # WBC 腿力（世界系）+ 地形阻抗（垂直跟随，横脊预抬经 terrain_h）
            fw = f_legs[leg * 3:leg * 3 + 3].copy()
            p = wheel_xyz[leg]
            # v220k: 单步跨越——迈步腿完全卸载（wrench 支撑力+地形阻抗都清零，
            # 否则 J^T 支撑力矩抵消 knee 位置 PD，轮抬不起来）
            _sl = float(np.asarray(
                cmd.get("step_lift", np.zeros(4)))[leg])
            fw[:] = fw * (1.0 - _sl)
            # v218q: 前轮 hop 冲量（世界 z 向上）——必须在卸载后加，
            # 否则被迈步腿清零乘掉（v220j 实测 hop 无效）
            _hop = cmd.get("hop")
            if _hop is not None:
                fw[2] += float(_hop[leg])
            pz_des = float(terrain_h[leg]) + self.fk.r
            fw[2] += (1.0 - _sl) * (
                self.kp_h * (pz_des - p[2])
                - self.kd_h * float(wheel_vel[leg, 2]))
            f_body = body["R"].T @ fw
            f_sag = np.array([f_body[0], -f_body[2]])       # [x, z_down]
            t_hipy, t_knee = J.T @ f_sag
            # hipx 侧向力经轮深杠杆 -> 力矩（wrench 已解出 fb[1]）
            # v218o: hipx 低通（wrench 侧向力随姿态抖动翻转 → 扑动）
            _hipx_f_on = float(os.environ.get("S10_VMC_HIPX_F", "1.0"))
            t_hipx = 0.30 * float(f_body[1]) * _hipx_f_on
            if not hasattr(self, "_hipx_f"):
                self._hipx_f = np.zeros(4)
            self._hipx_f[leg] += 0.15 * (t_hipx - self._hipx_f[leg])
            t_hipx = self._hipx_f[leg]
            # v218f: 姿态正则（零空间 PD 拉回蹲姿，防腿伸到近奇异）
            qhx = float(qpos[LEG_Q_IDX[b]])
            t_hipx += (self.kp_pose * (self.pose_target[b] - qhx)
                       - self.kd_pose * float(qvel[6 + LEG_QV_LEG[b]]))
            # v220g: 迈步腿 hipy 前摆+knee 伸直。前腿 q1 -1.16->-0.5(+0.66)、
            # 后腿 +1.16->+0.5(-0.66)，符号按腿分（此前统一减号=后摆 bug）
            _qs = -1.0 if leg in (0, 1) else 1.0
            _q1_tgt = self.pose_target[b + 1] - _sl * 0.66 * _qs
            _q2_tgt = self.pose_target[b + 2] + _sl * 0.42
            t_hipy += (self.kp_pose * (_q1_tgt - q1)
                       - self.kd_pose * float(qvel[6 + LEG_QV_LEG[b + 1]]))
            t_knee += (self.kp_pose * (_q2_tgt - q2)
                       - self.kd_pose * float(qvel[6 + LEG_QV_LEG[b + 2]]))

            # v218f: hipx 由 wrench 侧向力接管；S10_VMC_HIPX_TORQUE=1 叠加姿态反馈
            side = -1.0 if leg in (0, 2) else 1.0
            wd_side = side   # v218j: 左转(ω>0)需左轮慢右轮快
            if os.environ.get("S10_VMC_HIPX_TORQUE", "1") == "1":
                t_hipx += side * (self.kp_roll * (self._roll_f - body["roll"])
                                  - self.kd_roll * roll_rate)

            # 轮：差速转向
            wq = float(qvel[WHEEL_QV_IDX[leg]])
            # v218j: 校准正 wq=倒车，前进速度 = -wq*r（此前符号反导致超速失控）
            v_wheel = -wq * self.fk.r
            v_ref = self._vx_f + wd_side * self._om_f * self.track_half
            # v218: 实测 +轮力矩=倒车（S10 轮轴符号），取反前进
            # v218h: 驱动按校准取反，阻尼必须始终反向（否则负转速时放大）
            # v218j: 直接 yaw 差速力矩（轮全幅差速，参考 dial-MPC）
            # v218k: 左转(ω>0)左轮需向后力矩——符号与 wd_side 相反
            # v219o: yaw 差速增益随 |omega 指令| 自适应——直行小增益防
            # 差速振荡（v219m/n 实测 60→5/15），转弯大增益保证转向力。
            _yk = self.yaw_k_wheel * (0.3 + 0.7 * min(
                abs(self._om_f) / 0.4, 1.0))
            t_yaw = (-_yk * (self._om_f - body["omega"]) * wd_side
                     * _ysc * getattr(self, "_ground_f", 1.0))
            t_wheel = (-(self.wheel_k * (v_ref - v_wheel))
                       - self.wheel_d * wq + t_yaw)

            tau[hipx_i] = float(np.clip(t_hipx, -20, 20))
            tau[hipy_i] = float(np.clip(t_hipy, -50, 50))
            tau[knee_i] = float(np.clip(t_knee, -50, 50))
            # v218x: 动态抓地钳制－－轮侧向摩擦受限，载荷转移时外侧轮
            # 获得更大 yaw 权限（4SWLR/轮腿转向文献）；S10_VMC_WHEEL_TMAX
            # 仍可显式覆盖为静态钳制（A/B 用）
            _wt_env = os.environ.get("S10_VMC_WHEEL_TMAX")
            if _wt_env is not None:
                _wt = float(_wt_env)
            else:
                # v218y: 载荷用“静载+地形阻抗力”而非 WBC pinv
                # 解出的 fw[2]（最小范数解会把垂直分量分散）
                _mu_w = float(os.environ.get("S10_VMC_WHEEL_MU", "0.9"))
                _fz_load = (self.m * self.g / 4.0
                            + self.kp_h * (pz_des - p[2])
                            - self.kd_h * float(wheel_vel[leg, 2]))
                # v219l: 钳制下限用 0.5×静载（防减载轮
                # 推力崩溃到 0.36Nm 引发正反馈振荡）
                _wt = float(np.clip(
                    _mu_w * max(_fz_load, 0.5 * self.m * self.g / 4.0)
                    * self.fk.r, -14.0, 14.0))
            tau[WHEEL_Q_IDX[leg]] = float(np.clip(t_wheel, -_wt, _wt))
        tau[LEG_CTRL_IDX] = np.clip(tau[LEG_CTRL_IDX], -50, 50)
        return tau



    def _static_pose_tau(self, qpos, qvel, cmd, dt):
        """静态支撑（J^T·m·g/4）+ 姿态 PD 锁蹲姿 + 轮差速（最简稳定基线）。"""
        k = min(1.0, dt / 0.25)
        self._vx_f += (float(cmd["vx"]) - self._vx_f) * k
        self._om_f += (float(cmd["omega"]) - self._om_f) * k
        tau = np.zeros(16, dtype=np.float64)
        for leg in range(4):
            b = leg * 3
            q1 = float(qpos[LEG_Q_IDX[b + 1]])
            q2 = float(qpos[LEG_Q_IDX[b + 2]])
            J = self.fk.jac(q1, q2)
            f = np.array([0.0, self.m * self.g / 4.0])
            t = J.T @ f
            tau[LEG_CTRL_IDX[b + 1]] = t[0]
            tau[LEG_CTRL_IDX[b + 2]] = t[1]
            for j in range(3):
                qi = LEG_Q_IDX[b + j]
                ci = LEG_CTRL_IDX[b + j]
                tau[ci] += (self.kp_pose
                            * (self.pose_target[b + j] - float(qpos[qi]))
                            - self.kd_pose
                            * float(qvel[6 + LEG_QV_LEG[b + j]]))
            wq = float(qvel[WHEEL_QV_IDX[leg]])
            side = -1.0 if leg in (0, 2) else 1.0
            v_ref = self._vx_f + side * self._om_f * self.track_half
            tau[WHEEL_Q_IDX[leg]] = float(np.clip(
                -(self.wheel_k * (v_ref + wq * self.fk.r))
                - self.wheel_d * wq, -14, 14))
        tau[LEG_CTRL_IDX] = np.clip(tau[LEG_CTRL_IDX], -50, 50)
        return tau

# ==================== 已知地图地形栅格（预计算） ====================


    def _solve_wbc_qp(self, A6, W, f0):
        """v218t: 力分配 QP——摩擦锥 + 腿力上限 + 最小范数（SLSQP，12 变量）。"""
        from scipy.optimize import minimize
        mu = float(os.environ.get("S10_VMC_MU", "0.8"))
        f_max = float(os.environ.get("S10_VMC_F_LEG_MAX", "120.0"))

        def obj(f):
            d = A6 @ f - W
            return 0.5 * float(d @ d) + 1e-5 * float(f @ f)

        cons = []
        for l in range(4):
            b = l * 3
            cons.append({"type": "ineq",
                         "fun": lambda f, b=b: (mu * f[b + 2]
                                                 - abs(f[b]) - abs(f[b + 1]))})
            cons.append({"type": "ineq",
                         "fun": lambda f, b=b: f[b + 2]})
            cons.append({"type": "ineq",
                         "fun": lambda f, b=b: (f_max - np.sqrt(
                    f[b] ** 2 + f[b + 1] ** 2 + f[b + 2] ** 2))})
        try:
            res = minimize(obj, f0, method="SLSQP", constraints=cons,
                           options={"maxiter": 40, "ftol": 1e-4})
            if res.success or np.isfinite(res.fun):
                return np.asarray(res.x, dtype=np.float64)
        except Exception:
            pass
        return f0


    def _project_cone(self, f_legs):
        """v218u: 快速锥投影——只削水平分量、保持垂直支撑（防坡上卸载）。"""
        mu = float(os.environ.get("S10_VMC_MU", "0.8"))
        f_max = float(os.environ.get("S10_VMC_F_LEG_MAX", "140.0"))
        out = f_legs.copy()
        for l in range(4):
            b = l * 3
            fx, fy, fz = out[b], out[b + 1], out[b + 2]
            fz = max(fz, 5.0)                      # 保持最小支撑
            h = float(np.hypot(fx, fy))
            lim = mu * fz
            if h > lim and h > 1e-6:
                s = lim / h
                fx, fy = fx * s, fy * s
            fn = float(np.sqrt(fx * fx + fy * fy + fz * fz))
            if fn > f_max:
                s = f_max / fn
                fx, fy, fz = fx * s, fy * s, fz * s
            out[b], out[b + 1], out[b + 2] = fx, fy, fz
        return out

class TerrainMap:
    """启动时一次性 raycast 赛道地形（机器人移走），运行期 O(1) 查表。"""

    def __init__(self, model, data, x0=-22.0, x1=37.0, y0=-4.0, y1=50.0,
                 res=0.10):
        import mujoco
        self.res = res
        self.ox, self.oy = x0, y0
        self.nx = int(round((x1 - x0) / res)) + 1
        self.ny = int(round((y1 - y0) / res)) + 1
        # 机器人移走，避免射线打到自身
        data.qpos[0:3] = [500.0, 500.0, 10.0]
        mujoco.mj_forward(model, data)
        h = np.full((self.ny, self.nx), -1.0, dtype=np.float64)
        xs = x0 + np.arange(self.nx) * res
        ys = y0 + np.arange(self.ny) * res
        g = np.array([-1], dtype=np.int32)
        dist = np.zeros(1)
        nrm = np.zeros(3)
        for iy in range(self.ny):
            y = ys[iy]
            for ix in range(self.nx):
                x = xs[ix]
                hit = mujoco.mj_ray(model, data, [x, y, 8.0], [0, 0, -1],
                                    None, True, -1, g, nrm)
                h[iy, ix] = (8.0 - hit) if hit > 0 else 0.0
        self.h = h
        # 还原机器人（调用方随后重设位姿）
        data.qpos[0:3] = [0.0, -2.5, 0.2]
        mujoco.mj_forward(model, data)

    def height(self, x, y):
        ix = int(np.floor((x - self.ox) / self.res))
        iy = int(np.floor((y - self.oy) / self.res))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            v = float(self.h[iy, ix])
            return v if v >= 0.0 else 0.0
        return 0.0


# ==================== PD 站姿 + 轮驱动模式（v218k，稳定基线） ====================

class LidarTerrain:
    """v219g: lidar 传感器视角高程（替代上帝视角 raycast/TerrainMap）。

    从 lidar_site 按机器人航向发射扇形射线（mj_multiRay 批量），命中点
    写入以 lidar 为原点的局部栅格；height(x,y) 与 TerrainMap 接口一致
    （O(1) 查表），未覆盖区域返回 0。

    与 ROS2 mujoco-lidar 一致的传感器建模：
      - geomgroup 只留 group 0（地形），排除机器人(group 1)/赛道标记(group 2)；
      - 排除 base_link 自身；
      - 前向扇形 ±fov_h × 俯仰 +10°~-55°（近场地面到远场高台）；
      - 10Hz 更新，近处盲区/遮挡/延迟与真 lidar 一致。
    """

    def __init__(self, model, data, half=6.0, res=0.10,
                 th_n=32, phi_n=12,
                 fov_h=None, cutoff=20.0):
        import mujoco
        self.m, self.d = model, data
        self.res = float(res)
        self.n = int(2.0 * half / res) + 1
        self.h = np.zeros((self.n, self.n), dtype=np.float64)
        self.valid = np.zeros((self.n, self.n), dtype=np.int32)
        self.sid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
        if self.sid < 0:
            raise ValueError("lidar_site not found in model")
        self.cutoff = float(cutoff)
        if fov_h is None:
            fov_h = float(np.radians(55))
        # 俯仰：+10°(远处高台) .. -55°(近场地面)
        ths = np.linspace(-fov_h, fov_h, int(th_n))
        phs = np.linspace(np.radians(10.0), np.radians(-55.0), int(phi_n))
        dirs = []
        for ph in phs:
            for th in ths:
                dirs.append([float(np.cos(ph) * np.cos(th)),
                             float(np.cos(ph) * np.sin(th)),
                             float(np.sin(ph))])
        self.dirs_local = np.asarray(dirs, dtype=np.float64)
        self.ox = 0.0
        self.oy = 0.0
        self.geomgroup = np.zeros((mujoco.mjNGROUP,), dtype=np.ubyte)
        self.geomgroup[0] = 1

    def _yaw(self):
        q = self.d.xquat[1]
        return float(np.arctan2(
            2.0 * (q[3] * q[0] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))

    def update(self):
        """发射一帧射线并刷新栅格（10Hz 调用）。"""
        import mujoco
        m, d = self.m, self.d
        pos = np.asarray(d.site_xpos[self.sid], dtype=np.float64)
        self.ox = pos[0]
        self.oy = pos[1]
        self.h.fill(0.0)
        self.valid.fill(0)
        yaw = self._yaw()
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        fwd = np.array([c, s, 0.0])
        right = np.array([-s, c, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        L = self.dirs_local
        # 局部扇形 -> 世界：x*前向 + y*右向 + z*上
        vec = (L[:, 0:1] * fwd[None, :] + L[:, 1:2] * right[None, :]
               + L[:, 2:3] * up[None, :])
        n = len(L)
        pnt = pos.copy()          # mj_multiRay: 单起点 + (nray,3) 方向
        geomid = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, -1.0, dtype=np.float64)
        # vec 需展平 (nray*3,)（mujoco bindings_test 示例）
        mujoco.mj_multiRay(m, d, pnt, vec.reshape(-1), self.geomgroup,
                           True, 1, geomid, dist, None, n, self.cutoff)
        hit = dist > 0.0
        if hit.any():
            pts = pnt + dist[:, None] * vec
            for i in np.where(hit)[0]:
                p = pts[i]
                ix = int(np.floor((p[0] - self.ox) / self.res + 0.5 * self.n))
                iy = int(np.floor((p[1] - self.oy) / self.res + 0.5 * self.n))
                if 0 <= ix < self.n and 0 <= iy < self.n:
                    # min-z：同格多条射线取最低（地面优先于障碍顶）
                    if not self.valid[iy, ix] or p[2] < self.h[iy, ix]:
                        self.h[iy, ix] = p[2]
                    self.valid[iy, ix] = 1

    def height(self, x, y):
        ix = int(np.floor((x - self.ox) / self.res + 0.5 * self.n))
        iy = int(np.floor((y - self.oy) / self.res + 0.5 * self.n))
        if 0 <= ix < self.n and 0 <= iy < self.n and self.valid[iy, ix]:
            return float(self.h[iy, ix])
        return 0.0


class LegPDDrive:
    """最简可靠执行层：腿 PD 锁站姿（固定悬挂），轮差速驱动。

    身体层 MPPI 给 [vx, omega]；腿保持蹲姿（kp=80/kd=2，与站起 PD 一致），
    轮子按 v_ref 差速。平地/缓坡稳定（车式），横脊靠"膝目标预抬"。
    """

    LEG_TARGET = np.array([-0.05, -1.16, 2.30,
                            0.05, -1.16, 2.30,
                           -0.05,  1.16, -2.30,
                            0.05,  1.16, -2.30], dtype=np.float64)

    def __init__(self, kp_leg=30.0, kd_leg=3.0, wheel_k=0.30, wheel_d=0.03,
                 track_half=0.24, r=0.081):
        self.kp_leg, self.kd_leg = kp_leg, kd_leg
        self.wheel_k, self.wheel_d = wheel_k, wheel_d
        self.track_half = track_half
        self.r = r
        self._vx_f, self._om_f = 0.0, 0.0

    def compute_tau(self, qpos, qvel, wheel_xyz=None, wheel_vel=None,
                    cmd=None, terrain_h=None, dt=0.005):
        k = min(1.0, dt / 0.80)
        self._vx_f += (float(cmd["vx"]) - self._vx_f) * k
        self._om_f += (float(cmd["omega"]) - self._om_f) * k
        tau = np.zeros(16, dtype=np.float64)
        # 腿 PD（12 关节，索引见 LEG_Q_IDX/LEG_CTRL_IDX）
        for leg in range(4):
            b = leg * 3
            for j in range(3):
                qi = LEG_Q_IDX[b + j]
                ci = LEG_CTRL_IDX[b + j]
                tau[ci] = (self.kp_leg
                           * (self.LEG_TARGET[b + j] - float(qpos[qi]))
                           - self.kd_leg * float(qvel[6 + j + leg * 3]))
        # 轮差速（索引 WHEEL_QV_IDX 是 qvel[6:22] 内，绝对索引需 +6）
        for leg in range(4):
            wq = float(qvel[WHEEL_QV_IDX[leg]])
            side = -1.0 if leg in (0, 2) else 1.0
            v_ref = self._vx_f + side * self._om_f * self.track_half
            tau[WHEEL_Q_IDX[leg]] = float(np.clip(
                -(self.wheel_k * (v_ref + wq * self.r)
                  - self.wheel_d * wq), -14, 14))
        tau[LEG_CTRL_IDX] = np.clip(tau[LEG_CTRL_IDX], -50, 50)
        return tau
