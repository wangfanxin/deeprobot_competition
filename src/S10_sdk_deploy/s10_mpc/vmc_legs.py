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
        self.kp_pitch = float(os.environ.get(
            "S10_VMC_WBC_KP_PITCH", str(kp_pitch)))
        self.kd_pitch, self.pitch_ff = kd_pitch, pitch_ff
        self.kp_pose, self.kd_pose = kp_pose, kd_pose
        # v486: WBC 腿姿态 PD 可覆盖——后轮抬升量 0.07m 不足 0.13m 台阶
        # （kp=80 + 48Nm 限幅到达不了目标），提高 kp 让抬放姿态到位。
        self.kp_pose = float(os.environ.get("S10_VMC_KP_POSE", str(self.kp_pose)))
        self.kd_pose = float(os.environ.get("S10_VMC_KD_POSE", str(self.kd_pose)))
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
                    omega=float(qvel[5]), R=R,
                    omega_body=float(np.dot(qvel[3:6], R[:, 2])))

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
        # v467: WBC 站高偏移可调（原 0.205 是 dial-MPC 时代低站姿——当前
        # CarVMC 巡航站姿 body≈terrain+0.74，WBC 按 0.205 会把狗压到离地
        # 0.12m 腿折叠爬行卡死，v466 双技能卡 y=36.5 实测）。
        # v598: 楼梯区 z_des 用**最低地形**（车身后端贴地保牵引）——mean 会
        # 随前轮落脚点地形升高、把后轮悬空失牵引（wt=1.7Nm 空转实测）；
        # 前髋高度由 pitch（轴距坡度）+ 前轮落脚点地形负责。
        _zm9 = float(cmd.get("z_min", 0.0))
        z_des = float((np.min(terrain_h) if _zm9 > 0.0
                       else np.mean(terrain_h))) + float(os.environ.get(
            "S10_VMC_Z_DES_OFFSET", "0.205"))
        # v502: 抬轮时身体同步抬高（S10_VMC_WBC_LIFT_BODY）——楼梯抬轮姿态
        # 只抬轮不抬身，车身塌 0.6m 拖地卡死（v500 实测）。z_des 随平均
        # step_lift 上升，身体与轮同步爬升。
        _lb = float(os.environ.get("S10_VMC_WBC_LIFT_BODY", "0.0"))
        if _lb > 0.0:
            z_des += _lb * float(np.mean(
                np.asarray(cmd.get("step_lift", np.zeros(4)))))
        # v605: 楼梯区（z_min>0）**取消全局 z 抬升**——只留重力+阻尼，车高
        # 由逐轮落脚点阻抗决定（前轮=台面高、后轮=地面高，姿态自然形成），
        # 后轮保载荷不失牵引（此前 kp_z 抬车身→后轮 wt=1.7Nm 空转实测）。
        F_des_w = np.array([
            0.0, 0.0,
            self.m * self.g
            + (0.0 if _zm9 > 0.0
               else self.kp_z * (z_des - body["pos"][2]))
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
            # v491: 垂直支撑保留（S10_VMC_WBC_Z_KEEP，默认 0.5）——原 (1-sl)
            # 在 pinv 求解后清零抬轮腿力，总支撑不再满足 wrench，车身塌到
            # 0.11m 离地卡死（wp6→7 第一级前 0.4m 实测）。水平力照常衰减，
            # 垂直力保留部分防塌陷；姿态 PD 仍抬轮。
            _zk = float(os.environ.get("S10_VMC_WBC_Z_KEEP", "0.5"))
            fw[0:2] *= (1.0 - _sl)
            fw[2] *= (1.0 - _zk * _sl)
            # v218q: 前轮 hop 冲量（世界 z 向上）——必须在卸载后加，
            # 否则被迈步腿清零乘掉（v220j 实测 hop 无效）
            _hop = cmd.get("hop")
            if _hop is not None:
                fw[2] += float(_hop[leg])
            pz_des = float(terrain_h[leg]) + self.fk.r
            # v602: 抬轮时地形阻抗**不随 sl 衰减**（(1-zk*sl)）——楼梯落脚点
            # 把前轮地形置为台面高，阻抗把轮拉到 pz_des=台面+r，轮直接
            # 落上台面（原 (1-sl) 在 sl=1 时清零，抬轮只剩姿态 PD、够不到）
            fw[2] += (1.0 - _zk * _sl) * (
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
            _qs = -1.0 if leg in (0, 1) else 1.0
            if float(cmd.get("stair_lift", 0.0)) > 0.0:
                # v236: 台阶抬放姿态（FK 验证：前轮抬 0.141m、后轮抬 0.144m，
                # 够 0.125m riser+余量；后轮=髋前摆+膝直，修复 v220g 后轮方向）
                _amp = float(os.environ.get(
                    "S10_VMC_STAIR_LIFT_AMP", "1.0"))
                if _qs < 0.0:   # 前腿：髋大幅前摆过竖直+膝微屈——轮心抬到
                    # 髋上（FK: q1≈0.4/q2≈2.7 轮高+1.4cm），清 0.13m 棱
                    _q1_tgt = self.pose_target[b + 1] + _sl * 1.55 * _amp
                    _q2_tgt = self.pose_target[b + 2] + _sl * 0.42 * _amp
                else:           # 后腿：髋大幅前摆+膝伸直
                    _q1_tgt = self.pose_target[b + 1] + _sl * 1.10 * _amp
                    _q2_tgt = self.pose_target[b + 2] - _sl * 0.65 * _amp
            else:
                # v220g: 横脊迈步（保持原行为）。前腿 q1 -1.16->-0.5(+0.66)、
                # 后腿 +1.16->+0.5(-0.66)，符号按腿分
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
            # v589: 抬轮期间差速淡出（ysc 由 cmd.yaw_scale 传入）——前轮
            # 抬起时左右对转只会空耗推力，四轮统一向前推才能跨棱
            # v625: 楼梯攀爬区纯前向轮速控制（cmd.pure_fwd=1）——去掉差速
            # 与 yaw 反馈，打破"驱动-yaw-滑转"死结（62 组实验定位：差速/yaw
            # 反馈在轮速滑至 v_ref 后只剩对转，车原地不动）
            _pf9 = float(cmd.get("pure_fwd", 0.0))
            if _pf9 > 0.0:
                v_ref = self._vx_f
            else:
                v_ref = (self._vx_f
                         + wd_side * self._om_f * self.track_half * _ysc)
            # v218: 实测 +轮力矩=倒车（S10 轮轴符号），取反前进
            # v218h: 驱动按校准取反，阻尼必须始终反向（否则负转速时放大）
            # v218j: 直接 yaw 差速力矩（轮全幅差速，参考 dial-MPC）
            # v218k: 左转(ω>0)左轮需向后力矩——符号与 wd_side 相反
            # v219o: yaw 差速增益随 |omega 指令| 自适应——直行小增益防
            # 差速振荡（v219m/n 实测 60→5/15），转弯大增益保证转向力。
            _yk = float(os.environ.get(
                "S10_VMC_WBC_YAW_K", str(self.yaw_k_wheel))) * (
                    0.3 + 0.7 * min(abs(self._om_f) / 0.4, 1.0))
            # v237: yaw 高频阻尼——v235 符号修正（-kd 削弱恢复力矩）
            _kd_yaw = float(os.environ.get("S10_CAR_KD_YAW", "2.0"))
            _om_b = body.get("omega_body", body["omega"])
            if _pf9 > 0.0:
                t_yaw = 0.0
            else:
                t_yaw = ((-_yk * (self._om_f - _om_b)
                          + _kd_yaw * _om_b) * wd_side
                         * _ysc * getattr(self, "_ground_f", 1.0))
            t_wheel = (-(self.wheel_k * (v_ref - v_wheel))
                       - self.wheel_d * wq + t_yaw)

            tau[hipx_i] = float(np.clip(t_hipx, -20, 20))
            tau[hipy_i] = float(np.clip(t_hipy, -48, 48))
            tau[knee_i] = float(np.clip(t_knee, -48, 48))
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
                    * self.fk.r, -13.5, 13.5))
            tau[WHEEL_Q_IDX[leg]] = float(np.clip(t_wheel, -_wt, _wt))
            if (os.environ.get('S10_WBC_DEBUG', '0') == '1'
                    and body['pos'][1] > 36.0 and leg == 2):
                print('[WBCD] y=%.2f vx_f=%.2f om_f=%.2f wq=%.1f '
                      'v_ref=%.2f v_wheel=%.2f wk=%.2f t_yaw=%.2f '
                      't_wheel=%.2f wt=%.2f gf=%.2f bodyz=%.3f'
                      % (body['pos'][1], self._vx_f, self._om_f, wq,
                         v_ref, v_wheel, self.wheel_k, t_yaw, t_wheel,
                         _wt, getattr(self, '_ground_f', 1.0),
                         body['pos'][2]),
                      flush=True)
        tau[LEG_CTRL_IDX] = np.clip(tau[LEG_CTRL_IDX], -48, 48)
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
        tau[LEG_CTRL_IDX] = np.clip(tau[LEG_CTRL_IDX], -48, 48)
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
    """v223: lidar 传感器增量建世界高程图（可部署，地图变化鲁棒）。

    从 lidar_site 按机器人航向发射扇形射线（mj_multiRay 批量），命中点
    累积写入**固定世界栅格**（min-z 去噪）——像真机 lidar SLAM 建图：
      - 数据随时间累积，起步坡/远处地形保留（此前局部栅格每帧清空，
        10Hz 更新间隔 0.15m 导致脚下无数据 → 腿塌，wp0→1 卡死根因）
      - geomgroup 只留 group 0（地形），排除机器人/赛道标记
      - 前向扇形 ±fov_h × 俯仰 +10°~-55°（近场地面到远场高台）
      - 高度查询 O(1)；未覆盖格返回 0（真机近场盲区同样存在）
    """

    def __init__(self, model, data, x0=-25.0, x1=40.0, y0=-5.0, y1=55.0,
                 res=0.10, th_n=64, phi_n=32, fov_h=None, cutoff=20.0):
        import mujoco
        self.m, self.d = model, data
        self.res = float(res)
        self.ox, self.oy = float(x0), float(y0)
        self.nx = int(round((x1 - x0) / res)) + 1
        self.ny = int(round((y1 - y0) / res)) + 1
        self.h = np.full((self.ny, self.nx), np.inf, dtype=np.float64)
        self.valid = np.zeros((self.ny, self.nx), dtype=np.int32)
        self.sid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
        if self.sid < 0:
            raise ValueError("lidar_site not found in model")
        self.cutoff = float(cutoff)
        if fov_h is None:
            fov_h = float(np.radians(55))
        ths = np.linspace(-fov_h, fov_h, int(th_n))
        # v293c: 上缘 +10->+45 deg——配 40 deg 前倾安装后中心线仍保留~0 deg 世界俯仰，
        # 远场恢复到 20m cutoff（只加大安装角会把远场压到 ~1m）
        phs = np.linspace(np.radians(45.0), np.radians(-55.0), int(phi_n))
        dirs = []
        for ph in phs:
            for th in ths:
                dirs.append([float(np.cos(ph) * np.cos(th)),
                             float(np.cos(ph) * np.sin(th)),
                             float(np.sin(ph))])
        self.dirs_local = np.asarray(dirs, dtype=np.float64)
        self.geomgroup = np.zeros((mujoco.mjNGROUP,), dtype=np.ubyte)
        self.geomgroup[0] = 1

    def _yaw(self):
        q = self.d.xquat[1]
        return float(np.arctan2(
            2.0 * (q[3] * q[0] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))

    def update(self):
        """发射一帧射线，命中点累积入世界栅格（min-z）。"""
        import mujoco
        m, d = self.m, self.d
        pos = np.asarray(d.site_xpos[self.sid], dtype=np.float64)
        # v293: 射线用 lidar_site 世界姿态（site_xmat）——安装位置/俯仰
        # 真正生效（Go2 式头部安装：base 前 0.20m、上 0.30m、前下 8.6°）；
        # 此前只用机体 yaw，site euler 只是摆设
        xmat = np.asarray(d.site_xmat[self.sid], dtype=np.float64).reshape(3, 3)
        L = self.dirs_local
        vec = (L @ xmat.T)
        n = len(L)
        geomid = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, -1.0, dtype=np.float64)
        norm = np.zeros((n * 3,), dtype=np.float64)
        mujoco.mj_multiRay(m, d, pos.copy(), vec.reshape(-1),
                           self.geomgroup, True, 1, geomid, dist, norm,
                           n, self.cutoff)
        hit = dist > 0.0
        # v369: 只建"近水平面"图——高程图服务于腿阻抗/台阶踏面，必须排除
        # 竖直结构（起步坡门架壁/边缘掠射 0.6-1.0m 虚高读数 -> 前腿过度
        # 伸展 -> yaw 自旋）。|nz|>0.6 视为地面/踏面；竖直面 nz~0 丢弃。
        _nz_min = float(os.environ.get("S10_LIDAR_NZ_MIN", "0.6"))
        _nz = np.abs(norm.reshape(n, 3)[:, 2])
        hit = hit & (_nz >= _nz_min)
        if hit.any():
            pts = pos + dist[:, None] * vec
            for i in np.where(hit)[0]:
                p = pts[i]
                ix = int(np.floor((p[0] - self.ox) / self.res))
                iy = int(np.floor((p[1] - self.oy) / self.res))
                if 0 <= ix < self.nx and 0 <= iy < self.ny:
                    # min-z：地面优先（多条射线/多帧取最低）
                    if p[2] < self.h[iy, ix]:
                        self.h[iy, ix] = p[2]
                    self.valid[iy, ix] = 1

    def has(self, x, y):
        """精确格是否有 lidar 数据（供运动学 fallback）。"""
        ix = int(np.floor((x - self.ox) / self.res))
        iy = int(np.floor((y - self.oy) / self.res))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return bool(self.valid[iy, ix])
        return False

    def height(self, x, y):
        """精确格优先（射线 64x32 加密后覆盖率足够），无数据返回 0。
        3x3 空间扩散会把脊值混入脊前格致转弯差速失效（car25 偏出根因）。"""
        ix = int(np.floor((x - self.ox) / self.res))
        iy = int(np.floor((y - self.oy) / self.res))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            if self.valid[iy, ix]:
                return float(self.h[iy, ix])
        return 0.0


class CarVMC:
    """v221: 车化巡航控制器——轮=驱动+差速转向，腿=刚性主动悬架(姿态)。

    文献依据：
    - BIT "Enhancing high-speed steering stability of wheel-legged vehicles by
      active roll control"（yaw-roll 耦合模型，主动 roll 控制=压弯）：
      高速转向稳定性核心是 roll 控制，不是腿力 wrench。
    - "Static Stability ... Wheel-Legged Agricultural Robot"：leg-height 主动
      调节提高最大可容许 roll 角 85-90%、改善 pitch——腿长差=姿态悬架。
    - skid-steering 简单控制（BIT 2023）：vx PID + yaw rate 反馈即可。

    结构（比 wrench/pinv 简单一个量级）：
    1. 轮：v_ref = vx +- omega*track_half；vx 速度跟踪 + yaw rate 差速反馈
    2. 腿：每腿独立垂直力 F = mg/4 + roll分配 + pitch分配 + 地形阻抗
       roll: 左+右- 差 = kp_roll*(roll_tar-roll) - kd_roll*roll_rate
       pitch: 前-后+ 差 = kp_pitch*(pitch_tar-pitch) - kd_pitch*pitch_rate
    3. 压弯：roll_tar 随 vx*omega（弯内倾），由腿长差执行
    4. hipx：位置 PD + 侧身微调（随 roll_tar）
    5. 轮力矩动态抓地钳制（按 F_leg 实际载荷）
    """

    def __init__(self, mass=19.0, g=9.81, L1=0.18, L2=0.18, r=0.081,
                 track_half=0.24, kp_leg=80.0, kd_leg=6.0,
                 kp_h=200.0, kd_h=40.0,
                 kp_roll=120.0, kd_roll=12.0,
                 kp_pitch=150.0, kd_pitch=15.0,
                 wheel_k=4.0, wheel_d=0.08):
        import os
        self.m, self.g = mass, g
        self.fk = S10LegFK(L1, L2, r)
        self.track_half = track_half
        self.kp_leg, self.kd_leg = kp_leg, kd_leg
        self.kp_h = float(os.environ.get("S10_VMC_KPH", str(kp_h)))
        self.kd_h = float(os.environ.get("S10_VMC_KDH", str(kd_h)))
        self.kp_roll = float(os.environ.get("S10_CAR_KP_ROLL", str(kp_roll)))
        self.kd_roll = float(os.environ.get("S10_CAR_KD_ROLL", str(kd_roll)))
        self.kp_pitch = float(os.environ.get("S10_CAR_KP_PITCH", str(kp_pitch)))
        self.kd_pitch = float(os.environ.get("S10_CAR_KD_PITCH", str(kd_pitch)))
        self.wheel_k = float(os.environ.get("S10_VMC_WHEEL_K", str(wheel_k)))
        self.wheel_d = float(os.environ.get("S10_VMC_WHEEL_D", str(wheel_d)))
        self.yaw_k_wheel = float(os.environ.get(
            "S10_VMC_YAW_K_WHEEL", "60.0"))
        # v234: 巡航半蹲（轮足姿态总结）——knee 2.30->1.90 降质心 ~6cm，
        # 减侧翻矩、保四轮法向均载、弱化微起伏传递。S10_CAR_SQUAT=0 回站立
        if os.environ.get("S10_CAR_SQUAT", "1") == "1":
            self.pose_target = np.array([-0.05, -1.10, 1.90,
                                          0.05, -1.10, 1.90,
                                         -0.05,  1.10, -1.90,
                                          0.05,  1.10, -1.90],
                                        dtype=np.float64)
        else:
            self.pose_target = np.array([-0.05, -1.16, 2.30,
                                          0.05, -1.16, 2.30,
                                         -0.05,  1.16, -2.30,
                                          0.05,  1.16, -2.30],
                                        dtype=np.float64)
        # roll 分配符号：左腿 +，右腿 -
        self.roll_sign = np.array([1.0, -1.0, 1.0, -1.0])
        # pitch 分配符号：前腿 -，后腿 +
        self.pitch_sign = np.array([-1.0, -1.0, 1.0, 1.0])
        self._vx_f, self._om_f = 0.0, 0.0
        self._roll_f, self._pitch_f = 0.0, 0.0
        self._roll_prev = None
        self._pitch_prev = None
        self._ground_f = 1.0

    def _body_state(self, qpos, qvel):
        q = qpos[3:7]
        w, x, y, z = q
        yaw = float(np.arctan2(2.0 * (w * z + x * y),
                               1.0 - 2.0 * (y * y + z * z)))
        roll = float(np.arctan2(2.0 * (w * x + y * z),
                                1.0 - 2.0 * (x * x + y * y)))
        pitch = float(np.arctan2(2.0 * (w * y - z * x),
                                 1.0 - 2.0 * (y * y + x * x)))
        # v237: body 系 yaw 率（世界系 qvel[5] 会被地形俯仰/横滚污染）
        w, x, y, z = q
        z_axis = np.array([2.0 * (x * z + w * y),
                           2.0 * (y * z - w * x),
                           1.0 - 2.0 * (x * x + y * y)])
        omega_body = float(np.dot(qvel[3:6], z_axis))
        return dict(yaw=yaw, roll=roll, pitch=pitch,
                    omega=float(qvel[5]), omega_body=omega_body)

    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel,
                    cmd, terrain_h, dt=0.005):
        body = self._body_state(qpos, qvel)
        k = min(1.0, dt / 0.10)
        self._vx_f += (float(cmd["vx"]) - self._vx_f) * k
        self._om_f += (float(cmd["omega"]) - self._om_f) * k
        roll_tar = float(cmd.get("roll_tar", 0.0))
        pitch_tar = float(cmd.get("pitch_tar", 0.0))
        self._roll_f += (roll_tar - self._roll_f) * k
        self._pitch_f += (pitch_tar - self._pitch_f) * k
        roll_rate = ((body["roll"] - self._roll_prev) / max(dt, 1e-4)
                     if self._roll_prev is not None else 0.0)
        pitch_rate = ((body["pitch"] - self._pitch_prev) / max(dt, 1e-4)
                      if self._pitch_prev is not None else 0.0)
        self._roll_prev = body["roll"]
        self._pitch_prev = body["pitch"]

        # 腾空衰减（横脊/跳跃时关 yaw 反馈）
        if wheel_xyz is not None and terrain_h is not None:
            _lift_amt = float(np.mean(
                wheel_xyz[:, 2] - (np.asarray(terrain_h) + self.fk.r)))
            self._ground_f = float(np.clip(
                1.0 - max(0.0, _lift_amt - 0.02) / 0.05, 0.0, 1.0))
        else:
            self._ground_f = 1.0

        # 姿态力矩（腿长差）
        R = (self.kp_roll * (self._roll_f - body["roll"])
             - self.kd_roll * roll_rate)
        P = (self.kp_pitch * (self._pitch_f - body["pitch"])
             - self.kd_pitch * pitch_rate)
        _tmax = float(os.environ.get("S10_CAR_ATT_TMAX", "40.0"))
        R = float(np.clip(R, -_tmax, _tmax))
        P = float(np.clip(P, -_tmax, _tmax))

        # 轮差速 yaw 反馈（自适应：转弯大、直行小）
        _ysc = float(cmd.get("yaw_scale", 1.0))
        _yk = self.yaw_k_wheel * (0.3 + 0.7 * min(abs(self._om_f) / 0.4, 1.0))

        tau = np.zeros(16, dtype=np.float64)
        step_lift = np.asarray(cmd.get("step_lift", np.zeros(4)))
        hop = cmd.get("hop")
        for leg in range(4):
            b = leg * 3
            hipx_i, hipy_i, knee_i = (
                LEG_CTRL_IDX[b], LEG_CTRL_IDX[b + 1], LEG_CTRL_IDX[b + 2])
            q0 = float(qpos[LEG_Q_IDX[b]])
            q1 = float(qpos[LEG_Q_IDX[b + 1]])
            q2 = float(qpos[LEG_Q_IDX[b + 2]])
            J = self.fk.jac(q1, q2)
            sl = float(step_lift[leg])

            # 腿垂直支撑力：mg/4 + 姿态分配 + 地形阻抗
            F = self.m * self.g / 4.0
            # v221c: 压弯——符号以实验为准（F+roll_sign 在 car1 全程无侧翻；
            # 取反后实测更右倾+wq 振荡，回退）
            F += self.roll_sign[leg] * R
            F += self.pitch_sign[leg] * P
            p = wheel_xyz[leg] if wheel_xyz is not None else None
            # v449/v451: 软抬轮技能（台阶/陡升段）——lift_f_scale<1 时保留
            # 部分支撑力+抬升力（保牵引爬 0.125m 台阶）；巡航平脊段保持原
            # 硬抬轮（max 后再乘 (1-sl)，与 v445 提交版逐字一致）。
            _fscale = float(cmd.get(
                "lift_f_scale", os.environ.get("S10_VMC_LIFT_F_SCALE", "1.0")))
            if _fscale < 1.0:
                if p is not None and terrain_h is not None:
                    pz_des = float(terrain_h[leg]) + self.fk.r
                    F += (self.kp_h * (pz_des - p[2])
                          - self.kd_h * float(wheel_vel[leg, 2]))
                F = max(F, 2.0) * (1.0 - _fscale * sl)
            else:
                if p is not None and terrain_h is not None:
                    pz_des = float(terrain_h[leg]) + self.fk.r
                    F += (1.0 - sl) * (
                        self.kp_h * (pz_des - p[2])
                        - self.kd_h * float(wheel_vel[leg, 2]))
                F = max(F, 2.0) * (1.0 - sl)
            if hop is not None:
                F += float(hop[leg])

            # 腿关节力矩（J^T 垂直力）+ 位置 PD
            t_hipy, t_knee = J.T @ np.array([0.0, F])
            # 迈步：hipy 前摆 + knee 伸直
            # v453: 抬轮摆动幅度可调（S10_VMC_LIFT_SWING 默认 0.66）——
            # wp5→6 台阶实测 0.66rad 只抬轮 0.045m，不够 0.125m 台阶。
            _qs = -1.0 if leg in (0, 1) else 1.0
            _lsw = float(cmd.get(
                "lift_swing", os.environ.get("S10_VMC_LIFT_SWING", "0.66")))
            _q1_tgt = self.pose_target[b + 1] - sl * _lsw * _qs
            _q2_tgt = self.pose_target[b + 2] + sl * 0.42
            # v221i: 车身抬升（过脊用）——0.05 只到 0.68(差 5mm)，改 0.08
            _bl = float(cmd.get("body_lift", 0.0))
            _q2_tgt += _bl * 0.08
            t_hipy += (self.kp_leg * (_q1_tgt - q1)
                       - self.kd_leg * float(qvel[6 + LEG_QV_LEG[b + 1]]))
            t_knee += (self.kp_leg * (_q2_tgt - q2)
                       - self.kd_leg * float(qvel[6 + LEG_QV_LEG[b + 2]]))
            # hipx：位置 PD + 侧身（压弯）+ v232 扭胯 yaw 辅助——
            # 左转(om_f>0)时左腿 hipx 外展/右腿内收，产生 yaw 力矩
            # （用户"扭肩/胯转向"思路，突破轮差速 0.65rad/s 物理上限）
            _q0_tgt = self.pose_target[b] - 0.12 * self.roll_sign[leg] * self._roll_f
            _hipx_yaw = float(os.environ.get("S10_CAR_HIPX_YAW", "0.0"))
            if _hipx_yaw > 0.0:
                _ys = -1.0 if leg in (0, 1) else 1.0   # 前腿/后腿
                _q0_tgt += _hipx_yaw * _ys * self._om_f * self._ground_f
            t_hipx = (self.kp_leg * (_q0_tgt - q0)
                      - self.kd_leg * float(qvel[6 + LEG_QV_LEG[b]]))

            tau[hipx_i] = float(np.clip(t_hipx, -20, 20))
            tau[hipy_i] = float(np.clip(t_hipy, -48, 48))
            tau[knee_i] = float(np.clip(t_knee, -48, 48))

            # 轮：差速 + yaw 反馈 + 动态抓地钳制
            wq = float(qvel[WHEEL_QV_IDX[leg]])
            v_wheel = -wq * self.fk.r
            side = -1.0 if leg in (0, 2) else 1.0
            # v237: yaw 高频阻尼——修正 v235 符号（-kd 削弱恢复力矩），
            # body 系 yaw 率 + 高通（只阻尼振荡，稳态零偏置；RobuROC6 思路）
            _kd_yaw = float(os.environ.get("S10_CAR_KD_YAW", "2.0"))
            _om_b = body.get("omega_body", body["omega"])
            if not hasattr(self, "_omega_lp"):
                self._omega_lp = _om_b
            _om_hf = _om_b - self._omega_lp
            self._omega_lp += (_om_b - self._omega_lp) * min(1.0, dt / 0.05)
            # v249/v251: yaw 超速保护——|ω| 超 a_lat 安全包线
            # （S10_AUTO_LAT_MAX/v）时：差速参考**反向**（按安全转速反向给
            # 速度指令，硬刹）+ 阻尼+8。
            _latmax = float(os.environ.get("S10_AUTO_LAT_MAX", "5.0"))
            # v283: 保护上限 = min(a_lat/v, 绝对 ω 上限 2.0)——低速自旋
            # （ω 2.2, v 0.4）时 a_lat 很小但自旋本身失稳，需绝对上限
            _om_safe = min(
                _latmax / max(abs(self._vx_f), 0.5),
                float(os.environ.get("S10_VMC_OM_ABS_MAX", "2.0")))
            _kd_eff = _kd_yaw
            # v284: v_ref 差速参考用快速低通（τ=0.02）——即时指令在导航
            # 方向翻转时太暴力（轮差速瞬翻致 S 弯自旋）；0.1s 慢低通又滞后
            # 推旧方向。0.02s 兼顾方向响应与平滑。
            _om_ref = float(cmd.get("omega", 0.0))
            _om_ref_tau = float(os.environ.get("S10_CAR_OM_REF_TAU", "0.02"))
            if not hasattr(self, "_om_ref_f"):
                self._om_ref_f = _om_ref
            self._om_ref_f += (_om_ref - self._om_ref_f) * min(1.0, dt / _om_ref_tau)
            _om_ref = self._om_ref_f
            # v428: Tube-MPPI 灵感（RSS18 Williams et al.）——MPPI 规划
            # 标称轨迹，低层跟踪器保证实际轨迹在"管"内：管内不干预（让差速
            # 前馈自然执行），只有偏差超管才拉回。对应 yaw 通道：
            # |ω_act-ω_cmd| ≤ TUBE 时不加滑模反馈、不触发 om_safe 反向刹车
            # （转弯 cmd1.4/act1.6 的轻微过速不再反向硬刹 → 消除 wp4→5
            # 过冲振荡翻车）；超管才用 tanh 饱和拉回。默认 0.0 = 原行为。
            # v429: 刹车管（独立旋钮）——om_safe 反向硬刹只在实际 ω 偏差
            # 超过 S10_CAR_YAW_BRAKE_TUBE 时才触发（弯道 cmd1.8/act2.6 的
            # 合理过速不再瞬间反向差速 → 消除振荡翻车）；S10_CAR_YAW_TUBE
            # 单独控制滑模反馈死区。默认 0 = 原行为。
            _tube = float(os.environ.get("S10_CAR_YAW_TUBE", "0.0"))
            _btube = float(os.environ.get("S10_CAR_YAW_BRAKE_TUBE", "0.0"))
            _err_y = self._om_f - body["omega"]
            if abs(_om_b) > _om_safe and abs(_err_y) > _btube:
                _om_ref = -float(np.clip(_om_b, -_om_safe, _om_safe))
                _kd_eff = _kd_yaw + 8.0
            # v252: 差速参考用**即时指令**（导航已 slew 0.8/s，够平滑）——
            # 低通 _om_f 在指令方向翻转时滞后 ~0.3s，继续推旧方向致过冲
            # （wp4→5 指令+1.72 实际-2.86 翻车实测）；低通只留反馈项用。
            # v317: 差速参考随抓地系数衰减——轮子因地形/抬轮离地时保持
            # 差速会在落地瞬间产生 yaw 冲击（起步坡 wz0.6 实测 yaw 1.6->0.2
            # 自旋）；与 yaw 反馈一致用 ground_f 淡出差速，接地后恢复。
            v_ref = (self._vx_f
                     + side * _om_ref * _ysc * self._ground_f
                     * self.track_half)
            # v241/v242: yaw 摩擦前馈（RobuROC6 库仑摩擦补偿）——差速转向需
            # 先克服侧向滑移阻力才有 yaw 运动，纯误差反馈有死区滞后；按指令
            # 方向给基础差速力矩。**默认 0**：v241 线性 FF 在导航指令突变时
            # 过驱动（wp1→2 振荡侧翻实测）；启用时加 0.15s 低通防瞬翻。
            _kff = float(os.environ.get("S10_CAR_YAW_FF", "0.0"))
            if not hasattr(self, "_om_ff_lp"):
                self._om_ff_lp = 0.0
            self._om_ff_lp += (self._om_f - self._om_ff_lp) * min(1.0, dt / 0.15)
            # v289: 滑模式 yaw（RobuROC6）——饱和 tanh 误差项（小误差高增益
            # 快收敛、大误差饱和不过冲，替代纯比例→消除低速极限环）+
            # 库仑摩擦前馈（克服低速静摩擦）+ 高频阻尼。
            _k_sm = float(os.environ.get("S10_CAR_YAW_K_SM", "30.0"))
            _phi = float(os.environ.get("S10_CAR_YAW_PHI", "0.5"))
            # v436: 滑模误差改用 body 系 yaw 率（S10_CAR_YAW_OM_BODY 默认
            # 0=原世界系 qvel[5]）。v237 注释即指出世界系被地形俯仰/横滚
            # 污染——脊上 θ̇ 会混进 qvel[5]，滑模误判 yaw 误差→脊期需要
            # 冻结兜底；body 系 ωz≈ψ̇·cosθ 隔离俯仰，脊上不误动。
            if float(os.environ.get("S10_CAR_YAW_OM_BODY", "0")) > 0:
                _err_y = self._om_f - body["omega_body"]
            # v428: 管误差 = 超出 |误差|≤TUBE 的部分（死区），管内误差=0
            _err_t = _err_y
            if _tube > 0.0:
                _err_t = float(np.sign(_err_y)) * max(
                    abs(_err_y) - _tube, 0.0)
            _ff_sign = 0.0
            if abs(self._om_f) > 0.05:
                _ff_sign = float(np.sign(self._om_f)) * min(
                    abs(self._om_f) / 0.3, 1.0)
            t_yaw = ((-_k_sm * float(np.tanh(_err_t / _phi))
                      - _kff * _ff_sign
                      + _kd_eff * _om_hf) * side
                     * _ysc * self._ground_f)
            # v246: yaw 力矩限速——滑移权威（YAW_TMAX）下首次过冲瞬态太快
            # （实测 ω 冲到 3.6 翻车）；限 t_yaw 变化率，平滑起转与刹车。
            # v280: 动态 yaw slew——|yaw 误差|大时放开（高速/大 err 需激进
            # 转向，30→60+；连续 err 驱动，非门控）
            _yerr = abs(self._om_f - body["omega"])
            _tslew = float(os.environ.get("S10_CAR_YAW_SLEW", "30.0")) * float(
                np.clip(1.0 + _yerr / float(os.environ.get(
                    "S10_CAR_YAW_SLEW_K", "1.0")), 1.0, 4.0))
            if not hasattr(self, "_t_yaw_prev"):
                self._t_yaw_prev = np.zeros(4)
            _ty = float(np.clip(
                t_yaw,
                self._t_yaw_prev[leg] - _tslew * dt,
                self._t_yaw_prev[leg] + _tslew * dt))
            self._t_yaw_prev[leg] = _ty
            t_yaw = _ty
            t_wheel = (-(self.wheel_k * (v_ref - v_wheel))
                       - self.wheel_d * wq + t_yaw)
            # 动态钳制：按该腿实际分配载荷
            _fz_load = max(F, 0.5 * self.m * self.g / 4.0)
            _mu_w = float(os.environ.get("S10_VMC_WHEEL_MU", "0.9"))
            # v244: 差速滑移余量——牵引钳制 μN·r 掐死差速滑移（yaw 权威上限
            # ~0.75 rad/s）；允许差速分量额外力矩（执行器上限 14Nm 内），
            # 使轮子受控滑移转向（dial-MPC 轮速控制时代 1.5+ rad/s 的原理）。
            _ytm = float(os.environ.get("S10_VMC_YAW_TMAX", "0.0"))
            _wt = float(np.clip(
                _mu_w * _fz_load * self.fk.r
                + _ytm * min(abs(self._om_f) / 0.5, 1.0), -13.5, 13.5))
            tau[WHEEL_Q_IDX[leg]] = float(np.clip(t_wheel, -_wt, _wt))
        tau[LEG_CTRL_IDX] = np.clip(tau[LEG_CTRL_IDX], -48, 48)
        return tau


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
        tau[LEG_CTRL_IDX] = np.clip(tau[LEG_CTRL_IDX], -48, 48)
        return tau
