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
        # v746: 楼梯区 vx 低通加快（0.04s）——0.25s 斜坡让狗在楼梯前慢速
        # 蠕动 v_ref 0.03（实测 WBC 卡死），CPG 抬轮永远等不到。巡航保持 0.25s
        # 温和加速（防轮腿猛推抬头）。
        _vt = float(os.environ.get("S10_VMC_VX_TAU", "0.25"))
        if float(cmd.get("z_min", 0.0)) > 0.0:
            _vt = 0.04
        k = min(1.0, dt / _vt)
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
        # v792: 楼梯区 z_des 模式（S10_VMC_WBC_Z_MODE）——0=min（后轮贴地
        # 保牵引，但前上台面时后腿被顶到极限→body 低头钉死实测）；1=mean
        # （前后轴平均，body 随前轮升高、后腿可收缩，配合抬头 pitch 实现
        # USC 关键姿态）；2=max（前轮主导，后腿可能够不到地面）。
        _zmode = float(os.environ.get("S10_VMC_WBC_Z_MODE", "0"))
        if _zm9 > 0.0 and _zmode == 1:
            _zt9 = float(np.mean(terrain_h))
        elif _zm9 > 0.0 and _zmode == 2:
            _zt9 = float(np.max(terrain_h))
        elif _zm9 > 0.0:
            _zt9 = float(np.min(terrain_h))
        else:
            _zt9 = float(np.mean(terrain_h))
        z_des = _zt9 + float(os.environ.get(
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
        # v755e: 可选恢复楼梯区 z 位置控制（S10_VMC_WBC_Z_KP>0）——抬腿的
        # 反作用力把 body 顶起→后轮悬空无推力死锁实测；z 控制按住 body
        # 让抬升作用在轮上而非车身上（默认 0 保持 v605 行为）。
        _zkp9 = float(os.environ.get("S10_VMC_WBC_Z_KP", "0.0"))
        F_des_w = np.array([
            0.0, 0.0,
            self.m * self.g
            + (0.0 if _zm9 > 0.0 and _zkp9 <= 0.0
               else (_zkp9 if _zm9 > 0.0 else self.kp_z)
               * (z_des - body["pos"][2]))
            - self.kd_z * float(qvel[2])])
        # v785: 抬轮反作用补偿（楼梯区）——前轮抬升把 body 顶起、后轮
        # 悬空无牵引实测；按平均抬轮幅度给 body 加下压力（支撑腿执行），
        # 后轮保持贴地推力。S10_VMC_WBC_Z_LCOMP 默认 0 关闭。
        _zlc = float(os.environ.get("S10_VMC_WBC_Z_LCOMP", "0.0"))
        if _zm9 > 0.0 and _zlc > 0.0:
            F_des_w[2] -= _zlc * float(np.mean(
                np.asarray(cmd.get("step_lift", np.zeros(4)))))
        # v218k: 驱动 25% 由腿分担（轮为主），全轮在坡上推力不足
        _dsh = float(os.environ.get("S10_VMC_DRIVE_SHARE", "0.0"))
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
        # v770: 楼梯区（z_min>0）关闭加速俯冲前馈（加速时 -ff·m·ax 把
        # body 压成低头，与爬楼抬头姿态相反实测）+ 姿态力矩上限放开
        # （巡航 25Nm 被抬轮反作用饱和，pitch 无法抬头实测）。
        _pff9 = 0.0 if _zm9 > 0.0 else self.pitch_ff
        T_pitch_b = (self.kp_pitch * (pitch_tar - body["pitch"])
                     - self.kd_pitch * pitch_rate
                     - _pff9 * self.m * ax * 0.20)
        # v218h: 力矩钳制在支撑多边形可行域内（后轮无法上拉，|T|≤mg·lever/2）
        _tmax = float(os.environ.get("S10_VMC_TMAX", "25.0"))
        if _zm9 > 0.0:
            _tmax = float(os.environ.get("S10_VMC_STAIR_TMAX", "80.0"))
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
        _sl_all = np.asarray(cmd.get("step_lift", np.zeros(4)))
        _support_legs = []
        for leg in range(4):
            rw = wheel_xyz[leg] - body["pos"]
            S = np.array([
                [0.0, -rw[2], rw[1]],
                [rw[2], 0.0, -rw[0]],
                [-rw[1], rw[0], 0.0]])
            # v746: 抬轮腿（sl>0.3）从 wrench 求解中屏蔽——支撑腿承担全部
            # 载荷。pinv 后乘 (1-sl) 会破坏解使总支撑不足、身体塌陷悬空
            # （WBC fn=0 卡死实测）。屏蔽后支撑腿力自然加大，抬轮腿=0。
            # v767: 楼梯区（z_min>0）可保留抬轮腿进 wrench（S10_VMC_WBC_
            # KEEP_LIFT=1）——屏蔽后 pitch/z 控制对前腿失去权威，body 被
            # 抬轮反作用顶成低头悬空实测（pitch -0.3）；保留后姿态可控。
            if (float(_sl_all[leg]) > 0.3
                    and not (_zm9 > 0.0 and float(os.environ.get(
                        "S10_VMC_WBC_KEEP_LIFT", "0")) > 0)):
                continue
            _support_legs.append(leg)
            A6[0:3, leg * 3:leg * 3 + 3] = -np.eye(3)
            A6[3:6, leg * 3:leg * 3 + 3] = -S
        try:
            f_legs = np.linalg.pinv(A6) @ W
            # 抬轮腿力归零（未参与求解）
            for leg in range(4):
                if leg not in _support_legs:
                    f_legs[leg * 3:leg * 3 + 3] = 0.0
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
            # v764: 楼梯区单侧阻抗目标下调下压余量——mujoco 接触间隙 3-8mm，
            # 轮高到位但 fn=0 无牵引实测；下压压实台面恢复接触
            _press9 = (float(os.environ.get("S10_VMC_WBC_PRESS", "0.006"))
                       if _zm9 > 0.0 else 0.0)
            pz_des = float(terrain_h[leg]) + self.fk.r - _press9
            # v602: 抬轮时地形阻抗**不随 sl 衰减**（(1-zk*sl)）——楼梯落脚点
            # 把前轮地形置为台面高，阻抗把轮拉到 pz_des=台面+r，轮直接
            # 落上台面（原 (1-sl) 在 sl=1 时清零，抬轮只剩姿态 PD、够不到）
            # v763: 楼梯区（z_min>0）垂直阻抗改**单侧**（只下压不吸抬）——
            # 双侧弹簧在 ramp 目标高于轮位时把前轮"吸"离台面悬空 fn=0 卡死
            # 实测；下压保持贴地牵引，抬升由 CPG 抬轮负责（USC 贴面滚上）。
            _imp_w = (1.0 - _zk * _sl)
            _dz_h = pz_des - p[2]
            # v813: 爬升窗双侧/台面单侧——轮在 riser 爬升窗内（climb_mask）
            # 用双侧（沿面拉高+面摩擦=贴面爬升）；台面上单侧（只下压贴地
            # 抓地，防吸抬悬空 fn=0 打滑）。默认 ONESIDED=1 时台面单侧、
            # 爬升窗双侧（v790 双侧全开在台面吸抬、单侧全关拉不动面的
            # 各自缺陷合并解决）。
            _climb_l = np.asarray(cmd.get("climb_mask", np.zeros(4)))
            _oneside_eff = (float(os.environ.get(
                "S10_VMC_WBC_ONESIDED", "1")) == "1"
                and float(_climb_l[leg]) < 0.5)
            if _oneside_eff and _zm9 > 0.0 and _dz_h > 0.0:
                fw[2] += _imp_w * (-self.kd_h * float(wheel_vel[leg, 2]))
            else:
                fw[2] += _imp_w * (
                    self.kp_h * _dz_h
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
            # v789: v752 USC 关键姿态——前轴近棱时前腿膝伸直(z_down 增)+
            # 后腿膝屈(z_down 减)，几何 body 前倾让前轮接触台面。
            _sp3 = float(cmd.get("stair_pose", 0.0))
            if _sp3 > 0.0 and float(cmd.get("z_min", 0.0)) > 0.0:
                _spamp = float(os.environ.get("S10_VMC_STAIR_POSE", "1.0"))
                if _qs < 0.0:
                    _q2_tgt += _sp3 * -0.45 * _spamp
                else:
                    _q2_tgt += _sp3 * -0.22 * _spamp
            # v773: 楼梯区抬轮腿（sl>0.3）姿态 PD 软增益——满增益 kp=80 的
            # 抬轮力(~880N) 超过后轮支撑力(~530N)，把 body 顶起全轮悬空
            # 卡死实测；软增益让抬轮温和，后轮压住 body，轮缓慢抬起。
            _kpp = self.kp_pose
            if (_zm9 > 0.0 and _sl > 0.3
                    and float(os.environ.get(
                        "S10_VMC_WBC_SOFT_LIFT", "0.0")) > 0.0):
                _kpp *= float(os.environ.get("S10_VMC_WBC_SOFT_LIFT", "0.35"))
            t_hipy += (_kpp * (_q1_tgt - q1)
                       - self.kd_pose * float(qvel[6 + LEG_QV_LEG[b + 1]]))
            t_knee += (_kpp * (_q2_tgt - q2)
                       - self.kd_pose * float(qvel[6 + LEG_QV_LEG[b + 2]]))
            # v783: 楼梯区抬轮腿力矩限幅（力限抬轮）——pose PD 刚度高时
            # 抬轮力(~880N) 超过后轮支撑力把 body 顶起全轮悬空卡死实测；
            # 限幅到 S10_VMC_WBC_LIFT_TMAX(默认15Nm→~83N/腿) 让抬轮温和、
            # 后轮压住 body，轮缓慢抬起（配合 STAIR_WIN_VX 降低给足时间）。
            if (_zm9 > 0.0 and _sl > 0.3
                    and float(os.environ.get(
                        "S10_VMC_WBC_LIFT_TMAX", "0.0")) > 0.0):
                _lt9 = float(os.environ.get("S10_VMC_WBC_LIFT_TMAX", "15.0"))
                t_hipy = float(np.clip(t_hipy, -_lt9, _lt9))
                t_knee = float(np.clip(t_knee, -_lt9, _lt9))

            # v218f: hipx 由 wrench 侧向力接管；S10_VMC_HIPX_TORQUE=1 叠加姿态反馈
            side = -1.0 if leg in (0, 2) else 1.0
            wd_side = side   # v218j: 左转(ω>0)需左轮慢右轮快
            if os.environ.get("S10_VMC_HIPX_TORQUE", "0") == "1":
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
            # v737: 抬升腿轮速归零（后推前抬）——抬轮时轮已离地，继续
            # 指令 1.8 只会空转打滑（wq=-41 实测卡死）；支撑腿保持驱动
            if _sl > 0.1:
                v_ref = 0.0
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
                # v635: yaw 反馈死区（S10_VMC_WBC_YAW_TUBE）——小误差不驱动
                # 差速（实测 om_f≈0.2 时 t_yaw 仍覆盖 10Nm 驱动，车原地对转）；
                # 大误差照常纠偏。连续量，无门控。
                _ytube = float(os.environ.get("S10_VMC_WBC_YAW_TUBE", "0.15"))
                _yerr = self._om_f - _om_b
                _yerrd = float(np.sign(_yerr)) * max(
                    abs(_yerr) - _ytube, 0.0)
                t_yaw = ((-_yk * _yerrd
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
                _mu_w = float(os.environ.get("S10_VMC_WHEEL_MU", "0.9"))
                _fz_load = (self.m * self.g / 4.0
                            + self.kp_h * (pz_des - p[2])
                            - self.kd_h * float(wheel_vel[leg, 2]))
                _wt_curve = (_mu_w * max(
                    _fz_load, 0.5 * self.m * self.g / 4.0) * self.fk.r)
                _wt_straight = float(os.environ.get(
                    "S10_VMC_WBC_WHEEL_TMAX", "13.5"))
                # v746: 直线全力/弯道收敛（同 CarVMC）——WBC 全程 μN·r
                # 只有 1.7-3.4Nm，楼梯前推进力不足 fn 单轮卡死实测
                _yerr = abs(float(cmd.get("yaw_err", 0.0)))
            # v869: wheel torque gain fused - err large -> friction cone;
            # ridge near -> friction cone (continuous, S10_VMC_WT_RIDGE_D
            # removed; ridge<0.5m fades to 0, restored by 0.8m).
            _w_ge = float(np.clip(
                1.0 - _yerr / float(os.environ.get(
                    "S10_VMC_WT_ERR_GATE", "0.4")), 0.0, 1.0))
            _rd = float(cmd.get("ridge_dist", 99.0))
            _w_gr = float(np.clip((_rd - 0.5) / 0.3, 0.0, 1.0))
            _w_f = _w_ge * _w_gr
            _wt = float(np.clip(
                    _w_f * _wt_straight + (1.0 - _w_f) * _wt_curve,
                    -13.5, 13.5))
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

    def stair_confirmed(self, robot_xy, yaw,
                        fwd_dists=(0.3, 0.6, 0.9, 1.2, 1.5),
                        rise=0.08):
        """v871: 感知确认——前方 yaw 窗口内离散台阶（相邻格高差 >= rise
        的上升沿）>=1 处即判定楼梯区。世界坐标，与机体倾斜无关。"""
        fx = float(np.cos(yaw)); fy = float(np.sin(yaw))
        for d in fwd_dists:
            x = float(robot_xy[0]) + fx * d
            y = float(robot_xy[1]) + fy * d
            if self._step_at(x, y, rise):
                return True
        return False

    def _step_at(self, x, y, rise=0.08):
        ix = int(np.floor((x - self.ox) / self.res))
        iy = int(np.floor((y - self.oy) / self.res))
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return False
        if not self.valid[iy, ix]:
            return False
        h0 = float(self.h[iy, ix])
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i2 = iy + di
                j2 = ix + dj
                if 0 <= i2 < self.ny and 0 <= j2 < self.nx                         and self.valid[i2, j2]:
                    if float(self.h[i2, j2]) - h0 >= rise:
                        return True
        return False


def car_omega_limit(vx):
    # CarVMC 极限转向能力表（cap 测试实测：vx<=2 ω=3.0；vx=3→1.5；
    # vx=4→1.2 超过侧翻）。保守取值防翻，压弯加强后可上修。
    import os as _os
    _tbl = [(0.0, 3.0), (2.0, 3.0), (3.0, 1.5), (3.5, 1.5),
            (4.0, 1.2), (5.0, 1.0), (6.0, 0.8)]
    _env = _os.environ.get("S10_CAR_OMEGA_TBL", "")
    if _env:
        try:
            _tbl = [(float(a.split(":")[0]), float(a.split(":")[1]))
                    for a in _env.split(",") if ":" in a]
        except Exception:
            pass
    import numpy as _np
    _v0 = _np.array([t[0] for t in _tbl], dtype=_np.float64)
    _w0 = _np.array([t[1] for t in _tbl], dtype=_np.float64)
    return _np.interp(_np.abs(_np.asarray(vx, dtype=_np.float64)),
                      _v0, _w0, left=_w0[0], right=_w0[-1])


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

    def _ik(self, xd, zd, q1, q2):
        """2D IK：轮目标 (xd, zd)（相对髋 sagittal，zd 向上）→ q1/q2。"""
        for _ in range(8):
            p = self.fk.wheel_pos(q1, q2)
            err = np.array([xd - p[0], zd + p[1]])
            J = self.fk.jac(q1, q2)
            dq = np.linalg.lstsq(J, err, rcond=None)[0]
            dq = np.clip(dq, -0.3, 0.3)
            q1 += float(dq[0]); q2 += float(dq[1])
        return q1, q2

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
            # v870: ground_f more sensitive (0.01m onset, 0.04m zero) - MU=0.8
            # MPPI outputs bigger omega at startup; airborne 0.04m kept 60%
            # differential and spun (om -4.16 flip). Cruise lift<0.01 unaffected.
            self._ground_f = float(np.clip(
                1.0 - max(0.0, _lift_amt - 0.01) / 0.03, 0.0, 1.0))
        else:
            self._ground_f = 1.0

        # 姿态力矩（腿长差）——楼梯区 att_scale 加倍保持水平（载荷分配，
        # 防单轮着地 fn=[43,0,0,0]）
        _asc = float(cmd.get("att_scale", 1.0))
        R = _asc * (self.kp_roll * (self._roll_f - body["roll"])
                    - self.kd_roll * roll_rate)
        # v869: KD_PITCH split - base pitch damping + STARTUP anti-bounce
        # (added when longitudinal accel large; global 35 made body stiff).
        _kd_p_eff = self.kd_pitch
        _bsp = float(os.environ.get("S10_CAR_KD_PITCH_STARTUP", "0.0"))
        if _bsp > 0.0:
            _bq = qpos[3:7]
            _f2p = np.array([
                1.0 - 2.0 * (_bq[2] ** 2 + _bq[3] ** 2),
                2.0 * (_bq[1] * _bq[2] + _bq[0] * _bq[3]), 0.0])
            _vx_bp = float(np.dot(qvel[0:3], _f2p))
            _ax_b = abs(self._vx_f - _vx_bp) / 0.15  # CarVMC vx LP ~0.1s
            _kd_p_eff += _bsp * float(np.clip(_ax_b / 5.0, 0.0, 1.0))
        P = _asc * (self.kp_pitch * (self._pitch_f - body["pitch"])
                    - _kd_p_eff * pitch_rate)
        _tmax = float(os.environ.get("S10_CAR_ATT_TMAX", "40.0"))
        # v748: 大 lean-in 压弯时 roll 分配在减载方向钳到
        # S10_CAR_ROLL_MAX_DL×mg/4（默认 0 = 跳过 = v746 恒等 ±_tmax；
        # 开大压弯时设 0.5 防 μN 钳制崩推力；加载方向不限，弯内轮可多承）。
        _base_leg = self.m * self.g / 4.0
        _roll_dl = float(os.environ.get("S10_CAR_ROLL_MAX_DL", "0.0"))
        if _roll_dl > 0.0:
            R = float(np.clip(R, -_base_leg * _roll_dl, _tmax))
        else:
            R = float(np.clip(R, -_tmax, _tmax))
        P = float(np.clip(P, -_tmax, _tmax))

        # 轮差速 yaw 反馈（自适应：转弯大、直行小）
        _ysc = float(cmd.get("yaw_scale", 1.0))
        _yk = self.yaw_k_wheel * (0.3 + 0.7 * min(abs(self._om_f) / 0.4, 1.0))
        # v750: 差速反馈随实际速度缩放——低速/倒车时轮速差相对 vx 过大，
        # yaw 正反馈自旋（起步坡实际 ω -4.9 翻车实测，om_ref 0.96 → 轮子
        # 对转极限环）。连续量；默认 0 = v746 恒等，开大压弯/提速时建议
        # S10_CAR_YAW_VX_GATE=1.5（vx<1.5 差速反馈降至 15%）。
        _yvg = float(os.environ.get("S10_CAR_YAW_VX_GATE", "0.0"))
        if _yvg > 0.0:
            # 用实际 body 前向速度（_vx_f 是 cmd 低通，起步段=4 让缩放失效）
            _bq = qpos[3:7]
            _f2 = np.array([
                1.0 - 2.0 * (_bq[2] ** 2 + _bq[3] ** 2),
                2.0 * (_bq[1] * _bq[2] + _bq[0] * _bq[3]), 0.0])
            _vx_b = float(np.dot(qvel[0:3], _f2))
            _yv_sc = float(np.clip(abs(_vx_b) / _yvg, 0.0, 1.0))
            _yk *= (0.15 + 0.85 * _yv_sc)

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
                    # v716: 动态压轮——轮目标压到地面以下（卡死根因=轮子悬空
                    # 零接触力 fn=0）。对侧轴抬轮时本轴加压（保牵引），本轴
                    # 抬轮时卸载（保摆动自由）。
                    _wp = float(cmd.get("wheel_press", 0.0))
                    _wp_eff = 0.0
                    if _wp > 0.0:
                        _sl_all = np.asarray(cmd.get("step_lift", np.zeros(4)))
                        _own = float(_sl_all[leg])
                        _opp = (float(np.max(_sl_all[2:4])) if leg in (0, 1)
                                else float(np.max(_sl_all[0:2])))
                        # v728: 本轴不抬即满压（原 0.3 稀释只给 7N，
                        # 轮子承重不足腿撑车身）；本轴抬时卸载（对侧抬则
                        # 对侧也压不住）
                        _wp_eff = _wp * max(1.0 - _own, _opp)
                    pz_des = (float(terrain_h[leg]) + self.fk.r - _wp_eff)
                    F += (self.kp_h * (pz_des - p[2])
                          - self.kd_h * float(cmd.get("kd_scale", 1.0))
                          * float(wheel_vel[leg, 2]))
                F = max(F, 2.0) * (1.0 - _fscale * sl)
            else:
                if p is not None and terrain_h is not None:
                    pz_des = (float(terrain_h[leg]) + self.fk.r
                              - float(cmd.get("wheel_press", 0.0)))
                    F += (1.0 - sl) * (
                        self.kp_h * (pz_des - p[2])
                        - self.kd_h * float(cmd.get("kd_scale", 1.0))
                        * float(wheel_vel[leg, 2]))
                F = max(F, 2.0) * (1.0 - sl)
            # v748: 弯外轮保载——非抬轮腿至少保留
            # S10_CAR_ROLL_MIN_FRAC×mg/4 支撑力（默认 0 完全跳过=v746 恒等；
            # 开大压弯时设 0.25 防弯外轮减载崩推力）。
            _fmin = (_base_leg * float(os.environ.get(
                "S10_CAR_ROLL_MIN_FRAC", "0.0")))
            if _fmin > 0.0 and sl < 0.3:
                F = max(F, _fmin * (1.0 - sl))
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
            # v664: IK 落脚点覆盖（cmd.fp_place=1 且抬轮 sl>0.3）——轮直接
            # 放到 terrain+r（脚本落脚点地形已传台面高），姿态由 R/P 保持
            _fp = float(cmd.get("fp_place", 0.0))
            if False and _fp > 0.0 and sl > 0.3 and terrain_h is not None:
                # v732: 启用 IK 落脚（原 if False）——抬轮腿直接放到
                # 台面+半径（place_z 由脚本 stair 表给，terr 已覆盖）。
                # 投影用 body 前向 + 世界 z 差（同 FootPlace 修正）。
                _q4 = qpos[3:7]
                _w, _x, _y, _z = _q4
                _Rb = np.asarray([
                    [1 - 2*(_y*_y + _z*_z), 2*(_x*_y - _w*_z), 2*(_x*_z + _w*_y)],
                    [2*(_x*_y + _w*_z), 1 - 2*(_x*_x + _z*_z), 2*(_y*_z - _w*_x)],
                    [2*(_x*_z - _w*_y), 2*(_y*_z + _w*_x), 1 - 2*(_x*_x + _y*_y)],
                ], dtype=np.float64)
                _hip_w = qpos[0:3] + _Rb @ np.array(
                    [LEG_ATTACH[leg, 0], LEG_ATTACH[leg, 1], 0.0])
                _wz4 = float(terrain_h[leg]) + self.fk.r
                _dwh = np.array([wheel_xyz[leg, 0] - _hip_w[0],
                                 wheel_xyz[leg, 1] - _hip_w[1],
                                 _wz4 - _hip_w[2]])
                _rbf = _Rb.T @ _dwh
                _relf4 = float(_rbf[0])
                _relz4 = float(np.clip(_dwh[2], -0.34, 0.0))
                _q1_tgt, _q2_tgt = self._ik(_relf4, _relz4, q1, q2)
            # v221i: 车身抬升（过脊用）——0.05 只到 0.68(差 5mm)，改 0.08
            # v876: stair 副本——body_lift 只作用于**抬升腿**(sl>0.3)，
            # 原全腿生效会让后腿同伸→翘头轮式(pitch -1.4 实测)；前轮额外
            # 伸膝 0.08 补足清 riser 棱边的最后几毫米
            _bl = float(cmd.get("body_lift", 0.0))
            if _bl > 0.0 and sl > 0.3:
                _q2_tgt += _bl * 0.08
            t_hipy += (self.kp_leg * (_q1_tgt - q1)
                       - self.kd_leg * float(qvel[6 + LEG_QV_LEG[b + 1]]))
            t_knee += (self.kp_leg * (_q2_tgt - q2)
                       - self.kd_leg * float(qvel[6 + LEG_QV_LEG[b + 2]]))
            # hipx：位置 PD + 侧身（压弯）+ v232 扭胯 yaw 辅助——
            # 左转(om_f>0)时左腿 hipx 外展/右腿内收，产生 yaw 力矩
            # （用户"扭肩/胯转向"思路，突破轮差速 0.65rad/s 物理上限）
            _q0_tgt = self.pose_target[b] - 0.12 * self.roll_sign[leg] * self._roll_f
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
            # v855: om_safe 反向硬刹恢复——实测删除后 wp1→2 yaw 失控翻车，
            # 该刹是转弯自旋的安全网（用户确认后再软化/删除）
            _latmax = float(os.environ.get("S10_AUTO_LAT_MAX", "5.0"))
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
            # v750: 低速滑模 yaw 反馈缩放因子（连续量）——起步/倒车时
            # err_y 反馈正反馈自旋（起步坡实际 ω -4.9 翻车实测）。只缩
            # **反馈**（_k_sm），差速前馈（_om_ref）保留航向保持能力。
            # 默认 0 = v746 恒等；提速测试建议 S10_CAR_YAW_VX_GATE=2.0
            # （vx<2.0 时滑模反馈线性降至 50%）。
            # v869: gate uses ACTUAL body forward speed (qvel projection),
            # NOT _vx_f (low-passed cmd, useless at startup). Both gates same.
            _yvg = float(os.environ.get("S10_CAR_YAW_VX_GATE", "0.0"))
            self._yv_scale = 1.0
            if _yvg > 0.0:
                _bq = qpos[3:7]
                _f2 = np.array([
                    1.0 - 2.0 * (_bq[2] ** 2 + _bq[3] ** 2),
                    2.0 * (_bq[1] * _bq[2] + _bq[0] * _bq[3]), 0.0])
                _vx_b = float(np.dot(qvel[0:3], _f2))
                self._yv_scale = float(np.clip(abs(_vx_b) / _yvg, 0.0, 1.0))
            # v855: 反向硬刹恢复（转弯自旋安全网，实测必需）
            _err_y = self._om_f - body["omega"]
            if abs(_om_b) > _om_safe:
                _om_ref = -float(np.clip(_om_b, -_om_safe, _om_safe))
                _kd_eff = _kd_yaw + 8.0
            # v252: 差速参考用**即时指令**（导航已 slew 0.8/s，够平滑）——
            # 低通 _om_f 在指令方向翻转时滞后 ~0.3s，继续推旧方向致过冲
            # （wp4→5 指令+1.72 实际-2.86 翻车实测）；低通只留反馈项用。
            # v317: 差速参考随抓地系数衰减——轮子因地形/抬轮离地时保持
            # 差速会在落地瞬间产生 yaw 冲击（起步坡 wz0.6 实测 yaw 1.6->0.2
            # 自旋）；与 yaw 反馈一致用 ground_f 淡出差速，接地后恢复。
            # v807: 低速差速放大（S10_CAR_LOWSPD_TURN）——vx 低 + 偏航需求大
            # 时轮差速放大（内轮减速近停转，必要时反向=点转），提高低速转向
            # 权威（wp16→17 低速打转卡死实测：1.2-1.8m/s 差速不足 yaw 极限环
            # 振荡）。连续量：放大随 |vx| 降低增大，上限 4x。
            _lowb = float(os.environ.get("S10_CAR_LOWSPD_TURN", "0.0"))
            _om_ref2 = _om_ref
            if _lowb > 0.0:
                _vr_abs = abs(self._vx_f)
                if _vr_abs < 2.5:
                    _om_ref2 = _om_ref * float(np.clip(
                        1.0 + _lowb * (2.5 - _vr_abs) / 2.5, 1.0, 4.0))
            # v870: differential feedforward also scaled by _yv_scale (actual
            # vx) - MU=0.8 MPPI outputs om 0.83 at startup vs 0.36's 0.22,
            # unscaled FF spun (om -4.16 flip). Feedback already uses yv_scale.
            # v870: differential FF scaled by _yv_scale (actual vx) - MU=0.8
            # MPPI om 0.83 at startup spun (om -4.16). Use YAW_VX_GATE=2.0
            # so corners (vx>=2) keep full differential, startup damped.
            v_ref = (self._vx_f
                     + side * _om_ref2 * _ysc * self._ground_f
                     * getattr(self, "_yv_scale", 1.0)
                     * self.track_half)
            # v241/v242: yaw 摩擦前馈（RobuROC6 库仑摩擦补偿）——差速转向需
            # 先克服侧向滑移阻力才有 yaw 运动，纯误差反馈有死区滞后；按指令
            # 方向给基础差速力矩。**默认 0**：v241 线性 FF 在导航指令突变时
            # 过驱动（wp1→2 振荡侧翻实测）；启用时加 0.15s 低通防瞬翻。
            # v869: S10_CAR_YAW_FF env removed but friction feedforward kept
            # at fixed 1.0 (was enabled=1.0 in test script; deleting it broke
            # yaw response: startup spin + wp3->4 line loss). Fixed value,
            # no longer tunable. _vspd kept for slew speed scaling.
            _kff = 1.0
            if not hasattr(self, "_om_ff_lp"):
                self._om_ff_lp = 0.0
            self._om_ff_lp += (self._om_f - self._om_ff_lp) * min(1.0, dt / 0.15)
            _vspd = float(abs(getattr(self, "_vx_f", 0.0)))
            _kff *= float(np.clip(
                1.0 - max(0.0, _vspd - 3.0) / 2.0, 0.0, 1.0))
            _k_sm = float(os.environ.get("S10_CAR_YAW_K_SM", "30.0"))
            # v750: 滑模 yaw 力矩随实际速度缩放（低速抑制正反馈自旋，
            # 下限 0.5 保留基本航向保持——0.15 时起步 yaw 漂 57° 转不回）
            _k_sm *= (0.5 + 0.5 * float(getattr(self, "_yv_scale", 1.0)))
            _phi = float(os.environ.get("S10_CAR_YAW_PHI", "0.5"))
            # v854: 删除 OM_BODY 模式开关与 TUBE 死区（用户：无离散门控）——
            # 滑模误差固定用世界系 yaw 率，误差直接进 tanh
            _err_t = _err_y
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
            # v797: slew 速度放大——低速小（防过冲瞬态，巡航 wp1 弯），
            # 高速大（高架 4.9m/s 需 ~200 才不极限环，wp9→10 实测）。
            _tslew = float(os.environ.get("S10_CAR_YAW_SLEW", "30.0")) * float(
                np.clip(1.0 + _yerr / float(os.environ.get(
                    "S10_CAR_YAW_SLEW_K", "1.0")), 1.0, 4.0)) * float(
                np.clip(1.0 + max(0.0, _vspd - 4.2) * 5.0, 1.0, 7.0))
            if not hasattr(self, "_t_yaw_prev"):
                self._t_yaw_prev = np.zeros(4)
            _ty = float(np.clip(
                t_yaw,
                self._t_yaw_prev[leg] - _tslew * dt,
                self._t_yaw_prev[leg] + _tslew * dt))
            self._t_yaw_prev[leg] = _ty
            t_yaw = _ty
            # v865: vx wheel torque * ground_f - airborne wheels spin at full
            # torque, touchdown yaw impact flips (wp1 om -4.63 measured).
            _gf_w = float(getattr(self, "_ground_f", 1.0))
            _gf_w = float(os.environ.get("S10_CAR_WHEEL_GF", str(_gf_w)))
            t_wheel = ((-(self.wheel_k * (v_ref - v_wheel))
                        - self.wheel_d * wq) * _gf_w + t_yaw)
            # v743: 直线全力/弯道按 err 收敛（提速）——旧版全程 μN·r≈3.4Nm
            # 直线加速到 vref=4 需 16Nm 被压死（各段实际≈vref 一半，实测
            # wp5→6 vlim3.0 实际1.34）。err 小（直线）给满轮矩，err 大
            # （弯道）收敛到摩擦锥 μN·r。
            _fz_load = max(F, 0.5 * self.m * self.g / 4.0)
            _mu_w = float(os.environ.get("S10_VMC_WHEEL_MU", "0.9"))
            # v869: deleted S10_VMC_YAW_TMAX (default 0)
            _wt_curve = _mu_w * _fz_load * self.fk.r
            _wt_straight = float(os.environ.get("S10_VMC_WHEEL_TMAX", "13.5"))
            # 门限用导航偏航误差 err（比 om_cmd 更本质）——S弯 om 交替快、
            # om 门限来不及切换实测翻车；err 大立即收敛。
            _yerr = abs(float(cmd.get("yaw_err", 0.0)))
            _w_f = float(np.clip(
                1.0 - _yerr / float(os.environ.get(
                    "S10_VMC_WT_ERR_GATE", "0.4")), 0.0, 1.0))
            # v746: 横脊窗口（路径距离<0.8m）轮矩恢复 μN——发卡弯+横脊
            # 复合段 err 门控全力切换导致弹跳侧翻（wp4→5 实测）；连续距离量
            _rd = float(cmd.get("ridge_dist", 99.0))
            if _rd < float(os.environ.get("S10_VMC_WT_RIDGE_D", "0.8")):
                _w_f = 0.0
            _wt = float(np.clip(
                _w_f * _wt_straight + (1.0 - _w_f) * _wt_curve,
                -13.5, 13.5))
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


class FootPlaceVMC:
    """v660: 逐轮落脚点位置控制（楼梯/台阶）——每腿 IK 把轮放到目标高度
    （terrain_h+r，脚本的落脚点地形输入自动给出台面目标），身体姿态由四轮
    位置自然形成（前轮上台面→前髋抬高→车身抬头）；轮驱动纯前向。彻底去掉
    全局 wrench/z/pitch 控制，规避"身体悬空"死结（WBC 84 组实验失败）。"""

    def __init__(self, mass=19.0, g=9.81, L1=0.18, L2=0.18, r=0.081,
                 track_half=0.24, kp=220.0, kd=6.0,
                 wheel_k=4.0, wheel_d=0.08):
        import os
        self.m, self.g = mass, g
        self.fk = S10LegFK(L1, L2, r)
        self.track_half = track_half
        self.kp = float(os.environ.get("S10_FP_KP", str(kp)))
        self.kd = float(os.environ.get("S10_FP_KD", str(kd)))
        self.wheel_k = float(os.environ.get("S10_FP_WHEEL_K", str(wheel_k)))
        self.wheel_d = float(os.environ.get("S10_FP_WHEEL_D", str(wheel_d)))
        self.pose_target = np.array([-0.05, -1.10, 1.90,
                                     0.05, -1.10, 1.90,
                                    -0.05,  1.10, -1.90,
                                     0.05,  1.10, -1.90], dtype=np.float64)
        self._vx_f = 0.0
        self.kp_roll = float(os.environ.get("S10_FP_KP_ROLL", "400.0"))
        self.kp_pitch = float(os.environ.get("S10_FP_KP_PITCH", "300.0"))

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
                    vx=float(vw[0]), R=R)

    def _ik(self, xd, zd, q1, q2, lift=False):
        """2D IK：轮目标 (xd, zd)（相对髋，sagittal，zd 向上）→ q1/q2。
        v733: 加关节范围钳制——楼梯抬升目标接近工作空间边界时裸迭代会
        翻到镜像解（q1=2.7 实测）。lift=True（抬升腿）用迈步举轮姿态
        （v236 FK 验证 q1≈0.4/q2≈2.7 轮高+1.4cm）；支撑腿保持正常半蹲。"""
        _q1_lo, _q1_hi = (-0.35, 0.9) if lift else (-1.7, -0.35)
        _q2_lo = 1.8 if lift else -0.2
        for _ in range(10):
            p = self.fk.wheel_pos(q1, q2)
            err = np.array([xd - p[0], zd + p[1]])
            J = self.fk.jac(q1, q2)
            dq = np.linalg.lstsq(J, err, rcond=None)[0]
            dq = np.clip(dq, -0.25, 0.25)
            q1 += float(dq[0]); q2 += float(dq[1])
            q1 = float(np.clip(q1, _q1_lo, _q1_hi))
            q2 = float(np.clip(q2, _q2_lo, 3.0))
        return q1, q2

    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel,
                    cmd, terrain_h, dt=0.005):
        body = self._body_state(qpos, qvel)
        _posmode_fp = float(os.environ.get('S10_FP_POSMODE', '0'))
        _fp_press = (float(os.environ.get('S10_FP_PRESS', '0.005'))
                     if _posmode_fp > 0 else 0.0)
        self._vx_f += (float(cmd.get("vx", 0.0)) - self._vx_f) * min(
            1.0, dt / 0.10)
        self._om_f = getattr(self, "_om_f", 0.0)
        self._om_f += (float(cmd.get("omega", 0.0)) - self._om_f) * min(
            1.0, dt / 0.10)
        _rp = getattr(self, "_roll_prev", None)
        _pp = getattr(self, "_pitch_prev", None)
        roll_rate = ((body["roll"] - _rp) / max(dt, 1e-4)
                     if _rp is not None else 0.0)
        pitch_rate = ((body["pitch"] - _pp) / max(dt, 1e-4)
                      if _pp is not None else 0.0)
        self._roll_prev = body["roll"]
        self._pitch_prev = body["pitch"]
        _om_body = float(qvel[5])
        # v732: CPG 抬放量 + 落脚点台面高（脚本 stair_zone 由 riser 表生成）
        _sl_all = np.asarray(cmd.get("step_lift", np.zeros(4)), dtype=np.float64)
        _pz_all = np.asarray(cmd.get("place_z", np.zeros(4)), dtype=np.float64)
        _margin = float(cmd.get("place_margin", 0.04))
        if getattr(self, "_wz_f", None) is None:
            self._wz_f = np.array([terrain_h[leg] + self.fk.r
                                   for leg in range(4)], dtype=np.float64)
        tau = np.zeros(16, dtype=np.float64)
        # v823: body 姿态解析（用户架构核心）——posmode 下先算 4 轮目标，
        # 再解期望 body z/pitch（腿目标 = 期望 body 位姿 + 足端相对位，
        # 替代用实际 body 位姿导致高度漂移弹射）
        _bdes_z = None
        _bdes_pitch = 0.0
        if _posmode_fp > 0:
            _wz_all = np.zeros(4)
            for _leg0 in range(4):
                _sl0 = float(_sl_all[_leg0])
                _pz0 = float(_pz_all[_leg0])
                _wg0 = float(terrain_h[_leg0]) + self.fk.r
                if _posmode_fp > 0:
                    _wg0 -= float(os.environ.get('S10_FP_PRESS', '0.005'))
                _wz_all[_leg0] = _wg0
                if _pz0 > 0.01 and _sl0 > 0.02:
                    _wz_all[_leg0] = min(
                        _pz0 + self.fk.r + _margin,
                        float(body["pos"][2]) + 0.25)
            _bdes_z = float(np.mean(_wz_all)) + float(os.environ.get(
                'S10_FP_STAND_DROP', '0.26'))
            _wz_fm = float(np.mean(_wz_all[0:2]))
            _wz_rm = float(np.mean(_wz_all[2:4]))
            _bdes_pitch = -float(np.arctan2(_wz_fm - _wz_rm, 0.456))
        for leg in range(4):
            b = leg * 3
            hipx_i, hipy_i, knee_i = (LEG_CTRL_IDX[b],
                                      LEG_CTRL_IDX[b + 1],
                                      LEG_CTRL_IDX[b + 2])
            qhx = float(qpos[LEG_Q_IDX[b]])
            q1 = float(qpos[LEG_Q_IDX[b + 1]])
            q2 = float(qpos[LEG_Q_IDX[b + 2]])
            # v736: 支撑腿髋位置用 yaw-only（髋在 body z 高度）——完整 R
            # 在 pitch 下 hip z 偏移会让 IK 的 rel_z 错位（v768 平地侧翻）。
            # 抬升腿走关节空间目标（不走 IK），不受此影响。
            _cy = float(np.cos(body["yaw"])); _sy = float(np.sin(body["yaw"]))
            # v823: posmode 用期望 body 位姿（z=均值轮高+站立落差，pitch=前后
            # 轮坡度）算髋，腿目标=期望body+足端相对位，body 被拉向期望姿态
            _bposz = (float(_bdes_z) if _bdes_z is not None
                      else float(body["pos"][2]))
            _bposp = (_bdes_pitch if _bdes_z is not None
                      else float(body["pitch"]))
            _attach = LEG_ATTACH[leg]
            _hip_off = np.array([_cy * _attach[0] - _sy * _attach[1],
                                 _sy * _attach[0] + _cy * _attach[1],
                                 0.0])
            # 期望 pitch 下髋的 z 偏移（前腿高、后腿低）
            _hip_off[2] = -_attach[0] * float(np.sin(_bposp))
            hip_w = np.array([body["pos"][0] + _hip_off[0],
                              body["pos"][1] + _hip_off[1],
                              _bposz + _hip_off[2]])
            sl = float(_sl_all[leg])
            pz = float(_pz_all[leg])
            # 落脚目标：支撑腿 = 轮下地形+半径；抬升腿按 CPG 波形插值到
            # 台面高+半径+margin（v732 连续抬升轨迹，非二值跳变）
            # v822: 位置基模式（S10_FP_POSMODE=1）——支撑腿目标下调静压
            # 余量 S10_FP_PRESS（用户方案：位置控制下轮"硬顶"台面，接触力
            # 自然够，不用力控下压）
            wz_ground = float(terrain_h[leg]) + self.fk.r
            if _posmode_fp > 0:
                wz_ground -= float(os.environ.get('S10_FP_PRESS', '0.005'))
            # v736: 楼梯区不整体抬身——关节空间举轮（q1+1.55）已把轮抬到
            # 髋附近，z_des 抬身会让 body 顶高 roll 失稳（v769 实测）。
            # 支撑腿 wz 用轮下地形+半径（贴地），抬升腿走关节目标不走 wz。
            wz = wz_ground
            if pz > 0.01 and sl > 0.02:
                wz = min(pz + self.fk.r + _margin, float(hip_w[2]) + 0.15)
            # 目标低通（τ=0.08s）——CPG 波形已连续，低通只滤地形跳变
            self._wz_f[leg] += (wz - self._wz_f[leg]) * min(1.0, dt / 0.08)
            wz = float(self._wz_f[leg])
            # v734: roll 修正对所有腿生效（抬升腿也纠侧翻，防 roll 放大
            # IK 失真正反馈）；pitch 修正只作用于支撑腿（避免对抗抬放）。
            # 修正量 clamp ±0.05m 防正反馈猛伸。
            _side = -1.0 if leg in (0, 1) else 1.0
            _front = 1.0 if leg in (0, 1) else -1.0
            _kdr = float(os.environ.get('S10_FP_ROLL_KD', '20.0'))
            _rc = float(np.clip((self.kp_roll * (-float(body["roll"]))
                                 - _kdr * roll_rate) * 0.0025,
                                -0.05, 0.05))
            wz += _side * _rc
            if sl < 0.3:
                _pc = float(np.clip(
                    (self.kp_pitch * (float(cmd.get("pitch_tar", 0.0))
                                      - float(body["pitch"]))
                     - 6.0 * pitch_rate) * 0.0025,
                    -0.05, 0.05))
                wz += _front * _pc
            # v827: body 姿态闭环（用户架构核心）——期望 body z/pitch 由
            # 4 轮目标解算（v823），误差修正轮目标把 body 拉向期望姿态。
            # body 过高→轮目标上抬（腿缩短→body 下降）；pitch 误差→前/后
            # 轮差分修正。连续量，S10_FP_BODY_K 可调。
            if _posmode_fp > 0 and _bdes_z is not None:
                _bk = float(os.environ.get('S10_FP_BODY_K', '0.4'))
                _bz_err = float(_bdes_z - body["pos"][2])
                # v827b: 符号修正——body 过高(err<0)时轮目标**上抬**（腿缩短
                # →body 下降，轮贴地）；原 wz+=err 在 body 过高时下压→腿伸长
                # →body 更高正反馈弹射实测
                wz -= _bz_err * _bk
                # v827c: body z 速率阻尼（S10_FP_BODY_KD）——上抬速度大时
                # 轮目标反向压，防位置控制对地形跳变过冲弹射
                _bkd = float(os.environ.get('S10_FP_BODY_KD', '0.06'))
                wz += float(qvel[2]) * _bkd
                _bp_err = float(body["pitch"] - _bdes_pitch)
                wz += _front * _bp_err * 0.3 * _bk
            _dw = np.array([wheel_xyz[leg, 0] - hip_w[0],
                           wheel_xyz[leg, 1] - hip_w[1],
                           wz - hip_w[2]])
            # v736: IK 坐标 = yaw 前向投影 + 世界 z 差（稳定可达）
            rel = np.array([_cy * _dw[0] + _sy * _dw[1],
                            -_sy * _dw[0] + _cy * _dw[1],
                            _dw[2]])
            # v732: 抬升腿放宽 IK 伸展钳制（上 0.125m 台面需轮相对髋
            # z≈-0.25~-0.31，原 -0.16 锁死抬升）；支撑腿保持 -0.16 防猛伸
            # v810: 支撑腿腿长钳制可调（S10_FP_REACH）——断崖 0.377m 落差 >
            # 支撑腿默认 -0.16 行程，FP 无法放轮卡崖边实测；放宽到 -0.36
            # （腿行程极限）让前轮能落到低地。默认 -0.16 保 stair 行为。
            _lo = (-0.34 if sl > 0.1
                   else float(os.environ.get('S10_FP_REACH', '-0.16')))
            _hi = 0.15 if sl > 0.1 else 0.02
            _rz = float(np.clip(rel[2], _lo, _hi))
            q1t, q2t = self._ik(float(rel[0]), _rz, q1, q2, lift=(sl > 0.1))
            # v736: 抬升腿真正卸载——PD 增益降到 1/10（kp 220→22），
            # 轮近乎自由摆动（避免硬顶地面→反力顶起全身→四轮离地，
            # v725-728 耦合结论）。支撑腿承担全部载荷。
            # v827: 位置基模式腿增益（S10_FP_KP_POS 可调，默认全增益；
            # 全增益 220 在 body 过渡期暴力饱和翻转实测，可降至 120 平滑）
            if _posmode_fp > 0:
                _kpp9 = float(os.environ.get('S10_FP_KP_POS', '0'))
                _kp_leg = (float(_kpp9) if _kpp9 > 0 else self.kp)
                _kd_leg = self.kd
            else:
                _kp_leg = self.kp * (0.10 if sl > 0.1 else 1.0)
                _kd_leg = self.kd * (0.3 if sl > 0.1 else 1.0)
            if os.environ.get('S10_FP_DEBUG', '0') == '1' and leg == 0:
                print('[FP] t=%.2f sl=%.2f pz=%.3f wz_tgt=%.3f wz_act=%.3f '
                      'body_z=%.3f q1t=%.2f q2t=%.2f q1=%.2f q2=%.2f'
                      % (getattr(self, '_t', 0.0), sl, pz, wz,
                         float(wheel_xyz[leg, 2]), float(body["pos"][2]),
                         q1t, q2t, q1, q2), flush=True)
            tau[hipx_i] = (self.kp * (self.pose_target[b] - qhx)
                           - self.kd * float(qvel[6 + LEG_QV_LEG[b]]))
            # v828: posmode 支撑腿用单侧垂直阻抗（用户：力控只做阻抗）——
            # 轮贴地滚动，腿只下压不吸抬（位置锁定支撑腿在斜坡上过冲弹射
            # 实测）；抬升腿仍位置控制（IK 全增益）
            if _posmode_fp > 0 and sl <= 0.5:
                # v871: 终版——支撑腿 = 位置 PD 拉满 + 静压(-5mm) + 微阻抗
                # （原 v828 只用单侧阻抗→支撑力≈0 身体瘫坐 fn=0 实测；
                # 终版用户方案：位置控制下轮"硬顶"台面，接触力自然够，
                # 阻抗项只做柔顺补充，不主导）
                pz_des9 = float(terrain_h[leg]) + self.fk.r - _fp_press
                _dz9 = pz_des9 - float(wheel_xyz[leg, 2])
                _F9 = float(os.environ.get('S10_FP_KPH', '300')) * min(_dz9, 0.0)
                _F9 = max(_F9, 2.0)
                J9 = self.fk.jac(q1, q2)
                f_b9 = body["R"].T @ np.array([0.0, 0.0, _F9], dtype=np.float64)
                f_s9 = np.array([f_b9[0], -f_b9[2]])
                _th9, _tk9 = J9.T @ f_s9
                tau[hipy_i] = (_kp_leg * (q1t - q1)
                               - _kd_leg * float(qvel[6 + LEG_QV_LEG[b + 1]])
                               + float(_th9))
                tau[knee_i] = (_kp_leg * (q2t - q2)
                               - _kd_leg * float(qvel[6 + LEG_QV_LEG[b + 2]])
                               + float(_tk9))
            else:
                tau[hipy_i] = (_kp_leg * (q1t - q1)
                               - _kd_leg * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                tau[knee_i] = (_kp_leg * (q2t - q2)
                               - _kd_leg * float(qvel[6 + LEG_QV_LEG[b + 2]]))
            wq = float(qvel[WHEEL_QV_IDX[leg]])
            v_wheel = -wq * self.fk.r
            # 差速保持航向 + yaw 率阻尼
            _side = -1.0 if leg in (0, 1) else 1.0
            v_ref = (self._vx_f
                     + _side * self._om_f * self.track_half)
            # v732: 抬升腿轮速轻驱（防悬空空转/冲击），支撑腿全驱
            if sl > 0.5:
                v_ref *= 0.35
            t_yaw = -6.0 * _om_body * _side
            # v822: 位置基模式——抬升腿轮矩=0（前轮离地期间驱动力全给后轮，
            # 用户方案改进4）；支撑腿开环轮速 PID（位置控制下腿不顶 body，
            # 轮矩可开到 T_max，不用 μN 钳制）
            if _posmode_fp > 0 and sl > 0.5:
                tau[WHEEL_Q_IDX[leg]] = 0.0
            else:
                tau[WHEEL_Q_IDX[leg]] = (
                    -(self.wheel_k * (v_ref - v_wheel))
                    - self.wheel_d * wq + t_yaw)
        tau[LEG_CTRL_IDX] = np.clip(tau[LEG_CTRL_IDX], -48, 48)
        tau[WHEEL_Q_IDX] = np.clip(tau[WHEEL_Q_IDX], -13.5, 13.5)
        return tau
        return tau
