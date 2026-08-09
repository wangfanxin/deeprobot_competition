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
                 kp_roll=30.0, kd_roll=4.0,
                 kp_pitch=80.0, kd_pitch=12.0, pitch_ff=0.8,
                 wheel_k=1.2, wheel_d=0.05,
                 track_half=0.24):
        self.m, self.g = mass, g
        self.fk = S10LegFK(L1, L2, r)
        self.tau_v = tau_v
        self.kp_h, self.kd_h = kp_h, kd_h
        self.kp_z, self.kd_z = kp_z, kd_z
        self.kp_roll, self.kd_roll = kp_roll, kd_roll
        self.kp_pitch, self.kd_pitch, self.pitch_ff = kp_pitch, kd_pitch, pitch_ff
        self.wheel_k, self.wheel_d = wheel_k, wheel_d
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
        body = self._body_state(qpos, qvel)
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
        # v218d: 身体级 VMC——高度/俯仰/侧倾整体控制，按支撑几何分配到 4 腿；
        # 轮端阻抗只负责地形跟随（含横脊预抬）。4 轮独立重力补偿会跷跷板失稳。
        z_des = float(np.mean(terrain_h)) + 0.205   # 身体标称高（非轮心）
        if os.environ.get("S10_VMC_KPZ", "1") == "1":
            Fz_total = (self.m * self.g
                        + self.kp_z * (z_des - body["pos"][2])
                        - self.kd_z * float(qvel[2]))
        else:
            Fz_total = self.m * self.g
        roll_rate = ((body["roll"] - self._roll_prev) / max(dt, 1e-4)
                     if self._roll_prev is not None else 0.0)
        self._roll_prev = body["roll"]
        pitch_rate = ((body["pitch"] - self._pitch_prev) / max(dt, 1e-4)
                      if self._pitch_prev is not None else 0.0)
        self._pitch_prev = body["pitch"]
        ax = (self._vx_f - body["vx"]) / self.tau_v
        if os.environ.get("S10_VMC_PITCH", "1") == "1":
            T_pitch = (self.kp_pitch * (pitch_tar - body["pitch"])
                       - self.kd_pitch * pitch_rate
                       - self.pitch_ff * self.m * ax * 0.20)
        else:
            T_pitch = 0.0
        if os.environ.get("S10_VMC_HIPX", "1") == "1":
            # v218g: 校准 +T_roll→roll 更负，反馈需 (roll - tar)
            T_roll = (self.kp_roll * (body["roll"] - self._roll_f)
                      - self.kd_roll * roll_rate)
        else:
            T_roll = 0.0
        # 每腿分担：重力/4 + 俯仰力矩（前+/后-）+ 侧倾力矩（左-/右+，实测符号）
        fz_dist = np.full(4, Fz_total / 4.0)
        fz_dist[0:2] += T_pitch / self.wheelbase
        fz_dist[2:4] -= T_pitch / self.wheelbase
        # v218e: 侧倾分配符号（实测：roll 负=左侧抬，需左侧加支撑）
        fz_dist[0] += T_roll / (2.0 * self.track_half)
        fz_dist[2] += T_roll / (2.0 * self.track_half)
        fz_dist[1] -= T_roll / (2.0 * self.track_half)
        fz_dist[3] -= T_roll / (2.0 * self.track_half)

        for leg in range(4):
            b = leg * 3
            hipx_i, hipy_i, knee_i = (
                LEG_CTRL_IDX[b], LEG_CTRL_IDX[b + 1], LEG_CTRL_IDX[b + 2])
            q1 = float(qpos[LEG_Q_IDX[b + 1]])
            q2 = float(qpos[LEG_Q_IDX[b + 2]])
            J = self.fk.jac(q1, q2)
            # 轮端力：身体级分配 + 地形阻抗（横脊预抬经 terrain_h）
            p = wheel_xyz[leg]
            pz_des = float(terrain_h[leg]) + self.fk.r
            Fz_w = (fz_dist[leg]
                    + self.kp_h * (pz_des - p[2])
                    - self.kd_h * float(wheel_vel[leg, 2]))
            # v218b: 腿水平力是内力，不驱动身体——速度完全由轮电机负责。
            Fx_w = 0.0
            # 轮端力（世界）-> body 矢状面（x, z_down）
            # v218f: Fz_w 已是世界系向上力（负=下压），不要再取反
            f_body = body["R"].T @ np.array([Fx_w, 0.0, Fz_w])
            f_sag = np.array([f_body[0], -f_body[2]])       # [x, z_down]
            t_hipy, t_knee = J.T @ f_sag

            # hipx：侧倾（压弯）——左腿正/右腿负（符号测试校准）
            # v218c: hipx 侧倾纠偏符号（实测：左腿负/右腿正才能回正）
            side = -1.0 if leg in (0, 2) else 1.0
            wd_side = -side   # v218: 差速符号与 hipx 相反（实测）
            if os.environ.get("S10_VMC_HIPX", "1") != "1":
                t_hipx = 0.0
            else:
                t_hipx = side * (self.kp_roll * (self._roll_f - body["roll"])
                                 - self.kd_roll * roll_rate)

            # 轮：差速转向
            wq = float(qvel[WHEEL_QV_IDX[leg]])
            v_wheel = wq * self.fk.r
            v_ref = self._vx_f + wd_side * self._om_f * self.track_half
            # v218: 实测 +轮力矩=倒车（S10 轮轴符号），取反前进
            t_wheel = -(self.wheel_k * (v_ref - v_wheel)
                        - self.wheel_d * wq)

            tau[hipx_i] = float(np.clip(t_hipx, -20, 20))
            tau[hipy_i] = float(np.clip(t_hipy, -50, 50))
            tau[knee_i] = float(np.clip(t_knee, -50, 50))
            tau[WHEEL_Q_IDX[leg]] = float(np.clip(t_wheel, -14, 14))
        return tau


# ==================== 已知地图地形栅格（预计算） ====================

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
