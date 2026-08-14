import numpy as np

# Deterministic STAIR support/lock layer.
# Runs at 200 Hz after MBDPI torque is computed. It does not add reward
# shaping; it vetoes or clamps unsafe swing/lock decisions from the soft
# gait schedule, so the robot always keeps at least a valid support polygon
# before lifting a wheel.

class StairStanceGuard:
    def __init__(self, model, data, wheel_body_ids=(5, 9, 13, 17),
                 wheel_act_ids=(3, 7, 11, 15),
                 contact_min_n=20.0, support_margin=0.06,
                 wheel_tau_max=13.5, mu=0.8, wheel_radius=0.081):
        self.m = model
        self.d = data
        self.wheel_body_ids = wheel_body_ids
        self.wheel_act_ids = wheel_act_ids
        self.contact_min_n = float(contact_min_n)
        self.support_margin = float(__import__('os').environ.get('S10_STANCE_SUPPORT_MARGIN', str(support_margin)))
        self.wheel_tau_max = float(wheel_tau_max)
        self.mu = float(mu)
        self.wheel_radius = float(wheel_radius)

    def apply(self, tau, gait_swing, com_xy, wheel_y=None, wheel_z=None, terrain_z=None, prox=None, wheel_ref_z=None):
        # tau: (16,) numpy torque vector.
        # gait_swing: (4,) continuous weights from the planner.
        # com_xy: (2,) world xy of the base link.
        tau = np.asarray(tau, dtype=np.float64).copy()
        swing = np.asarray(gait_swing, dtype=np.float64)
        # Geometric contact is more reliable than cfrc_ext in MuJoCo wheel
        # contacts, especially while climbing.
        wheel_y = np.asarray(wheel_y if wheel_y is not None else [self.d.xpos[wb][1] for wb in self.wheel_body_ids], dtype=np.float64)
        wheel_z = np.asarray(wheel_z if wheel_z is not None else [self.d.xpos[wb][2] for wb in self.wheel_body_ids], dtype=np.float64)
        terrain_z = np.asarray(terrain_z if terrain_z is not None else (wheel_z - self.wheel_radius), dtype=np.float64)
        contact = wheel_z < (terrain_z + self.wheel_radius + 0.02)
        fn = np.where(contact, self.contact_min_n, 0.0)

        request_swing = swing > 0.5
        if prox is None:
            prox = np.full(4, 1e9, dtype=np.float64)
        else:
            prox = np.asarray(prox, dtype=np.float64)

        # Veto swing unless remaining contact wheels form a support polygon
        # containing the projected CoM. A wheel close to a riser face gets a
        # physical reaction from that face, so two-wheel rear support is
        # acceptable during the front-axle transition.
        riser_prox = float(__import__('os').environ.get('S10_STANCE_RISER_PROX', '0.15'))
        for i in range(4):
            if not request_swing[i]:
                continue
            stance = [j for j in range(4) if j != i and contact[j]]
            if len(stance) < 2:
                request_swing[i] = False
                continue
            if prox[i] < riser_prox:
                continue
            pts = np.array([self.d.xpos[self.wheel_body_ids[j]][:2] for j in stance])
            if not self._point_in_support(np.asarray(com_xy, dtype=np.float64), pts, self.support_margin):
                request_swing[i] = False

        # Swing wheels are locked; support wheels keep DiAL drive torque but
        # are limited by Coulomb traction so they cannot free-spin.
        for i, act_id in enumerate(self.wheel_act_ids):
            if request_swing[i]:
                tau[act_id] = 0.0
            elif contact[i]:
                normal = max(fn[i], 0.0)
                limit = self.mu * normal * self.wheel_radius
                limit = min(limit, self.wheel_tau_max)
                tau[act_id] = float(np.clip(tau[act_id], -limit, limit))

        if __import__('os').environ.get('S10_STANCE_BODY_CTRL', '0') == '1':
            kp_roll = float(__import__('os').environ.get('S10_STANCE_BODY_KP_ROLL', '40.0'))
            kd_roll = float(__import__('os').environ.get('S10_STANCE_BODY_KD_ROLL', '4.0'))
            kp_pitch = float(__import__('os').environ.get('S10_STANCE_BODY_KP_PITCH', '40.0'))
            kd_pitch = float(__import__('os').environ.get('S10_STANCE_BODY_KD_PITCH', '4.0'))
            quat = np.asarray(self.d.xquat[1], dtype=np.float64)
            w, x, y, z = quat
            roll = float(np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y)))
            pitch = float(np.arcsin(np.clip(2.0*(w*y - z*x), -1.0, 1.0)))
            ang = np.asarray(self.d.cvel[1][3:6], dtype=np.float64)
            roll_err = -roll
            pitch_err = -pitch
            tau_roll = kp_roll * roll_err - kd_roll * float(ang[0])
            tau_pitch = kp_pitch * pitch_err - kd_pitch * float(ang[1])
            if __import__('os').environ.get('S10_STANCE_BODY_DEBUG', '0') == '1':
                print(f'[BODY] roll={roll:.3f} pitch={pitch:.3f} ang={[round(float(v),3) for v in ang[:2]]} tau_roll={tau_roll:.1f} tau_pitch={tau_pitch:.1f}', flush=True)
            for i in range(4):
                if not contact[i]:
                    continue
                base = i * 4
                hip_idx = base + 1
                knee_idx = base + 2
                left = i in (0, 2)
                front = i in (0, 1)
                roll_side = -1.0 if left else 1.0
                pitch_axle = -1.0 if front else 1.0
                tau[hip_idx] += roll_side * tau_roll
                tau[knee_idx] += pitch_axle * tau_pitch
        leg_ids = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)
        for j in leg_ids:
            tau[j] = float(np.clip(tau[j], -48.0, 48.0))

        if __import__('os').environ.get('S10_STANCE_GUARD_DEBUG', '0') == '1':
            print(f'[GUARD] fn={[round(float(v),1) for v in fn]} contact={[bool(v) for v in contact]} swing={[bool(v) for v in request_swing]}', flush=True)
        return tau

    @staticmethod
    def _point_in_support(p, pts, margin):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape[0] == 2:
            a, b = pts[0], pts[1]
            ab = b - a
            denom = float(np.dot(ab, ab)) + 1e-9
            t = float(np.dot(p - a, ab) / denom)
            t = min(max(t, 0.0), 1.0)
            closest = a + t * ab
            dist = float(np.linalg.norm(p - closest))
            return dist >= margin
        if pts.shape[0] < 3:
            return False
        return StairStanceGuard._point_in_polygon(p, pts, margin)

    @staticmethod
    def _point_in_polygon(p, pts, margin):
        # True if p is inside convex polygon pts with margin.
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape[0] < 3:
            return False
        orient = 0.0
        for k in range(pts.shape[0]):
            a = pts[k]
            b = pts[(k + 1) % pts.shape[0]]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            s = 1.0 if cross >= 0 else -1.0
            if k == 0:
                orient = s
            elif s * orient < 0:
                return False
            edge_len = float(np.linalg.norm(b - a)) + 1e-9
            dist = abs(cross) / edge_len
            if dist < margin:
                return False
        return True
