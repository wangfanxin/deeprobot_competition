"""StairContactPlanner: lidar-elevation based continuous contact planner for DiAL.

Keeps the discrete gait/foothold decision outside MBDPI sampling:
- LidarTerrainV2 builds an incremental elevation map from simulated LiDAR rays.
- Riser table is detected from that elevation map along the navigation path.
- A 60x60 world-aligned tile is fed to MPCController.set_elevation_map().
- wheel_ref / foothold_y fields are generated from the detected riser table.
- Gait is a continuous swing weight injected into MPCController._gait_swing.
"""

import os
import numpy as np

from s10_mpc.lidar_terrain_v2 import LidarTerrainV2
from perception.local_map import LocalMapTile, compute_terrain_features


class StairContactPlanner:
    def __init__(self, model, data, follower, tile_half=3.0, res=0.10):
        self.model = model
        self.data = data
        self.follower = follower
        self.res = float(res)
        self.tile_half = float(tile_half)
        self.n_out = int(round(2.0 * self.tile_half / self.res))
        self.lidar = LidarTerrainV2(model, data, res=self.res)
        self.riser_records = []

    def update_perception(self, t):
        self.lidar.update()

    def update_risers(self, s_cur):
        """Detect risers from lidar max-z along the local path window.

        The follower's hard-coded stair tables are only a fallback; once
        enough risers are detected we overwrite S10_STAIR_RISERS / TOPS so all
        follower geometry functions consume lidar-derived data.
        """
        fol = self.follower
        pts = getattr(fol, "path_pts", None)
        cum = getattr(fol, "path_cum", None)
        if pts is None or cum is None or len(pts) == 0:
            return
        s_lo = max(float(cum[0]), float(s_cur) - 1.0)
        s_hi = float(s_cur) + 6.0
        det = self.lidar.detect_risers(pts, cum, s_lo, s_hi,
                                       rise=0.05, max_dh=0.16)
        changed = False
        for s, dh, top in det:
            k = int(np.searchsorted(cum, s, side="right") - 1)
            k = min(max(k, 0), len(pts) - 1)
            y = float(pts[k, 1])
            if any(abs(y - rr[3]) < 0.05 for rr in self.riser_records):
                continue
            self.riser_records.append((float(s), float(dh), float(top), y))
            changed = True
        if not changed or len(self.riser_records) < 1:
            return
        # Sort by path arc and keep unique tread y positions.
        recs = sorted(self.riser_records, key=lambda r: r[0])
        ys = []
        tops = []
        for s, dh, top, y in recs:
            if not ys or abs(y - ys[-1]) > 0.03:
                ys.append(y)
                tops.append(top)
        os.environ["S10_STAIR_RISERS"] = ",".join(
            [f"{v:.4f}" for v in ys])
        os.environ["S10_STAIR_TOPS"] = ",".join(
            [f"{v:.4f}" for v in tops])
        fol._stair_last_arc = float(recs[-1][0])

    def get_tile(self, robot_xy, t=0.0):
        """Crop the lidar map to a fixed (n_out,n_out) tile around the robot."""
        ox, oy = float(self.lidar.ox), float(self.lidar.oy)
        x0 = float(np.floor((robot_xy[0] - self.tile_half) / self.res) * self.res)
        y0 = float(np.floor((robot_xy[1] - self.tile_half) / self.res) * self.res)
        ix0 = int(round((x0 - ox) / self.res))
        iy0 = int(round((y0 - oy) / self.res))
        n = self.n_out
        valid = np.zeros((n, n), dtype=np.bool_)
        height = np.full((n, n), 10.0, dtype=np.float32)
        src = self.lidar.valid
        hgt = self.lidar.h
        for i in range(n):
            for j in range(n):
                si = iy0 + i
                sj = ix0 + j
                if 0 <= si < src.shape[0] and 0 <= sj < src.shape[1]:
                    if bool(src[si, sj]):
                        valid[i, j] = True
                        height[i, j] = float(hgt[si, sj])
        tile = LocalMapTile(height, valid, (x0, y0), self.res, n, n,
                            float(t))
        features = compute_terrain_features(tile, step_threshold=0.18)
        out = {
            "heightmap": height,
            "valid": valid,
            "origin": np.array([x0, y0], dtype=np.float32),
            "resolution": self.res,
            "nx": n,
            "ny": n,
            "features": features,
        }
        fol = self.follower
        if hasattr(fol, "stair_wheel_ref_grid"):
            wr, wr_ok = fol.stair_wheel_ref_grid(x0, y0, n, n, self.res)
            features["wheel_ref"] = wr.astype(np.float32)
            features["wheel_ref_valid"] = wr_ok
        if hasattr(fol, "stair_foothold_y_grid"):
            fy, fy_ok = fol.stair_foothold_y_grid(x0, y0, n, n, self.res)
            features["foothold_y"] = fy.astype(np.float32)
            features["foothold_valid"] = fy_ok
        if os.environ.get("S10_KNOWN_TERRAIN", "0") == "1" and hasattr(fol, "stair_known_tile"):
            kt = fol.stair_known_tile(x0, y0, n, n, self.res)
            if kt is not None:
                mk = kt["valid"]
                out["heightmap"] = np.where(mk, kt["heightmap"], out["heightmap"])
                out["valid"] = out["valid"] | mk
                for _key in ("slope", "roughness", "step", "step_flag"):
                    features[_key] = np.where(mk, kt[_key], features[_key])
        return out

    def compute_hard_mode(self, wheel_y, wheel_z):
        fol = self.follower
        rs, ts = fol._stair_tables()
        wheel_y = np.asarray(wheel_y, dtype=np.float64)
        wheel_z = np.asarray(wheel_z, dtype=np.float64)
        mode = np.zeros(4, dtype=np.float32)
        foothold_z = fol.stair_terrain(wheel_y) + 0.081
        # Front wheels swing when they are close to the next riser and the
        # next tread is clearly above the current tread.
        for i in (0, 1):
            idx = int(np.searchsorted(rs, wheel_y[i]))
            if idx < len(rs):
                d = float(rs[idx]) - float(wheel_y[i])
                target_z = float(ts[idx]) + 0.081
                need = target_z - float(wheel_z[i])
                if d < float(os.environ.get('S10_HARD_FRONT_PROX', '0.15')) and need > 0.02:
                    mode[i] = 1.0
                    foothold_z[i] = target_z
        # Rear wheels swing after the front wheels have reached a higher tread.
        front_terr = max(float(fol.stair_terrain(np.array([wheel_y[0]]))[0]),
                         float(fol.stair_terrain(np.array([wheel_y[1]]))[0]))
        rear_terr = float(fol.stair_terrain(np.array([wheel_y[2]]))[0])
        for i in (2, 3):
            idx = int(np.searchsorted(rs, wheel_y[i]))
            if idx < len(rs):
                d = float(rs[idx]) - float(wheel_y[i])
                target_z = front_terr + 0.081
                need = target_z - float(wheel_z[i])
                if (front_terr - rear_terr) > 0.06 and d < float(os.environ.get('S10_HARD_REAR_PROX', '0.15')) and need > 0.02:
                    mode[i] = 1.0
                    foothold_z[i] = target_z
        return mode, foothold_z.astype(np.float32)

    def stair_confirmed(self, robot_xy, yaw):
        """Small forward window over max-z detects at least one riser."""
        return self.lidar.stair_confirmed(robot_xy, yaw, rise=0.06,
                                          need=1, span=2.0)

    def apply_contact(self, mpc, wheel_y, wheel_z, t, body_y):
        """Write continuous gait/prox/geometry references into MPCController."""
        fol = self.follower
        rs, _ = fol._stair_tables()
        prox = np.full(4, 1e9, dtype=np.float64)
        for k in range(4):
            ix = int(np.searchsorted(rs, float(wheel_y[k])))
            if ix < len(rs):
                prox[k] = float(rs[ix]) - float(wheel_y[k])
        mpc._stair_prox = np.asarray(prox, dtype=np.float32)

        if (os.environ.get("S10_GAIT", "0") == "1"
                or os.environ.get("S10_GAIT_UTIL", "0") == "1"):
            mpc._gait_swing = np.asarray(
                fol.gait_schedule(wheel_y, wheel_z, t), dtype=np.float32)

        y_arr = np.asarray([body_y], dtype=np.float64)
        pitch = float(fol.stair_pitch_ref(y_arr)[0])
        base_z = float(fol.stair_base_z_ref(y_arr)[0])
        mpc.set_stair_ref(pitch, base_z)
        wr_now = np.asarray(fol.stair_wheel_ref(np.asarray(wheel_y, dtype=np.float64)), dtype=np.float64)
        lift_now = np.clip(wr_now - np.asarray(wheel_z, dtype=np.float64), 0.0, 0.25)
        lneed = max(float(lift_now[0]), float(lift_now[2]))
        rneed = max(float(lift_now[1]), float(lift_now[3]))
        _imb = float(np.clip((lneed - rneed) * float(os.environ.get("S10_ROLL_IMB_GAIN", "0.8")), -0.15, 0.15))
        mpc._stair_roll_override = _imb
        if os.environ.get('S10_STAIR_PLANNER_DEBUG', '0') == '1':
            wr = np.asarray(fol.stair_wheel_ref(np.asarray(wheel_y, dtype=np.float64)), dtype=np.float64)
            if int(t * 20) % 20 == 0:
                print(f'[PLANNER] t={t:.1f} wy={[round(float(v),2) for v in wheel_y]} wz={[round(float(v),2) for v in wheel_z]} wr={[round(float(v),2) for v in wr]} prox={[round(float(v),2) if v < 1e8 else -1 for v in prox]}', flush=True)

        # Soft time-varying action bias: keep MBDPI sampling mean in the
        # direction of the geometric lift field. It is a prior, not a gate.
        Hn = int(getattr(mpc, 'Y', None).shape[0] if hasattr(mpc, 'Y') else 0)
        if Hn <= 0 or not hasattr(mpc, 'set_stair_action_bias'):
            return
        dt = float(getattr(mpc, 'dt', 0.02))
        vx = max(abs(float(np.asarray(mpc.cmd_vel)[0])), 0.1)
        sww = np.asarray(getattr(mpc, '_gait_swing', np.zeros(4)), dtype=np.float32)
        bprof = float(os.environ.get('S10_BIAS_T_PROFILE', '0'))
        lift_min = float(os.environ.get('S10_BIAS_LIFT_MIN', '0.05'))
        full_ref = os.environ.get('S10_BIAS_FULL_REF', '0') == '1'
        bc = np.zeros(8, dtype=np.float32)
        bc[0] = float(os.environ.get('S10_BIAS_FL_HIPY', '0.20'))
        bc[1] = float(os.environ.get('S10_BIAS_FL_KNEE', '-0.50'))
        bc[2] = float(os.environ.get('S10_BIAS_FR_HIPY', '0.20'))
        bc[3] = float(os.environ.get('S10_BIAS_FR_KNEE', '-0.50'))
        bc[4] = float(os.environ.get('S10_BIAS_HL_HIPY', '-0.10'))
        bc[5] = float(os.environ.get('S10_BIAS_HL_KNEE', '0.45'))
        bc[6] = float(os.environ.get('S10_BIAS_HR_HIPY', '-0.10'))
        bc[7] = float(os.environ.get('S10_BIAS_HR_KNEE', '0.45'))
        bH = np.zeros((Hn, 12), dtype=np.float32)
        for k in range(Hn):
            tt = k / max(Hn - 1, 1)
            pro = 1.0
            if bprof == 1.0:
                pro = float(np.cos(np.pi * tt / 2.0))
            elif bprof == 2.0:
                pro = float(np.sin(np.pi * tt))
            pw = np.array([pro if float(sww[i]) > 0.3 else 1.0
                           for i in range(4)], dtype=np.float32)
            yk = np.asarray(wheel_y, dtype=np.float64) + vx * k * dt
            wrk = np.asarray(fol.stair_wheel_ref(yk), dtype=np.float64)
            if full_ref:
                nrr = np.asarray(fol.stair_next_riser_ref(yk), dtype=np.float64)
                wrk = np.maximum(wrk, nrr)
            lk = np.clip(wrk - np.asarray(wheel_z, dtype=np.float64), 0.0, 0.25)
            lk = np.where(lk < lift_min, 0.0, lk)
            nk = np.clip(lk / 0.15, 0.0, 1.0)
            b12 = np.zeros(12, dtype=np.float32)
            b12[1] = bc[0] * nk[0] * pw[0]
            b12[2] = bc[1] * nk[0] * pw[0]
            b12[4] = bc[2] * nk[1] * pw[1]
            b12[5] = bc[3] * nk[1] * pw[1]
            b12[7] = bc[4] * nk[2] * pw[2]
            b12[8] = bc[5] * nk[2] * pw[2]
            b12[10] = bc[6] * nk[3] * pw[3]
            b12[11] = bc[7] * nk[3] * pw[3]
            bH[k] = b12
        mpc.set_stair_action_bias(bH)
