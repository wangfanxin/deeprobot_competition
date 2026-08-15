import os
import numpy as np


class LidarTerrainV2:
    '''Incremental lidar elevation map (deployable: no god-view).

    Dual per-column height:
      - h   = min-z (clearance): a surface hit implies solid below it, so a
        vertical riser face reads as the face-bottom / lower tread. Used for
        wheel terrain.
      - hmax = max-z (surface): the top of the solid column, so a riser face
        reads as the upper tread. Used for riser detection (tread tops).
    Side-fill: fill the +-x (track-lateral) neighbours of fresh hits, closing
    cross-track holes without smearing the along-path risers.
    '''

    def __init__(self, model, data, x0=-25.0, x1=40.0, y0=-5.0, y1=55.0,
                 res=0.05, th_n=96, phi_n=48, fov_h=None, cutoff=20.0):
        import mujoco
        self.m, self.d = model, data
        self.res = float(res)
        self.ox, self.oy = float(x0), float(y0)
        self.nx = int(round((x1 - x0) / res)) + 1
        self.ny = int(round((y1 - y0) / res)) + 1
        self.h = np.full((self.ny, self.nx), np.inf, dtype=np.float64)
        self.hmax = np.full((self.ny, self.nx), -np.inf, dtype=np.float64)
        self.valid = np.zeros((self.ny, self.nx), dtype=np.int32)
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'lidar_site')
        if self.sid < 0:
            raise ValueError('lidar_site not found in model')
        self.cutoff = float(cutoff)
        if fov_h is None:
            # USER-DIRECTED 2026-08-16: increase lidar scan area (55deg -> 90deg) so the
            # elevation map covers the path ahead (incl. the wp4->5 turn where the robot drifts).
            fov_h = float(np.radians(float(os.environ.get("S10_LIDAR_FOV_H", "90"))))
        ths = np.linspace(-fov_h, fov_h, int(th_n))
        phs = np.linspace(np.radians(45.0), np.radians(-55.0), int(phi_n))
        dirs = []
        for ph in phs:
            for th in ths:
                dirs.append([float(np.cos(ph) * np.cos(th)),
                             float(np.cos(ph) * np.sin(th)),
                             float(np.sin(ph))])
        self.dirs_local = np.asarray(dirs, dtype=np.float64)
        self.geomgroup = np.zeros((mujoco.mjNGROUP,), dtype=np.ubyte)
        # USER-DIRECTED 2026-08-16 (GOAL #1): the real terrain (group 0) IS ray-hittable
        # once the lidar origin is raised (S10_LIDAR_RAISE_Z): shallow grazing rays from
        # the low stock mount passed THROUGH the STL mesh (garbage z=0.10-0.48); from
        # +0.6m they hit real treads (z 0.48->1.17, verified). Robot body geoms are
        # group 1/2 (NOT 0), so group 0 = terrain only -> no self-occlusion by group.
        # S10_LIDAR_PATH_ONLY=1 keeps the old group-2 path-capsule mode as fallback.
        if os.environ.get("S10_LIDAR_PATH_ONLY", "0") == "1":
            self.geomgroup[2] = 1
        else:
            self.geomgroup[0] = 1

    def update(self):
        import mujoco
        m, d = self.m, self.d
        pos = np.asarray(d.site_xpos[self.sid], dtype=np.float64)
        # USER-DIRECTED 2026-08-16 (GOAL #1): raised lidar mount (origin +z) so the
        # 96-line rays hit the terrain mesh at non-grazing incidence. Ray DIRECTIONS
        # unchanged (still follow the body); only the origin is raised -> physically a
        # sensor mast, no scene file change.
        _rz = float(os.environ.get("S10_LIDAR_RAISE_Z", "0.6"))
        if _rz > 0.0:
            pos = pos + np.array([0.0, 0.0, _rz])
        xmat = np.asarray(d.site_xmat[self.sid], dtype=np.float64).reshape(3, 3)
        vec = (self.dirs_local @ xmat.T)
        n = len(vec)
        geomid = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, -1.0, dtype=np.float64)
        norm = np.zeros((n * 3,), dtype=np.float64)
        mujoco.mj_multiRay(m, d, pos.copy(), vec.reshape(-1),
                           self.geomgroup, True, 1, geomid, dist, norm,
                           n, self.cutoff)
        hit = dist > 0.0
        # hit filter by mode: group-2 (path capsules) -> keep ONLY track_segment_*
        # (waypoint spheres / height posts / robot body geoms create false rises);
        # group-0 (real terrain) -> keep all (robot is group 1/2, already excluded).
        if int(self.geomgroup[2]) == 1:
            hit_idx = [_i for _i in np.where(hit)[0]
                       if (lambda _nm: _nm is not None and _nm.startswith("track_segment_"))(
                           mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(geomid[_i])) if geomid[_i] >= 0 else None)]
        else:
            # group-0 real terrain: keep near-horizontal surfaces only (|nz|>=nz_min).
            # Discards vertical structures (start gate walls, riser faces, overhead)
            # which otherwise read as false "steps" in the elevation map. Ground /
            # tread tops (nz~1) are kept -> the discrete stair profile survives.
            _nz_min = float(os.environ.get("S10_LIDAR_NZ_MIN", "0.6"))
            _nz = np.abs(norm.reshape(n, 3)[:, 2])
            hit_idx = [_i for _i in np.where(hit)[0] if _nz[_i] >= _nz_min]
        pts = pos + dist[hit_idx, None] * vec[hit_idx]
        fresh = []
        for i in hit_idx:
            p = pos + dist[i] * vec[i]
            if p[2] > pos[2] + 0.5:
                continue
            ix = int(np.floor((p[0] - self.ox) / self.res))
            iy = int(np.floor((p[1] - self.oy) / self.res))
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                if p[2] < self.h[iy, ix]:
                    self.h[iy, ix] = p[2]
                if p[2] > self.hmax[iy, ix]:
                    self.hmax[iy, ix] = p[2]
                if not self.valid[iy, ix]:
                    self.valid[iy, ix] = 1
                    fresh.append((ix, iy))
        for (ix, iy) in fresh:
            h0 = self.h[iy, ix]
            hm0 = self.hmax[iy, ix]
            for jx in (ix - 1, ix + 1):
                if 0 <= jx < self.nx and not self.valid[iy, jx]:
                    self.valid[iy, jx] = 1
                    self.h[iy, jx] = h0
                    self.hmax[iy, jx] = hm0

    def has(self, x, y):
        ix = int(np.floor((x - self.ox) / self.res))
        iy = int(np.floor((y - self.oy) / self.res))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return bool(self.valid[iy, ix])
        return False

    def height(self, x, y):
        ix = int(np.floor((x - self.ox) / self.res))
        iy = int(np.floor((y - self.oy) / self.res))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            if self.valid[iy, ix]:
                return float(self.h[iy, ix])
        return 0.0

    def height_or_none(self, x, y):
        ix = int(np.floor((x - self.ox) / self.res))
        iy = int(np.floor((y - self.oy) / self.res))
        if 0 <= ix < self.nx and 0 <= iy < self.ny and self.valid[iy, ix]:
            return float(self.h[iy, ix])
        return None

    def height_max_or_none(self, x, y):
        ix = int(np.floor((x - self.ox) / self.res))
        iy = int(np.floor((y - self.oy) / self.res))
        if 0 <= ix < self.nx and 0 <= iy < self.ny and self.valid[iy, ix]:
            return float(self.hmax[iy, ix])
        return None

    def stair_confirmed(self, robot_xy, yaw, rise=0.06, need=2, span=4.0):
        fx = float(np.cos(yaw))
        fy = float(np.sin(yaw))
        hs = []
        for dd in np.arange(0.2, span + 0.05, 0.1):
            h = self.height_max_or_none(robot_xy[0] + fx * dd,
                                        robot_xy[1] + fy * dd)
            if h is not None:
                hs.append(float(h))
        n_rise = 0
        for i in range(1, len(hs)):
            if hs[i] - hs[i - 1] >= rise:
                n_rise += 1
        return n_rise >= need

    def detect_risers(self, pts, cum, s_lo, s_hi, rise=0.05, max_dh=0.16,
                      top_win=0.30):
        """沿路径窗口检测 riser。

        2026-08-14 精度修复：旧版把"跳变点 hmax"直接当台面顶，但后续
        riser 会遮挡当前踏面（lidar 前下 45° 扇形），实测 riser2 顶低
        0.05m -> wheel_ref 偏低 -> 前轮差 0.012m 卡在下一级立面。现在：
          - riser 位置取跳变起点（k-1 与 k 的中点，更接近立面）；
          - 台面顶 = 跳变后 top_win(0.30m) 内 hmax 最大值（踏面中段
            无阴影处），窗口 < 阶距 0.4m 不会串到下一级。
        """
        out = []
        n = len(cum)
        res = float(cum[1] - cum[0]) if n > 1 else 0.1
        prev_h = None
        for k in range(n):
            s = float(cum[k])
            if s < s_lo:
                continue
            if s > s_hi:
                break
            h = self.height_max_or_none(float(pts[k, 0]), float(pts[k, 1]))
            if h is None:
                prev_h = None
                continue
            if prev_h is not None:
                dh = float(h - prev_h)
                if rise <= dh <= max_dh:
                    top = float(h)
                    n_win = int(top_win / max(res, 1e-3)) + 1
                    for j in range(k, min(n, k + n_win)):
                        hj = self.height_max_or_none(
                            float(pts[j, 0]), float(pts[j, 1]))
                        if hj is not None and float(hj) > top:
                            top = float(hj)
                    # 位置取跳变点 k 本身（实测 k-1 中点会早 0.04~0.085m，
                    # 使前轮提前 swing 卡在真实立面前；k 点误差 ±0.04m 最小）
                    out.append((s, dh, top))
            prev_h = h
        return out
