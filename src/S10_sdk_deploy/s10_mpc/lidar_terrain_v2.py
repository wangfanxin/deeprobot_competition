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
                 res=0.10, th_n=64, phi_n=32, fov_h=None, cutoff=20.0):
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
            fov_h = float(np.radians(55))
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
        self.geomgroup[0] = 1

    def update(self):
        import mujoco
        m, d = self.m, self.d
        pos = np.asarray(d.site_xpos[self.sid], dtype=np.float64)
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
        pts = pos + dist[:, None] * vec
        fresh = []
        for i in np.where(hit)[0]:
            p = pts[i]
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

    def stair_confirmed(self, robot_xy, yaw, rise=0.06, need=2, span=2.0):
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

    def detect_risers(self, pts, cum, s_lo, s_hi, rise=0.05, max_dh=0.15):
        out = []
        n = len(cum)
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
                    out.append((s, dh, float(h)))
            prev_h = h
        return out
