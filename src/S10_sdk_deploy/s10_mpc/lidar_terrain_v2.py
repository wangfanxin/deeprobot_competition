import numpy as np


class LidarTerrainV2:
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
        for i in np.where(hit)[0]:
            p = pts[i]
            if p[2] > pos[2] + 0.5:
                continue
            ix = int(np.floor((p[0] - self.ox) / self.res))
            iy = int(np.floor((p[1] - self.oy) / self.res))
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                if p[2] < self.h[iy, ix]:
                    self.h[iy, ix] = p[2]
                self.valid[iy, ix] = 1

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
