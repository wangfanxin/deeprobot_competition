"""costmap2d.py: 2D truncated distance field (ESDF) for MPPI obstacle avoidance.

User goal 3 + friend review (2026-08-16):
  lidar wall/terrain channel
    -> obstacle classification (OFF-path walls/edges only; ON-path stairs excluded)
    -> inflation (robot half-width)
    -> truncated unsigned distance field (scipy EDT)
    -> BodyMPPI soft potential  rho = max(0, d_safe-d)^2 / d_safe^2
Only world-frame. Built from LidarTerrainV2 (lidar only) + wp-derived path (allowed).
"""
import os
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt, binary_dilation, minimum_filter
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


class CostMap2D:
    def __init__(self, d, origin, res, dmax):
        self.d = np.asarray(d, dtype=np.float64)   # (ny,nx) truncated distance [m]
        self.origin = np.asarray(origin, dtype=np.float64)  # (x0, y0)
        self.res = float(res)
        self.dmax = float(dmax)
        self.ny, self.nx = self.d.shape

    def query(self, xy):
        """xy: (...,2) world -> (...,) truncated distance to nearest obstacle.

        Bilinear interpolation; out-of-window -> dmax (treated as far / no penalty,
        the local window already covers the relevant path vicinity).
        """
        xy = np.asarray(xy, dtype=np.float64)
        gx = (xy[..., 0] - self.origin[0]) / self.res
        gy = (xy[..., 1] - self.origin[1]) / self.res
        ix = np.floor(gx).astype(np.int64)
        iy = np.floor(gy).astype(np.int64)
        fx = gx - ix
        fy = gy - iy
        inb = ((ix >= 0) & (ix < self.nx - 1)
               & (iy >= 0) & (iy < self.ny - 1))
        d = np.full(xy.shape[:-1], self.dmax, dtype=np.float64)
        ix0 = np.clip(ix, 0, self.nx - 1)
        iy0 = np.clip(iy, 0, self.ny - 1)
        ix1 = np.clip(ix + 1, 0, self.nx - 1)
        iy1 = np.clip(iy + 1, 0, self.ny - 1)
        d00 = self.d[iy0, ix0]
        d01 = self.d[iy0, ix1]
        d10 = self.d[iy1, ix0]
        d11 = self.d[iy1, ix1]
        top = d00 * (1.0 - fx) + d01 * fx
        bot = d10 * (1.0 - fx) + d11 * fx
        interp = top * (1.0 - fy) + bot * fy
        d = np.where(inb, interp, d)
        return d


def _path_window(path_pts, path_cum, s_cur, half, margin=6.0):
    """Indices of path points near the robot arc-length (for lateral-distance)."""
    if path_cum is None or len(path_cum) == 0:
        return np.arange(len(path_pts))
    lo = s_cur - (half + margin)
    hi = s_cur + (half + margin)
    i0 = int(np.searchsorted(path_cum, lo, side='left'))
    i1 = int(np.searchsorted(path_cum, hi, side='right'))
    i0 = max(0, i0 - 2); i1 = min(len(path_pts), i1 + 2)
    return np.arange(i0, i1)


def build_costmap(lterr, path_pts, path_cum, s_cur, cx, cy,
                  half=8.0, res=0.2, lat_min=0.5, inflate=0.3,
                  dmax=2.0, h_min=0.3, h_hard=0.5):
    """Build a local truncated distance field from the lidar wall channel.

    WALL vs STAIR discrimination (USER 2026-08-16, goal 3.1):
      - stair grows stepwise, each step <= 0.3m (ON-path) -> NOT obstacle;
      - wall is very tall (protrusion >> 0.3m) and on the path SIDES -> obstacle.
    So the PRIMARY criterion is the vertical PROTRUSION above the local ground:
      protrusion = wall_h - base_ground (base_ground = local min terrain hmax).
    Obstacle = protrusion > h_min AND (off-path OR protrusion > h_hard).
    A stair riser (<=0.3m) is excluded by height; a tall wall on the side is kept,
    a very tall wall on the path is hard-blocked. Returns CostMap2D or None.
    """
    if not _HAS_SCIPY:
        return None
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    nx = max(int((x1 - x0) / res), 2)
    ny = max(int((y1 - y0) / res), 2)
    obstacle = np.zeros((ny, nx), dtype=bool)

    # wall-channel cells inside the window (LidarTerrainV2 grid, res 0.05)
    wi0 = max(int(np.floor((y0 - lterr.oy) / lterr.res)), 0)
    wi1 = min(int(np.ceil((y1 - lterr.oy) / lterr.res)), lterr.ny - 1)
    wj0 = max(int(np.floor((x0 - lterr.ox) / lterr.res)), 0)
    wj1 = min(int(np.ceil((x1 - lterr.ox) / lterr.res)), lterr.nx - 1)
    if wi1 <= wi0 or wj1 <= wj0:
        return None
    wv = lterr.wall_valid[wi0:wi1, wj0:wj1]
    iy_w, ix_w = np.where(wv > 0)
    if len(ix_w) == 0:
        return None
    wx = lterr.ox + (wj0 + ix_w) * lterr.res
    wy = lterr.oy + (wi0 + iy_w) * lterr.res
    wh = lterr.wall_h[wi0:wi1, wj0:wj1][iy_w, ix_w]

    # base ground = local minimum terrain surface (invalid -> +inf). A small
    # min-filter window (5 cells = 0.25m) finds the lower side of the face, so
    # protrusion ~= face height. Stair steps (<=0.3m) fall below h_min.
    hm = lterr.hmax[wi0:wi1, wj0:wj1].astype(np.float64)
    hm[~np.isfinite(hm)] = np.inf
    base = minimum_filter(hm, size=5, mode='constant', cval=np.inf)
    protrusion = wh - base[iy_w, ix_w]

    # on/off-path lateral distance (walls on the sides; stairs on the path)
    pw = _path_window(path_pts, path_cum, s_cur, half)
    pp = np.asarray(path_pts)[pw]
    if len(pp) == 0:
        pp = np.asarray(path_pts)
    lat = np.full(len(wx), np.inf)
    CH = 512
    for k0 in range(0, len(wx), CH):
        wxs = wx[k0:k0 + CH]; wys = wy[k0:k0 + CH]
        d2 = ((pp[None, :, 0] - wxs[:, None]) ** 2
              + (pp[None, :, 1] - wys[:, None]) ** 2)
        lat[k0:k0 + CH] = np.sqrt(d2.min(axis=1))

    tall = protrusion > h_min
    hard = protrusion > h_hard
    offpath = lat > lat_min
    # USER 2026-08-17: path high walls must be bypassed. Only climb steps
    # whose rise is <= h_hard (default 0.5m). So:
    #   obstacle = very tall (hard) regardless of on/off path
    #            OR side wall taller than h_min
    # Stairs <=0.5m are NOT obstacles.
    keep = hard | (tall & offpath)
    wx, wy = wx[keep], wy[keep]
    if len(wx) == 0:
        return None

    # rasterise kept wall cells into the costmap grid
    gx = np.floor((wx - x0) / res).astype(np.int64)
    gy = np.floor((wy - y0) / res).astype(np.int64)
    m = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    obstacle[gy[m], gx[m]] = True

    # inflation (robot half-width + margin)
    r = max(int(np.ceil(inflate / res)), 1)
    if r > 0 and obstacle.any():
        it = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        it = (xx * xx + yy * yy) <= (r * r)
        obstacle = binary_dilation(obstacle, structure=it)

    # TEMP diagnostic (USER goal 3 tuning): log obstacle bounds when enabled.
    if os.environ.get("S10_OBST_DEBUG", "0") == "1" and obstacle.any():
        _ys, _xs = np.where(obstacle)
        print(f"[OBST-DBG] cells={int(obstacle.sum())} "
              f"x[{x0+_xs.min()*res:.1f},{x0+_xs.max()*res:.1f}] "
              f"y[{y0+_ys.min()*res:.1f},{y0+_ys.max()*res:.1f}]", flush=True)

    # truncated unsigned distance field (0 inside obstacle, free-space dist)    # truncated unsigned distance field (0 inside obstacle, free-space dist)
    d = distance_transform_edt(~obstacle) * res
    d = np.clip(d, 0.0, dmax)
    return CostMap2D(d, np.array([x0, y0]), res, dmax)
