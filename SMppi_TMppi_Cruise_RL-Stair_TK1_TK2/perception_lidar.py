"""perception_lidar.py -- 全局 lidar 感知模块。

提供 TK1/TK2/RL 共用的高程图、轮下 terrain_at、楼梯 heading、在线 riser 表。
禁止 god-view ray，禁止已知地图硬编码表。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PKG = os.path.join(REPO, 'src', 'S10_sdk_deploy')
for _p in (HERE, PKG, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from s10_mpc.lidar_terrain_v2 import LidarTerrainV2  # noqa: E402
from rl_stair.deploy.elev_tile import build_local_tile  # noqa: E402


class LidarPerception:
    def __init__(self, m, d):
        self.m = m
        self.d = d
        self.lterr = LidarTerrainV2(m, d)
        self._last_upd = -1e9

    def update(self, t):
        hz = float(os.environ.get('S10_ELEV_HZ', '4.0'))
        if t - self._last_upd >= 1.0 / max(hz, 1.0):
            self.lterr.update()
            self._last_upd = t

    def height(self, x, y, t, body_z):
        """轮下地形：只用 lidar；无数据/高架伪影用运动学兜底。"""
        self.update(t)
        if not self.lterr.has(x, y):
            return float(body_z - 0.55)
        h = self.lterr.height(x, y)
        if h > body_z + 1.0:
            return float(body_z - 0.55)
        return h

    def local_tile(self, body_xy, t):
        self.update(t)
        return build_local_tile(
            self.lterr, float(body_xy[0]), float(body_xy[1]),
            half=float(os.environ.get('S10_ELEV_HALF', '8.0')))

    def stair_heading(self, fol):
        """返回路径前方 riser 对应的路径航向；未检测到返回 None。"""
        sc = float(fol.s_cur)
        lo = sc + 0.5
        hi = sc + float(os.environ.get('S10_TK1_LOOKAHEAD', '5.0'))
        cum = fol.path_cum
        k0 = int(np.searchsorted(cum, lo))
        k1 = int(np.searchsorted(cum, hi))
        if k1 <= k0 + 3:
            return None
        try:
            rs = self.lterr.detect_risers(
                fol.path_pts[k0:k1], cum[k0:k1], lo, hi,
                rise=0.05, max_dh=0.16)
        except Exception:
            rs = []
        if rs:
            sm = float(np.mean([float(r[0]) for r in rs]))
            ki = int(np.searchsorted(cum, sm, side='right') - 1)
            ki = max(0, min(ki, len(fol.path_heading) - 1))
            return float(fol.path_heading[ki])

        # wall 通道：on-path 垂直面（六级楼梯）
        wpts = fol.path_pts[k0:k1]
        x0 = float(wpts[:, 0].min() - 1.0)
        x1 = float(wpts[:, 0].max() + 1.0)
        y0 = float(wpts[:, 1].min() - 1.0)
        y1 = float(wpts[:, 1].max() + 1.0)
        res = self.lterr.res
        wi0 = max(int(np.floor((y0 - self.lterr.oy) / res)), 0)
        wi1 = min(int(np.ceil((y1 - self.lterr.oy) / res)), self.lterr.ny - 1)
        wj0 = max(int(np.floor((x0 - self.lterr.ox) / res)), 0)
        wj1 = min(int(np.ceil((x1 - self.lterr.ox) / res)), self.lterr.nx - 1)
        if wi1 <= wi0 or wj1 <= wj0:
            return None
        wv = self.lterr.wall_valid[wi0:wi1, wj0:wj1]
        iy, ix = np.where(wv > 0)
        if len(ix) == 0:
            return None
        wx = self.lterr.ox + (wj0 + ix) * res
        wy = self.lterr.oy + (wi0 + iy) * res
        lat_min = float(os.environ.get('S10_OBST_LAT_MIN', '0.5'))
        d2 = ((wpts[None, :, 0] - wx[:, None]) ** 2
              + (wpts[None, :, 1] - wy[:, None]) ** 2)
        lat = np.sqrt(d2.min(axis=1))
        op = lat < lat_min
        if int(op.sum()) < int(os.environ.get('S10_TK1_MIN_CELLS', '8')):
            return None
        scs = []
        for ii in np.where(op)[0]:
            dd = ((wpts[:, 0] - wx[ii]) ** 2
                  + (wpts[:, 1] - wy[ii]) ** 2)
            scs.append(cum[k0 + int(np.argmin(dd))])
        sm = float(np.mean(scs))
        ki = int(np.searchsorted(cum, sm, side='right') - 1)
        ki = max(0, min(ki, len(fol.path_heading) - 1))
        return float(fol.path_heading[ki])

    def riser_table(self, fol):
        """沿原始折线在线检测 riser，返回 (xy(N,2), top(N,))。"""
        sc = float(fol.s_cur)
        lo = sc + 0.3
        hi = sc + max(6.0, float(os.environ.get('S10_TK1_LOOKAHEAD', '5.0')))
        k0 = int(np.searchsorted(fol.path_cum, lo))
        k1 = int(np.searchsorted(fol.path_cum, hi))
        if k1 <= k0 + 3:
            return None, None
        try:
            rs = self.lterr.detect_risers(
                fol.path_pts[k0:k1], fol.path_cum[k0:k1], lo, hi,
                rise=0.05, max_dh=0.16)
        except Exception:
            return None, None
        if not rs:
            return None, None
        xy = []
        tops = []
        for (sr, _dh, top) in rs:
            ki = int(np.searchsorted(fol.path_cum, sr, side='right') - 1)
            ki = max(0, min(ki, len(fol.path_pts) - 1))
            xy.append(fol.path_pts[ki, :2].copy())
            tops.append(float(top))
        return np.asarray(xy, dtype=np.float64), np.asarray(tops,
                                                            dtype=np.float64)
