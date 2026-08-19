"""stair_mode.py -- CRUISE/STAIR 判定（独立于 nav 层）。

只负责：根据 lidar 高程图更新 CRUISE/STAIR 与 TK1 交付门控。
不输出 [vx,vyaw]，不做 CTE/曲率控制。
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

from s10_mpc.auto_nav import AutoNavFollower  # noqa: E402


class StairGate:
    def __init__(self, waypoints):
        os.environ.setdefault('S10_GLOBAL_FILLET_R', '0')
        os.environ.setdefault('S10_STAIR_CORRIDOR_X', '0.0')
        self.fol = AutoNavFollower(waypoints)

    def update(self, pos2, next_idx, yaw, local_map, body_vx, wheel_z,
               heading=None):
        self._next_idx = next_idx
        # 只推进弧长游标供感知使用，不做速度/转向控制
        if hasattr(self.fol, 'path_pts'):
            k = int(np.argmin(np.sum(
                (self.fol.path_pts[:, :2] - np.asarray(pos2)[None, :]) ** 2,
                axis=1)))
            s_proj = float(self.fol.path_cum[k])
            self.fol._s_cur = max(float(getattr(self.fol, '_s_cur', 0.0)),
                                  s_proj)
        _ch = self._climb_heading()
        self.fol.update_mode(pos2, next_idx, yaw=yaw, local_map=local_map,
                             body_vx=body_vx, wheel_z=wheel_z,
                             heading=_ch)

    @property
    def mode(self):
        return self.fol.mode

    @property
    def decel_request(self):
        return self.fol.decel_request

    @property
    def stair_ahead_dist(self):
        return self.fol.stair_ahead_dist

    @property
    def drop_ahead_dist(self):
        return self.fol.drop_ahead_dist

    @property
    def stair_first_heading(self):
        return self.fol._stair_first_heading

    def _climb_heading(self):
        rs = getattr(self.fol, 'stair_rises_s', None)
        if not rs:
            return None
        # riser 点在路径上插值（顶点索引取到的是段起点顶点：
        # 平台角 riser 落在 wp3-4 段中部，顶点=wp3，算出的
        # riser->drop 方位变成整段西向，round86 实测 RL 西直行）
        rxy0 = np.asarray(self.fol._path_point_at(float(rs[0]))[:2])
        # 台阶立面法向 = riser -> 远沿方向（比路径航向准：路径斜穿
        # 台沿时爬升轴会被带偏，RL 沿 2.91 爬台西漂 3m 实测）
        # 选 riser 之后 >=0.8m 的跌落沿（台面远沿）：平台东角是
        # 0.44m 窄条，最近的 drop 是角自身西沿，riser->drop 方位
        # 变成西向（round86/87 实测 RL 西直行）；取远沿才是台面
        # 爬升轴（北向 1.67）
        dd = None
        _drops = getattr(self.fol, '_elev_drops', None)
        _rs0 = float(rs[0]) - float(getattr(self.fol, '_s_cur', 0.0))
        if _drops:
            # 取台面远沿：角部墙区 drop 噪声簇横跨 2.2~4.8m，
            # min 取到簇内中间值（方位 2.3~2.5 西偏），max 才
            # 落在真远沿（方位 ~1.8 北偏，被航线夹角门挡住）
            _cand = [x for x in _drops if x > _rs0 + 0.8]
            if _cand:
                dd = float(max(_cand))
        if dd is not None:
            s2 = float(getattr(self.fol, '_s_cur', 0.0)) + float(dd)
            xy2 = np.asarray(self.fol._path_point_at(s2)[:2])
            dv = xy2 - rxy0
            if np.linalg.norm(dv) > 0.3:
                return float(np.arctan2(dv[1], dv[0]))
        # 无台面远沿：riser 距段末航点 >=1.0m（航线过台后仍直行）
        # → 航线航向（6 级楼梯/正对台面，与航线夹角 ~0 放行）；
        # 距段末 <1.0m（航线即将转弯，riser 属路径外障碍——平台
        # 东角在 wp4 前 0.44m 实测）→ 瞄下一航点，与航线夹角大
        # 被 TK1 航线夹角门挡住，不交 RL
        _ni = getattr(self, '_next_idx', None)
        _ws = self.wp_s(_ni) if _ni is not None else None
        if _ws is not None and _ws - float(rs[0]) >= 1.0:
            k = int(np.searchsorted(self.fol.path_cum, float(rs[0]),
                                    side='right') - 1)
            k = min(max(k, 0), len(self.fol.path_heading) - 1)
            return float(self.fol.path_heading[k])
        if _ni is not None and _ni + 1 < len(self.fol.wp):
            _wxy = np.asarray(self.fol.wp[_ni + 1][:2])
            _dv2 = _wxy - rxy0
            if np.linalg.norm(_dv2) > 0.3:
                return float(np.arctan2(_dv2[1], _dv2[0]))
        k = int(np.searchsorted(self.fol.path_cum, float(rs[0]),
                                side='right') - 1)
        k = min(max(k, 0), len(self.fol.path_heading) - 1)
        return float(self.fol.path_heading[k])

    def wp_s(self, idx):
        if idx < 0 or idx >= len(self.fol.wp):
            return None
        k = int(np.argmin(np.sum(
            (self.fol.path_pts[:, :2] - self.fol.wp[idx][:2]) ** 2, axis=1)))
        return float(self.fol.path_cum[k])

    @property
    def climb_heading(self):
        return self._climb_heading()

    @property
    def first_riser_path_heading(self):
        rs = getattr(self.fol, 'stair_rises_s', None)
        if not rs:
            return None
        k = int(np.searchsorted(self.fol.path_cum, rs[0],
                                side='right') - 1)
        k = min(max(k, 0), len(self.fol.path_heading) - 1)
        return float(self.fol.path_heading[k])

    @property
    def s_cur(self):
        return self.fol._s_cur

    @property
    def path_pts(self):
        return self.fol.path_pts

    @property
    def path_cum(self):
        return self.fol.path_cum

    @property
    def path_heading(self):
        return self.fol.path_heading
