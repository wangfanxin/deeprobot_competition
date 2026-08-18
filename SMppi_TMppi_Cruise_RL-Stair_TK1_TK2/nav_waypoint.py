"""nav_waypoint.py -- 直线航点导航层（只输出直线，不做控制）。

职责：
  1. 读取 track_waypoint_* 航点；
  2. 输出当前航点到下一航点的直线段；
  3. 输出到下一航点的距离与直线 heading；
  4. 只按水平距离判点推进。

不做：vx/vyaw 控制、曲率 vlim、CTE、CRUISE/STAIR 判定。
"""
import numpy as np
import mujoco


def wrap_angle(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def extract_waypoints(m, d):
    """从 track_waypoint_* geom 读取航点（保持 XML 顺序）。"""
    wps = []
    for gid in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if name.startswith('track_waypoint_'):
            idx = int(name[len('track_waypoint_'):].split('_')[0])
            wps.append((idx, d.geom_xpos[gid].copy()))
    wps.sort()
    if not wps:
        raise RuntimeError('no track_waypoint_* geoms in model')
    return np.asarray([w for _, w in wps], dtype=np.float64)


class WaypointLineNav:
    """只输出原始航点直线。"""

    def __init__(self, waypoints):
        self.wp = np.asarray(waypoints, dtype=np.float64)

    def line(self, next_idx, robot_xy=None):
        """返回当前航段的直线信息。

        robot_xy 可选：需要到下一航点的距离时传入。
        """
        if next_idx >= len(self.wp):
            return None
        end = self.wp[next_idx, :2]
        if next_idx > 0:
            start = self.wp[next_idx - 1, :2]
        elif robot_xy is not None:
            start = np.asarray(robot_xy, dtype=np.float64)
        else:
            start = self.wp[0, :2]
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length < 1e-9:
            heading = 0.0
        else:
            heading = float(np.arctan2(vec[1], vec[0]))
        dist = None
        if robot_xy is not None:
            dist = float(np.linalg.norm(np.asarray(robot_xy) - end))
        return {
            'start': start.copy(),
            'end': end.copy(),
            'heading': heading,
            'length': length,
            'dist_to_wp': dist,
        }

    def reached(self, next_idx, robot_xy, radius=None):
        if next_idx >= len(self.wp):
            return True
        if radius is None:
            import os
            radius = float(os.environ.get('S10_WP_ADVANCE_DIST', '0.2'))
        pos = np.asarray(robot_xy, dtype=np.float64)
        if float(np.linalg.norm(pos - self.wp[next_idx, :2])) <= radius:
            return True
        # 过点兜底：沿上一航点->当前航点方向已越过当前点，且横向偏差不大，
        # 视为已通过。避免复杂地形上高速冲过 0.2m 判点圆后倒车找点。
        if next_idx > 0:
            seg = self.wp[next_idx, :2] - self.wp[next_idx - 1, :2]
            length = float(np.linalg.norm(seg))
            if length > 1e-9:
                u = seg / length
                proj = float(np.dot(pos - self.wp[next_idx - 1, :2], u))
                lat = float(np.linalg.norm(
                    pos - (self.wp[next_idx - 1, :2] + u * proj)))
                if proj > length - 0.5 and lat < 0.8:
                    return True
        return False
