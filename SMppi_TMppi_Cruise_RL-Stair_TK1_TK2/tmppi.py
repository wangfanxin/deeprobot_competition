"""tmppi.py -- TMppi 航点低速转向模块（独立实现，只做近点转向）。"""
import os
import numpy as np


def wrap_angle(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class TMppi:
    def __init__(self):
        self.split = os.environ.get('S10_TURN_SPLIT', '1') == '1'
        self.arrive_r = float(os.environ.get('S10_WP_ARRIVE_R', '0.2'))
        # 触发半径可大于判点半径：近点先原地转、再前插过点，避免
        # 终点代价绕点极限环（wp1 实测 6s 绕圈）。
        self.trig_r = float(os.environ.get('S10_TURN_ARRIVE_R',
                                           str(self.arrive_r)))
        self.v_max = float(os.environ.get('S10_TURN_V_MAX', '0.2'))
        self.vx_cmd = float(os.environ.get('S10_WP_TURN_VX', '0.2'))
        self.k = float(os.environ.get('S10_TURN_K', '3.0'))
        self.om_max = float(os.environ.get('S10_TURN_OM_MAX', '2.0'))
        self.err_deg = float(os.environ.get('S10_TURN_ERR_DEG', '10.0'))

    def try_plan(self, body_xy, yaw, speed, wp_cur, wp_next):
        """满足触发条件时返回 (True, vx, omega)，否则 (False, None, None)。"""
        if not self.split or wp_next is None:
            return False, None, None
        dist = float(np.linalg.norm(np.asarray(body_xy) - wp_cur[:2]))
        if dist >= self.trig_r or speed >= self.v_max:
            return False, None, None
        target = float(np.arctan2(wp_next[1] - wp_cur[1],
                                  wp_next[0] - wp_cur[0]))
        err = wrap_angle(target - yaw)
        if abs(err) <= np.radians(self.err_deg):
            return False, None, None
        vx = self.vx_cmd
        om = float(np.clip(self.k * err, -self.om_max, self.om_max))
        return True, vx, om
