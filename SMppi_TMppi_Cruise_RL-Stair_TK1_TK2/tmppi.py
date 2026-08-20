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
        self.kd = float(os.environ.get('S10_TURN_KD', '1.5'))
        self.om_max = float(os.environ.get('S10_TURN_OM_MAX', '2.0'))
        self.err_deg = float(os.environ.get('S10_TURN_ERR_DEG', '10.0'))

    def will_fire(self, body_xy, yaw, speed, wp_cur, wp_next):
        """触发条件镜像：TK1/TK2 门控用，判断当前拍是否 TMppi 模式
        （TMppi 只与 SMppi 互切，不切其它）。"""
        if not self.split or wp_next is None:
            return False
        dist = float(np.linalg.norm(np.asarray(body_xy) - wp_cur[:2]))
        if dist >= self.trig_r or speed >= self.v_max:
            return False
        target = float(np.arctan2(wp_next[1] - wp_cur[1],
                                  wp_next[0] - wp_cur[0]))
        err = wrap_angle(target - yaw)
        return abs(err) > np.radians(self.err_deg)

    def try_plan(self, body_xy, yaw, speed, omega, wp_cur, wp_next):
        """满足触发条件时返回 (True, vx, omega)，否则 (False, None, None)。"""
        if not self.will_fire(body_xy, yaw, speed, wp_cur, wp_next):
            return False, None, None
        target = float(np.arctan2(wp_next[1] - wp_cur[1],
                                  wp_next[0] - wp_cur[0]))
        err = wrap_angle(target - yaw)
        vx = self.vx_cmd
        # 终端角速度阻尼（用户指示）：转到目标角度时应收敛到停住，
        # 而不是带着角动量甩过——om = k·err − kd·ω，ω→0 时停稳
        om = float(np.clip(self.k * err - self.kd * float(omega),
                           -self.om_max, self.om_max))
        return True, vx, om
