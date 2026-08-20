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

    def will_fire(self, body_xy, yaw, speed, wp_cur, wp_next, wide=False):
        """触发条件镜像：TK1/TK2 门控用，判断当前拍是否 TMppi 模式
        （TMppi 只与 SMppi 互切，不切其它）。

        wide（楼梯交还期，用户指示）：出楼梯后由 TMppi 转过身子对准
        下一段航线——距当前 wp 2.5m 内或距下一 wp 6m 内且航向差>10°
        即触发，低速原地转（vx=0.2）对齐航线后再交还 SMppi。"""
        if not self.split or wp_next is None:
            return False
        body = np.asarray(body_xy)
        d_cur = float(np.linalg.norm(body - wp_cur[:2]))
        if wide:
            d_next = float(np.linalg.norm(body - wp_next[:2]))
            if d_cur >= 2.5 and d_next >= 6.0:
                return False
            if speed >= self.v_max:
                return False
            target = float(np.arctan2(wp_next[1] - wp_cur[1],
                                      wp_next[0] - wp_cur[0]))
            err = wrap_angle(target - yaw)
            return abs(err) > np.radians(10.0)
        # 近下一 wp 低速转向门（全局，S10_TURN_NEXT_R>0 开启，默认 0=关）：
        # wp3-4 实测 10.8s ±π 自旋卡——机器人 1.2~1.5m 外接近下一 wp、
        # 航向差大且低速时，原触发只看距上一 wp 距离（已远离>1.2m），
        # 永不接管，SMppi 沿 π 边界极限环滑转。此门与判点/段无关，
        # 对所有 wp 统一生效。
        _next_r = float(os.environ.get('S10_TURN_NEXT_R', '0.0'))
        if _next_r > 0.0:
            d_next = float(np.linalg.norm(body - wp_next[:2]))
            if d_next < _next_r and speed < self.v_max:
                _t2 = float(np.arctan2(wp_next[1] - body[1],
                                       wp_next[0] - body[0]))
                if abs(wrap_angle(_t2 - yaw)) > np.radians(self.err_deg):
                    return True
        if d_cur >= self.trig_r or speed >= self.v_max:
            return False
        target = float(np.arctan2(wp_next[1] - wp_cur[1],
                                  wp_next[0] - wp_cur[0]))
        err = wrap_angle(target - yaw)
        return abs(err) > np.radians(self.err_deg)

    def try_plan(self, body_xy, yaw, speed, omega, wp_cur, wp_next,
                 wide=False):
        """满足触发条件时返回 (True, vx, omega)，否则 (False, None, None)。"""
        _hold = os.environ.get('S10_TURN_HOLD', '0') == '1'
        _fired = self.will_fire(body_xy, yaw, speed, wp_cur, wp_next,
                                wide=wide)
        if _hold and not _fired and getattr(self, '_turn_active', False) \
                and speed < self.v_max:
            # 转向完成保持（全局，S10_TURN_HOLD=1 开启，默认关）：
            # wp3 后 104° 残余转向实测——TMppi 在距当前 wp>1.2m 即退出，
            # 剩余 25-40° 误差交还 SMppi，wp4 前慢转滑移 5.5s（r100 10.8s）。
            # 保持到 |err|<err_deg/2 才交还，对所有 wp 统一。
            target = float(np.arctan2(wp_next[1] - wp_cur[1],
                                      wp_next[0] - wp_cur[0]))
            err = wrap_angle(target - yaw)
            _hold_deg = float(os.environ.get('S10_TURN_HOLD_DEG', '30.0'))
            if abs(err) > np.radians(_hold_deg):
                _fired = True
        if not _fired:
            self._turn_active = False
            return False, None, None
        body = np.asarray(body_xy)
        self._turn_active = True
        target = float(np.arctan2(wp_next[1] - wp_cur[1],
                                  wp_next[0] - wp_cur[0]))
        _next_r = float(os.environ.get('S10_TURN_NEXT_R', '0.0'))
        if _next_r > 0.0:
            d_next = float(np.linalg.norm(body - wp_next[:2]))
            if d_next < _next_r and speed < self.v_max:
                target = float(np.arctan2(wp_next[1] - body[1],
                                          wp_next[0] - body[0]))
        err = wrap_angle(target - yaw)
        if os.environ.get('S10_TURN_DBG','0')=='1':
            print('[TURN] pos=(%.2f,%.2f) yaw=%.3f spd=%.2f dcur=%.2f target=%.3f err=%.3f om=%.3f'
                  % (float(body_xy[0]), float(body_xy[1]), float(yaw),
                     float(speed), float(np.linalg.norm(body - wp_cur[:2])),
                     float(target), float(err), float(self.k*err - self.kd*float(omega))),
                  flush=True)
        vx = self.vx_cmd
        # 终端角速度阻尼（用户指示）：转到目标角度时应收敛到停住，
        # 而不是带着角动量甩过——om = k·err − kd·ω，ω→0 时停稳
        om = float(np.clip(self.k * err - self.kd * float(omega),
                           -self.om_max, self.om_max))
        return True, vx, om
