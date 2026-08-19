"""smppi.py -- SMppi 直线保持模块（BodyMPPI 封装）。

只保留 BodyMPPI 采样规划；避障 costmap 已完全删除。
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

from s10_mpc.body_mppi import BodyMPPI  # noqa: E402


class SMppi:
    def __init__(self, vx_max):
        self.mppi = BodyMPPI(
            N=int(os.environ.get('VMC_MPPI_N', '512')),
            H=int(os.environ.get('VMC_MPPI_H', '20')),
            vx_max=float(vx_max))

    def plan(self, state, ref_path, v_ref, prev_u, guide_om, wp_dx=None):
        """state=[x,y,yaw,body_vx,body_vy,omega], ref_path=(R,3).

        wp_dx: 当前到下一 wp 的水平距离，终点代价激活距离（None 关闭）。
        """
        return self.mppi.plan(state, ref_path, float(v_ref), prev_u,
                              guide_om=float(guide_om), wp_dx=wp_dx)

    def sync_applied(self, u):
        """把主循环实际施加（经 roll 门控/omcap 钳制后）的指令同步给 MPPI，
        使输出 slew 基准 = 真实指令——门控释放时不会 0.4->4.0 阶跃激振。"""
        self.mppi._out_prev = np.asarray(u, dtype=np.float64)

    @property
    def plan_stats(self):
        return (self.mppi.plan_ms_ema, self.mppi.plan_ms_max,
                self.mppi.n_plan)
