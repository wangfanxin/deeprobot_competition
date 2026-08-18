"""smppi.py -- SMppi 直线保持模块（BodyMPPI 封装）。

只保留 BodyMPPI 采样规划；避障 costmap 已完全删除。
"""
import os
import sys

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

    def plan(self, state, ref_path, v_ref, prev_u, guide_om):
        """state=[x,y,yaw,body_vx,body_vy,omega], ref_path=(R,3)."""
        return self.mppi.plan(state, ref_path, float(v_ref), prev_u,
                              guide_om=float(guide_om))
