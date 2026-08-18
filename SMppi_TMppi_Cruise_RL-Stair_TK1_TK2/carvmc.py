"""carvmc.py -- CarVMC 巡航执行模块。

半蹲站姿由 S10_CAR_SQUAT=1 控制（hipy∓1.10 / knee±1.90）。
所有增益从环境变量读取，与启动 sh 保持一致。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PKG = os.path.join(REPO, 'src', 'S10_sdk_deploy')
for _p in (HERE, PKG, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from s10_mpc.vmc_legs import CarVMC  # noqa: E402


class CarVMCExecutor:
    def __init__(self):
        os.environ.setdefault('S10_CAR_SQUAT', '1')
        self.vmc = CarVMC()

    @property
    def pose_target(self):
        return self.vmc.pose_target

    @pose_target.setter
    def pose_target(self, value):
        self.vmc.pose_target = value

    def reset_state(self, vx, omega, roll, pitch):
        self.vmc.reset_state(vx=vx, omega=omega, roll=roll, pitch=pitch)

    def body_state(self, qpos, qvel):
        return self.vmc._body_state(qpos, qvel)

    def compute_tau(self, qpos, qvel, wheel_xyz, wheel_vel, cmd, terr, dt):
        return self.vmc.compute_tau(qpos, qvel, wheel_xyz, wheel_vel, cmd,
                                    terr, dt)
