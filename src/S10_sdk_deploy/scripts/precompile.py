# -*- coding: utf-8 -*-
"""最小预热脚本：初始化 MPC + plan_once 一次，打印 JIT 耗时。"""
import os, sys, time
import numpy as np
PKG = '/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy'
sys.path.insert(0, PKG)
sys.path.insert(0, '/home/wfx/DR_competition/dial-mpc')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', os.path.expanduser('~/.cache/s10_dial_mpc'))

import mujoco
from s10_mpc.mpc_controller import MPCController
XML = f'{PKG}/S10_description/s10_mjcf/mjcf/S10_track.xml'
MPC_YAML = '/home/wfx/DR_competition/0810new/deeprobot_competition/doc/s10_mpc_deploy.yaml'
JOINT_INIT = np.array([-0.438, -1.16, 2.45, 0.0, 0.438, -1.16, 2.45, 0.0,
                       -0.438, 1.16, -2.45, 0.0, 0.438, 1.16, -2.45, 0.0])
t0 = time.time()
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
d.qpos[7:23] = JOINT_INIT
d.qpos[0:3] = [0.0, -2.5, 0.2]
d.qpos[3:7] = [1, 0, 0, 0]
mujoco.mj_forward(m, d)
mpc = MPCController(MPC_YAML)
mpc.init_state(np.asarray(d.qpos[:23], dtype=np.float32),
               np.asarray(d.qvel[:22], dtype=np.float32))
mpc.set_cmd(0.0, 0.0, 0.0)
print(f'[PRE] MPC ready {time.time()-t0:.1f}s', flush=True)
t1 = time.time()
act = mpc.plan_once(np.asarray(d.qpos[:23], dtype=np.float32),
                    np.asarray(d.qvel[:22], dtype=np.float32), 3.0)
print(f'[PRE] first plan_once JIT {time.time()-t1:.1f}s', flush=True)
t2 = time.time()
act = mpc.plan_once(np.asarray(d.qpos[:23], dtype=np.float32),
                    np.asarray(d.qvel[:22], dtype=np.float32), 3.05)
print(f'[PRE] second plan_once {time.time()-t2:.1f}s', flush=True)
