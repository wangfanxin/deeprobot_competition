#!/usr/bin/env python3
"""dial-mpc 编译预热脚本（测量/验证 plan_once 编译耗时）。

用法（在安装了 .venv 的 WSL 环境）：
    ./.venv/bin/python deeprobot_competition/src/S10_sdk_deploy/scripts/precompile_mpc.py

说明：
- 实测本机 JAX 0.4.38 + GPU 上持久化编译缓存跨进程不生效（写盘但加载无效），
  因此"离线编译、启动即用"的实现方式改为：仿真启动时主线程自动预热
  （见 mujoco_simulation_ros2.py 的 _warmup_mpc，首次约 24s，之后按 c 即时）。
- 本脚本用于：验证编译链路 / 测量各变体耗时 / 作为手动预热参考。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]   # DR_competition/
PKG = ROOT / "deeprobot_competition" / "src" / "S10_sdk_deploy"

os.environ.setdefault(
    "S10_MPC_YAML",
    str(PKG.parent.parent / "doc" / "s10_mpc_deploy.yaml"))
sys.path.insert(0, str(ROOT / "dial-mpc"))
sys.path.insert(0, str(PKG))

from s10_mpc.mpc_controller import MPCController, _JAX_CACHE_DIR  # noqa: E402


def main():
    print(f"[precompile] cache dir: {_JAX_CACHE_DIR}")
    print("[precompile] 构建 MPCController ...", flush=True)
    t0 = time.time()
    ctrl = MPCController(os.environ["S10_MPC_YAML"])
    print(f"[precompile] 构建完成 {time.time()-t0:.1f}s", flush=True)

    # 站姿 qpos（与仿真初始一致）：base pos + quat + 16 关节
    qpos = np.zeros(23, dtype=np.float32)
    qpos[0:3] = [0.0, 0.0, 0.16]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos[7:23] = np.array([-0.10, -1.16, 2.45, 0.0,
                            0.10, -1.16, 2.45, 0.0,
                           -0.10,  1.16, -2.45, 0.0,
                            0.10,  1.16, -2.45, 0.0], dtype=np.float32)
    qvel = np.zeros(22, dtype=np.float32)

    ctrl.init_state(qpos, qvel)
    ctrl.set_cmd(0.0, 0.0, 0.0)
    print("[precompile] 首次 plan_once（JIT 编译，约 30-60s）...", flush=True)
    t1 = time.time()
    act = ctrl.plan_once(qpos, qvel, 0.0)
    dt1 = time.time() - t1
    print(f"[precompile] 首次 plan_once 耗时 {dt1:.1f}s", flush=True)

    # 再跑一次，验证磁盘缓存命中
    t2 = time.time()
    ctrl.plan_once(qpos, qvel, 0.02)
    dt2 = time.time() - t2
    print(f"[precompile] 二次 plan_once 耗时 {dt2*1000:.0f}ms"
          f"（缓存{'命中' if dt2 < dt1/5 else '未命中'}）", flush=True)
    print(f"[precompile] action 样本: {np.round(np.asarray(act), 3)}")
    print("[precompile] 完成。启动仿真后按 c 进入遥控将直接加载缓存。", flush=True)


if __name__ == "__main__":
    main()
