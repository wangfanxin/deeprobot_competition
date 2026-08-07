"""cruise 模式轻量测试：wp0→wp7，记录航点到达时间与崩溃。
用于 10 个 cruise 版本的速度对比（真实节点、无头、低噪声日志）。
"""
import os
import sys
import time
from pathlib import Path

os.environ["S10_USE_VIEWER"] = "0"
os.environ["S10_MPC_ENABLE"] = "1"
os.environ["S10_MODE"] = "auto_nav"
os.environ["S10_LIDAR_BACKEND"] = "cpu"
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.expanduser("~/.cache/s10_dial_mpc"))

SIM_DIR = str(Path(__file__).resolve().parents[1] /
              "interface/robot/simulation")
sys.path.insert(0, SIM_DIR)

import numpy as np                                  # noqa: E402
import mujoco                                        # noqa: E402
import rclpy                                         # noqa: E402
from mujoco_simulation_ros2 import (                 # noqa: E402
    MuJoCoSimulationNode, MPC_PLAN_INTERVAL)

DT = 0.005
MAX_SIM = float(os.environ.get("S10_TEST_MAX_SIM", "300.0"))
MAX_WP = int(os.environ.get("S10_AUTO_MAX_WP", "5"))


def main():
    rclpy.init()
    node = MuJoCoSimulationNode()
    t0 = time.time()
    while node.mpc is None and time.time() - t0 < 180:
        time.sleep(0.1)
    assert node.mpc is not None, "MPC 未构建"
    print(f"[CRUISE-T] MPC ready ({time.time()-t0:.1f}s) MAX_WP={MAX_WP}",
          flush=True)

    last_act = None
    wp_times = {}
    t_start = None
    t_cmd0 = None
    last_progress_t = None
    last_progress_idx = 0
    crashed = None
    while node.timestamp < MAX_SIM:
        node._apply_joint_torque()
        mujoco.mj_step(node.model, node.data)
        node.timestamp += DT
        step = int(node.timestamp / DT)

        if node.mpc_mode and node.mpc is not None:
            q = np.asarray(node.data.qpos[:23], dtype=np.float32)
            qd = np.asarray(node.data.qvel[:22], dtype=np.float32)
            if node.auto_nav_active and step % 10 == 0:
                node._update_auto_nav()
            plan_interval = int(os.environ.get(
                "S10_MPC_PLAN_INTERVAL_AUTO", "10")
                if node.auto_nav_active else str(MPC_PLAN_INTERVAL))
            if last_act is None or step % plan_interval == 0:
                last_act = node.mpc.plan_once(q, qd, node.timestamp)
                node.last_act = last_act
                if t_cmd0 is None:
                    t_cmd0 = node.timestamp
            if node.auto_nav_active:
                la = np.asarray(last_act).copy()
                la[:12] += np.asarray(node._leg_assist, np.float32)
                la[:12] = np.clip(
                    la[:12],
                    -float(os.environ.get("S10_AUTO_LEG_CLIP", "0.30")),
                    float(os.environ.get("S10_AUTO_LEG_CLIP", "0.30")))
                last_act = la
                node.last_act = la
            node.mpc.latest_tau = node.mpc.compute_tau(last_act, q, qd)
            if step % 25 == 0:
                try:
                    node._publish_robot_state(step)
                    if getattr(node, "lidar", None) is not None:
                        node._publish_lidar_data()
                        node._publish_lidar_tf()
                except Exception as _e:
                    print("[CRUISE-T] pub-err", _e, flush=True)
        else:
            if node.mpc is not None and not node._mpc_warmup_done:
                node._warmup_mpc()
            if (node.auto_nav and not node.auto_nav_active
                    and node._mpc_warmup_done):
                if node.auto_stand_t0 is None:
                    node._start_auto_nav()
                else:
                    node._maybe_enter_auto_mpc()
            if step % 5 == 0:
                try:
                    node._publish_robot_state(step)
                    if getattr(node, "lidar", None) is not None:
                        node._publish_lidar_data()
                        node._publish_lidar_tf()
                except Exception:
                    pass

        # 航点计时：wp0 进入（计时开始）→ 每推进一个航点记录
        idx = node.track_next_index
        if idx == 0 and t_start is None:
            t_start = node.timestamp
            print(f"[CRUISE-T] wp0 @ t={node.timestamp:.1f}s (计时开始)",
                  flush=True)
        if idx not in wp_times and t_start is not None and idx > 0:
            wp_times[idx] = node.timestamp - t_start
        cur_idx = node.track_next_index
        if cur_idx != last_progress_idx:
            last_progress_t = node.timestamp
            last_progress_idx = cur_idx
            print(f"[CRUISE-T] wp{idx} @ {wp_times[idx]:.1f}s "
                  f"(t={node.timestamp:.1f}s)", flush=True)

        # 崩溃检测
        qq = node.data.xquat[node.track_body_id]
        w, x, y, z = qq
        roll = float(np.arctan2(2.0 * (w * x + y * z),
                                1.0 - 2.0 * (x * x + y * y)))
        xyz = node.data.xpos[node.track_body_id]
        if abs(roll) > 0.7 or xyz[2] < 0.12:
            crashed = f"roll={roll:.2f} z={xyz[2]:.3f}"
            print(f"[CRUISE-T] *** 崩溃 *** {crashed} @ t={node.timestamp:.1f}s "
                  f"wp={idx}", flush=True)
            break

        if last_progress_t is not None and node.timestamp - last_progress_t > 15.0:
            print(f"[CRUISE-T] 卡住 15s（wp={idx} 无推进），提前结束", flush=True)
            break
        if node.track_complete:
            break
        if idx >= MAX_WP:
            print(f"[CRUISE-T] 到达目标航点 wp{MAX_WP}，结束 "
                  f"(t={node.timestamp:.1f}s)", flush=True)
            break

    if crashed is None and idx < MAX_WP and not node.track_complete:
        print(f"[CRUISE-T] 超时/卡死：wp={idx} t={node.timestamp:.1f}s",
              flush=True)
    wps = node.track_waypoint_positions
    route_len = float(np.sum(np.linalg.norm(
        np.diff(wps[:min(MAX_WP, len(wps)) + 1], axis=0), axis=1)))
    t_end = node.timestamp
    if t_cmd0 is not None:
        run_t = t_end - t_cmd0
    elif t_start is not None:
        run_t = t_end - t_start
    else:
        run_t = t_end
    avg_speed = route_len / run_t if run_t > 0 else 0.0
    print(f"[CRUISE-T] RESULT version={os.environ.get('S10_VER','?')} "
          f"wp_times={ {k: round(v,1) for k,v in wp_times.items()} } "
          f"crashed={crashed} final_wp={idx} route_len={route_len:.1f}m "
          f"run_t={run_t:.1f}s avg_speed={avg_speed:.2f}m/s",
          flush=True)


if __name__ == "__main__":
    main()
