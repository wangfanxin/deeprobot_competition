"""cruise 无 ROS 独立测试：wp0→MAX_WP，直接 mujoco + MPC + 导航。
不初始化 rclpy，可与 stair 会话并行（无 DDS 冲突）。"""
import os, sys, time
import numpy as np
import mujoco

PKG = '/home/wfx/DR_competition/deeprobot_competition/src/S10_sdk_deploy'
sys.path.insert(0, PKG)
sys.path.insert(0, '/home/wfx/DR_competition/dial-mpc')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',
                      os.path.expanduser('~/.cache/s10_dial_mpc'))

from s10_mpc.mpc_controller import MPCController
from s10_mpc.auto_nav import AutoNavFollower

DT = 0.005
MAX_SIM = float(os.environ.get('S10_TEST_MAX_SIM', '120'))
MAX_WP = int(os.environ.get('S10_AUTO_MAX_WP', '5'))
XML = f'{PKG}/S10_description/s10_mjcf/mjcf/S10_track.xml'
MPC_YAML = os.environ.get('S10_MPC_YAML',
    '/home/wfx/DR_competition/deeprobot_competition/doc/s10_mpc_deploy.yaml')

JOINT_INIT = np.array([-0.438, -1.16, 2.45, 0.0,
                        0.438, -1.16, 2.45, 0.0,
                       -0.438,  1.16, -2.45, 0.0,
                        0.438,  1.16, -2.45, 0.0], dtype=np.float64)
STAND_TARGET = np.array([-0.05, -1.16, 2.30, 0.0,
                          0.05, -1.16, 2.30, 0.0,
                         -0.05,  1.16, -2.30, 0.0,
                          0.05,  1.16, -2.30, 0.0], dtype=np.float64)


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    d.qpos[7:23] = JOINT_INIT
    d.qpos[0:3] = [0.0, -2.5, 0.2]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)

    waypoints = []
    for gid in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if name.startswith('track_waypoint_'):
            idx = int(name[len('track_waypoint_'):].split('_')[0])
            waypoints.append((idx, d.geom_xpos[gid].copy()))
    waypoints.sort()
    wps = np.array([w for _, w in waypoints], dtype=np.float64)
    print(f'[NOROS] waypoints: {len(wps)}', flush=True)
    track_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    assert track_body >= 0

    t0 = time.time()
    mpc = MPCController(MPC_YAML)
    mpc.init_state(np.asarray(d.qpos[:23], dtype=np.float32),
                   np.asarray(d.qvel[:22], dtype=np.float32))
    mpc.set_cmd(0.0, 0.0, 0.0)
    print(f'[NOROS] MPC ready ({time.time()-t0:.1f}s)', flush=True)

    fol = AutoNavFollower(
        wps,
        max_speed=float(os.environ.get('S10_AUTO_VMAX', '4.5')),
        vyaw_max=float(os.environ.get('S10_AUTO_VYAW_MAX', '1.5')),
        yaw_gain=float(os.environ.get('S10_AUTO_YAW_GAIN', '3.0')),
        lookahead=float(os.environ.get('S10_AUTO_LOOKAHEAD', '1.5')),
    )
    mpc.set_yaw_gain_lo(float(os.environ.get('S10_AUTO_YAW_FF_GAIN', '20.0')))

    next_idx = 0
    t = 0.0
    last_act = None
    auto_active = False
    wp_times = {}
    t_start = None
    t_cmd0 = None
    crashed = None
    last_progress_idx = 0
    last_progress_t = None
    plan_interval = int(os.environ.get('S10_MPC_PLAN_INTERVAL_AUTO', '10'))
    dbg_cnt = 0
    traj_file = os.environ.get('S10_TRAJ_FILE')
    if traj_file:
        _tf = open(traj_file, 'w')
        _tf.write('t,x,y,yaw,next_idx,err,d_wp,vx,vyaw,cte,s_cur,tgt_x,tgt_y\n')

    while t < MAX_SIM:
        step = int(t / DT)
        if not auto_active:
            # 站起 PD（3s）
            q = d.qpos[7:23].reshape(-1, 1)
            dq = d.qvel[6:22].reshape(-1, 1)
            tau = (80.0 * (STAND_TARGET.reshape(-1, 1) - q) - 2.0 * dq).flatten()
            tau[3::4] = -0.3 * dq[3::4].flatten()
            d.ctrl[:] = tau
            if t >= 3.0:
                qq = np.asarray(d.qpos[:23], dtype=np.float32)
                qqd = np.asarray(d.qvel[:22], dtype=np.float32)
                last_act = mpc.plan_once(qq, qqd, t)
                t_cmd0 = t
                auto_active = True
                print(f'[NOROS] auto_nav 启动 t={t:.1f}s', flush=True)
        else:
            qq = np.asarray(d.qpos[:23], dtype=np.float32)
            qqd = np.asarray(d.qvel[:22], dtype=np.float32)
            if step % 10 == 0:
                pos = d.xpos[track_body][:2]
                quat = d.xquat[track_body]
                yaw = float(np.arctan2(2.0*(quat[3]*quat[0]+quat[1]*quat[2]),
                                       1.0-2.0*(quat[2]**2+quat[3]**2)))
                fol.update_mode(pos, next_idx, yaw=yaw, local_map=None)
                mpc.set_mode(fol.mode)
                vx, vyaw = fol.compute_cmd(
                    pos, yaw, next_idx,
                    robot_z=float(d.xpos[track_body][2]), yaw_rate=0.0)
                mpc.set_cmd(vx, 0.0, vyaw)
                dbg_cnt += 1
                if traj_file and dbg_cnt % 10 == 0:
                    _tf.write(f'{t:.2f},{pos[0]:.3f},{pos[1]:.3f},'
                              f'{yaw:.3f},{next_idx},{fol._last_err:.3f},'
                              f'{fol._last_dwp:.3f},{vx:.3f},{vyaw:.3f},'
                              f'{getattr(fol,"_last_cte",0.0):.3f},'
                              f'{fol._s_cur:.3f},'
                              f'{getattr(fol,"_last_tgt",[0,0])[0]:.3f},'
                              f'{getattr(fol,"_last_tgt",[0,0])[1]:.3f}\n')
                    _tf.flush()
                if os.environ.get('S10_AUTO_DEBUG') == '1' and dbg_cnt % 40 == 1:
                    print(f'[NAVDBG] next={next_idx} vx={vx:.2f} '
                          f'vyaw={vyaw:.2f} err={fol._last_err:.2f} '
                          f'd_wp={fol._last_dwp:.2f} mode={fol.mode}',
                          flush=True)
            if step % plan_interval == 0:
                last_act = mpc.plan_once(qq, qqd, t)
                if t_cmd0 is None:
                    t_cmd0 = t
            mpc.latest_tau = mpc.compute_tau(last_act, qq, qqd)
            d.ctrl[:] = np.asarray(mpc.latest_tau, dtype=np.float64)

        mujoco.mj_step(m, d)
        t += DT

        # 航点推进（0.5m 半径，与节点一致）
        if next_idx < len(wps):
            rp = d.xpos[track_body][:2]
            dist = float(np.linalg.norm(rp - wps[next_idx][:2]))
            if dist <= 0.5:
                if next_idx == 0 and t_start is None:
                    t_start = t
                    print(f'[NOROS] wp0 @ t={t:.1f}s (计时开始)', flush=True)
                else:
                    wp_times[next_idx] = t - (t_start or 0.0)
                    print(f'[NOROS] wp{next_idx} @ {wp_times[next_idx]:.1f}s '
                          f'(t={t:.1f}s)', flush=True)
                next_idx += 1
                if next_idx >= MAX_WP:
                    print(f'[NOROS] 到达 wp{MAX_WP}，结束 (t={t:.1f}s)', flush=True)
                    break

        # 崩溃检测
        quat = d.xquat[track_body]
        w, x, y, z = quat
        roll = float(np.arctan2(2.0*(w*x + y*z), 1.0-2.0*(x*x + y*y)))
        xyz = d.xpos[track_body]
        if abs(roll) > 0.7 or xyz[2] < 0.12:
            crashed = f'roll={roll:.2f} z={xyz[2]:.3f}'
            print(f'[NOROS] *** 崩溃 *** {crashed} @ t={t:.1f}s wp={next_idx}',
                  flush=True)
            break

        # wp 计时 60s 上限（用户要求：到 wp5 最多 60s，不含预热）
        if t_start is not None and t - t_start > 60.0:
            print(f'[NOROS] wp 计时超 60s（wp={next_idx}），强制结束',
                  flush=True)
            break
        # 卡住检测（15s 无推进）
        if next_idx != last_progress_idx:
            last_progress_idx = next_idx
            last_progress_t = t
        if last_progress_t is not None and t - last_progress_t > 15.0:
            print(f'[NOROS] 卡住 15s（wp={next_idx} 无推进），提前结束',
                  flush=True)
            break

    if crashed is None and next_idx < MAX_WP:
        print(f'[NOROS] 超时/卡死：wp={next_idx} t={t:.1f}s', flush=True)
    if traj_file:
        _tf.close()
    route_len = float(np.sum(np.linalg.norm(
        np.diff(wps[:min(MAX_WP, len(wps)) + 1], axis=0), axis=1)))
    run_t = (t - t_cmd0) if t_cmd0 else (t - (t_start or 0.0))
    avg = route_len / run_t if run_t > 0 else 0.0
    print(f'[NOROS] RESULT version={os.environ.get("S10_VER","?")} '
          f'wp_times={ {k: round(v,1) for k,v in wp_times.items()} } '
          f'crashed={crashed} final_wp={next_idx} route_len={route_len:.1f}m '
          f'run_t={run_t:.1f}s avg_speed={avg:.2f}m/s', flush=True)


if __name__ == '__main__':
    main()