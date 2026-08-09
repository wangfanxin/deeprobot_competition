"""cruise_vmc_noros.py — v218 方案无 ROS 独立测试（wp0→MAX_WP，原始赛道）。

结构：导航（AutoNavFollower pursuit/vlim）→ 身体层 MPPI [vx,ω]（20Hz）
     → VMC/阻抗腿层（200Hz）→ mujoco。
已知地图：地形高 mj_ray 逐轮查询；横脊预扫描 dh>0.12 → 弧长表 → 抬轮前馈。
"""
import os, sys, time
import numpy as np
import mujoco

PKG = '/home/wfx/DR_competition/deeprobot_competition/src/S10_sdk_deploy'
sys.path.insert(0, PKG)
from s10_mpc.auto_nav import AutoNavFollower
from s10_mpc.body_mppi import BodyMPPI
from s10_mpc.vmc_legs import VMCController, LEG_ATTACH, WHEEL_BODY, WHEEL_Q_IDX

DT = 0.005
MAX_SIM = float(os.environ.get('S10_TEST_MAX_SIM', '90'))
MAX_WP = int(os.environ.get('S10_AUTO_MAX_WP', '8'))
STOP_AT = int(os.environ.get('S10_STOP_AT_WP', '0'))
XML = os.environ.get('S10_XML',
    f'{PKG}/S10_description/s10_mjcf/mjcf/S10_track.xml')

STAND_TARGET = np.array([-0.05, -1.16, 2.30, 0.0,
                          0.05, -1.16, 2.30, 0.0,
                         -0.05,  1.16, -2.30, 0.0,
                          0.05,  1.16, -2.30, 0.0], dtype=np.float64)


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    d.qpos[0:3] = [0.0, -2.5, 0.2]
    iy = float(os.environ.get('S10_INIT_YAW', '0'))
    if abs(iy) > 1e-3:
        d.qpos[3:7] = [np.cos(iy/2), 0, 0, np.sin(iy/2)]
    else:
        d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:23] = np.array([-0.438, -1.16, 2.45, 0.0,
                              0.438, -1.16, 2.45, 0.0,
                             -0.438,  1.16, -2.45, 0.0,
                              0.438,  1.16, -2.45, 0.0])
    mujoco.mj_forward(m, d)

    wps = []
    for gid in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if name.startswith('track_waypoint_'):
            idx = int(name[len('track_waypoint_'):].split('_')[0])
            wps.append((idx, d.geom_xpos[gid].copy()))
    wps.sort()
    wp = np.array([w for _, w in wps])

    fol = AutoNavFollower(
        wp,
        max_speed=float(os.environ.get('S10_AUTO_VMAX', '5.0')),
        vyaw_max=float(os.environ.get('S10_AUTO_VYAW_MAX', '3.5')),
        yaw_gain=float(os.environ.get('S10_AUTO_YAW_GAIN', '2.5')),
        lookahead=float(os.environ.get('S10_AUTO_LOOKAHEAD', '1.5')))

    # 已知地图横脊预扫描（与节点 _scan_ridge_zones 同法）
    ridge_arcs = []
    try:
        pts = fol.path_pts
        hs = np.empty(len(pts))
        for k, p in enumerate(pts):
            g = np.array([-1], dtype=np.int32); dist = np.zeros(1); nrm = np.zeros(3)
            hit = mujoco.mj_ray(m, d, [p[0], p[1], 8.0], [0, 0, -1],
                                None, False, -1, g, nrm)
            hs[k] = (8.0 - hit) if g[0] >= 0 else float(p[2])
        dh = np.abs(np.diff(hs))
        skip_s = float(fol.path_wp_s[1]) - 2.0
        ridge_idx = np.where((dh > 0.12) & (fol.path_cum[:len(dh)] > skip_s))[0]
        ridge_arcs = [(float(fol.path_cum[k]), float(dh[k])) for k in ridge_idx]
        print(f'[VMC] 预扫描横脊 {len(ridge_arcs)} 处', flush=True)
    except Exception as e:
        print('[VMC] 横脊预扫描失败', e, flush=True)

    mppi = BodyMPPI(
        N=int(os.environ.get('VMC_MPPI_N', '256')),
        H=int(os.environ.get('VMC_MPPI_H', '20')),
        vx_max=float(os.environ.get('S10_AUTO_VMAX', '5.0')))
    vmc = VMCController()

    # 站起
    t = 0.0
    while t < 2.0:
        q = d.qpos[7:23].reshape(-1, 1)
        dq = d.qvel[6:22].reshape(-1, 1)
        tau = (80.0 * (STAND_TARGET.reshape(-1, 1) - q) - 2.0 * dq).flatten()
        tau[WHEEL_Q_IDX] = -0.3 * dq[WHEEL_Q_IDX].flatten()
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        t += DT

    def terrain_at(x, y):
        g = np.array([-1], dtype=np.int32); dist = np.zeros(1); nrm = np.zeros(3)
        hit = mujoco.mj_ray(m, d, [x, y, 8.0], [0, 0, -1],
                            None, True, -1, g, nrm)
        return (8.0 - hit) if hit > 0 else 0.0

    next_idx = 0
    wp_times = {}
    t_start = None
    traj = []
    prev_u = np.zeros(2)
    dbg = 0
    last_log = 0.0
    while t < MAX_SIM:
        qpos = np.asarray(d.qpos, dtype=np.float64)
        qvel = np.asarray(d.qvel, dtype=np.float64)
        body_pos = d.xpos[1]
        quat = d.xquat[1]
        yaw = float(np.arctan2(
            2.0*(quat[3]*quat[0]+quat[1]*quat[2]),
            1.0-2.0*(quat[2]**2+quat[3]**2)))
        wheel_xyz = np.asarray([d.xpos[WHEEL_BODY[i]] for i in range(4)])
        wheel_vel = np.asarray([d.cvel[WHEEL_BODY[i]][0:3] for i in range(4)])

        # 20Hz 导航 + MPPI
        if int(t * 20) % 1 == 0 and (dbg == 0 or t - last_log >= 0.05):
            pos2 = body_pos[:2]
            vx, vyaw = fol.compute_cmd(
                pos2, yaw, next_idx,
                robot_z=float(body_pos[2]), yaw_rate=float(qvel[5]))
            v_ref = fol._last_vlim
            # 路径参考轨迹（弧长采样：当前位置起 8m，步长 0.5m）
            _ref = []
            _s0 = float(fol._s_cur)
            for _ds in np.arange(0.0, 8.0, 0.5):
                _sp = _s0 + _ds
                if _sp >= fol.path_total:
                    break
                _k = int(np.searchsorted(fol.path_cum, _sp, side="right") - 1)
                _k = min(max(_k, 0), len(fol.path_pts) - 2)
                _t = ((_sp - fol.path_cum[_k])
                      / max(fol.path_cum[_k + 1] - fol.path_cum[_k], 1e-6))
                _x = fol.path_pts[_k, 0] + _t * (fol.path_pts[_k + 1, 0] - fol.path_pts[_k, 0])
                _y = fol.path_pts[_k, 1] + _t * (fol.path_pts[_k + 1, 1] - fol.path_pts[_k, 1])
                _hd = fol.path_heading[min(_k, len(fol.path_heading) - 1)]
                _ref.append([_x, _y, _hd])
            _ref = np.array(_ref) if len(_ref) else np.array([[pos2[0], pos2[1], yaw]])
            st = np.array([pos2[0], pos2[1], yaw,
                           float(d.cvel[1][3]), float(d.cvel[1][4]), float(qvel[5])])
            vx_c, om_c = mppi.plan(st, _ref, v_ref, prev_u)
            prev_u = np.array([vx_c, om_c])
            last_log = t
        else:
            vx_c, om_c = prev_u

        # 地形 + 横脊预抬
        terr = np.array([terrain_at(wheel_xyz[i, 0], wheel_xyz[i, 1])
                         for i in range(4)])
        s_cur = float(getattr(fol, '_s_cur', 0.0))
        for (sr, dhv) in ridge_arcs:
            ds = s_cur - sr
            if -0.6 <= ds < 0.0:
                terr[:] = np.maximum(terr, terr + 0.10)   # 前轮预抬（全部）
            elif 0.0 <= ds < 0.5:
                terr[2:] = np.maximum(terr[2:], terr[2:] + 0.10)  # 后轮跟抬

        # 压弯 + 坡度
        roll_tar = float(np.clip(0.20 * om_c * abs(vx_c), -0.40, 0.40))
        pitch_tar = 0.0
        try:
            fwd = d.xmat[1][0:2]
            fx, fy = fwd[0], fwd[1]
            h_a = terrain_at(body_pos[0] + fx*0.6, body_pos[1] + fy*0.6)
            h_b = terrain_at(body_pos[0] - fx*0.6, body_pos[1] - fy*0.6)
            pitch_tar = float(np.clip(np.arctan2(h_a - h_b, 1.2), -0.35, 0.35))
        except Exception:
            pass

        if os.environ.get('VMC_STAND', '0') == '1':
            cmd = dict(vx=0.0, omega=0.0, roll_tar=0.0, pitch_tar=0.0)
        else:
            cmd = dict(vx=vx_c, omega=om_c, roll_tar=roll_tar, pitch_tar=pitch_tar)
        tau = vmc.compute_tau(qpos, qvel, wheel_xyz, wheel_vel, cmd, terr, DT)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        t += DT

        # 航点推进（0.5m + v204 捷径）
        if next_idx < len(wp):
            rp = d.xpos[1][:2]
            dist = float(np.linalg.norm(rp - wp[next_idx][:2]))
            reached = dist <= 0.5
            if not reached:
                adv = float(os.environ.get('S10_WP_ADVANCE_DIST', '2.5'))
                if (next_idx >= 1 and adv > 0.0
                        and next_idx < len(fol.path_wp_s)
                        and fol._s_cur > fol.path_wp_s[next_idx] - 0.05
                        and dist <= adv):
                    reached = True
            if reached:
                if next_idx == 0 and t_start is None:
                    t_start = t
                wp_times[next_idx] = t
                print(f'[VMC-T] wp{next_idx} @ t={t:.2f}s', flush=True)
                next_idx += 1
                if STOP_AT > 0 and next_idx > STOP_AT:
                    print(f'[VMC-T] 到达 wp{STOP_AT}，结束', flush=True)
                    break
                if next_idx >= MAX_WP:
                    print('[VMC-T] 到达最大航点，结束', flush=True)
                    break

        if int(t * 200) % 100 == 0:
            roll = float(np.arctan2(
                2.0*(quat[0]*quat[1]+quat[2]*quat[3]),
                1.0-2.0*(quat[1]**2+quat[2]**2)))
            print(f'[VMC-T] t={t:.0f}s wp={next_idx} pos=({body_pos[0]:.1f},'
                  f'{body_pos[1]:.1f},{body_pos[2]:.2f}) vx={float(d.cvel[1][3]):.2f} '
                  f'roll={roll:.2f} cmd=({vx_c:.2f},{om_c:.2f}) '
                  f'vref={v_ref:.2f} tau_max={np.abs(tau).max():.1f}', flush=True)
            if abs(roll) > 0.9 or body_pos[2] < 0.12:
                print('[VMC-T] *** 侧翻/摔倒 ***', flush=True)
                break
            traj.append([t, body_pos[0], body_pos[1], float(d.cvel[1][3])])
    print('=== VMC 全航点结果 ===')
    print(f'完成: {next_idx >= MAX_WP}，最终 wp={next_idx}/{MAX_WP}')
    if t_start is not None:
        print(f'wp0→wp{min(next_idx-1, MAX_WP-1)} 用时 {wp_times.get(max(next_idx-1,0),0)-t_start:.1f}s')
    if os.environ.get('VMC_TRAJ'):
        np.save(os.environ['VMC_TRAJ'], np.array(traj))


if __name__ == '__main__':
    main()
