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
from s10_mpc.vmc_legs import (VMCController, LEG_ATTACH, WHEEL_BODY,
    WHEEL_Q_IDX, LidarTerrain)

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
        # v218m: 横脊限速（同节点 _scan_ridge_zones）——防高速冲棱
        _rv = float(os.environ.get('S10_RIDGE_VX', '1.5'))
        for _k in ridge_idx:
            _lo = max(0, _k - int(2.0 / fol.path_res))
            _hi = min(len(fol.path_vlim), _k + int(1.2 / fol.path_res))
            fol.path_vlim[_lo:_hi] = np.minimum(
                fol.path_vlim[_lo:_hi], _rv)
        fol.ridge_s = [float(fol.path_cum[k]) for k in ridge_idx]
        print(f'[VMC] 预扫描横脊 {len(ridge_arcs)} 处', flush=True)
    except Exception as e:
        print('[VMC] 横脊预扫描失败', e, flush=True)

    mppi = BodyMPPI(
        N=int(os.environ.get('VMC_MPPI_N', '256')),
        H=int(os.environ.get('VMC_MPPI_H', '20')),
        vx_max=float(os.environ.get('S10_AUTO_VMAX', '5.0')))
    if os.environ.get('S10_VMC_MODE', 'wbc') == 'pd':
        from s10_mpc.vmc_legs import LegPDDrive
        vmc = LegPDDrive()
        print('[VMC] LegPDDrive 模式（腿锁蹲姿+轮驱动）', flush=True)
    else:
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

    # v219f: 地形感知来源。ray=上帝视角实时 raycast（调试，零噪声）；
    # lidar=lidar_site 扇形射线局部栅格（传感器视角，10Hz 更新，部署同款）
    if os.environ.get('S10_VMC_TERRAIN', 'ray') == 'lidar':
        lterr = LidarTerrain(m, d)
        _lupd = -1.0
        def terrain_at(x, y):
            nonlocal _lupd
            if t - _lupd >= 0.1:
                lterr.update()
                _lupd = t
            return lterr.height(x, y)
    else:
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
    # v220a: 单步跨越状态机（0=off, 1=前轮抬, 2=后轮抬）
    _step_state = 0
    _step_t0 = 0.0
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
            if os.environ.get('S10_VMC_USE_NAV', '0') == '1':
                vx_c, om_c = vx, vyaw   # 直接导航指令（无 MPPI 随机性）
            else:
                vx_c, om_c = mppi.plan(st, _ref, v_ref, prev_u)
            # v218p: omega 上限匹配 VMC yaw 能力（防指令远超执行导致振荡）
            _omcap = float(os.environ.get("S10_VMC_OM_CAP", "0.5"))
            om_c = float(np.clip(om_c, -_omcap, _omcap))
            prev_u = np.array([vx_c, om_c])
            last_log = t
        else:
            vx_c, om_c = prev_u

        # 地形 + 横脊预抬
        terr = np.array([terrain_at(wheel_xyz[i, 0], wheel_xyz[i, 1])
                         for i in range(4)])
        s_cur = float(getattr(fol, '_s_cur', 0.0))

        # v220a: 单步跨越状态机——横脊 0.125m > 轮半径 0.081，轮滚不上，
        # 前轮抬膝跨脊（0.45s）后轮跟抬（0.45s），迈步时关 RIDGE_LIFT 防叠加
        step_lift = np.zeros(4)
        if float(os.environ.get('S10_VMC_STEP_OVER', '0')) > 0:
            _near_ridge = any(
                0.2 <= sr - s_cur <= 0.9 for (sr, dhv) in ridge_arcs)
            _step_dur = float(os.environ.get('S10_VMC_STEP_DUR', '0.8'))
            if _near_ridge and _step_state == 0:
                _step_state = 1
                _step_t0 = t
            elif _step_state == 1 and t - _step_t0 > _step_dur:
                _step_state = 2
                _step_t0 = t
            elif _step_state == 2 and t - _step_t0 > _step_dur:
                _step_state = 0
            if _step_state == 1:
                step_lift[:] = [1.0, 1.0, 0.0, 0.0]
            elif _step_state == 2:
                step_lift[:] = [0.0, 0.0, 1.0, 1.0]

        _lift = float(os.environ.get('S10_VMC_RIDGE_LIFT', '0.12'))
        _lift_act = 0.0
        for (sr, dhv) in ridge_arcs:
            ds = s_cur - sr
            # v219u: 短促 bump——前轮棱前 0.8m 抬、过棱即放。
            if -0.8 <= ds < 0.08:
                f = float(np.clip((0.8 - abs(ds + 0.30)) / 0.8, 0.0, 1.0))
                terr[:] = np.maximum(terr, terr + _lift * f)
                _lift_act = max(_lift_act, f)
            elif _step_state == 0 and 0.08 <= ds < 0.55:
                # v220i: 迈步时后轮不抬（前轮由 step_lift 抬 + 前轮 bump 保留，
                # 组合抬升够脊顶；后轮必须着地推车身）
                f = float(np.clip((0.55 - ds) / 0.47, 0.0, 1.0))
                terr[2:] = np.maximum(terr[2:], terr[2:] + _lift * f)
                _lift_act = max(_lift_act, f)
        # v219i: 后轮跟抬——前轮已上棱即抬后轮。加 s_cur 接近横脊的条件：
        # 弯道/上坡时前轮天然高于后轮（>0.05），无条件触发会误抬后轮
        # 导致失去驱动力（v219a 实测 wp3→4 卡死根因）。
        _wz = np.asarray([d.xpos[WHEEL_BODY[i], 2] for i in range(4)])
        _lf = max(_wz[0], _wz[1]); _lr = min(_wz[2], _wz[3])
        _in_ridge = any(-0.8 <= s_cur - sr <= 1.2 for (sr, dhv) in ridge_arcs)
        # v220b: 迈步期间禁用后轮跟抬——前轮被迈步抬起时 wz 天然高于后轮，
        # 会误把后轮也抬离地（全轮无推力死锁）
        if _step_state == 0 and _in_ridge and _lf > _lr + 0.05:
            terr[2:] = np.maximum(terr[2:], terr[2:] + _lift * 0.8)


        # 压弯 + 坡度
        roll_tar = float(np.clip(0.20 * om_c * abs(vx_c), -0.40, 0.40))
        pitch_tar = 0.0
        try:
            fwd = d.xmat[1][0:2]
            fx, fy = fwd[0], fwd[1]
            h_a = terrain_at(body_pos[0] + fx*0.6, body_pos[1] + fy*0.6)
            h_b = terrain_at(body_pos[0] - fx*0.6, body_pos[1] - fy*0.6)
            pitch_tar = float(np.clip(np.arctan2(h_a - h_b, 1.2), -0.35, 0.35))
            # v218o: 横脊抬前轮时顺坡仰头（防 pitch 控制器对抗抬升导致腿饱和）
            if _lift_act > 0.05:
                pitch_tar = max(pitch_tar, 0.25 * _lift_act)
        except Exception:
            pass

        if os.environ.get('VMC_STAND', '0') == '1':
            cmd = dict(vx=0.0, omega=0.0, roll_tar=0.0, pitch_tar=0.0)
        else:
            # v218q: 前轮地形前瞻 hop——前方 0.3m 高差>0.08 给前轮向上冲量（不伸腿）
            hop = np.zeros(4, dtype=np.float64)
            _fwd2 = d.xmat[1][0:2]
            _fx, _fy = _fwd2[0], _fwd2[1]
            _fn = float(np.hypot(_fx, _fy)) + 1e-9
            _fx, _fy = _fx / _fn, _fy / _fn
            for _wi in (0, 1):
                _w0 = wheel_xyz[_wi]
                # v220l: hop 检测用原始地形（terr 已被 RIDGE_LIFT bump 抬到
                # 0.60，差值<0.08 导致 hop 永不触发）
                _h0 = terrain_at(_w0[0], _w0[1])
                # v219z: 前瞻 0.3->0.8m
                _ha = terrain_at(_w0[0] + _fx * 0.80, _w0[1] + _fy * 0.80)
                if _ha - _h0 > 0.08:
                    hop[_wi] = float(os.environ.get('S10_VMC_HOP_F', '180.0'))
            cmd = dict(vx=vx_c, omega=om_c, roll_tar=roll_tar,
                      pitch_tar=pitch_tar,
                      yaw_scale=1.0 - _lift_act, hop=hop,
                      step_lift=step_lift)
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
                  f'{body_pos[1]:.1f},{body_pos[2]:.2f}) yaw={yaw:.2f} vx_w={float(d.cvel[1][3]):.2f} '
                  f'roll={roll:.2f} cmd=({vx_c:.2f},{om_c:.2f}) '
                  f'vref={v_ref:.2f} tau_max={np.abs(tau).max():.1f} '
                  f'wz={np.round([d.xpos[WHEEL_BODY[i],2] for i in range(4)],2)} '
                  f'tauH={np.round(tau[[0,4,8,12]],0)} tauY={np.round(tau[[1,5,9,13]],0)} tauK={np.round(tau[[2,6,10,14]],0)} '
                  f'om={float(qvel[5]):.2f} tauW={np.round(tau[[3,7,11,15]],1)} '
                  f'stp={_step_state} sl={np.round(step_lift,1)}', flush=True)
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
