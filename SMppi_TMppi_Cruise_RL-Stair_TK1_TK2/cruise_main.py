"""cruise_main.py -- SMppi/TMppi Cruise + RL-Stair + TK1/TK2 主循环。"""
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PKG = os.path.join(REPO, 'src', 'S10_sdk_deploy')
for _p in (HERE, PKG, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nav_waypoint import extract_waypoints, WaypointLineNav
from stair_mode import StairGate
from perception_lidar import LidarPerception
from smppi import SMppi
from tmppi import TMppi
from carvmc import CarVMCExecutor
from rlstair_ctrl import RLStairCtrl
from s10_mpc.vmc_legs import WHEEL_BODY

DT = 0.005
XML = os.environ.get('S10_XML',
                     os.path.join(PKG, 'S10_description/s10_mjcf/mjcf/S10_track.xml'))


def quat_yaw(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)))


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    m.opt.timestep = DT
    d = mujoco.MjData(m)

    d.qpos[0:3] = [0.0, -2.5, float(os.environ.get('S10_INIT_Z', '0.2'))]
    iy = float(os.environ.get('S10_INIT_YAW', '1.5708'))
    d.qpos[3:7] = [np.cos(iy / 2), 0, 0, np.sin(iy / 2)]
    d.qpos[7:23] = np.array([-0.438, -1.16, 2.45, 0.0,
                              0.438, -1.16, 2.45, 0.0,
                             -0.438,  1.16, -2.45, 0.0,
                              0.438,  1.16, -2.45, 0.0])
    mujoco.mj_forward(m, d)

    wp = extract_waypoints(m, d)
    nav = WaypointLineNav(wp)
    stair = StairGate(wp)
    perc = LidarPerception(m, d)
    smppi = SMppi(float(os.environ.get('S10_AUTO_VMAX', '3.0')))
    tmppi = TMppi()
    carvmc = CarVMCExecutor()
    rl = RLStairCtrl(m)

    t = 0.0
    while t < 0.5:
        qpos = np.asarray(d.qpos, dtype=np.float64)
        qvel = np.asarray(d.qvel, dtype=np.float64)
        wheel_xyz = np.asarray([d.xpos[WHEEL_BODY[i]] for i in range(4)])
        wheel_vel = np.asarray([d.cvel[WHEEL_BODY[i]][0:3] for i in range(4)])
        terr = np.asarray([perc.height(float(wheel_xyz[i, 0]),
                                       float(wheel_xyz[i, 1]), t,
                                       float(d.xpos[1][2]),
                                       float(wheel_xyz[i, 2]) - 0.081)
                           for i in range(4)])
        tau = carvmc.compute_tau(qpos, qvel, wheel_xyz, wheel_vel,
                                 dict(vx=0.0, omega=0.0, roll_tar=0.0,
                                      pitch_tar=0.0), terr, DT)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        t += DT

    next_idx = 0
    prev_u = np.zeros(2)
    traj = []
    wp_times = {}
    t_start = None
    last_adv_t = t
    _rl_was_stair = False
    _tk2 = False
    _tk2_was_stair = False
    _post_stair_xy = None
    _rl_diag_done = False
    _max_tau_leg = 0.0
    _max_tau_wh = 0.0
    _over_run = 0.0
    _over_worst = 0.0
    _over_total = 0.0
    _nav_period = max(1, int(round(200.0 / float(
        os.environ.get('S10_NAV_HZ', '20')))))
    _ctrl_cnt = 0
    _correction = ''
    _planner = ''
    _prev_line_head = None

    while t < float(os.environ.get('S10_TEST_MAX_SIM', '600')):
        qpos = np.asarray(d.qpos, dtype=np.float64)
        qvel = np.asarray(d.qvel, dtype=np.float64)
        body_pos = d.xpos[1]
        yaw = quat_yaw(d.xquat[1])
        wheel_xyz = np.asarray([d.xpos[WHEEL_BODY[i]] for i in range(4)])
        wheel_vel = np.asarray([d.cvel[WHEEL_BODY[i]][0:3] for i in range(4)])
        _Rbm = np.asarray(d.xmat[1], dtype=np.float64).reshape(3, 3)
        body_vel = _Rbm.T @ np.asarray(d.cvel[1][3:6], dtype=np.float64)

        if int(t * 200) % _nav_period == 0:
            pos2 = body_pos[:2]
            _correction = ''
            wheel_z = np.asarray([d.xpos[WHEEL_BODY[i]][2]
                                  for i in range(4)], dtype=np.float64)
            local_map = None
            if os.environ.get('S10_RL_ELEV', '0') == '1':
                local_map = perc.local_tile(pos2, t)
            stair.update(pos2, next_idx, yaw, local_map,
                         float(body_vel[0]), wheel_z)

            if stair.mode != 'STAIR' and _tk2_was_stair:
                _post_stair_xy = np.asarray(pos2, dtype=np.float64)
                if os.environ.get('S10_TK2', '0') == '1':
                    _tk2 = True
            _tk2_was_stair = (stair.mode == 'STAIR')

            if os.environ.get('S10_RL_ELEV', '0') == '1':
                rxy, rtops = perc.riser_table(stair)
                if rxy is not None and len(rxy) >= 2:
                    h = stair.stair_first_heading if stair.mode == 'STAIR' else None
                    if h is None:
                        h = perc.stair_heading(stair)
                    rl.set_risers(rxy, rtops, heading=h)

            line = nav.line(next_idx, pos2)
            if line is None:
                vx, vyaw = 0.0, 0.0
            else:
                dist_wp = float(line['dist_to_wp'] or 0.0)
                # 直线导航只输出直线段；主循环做“航段方向 + 横向纠偏”，
                # 接近 wp 时再直接瞄准 wp，确保 0.2m 判点圆可达。
                _seg = np.asarray(line['end']) - np.asarray(line['start'])
                _segl = max(float(np.linalg.norm(_seg)), 1e-9)
                _ux, _uy = _seg / _segl
                _seg_head = float(np.arctan2(_uy, _ux))
                _rel = pos2 - np.asarray(line['start'])
                _cte = float(-_uy * _rel[0] + _ux * _rel[1])
                _cte_k = float(os.environ.get('S10_LINE_CTE_K', '1.0'))
                _des = _seg_head - _cte_k * float(np.clip(_cte, -1.0, 1.0))
                if dist_wp < 0.5:
                    _des = float(np.arctan2(line['end'][1] - pos2[1],
                                            line['end'][0] - pos2[0]))
                line_head = _des
                head_err = float(np.arctan2(np.sin(line_head - yaw),
                                            np.cos(line_head - yaw)))
                vyaw = float(np.clip(
                    float(os.environ.get('S10_LINE_YAW_GAIN', '2.5')) * head_err,
                    -float(os.environ.get('S10_LINE_YAW_MAX', '2.0')),
                    float(os.environ.get('S10_LINE_YAW_MAX', '2.0'))))
                brake = float(np.clip(
                    (dist_wp - float(os.environ.get('S10_WP_ARRIVE_R', '0.2')))
                    / max(float(os.environ.get('S10_LINE_BRAKE_DIST', '1.5')),
                          1e-3), 0.0, 1.0))
                vx = float(os.environ.get('S10_LINE_VMAX', '3.0')) * brake
                # 下一航段是锐角时，提前在当前段内减速，不能带 3m/s 冲入弯
                if next_idx + 1 < len(wp) and dist_wp < 3.0:
                    _next_vec = wp[next_idx + 1, :2] - wp[next_idx, :2]
                    _next_head = float(np.arctan2(_next_vec[1], _next_vec[0]))
                    _delta_next = abs(float(np.arctan2(
                        np.sin(_next_head - line_head),
                        np.cos(_next_head - line_head))))
                    if _delta_next > float(os.environ.get(
                            'S10_LINE_TURN_ANGLE', '0.5')):
                        vx = min(vx, float(os.environ.get(
                            'S10_LINE_TURN_VMAX', '1.2')))
                # 近点：未对准时允许降到 TMppi 触发速度；对准后给最低速度过点
                if abs(head_err) <= float(os.environ.get(
                        'S10_LINE_ALIGNED_DB', '0.15')):
                    vx = max(vx, float(os.environ.get('S10_LINE_MIN_VX', '0.5')))
                elif dist_wp > 0.35:
                    vx = max(vx, float(os.environ.get('S10_LINE_MIN_VX', '0.5')))
                if _prev_line_head is not None:
                    delta = abs(float(np.arctan2(
                        np.sin(line_head - _prev_line_head),
                        np.cos(line_head - _prev_line_head))))
                    if delta > float(os.environ.get('S10_LINE_TURN_ANGLE', '0.5')):
                        vx = min(vx, float(os.environ.get(
                            'S10_LINE_TURN_VMAX', '1.2')))
                _prev_line_head = line_head

            # TK1：CRUISE 中检测到前方楼梯，只做“限速 + 对准”，不改模式
            if (os.environ.get('S10_TK1', '0') == '1'
                    and os.environ.get('S10_RL_ELEV', '0') == '1'
                    and stair.mode == 'CRUISE'):
                th = perc.stair_heading(stair)
                if th is not None:
                    ey = float(np.arctan2(np.sin(th - yaw),
                                          np.cos(th - yaw)))
                    vx = min(vx, float(os.environ.get('S10_TK1_VX', '2.0')))
                    _correction += 'TK1'
                    if abs(ey) > float(os.environ.get('S10_TK1_YAW_DB', '0.20')):
                        ky = float(os.environ.get('S10_TK1_YAW_K', '2.5'))
                        ym = float(os.environ.get('S10_TK1_YAW_MAX', '1.5'))
                        vyaw = float(np.clip(ky * ey, -ym, ym))

            # TK2：STAIR→CRUISE 后立即对准下一航点；对齐后交回 TMppi/SMppi
            if (os.environ.get('S10_TK2', '0') == '1' and _tk2
                    and stair.mode == 'CRUISE'):
                _correction += 'TK2'
                th2 = float(np.arctan2(wp[next_idx, 1] - body_pos[1],
                                       wp[next_idx, 0] - body_pos[0]))
                ey2 = float(np.arctan2(np.sin(th2 - yaw),
                                       np.cos(th2 - yaw)))
                if abs(ey2) > float(os.environ.get('S10_TK2_YAW_DB', '0.15')):
                    ky2 = float(os.environ.get('S10_TK2_YAW_K', '2.5'))
                    ym2 = float(os.environ.get('S10_TK2_YAW_MAX', '1.5'))
                    vyaw = float(np.clip(ky2 * ey2, -ym2, ym2))
                    vx = min(vx, float(os.environ.get('S10_TK2_VX', '1.5')))
                else:
                    _tk2 = False

            v_ref = vx
            if stair.decel_request > 0.0:
                dv = float(os.environ.get('S10_ELEV_DECEL_VX', '2.0'))
                vx = vx * (1.0 - stair.decel_request) + dv * stair.decel_request
                v_ref = min(v_ref, vx)

            ref_pts = []
            if line is not None:
                u = np.asarray(line['end'] - line['start'], dtype=np.float64)
                u = u / max(np.linalg.norm(u), 1e-9)
                look = min(12.0, max(float(line['dist_to_wp'] or 0.0) + 1.5, 2.0))
                for ds in np.arange(0.0, look + 1e-6, 0.5):
                    pt = pos2 + u * ds
                    ref_pts.append([pt[0], pt[1], float(line['heading'])])
            if ref_pts:
                ref_path = np.asarray(ref_pts, dtype=np.float64)
            else:
                ref_path = np.asarray([[pos2[0], pos2[1], yaw]])

            state = np.asarray([pos2[0], pos2[1], yaw,
                                body_vel[0], body_vel[1], qvel[5]])
            # 巡航规划器二选一：
            # TMppi：已在当前 wp 0.2m 内、实际速度<0.2、且下一段方向误差>10°
            # SMppi：其余所有 CRUISE 时间
            wp_next = wp[next_idx + 1] if next_idx + 1 < len(wp) else None
            used_turn, vx_c, om_c = tmppi.try_plan(
                pos2, yaw, float(np.linalg.norm(d.cvel[1][3:6])),
                wp[next_idx], wp_next)
            if used_turn:
                _planner = 'TMppi'
            else:
                _planner = 'SMppi'
                vx_c, om_c = smppi.plan(state, ref_path, v_ref, prev_u,
                                        float(vyaw))
            omcap = float(os.environ.get('S10_VMC_OM_CAP', '2.0'))
            latmax = float(os.environ.get('S10_AUTO_LAT_MAX', '5.0'))
            omcap = min(omcap, latmax / max(abs(vx_c), 0.5))
            om_c = float(np.clip(om_c, -omcap, omcap))
            prev_u = np.asarray([vx_c, om_c])
        else:
            vx_c, om_c = prev_u

        # 全局 lidar 轮下地形
        terr = np.asarray([perc.height(float(wheel_xyz[i, 0]),
                                       float(wheel_xyz[i, 1]), t,
                                       float(body_pos[2]),
                                       float(wheel_xyz[i, 2]) - 0.081)
                           for i in range(4)], dtype=np.float64)
        roll_tar = float(np.clip(
            -float(os.environ.get('S10_CAR_ROLL_K', '0.06')) * om_c
            * abs(vx_c),
            -float(os.environ.get('S10_CAR_ROLL_AMP', '0.06')),
            float(os.environ.get('S10_CAR_ROLL_AMP', '0.06'))))
        cmd = dict(vx=vx_c, omega=om_c, roll_tar=roll_tar, pitch_tar=0.0)

        # PRETRANS：楼梯前按 riser 距离进入 RL 高站姿；楼梯后按 handback 距离退出
        if os.environ.get('S10_PRETRANS', '1') == '1' and stair.mode != 'STAIR':
            if os.environ.get('S10_CAR_SQUAT', '1') == '1':
                sq = np.asarray([-0.05, -1.10, 1.90, 0.05, -1.10, 1.90,
                                 -0.05, 1.10, -1.90, 0.05, 1.10, -1.90])
            else:
                sq = np.asarray([-0.05, -1.16, 2.30, 0.05, -1.16, 2.30,
                                 -0.05, 1.16, -2.30, 0.05, 1.16, -2.30])
            ta = np.asarray([-0.05, -0.60, 1.20, 0.05, -0.60, 1.20,
                             -0.05, 0.60, -1.20, 0.05, 0.60, -1.20])
            if _post_stair_xy is not None:
                elen = float(os.environ.get('S10_PRETRANS_EXIT_LEN', '2.0'))
                dex = float(np.hypot(body_pos[0] - _post_stair_xy[0],
                                     body_pos[1] - _post_stair_xy[1]))
                tpr = float(np.clip(1.0 - dex / max(elen, 1e-3), 0.0, 1.0))
            else:
                enter = float(os.environ.get('S10_PRETRANS_ENTER_DIST', '2.0'))
                blend = float(os.environ.get('S10_PRETRANS_BLEND_LEN', '1.0'))
                ad = stair.stair_ahead_dist
                tpr = 0.0 if ad is None else float(np.clip(
                    (enter + blend - ad) / max(blend, 1e-3), 0.0, 1.0))
            carvmc.pose_target = (1.0 - tpr) * sq + tpr * ta

        # 执行器切换：STAIR=RL，CRUISE=CarVMC
        if stair.mode != 'STAIR' and _rl_was_stair:
            hy = quat_yaw(d.xquat[1])
            hv = float(d.qvel[0] * np.cos(hy) + d.qvel[1] * np.sin(hy))
            bs = carvmc.body_state(qpos, qvel)
            carvmc.reset_state(vx=hv, omega=float(qvel[5]),
                               roll=bs['roll'], pitch=bs['pitch'])
            print('[RL-DIAG] RL->CRUISE at pos=(%.2f,%.2f,%.2f) yaw=%.3f vx=%.2f'
                  % (body_pos[0], body_pos[1], body_pos[2], hy, hv), flush=True)
        if stair.mode == 'STAIR' and not _rl_diag_done:
            _rl_diag_done = True
            if stair.stair_first_heading is not None:
                rl.set_heading(float(stair.stair_first_heading))
            dq = qpos[rl.idx['act2jnt']] - rl.default_dof
            print('[RL-DIAG] takeover pos=(%.2f,%.2f,%.2f) yaw=%.3f '
                  'max_leg_err=%.3f'
                  % (body_pos[0], body_pos[1], body_pos[2], yaw,
                     float(np.abs(dq[rl.idx['leg_idx']]).max())), flush=True)
        _rl_was_stair = (stair.mode == 'STAIR')

        if stair.mode == 'STAIR':
            tau = rl.compute_tau(qpos, qvel, wheel_xyz, wheel_vel, cmd,
                                 terr, DT)
        else:
            tau = carvmc.compute_tau(qpos, qvel, wheel_xyz, wheel_vel, cmd,
                                     terr, DT)

        # PRETRANS 腿 PD：进入前锁定 RL 高站姿；退出时按距离平滑交还 CarVMC
        if (os.environ.get('S10_PRETRANS', '1') == '1'
                and stair.mode != 'STAIR'):
            hold = float(os.environ.get('S10_PRETRANS_HOLD_DIST', '2.0'))
            approach_hold = (_post_stair_xy is None
                             and stair.stair_ahead_dist is not None
                             and stair.stair_ahead_dist <= hold)
            if approach_hold or _post_stair_xy is not None:
                li = rl.idx['leg_idx']
                lj = rl.idx['act2jnt'][li]
                lv = rl.idx['act2vel'][li]
                lt = np.clip(60.0 * (rl.default_dof[li] - qpos[lj])
                             - 4.0 * qvel[lv], -48.0, 48.0)
                if _post_stair_xy is None:
                    tau[li] = lt
                else:
                    elen = float(os.environ.get('S10_PRETRANS_EXIT_LEN', '2.0'))
                    dex = float(np.hypot(body_pos[0] - _post_stair_xy[0],
                                         body_pos[1] - _post_stair_xy[1]))
                    b = float(np.clip(dex / max(elen, 1e-3), 0.0, 1.0))
                    tau[li] = (1.0 - b) * lt + b * tau[li]

        tleg = float(np.abs(tau[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]]).max())
        twh = float(np.abs(tau[[3, 7, 11, 15]]).max())
        _max_tau_leg = max(_max_tau_leg, tleg)
        _max_tau_wh = max(_max_tau_wh, twh)
        if tleg >= 50.0 or twh >= 14.0:
            _over_run += DT
            _over_total += DT
            _over_worst = max(_over_worst, _over_run)
        else:
            _over_run = 0.0

        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        t += DT
        _ctrl_cnt += 1

        if int(t * 200) % 100 == 0:
            roll = float(np.arctan2(
                2.0 * (d.xquat[1][0] * d.xquat[1][1]
                       + d.xquat[1][2] * d.xquat[1][3]),
                1.0 - 2.0 * (d.xquat[1][1] ** 2 + d.xquat[1][2] ** 2)))
            spd = float(np.hypot(d.cvel[1][3], d.cvel[1][4]))
            print('[T] t=%.0f wp=%d pos=(%.1f,%.1f,%.2f) yaw=%.2f '
                  'spd=%.2f roll=%.2f cmd=(%.2f,%.2f) mode=%s corr=%s plan=%s'
                  % (t, next_idx, body_pos[0], body_pos[1], body_pos[2],
                     yaw, spd, roll, vx_c, om_c, stair.mode,
                     (_correction or 'NONE'), _planner), flush=True)
            if abs(roll) > 0.9 or body_pos[2] < 0.12:
                print('[T] *** 侧翻/摔倒 ***', flush=True)
                break
            if os.environ.get('S10_STUCK_TIMEOUT', '90') != '0' and \
                    t - last_adv_t > float(os.environ.get(
                        'S10_STUCK_TIMEOUT', '90')):
                print('[T] *** 卡死超时 wp=%d ***' % next_idx, flush=True)
                break

        if os.environ.get('S10_TRAJ_DENSE', '0') == '1':
            traj.append([t, body_pos[0], body_pos[1], body_pos[2], yaw,
                         1.0 if stair.mode == 'STAIR' else 0.0,
                         float(np.hypot(d.cvel[1][3], d.cvel[1][4]))])

        # 航点推进：只按原始折线水平距离
        if next_idx < len(wp):
            if nav.reached(next_idx, d.xpos[1][:2]):
                if next_idx == 0 and t_start is None:
                    t_start = t
                wp_times[next_idx] = t
                last_adv_t = t
                print('[T] wp%d @ t=%.2f' % (next_idx, t), flush=True)
                next_idx += 1
                if next_idx >= int(os.environ.get('S10_AUTO_MAX_WP', '33')):
                    print('[T] 到达最大航点，结束', flush=True)
                    break

    print('=== result ===')
    print('完成: %s, 最终 wp=%d/%d'
          % (next_idx >= int(os.environ.get('S10_AUTO_MAX_WP', '33')),
             next_idx, int(os.environ.get('S10_AUTO_MAX_WP', '33'))))
    if t_start is not None and next_idx > 0:
        print('wp0->wp%d 用时 %.1fs'
              % (min(next_idx - 1, int(os.environ.get('S10_AUTO_MAX_WP', '33'))
                     - 1), wp_times.get(next_idx - 1, t) - t_start))
    print('力矩合规: 腿max|tau|=%.1fNm(限48/50) 轮max|tau|=%.1fNm(限13.5/14) '
          '连续超限最长 %.2fs / 累计 %.2fs%s'
          % (_max_tau_leg, _max_tau_wh, _over_worst, _over_total,
             '  [超0.5s不合格!]' if _over_worst > 0.5 else ''), flush=True)
    if os.environ.get('VMC_TRAJ'):
        np.save(os.environ['VMC_TRAJ'], np.asarray(traj))


if __name__ == '__main__':
    main()
