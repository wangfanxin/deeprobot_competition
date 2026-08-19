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
    _post_stair_t = None
    _rl_diag_done = False
    _max_tau_leg = 0.0
    _max_tau_wh = 0.0
    _over_run = 0.0
    _over_worst = 0.0
    _over_total = 0.0
    _nav_period = max(1, int(round(200.0 / float(
        os.environ.get('S10_NAV_HZ', '40')))))
    _ctrl_cnt = 0
    _correction = ''
    _planner = ''
    _n_nav = 0
    _tk1_t0 = None
    _tk1_align_t0 = None
    _tk2_t0 = None
    _roll_gate = False
    _roll_gate_since = None
    _edge_lift = np.zeros(4)
    _vyaw_f = 0.0
    _lip_hold = False
    _lip_g0 = 0.0
    _lip_grind_since = None
    _last_lift = np.zeros(4)
    _last_lift_t = -1e9

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
            _n_nav += 1
            pos2 = body_pos[:2]
            _correction = ''
            wheel_z = np.asarray([d.xpos[WHEEL_BODY[i]][2]
                                  for i in range(4)], dtype=np.float64)
            local_map = None
            _ph = None
            if os.environ.get('S10_RL_ELEV', '0') == '1':
                local_map = perc.local_tile(pos2, t)
                _ph = perc.stair_heading(stair)
            stair.update(pos2, next_idx, yaw, local_map,
                         float(body_vel[0]), wheel_z, heading=_ph)
            if os.environ.get('S10_LIP_DEBUG', '0') == '1' and next_idx == 5:
                print('[LIPDBG] t=%.2f pos=(%.2f,%.2f) rises=%s ahead=%s '
                      'decel=%.2f'
                      % (t, pos2[0], pos2[1],
                         [round(v, 2) for v in stair.fol.stair_rises_s[:2]],
                         None if stair.stair_ahead_dist is None
                         else round(stair.stair_ahead_dist, 2),
                         stair.decel_request), flush=True)

            if stair.mode != 'STAIR' and _tk2_was_stair:
                _post_stair_xy = np.asarray(pos2, dtype=np.float64)
                _post_stair_t = t
                if os.environ.get('S10_TK2', '0') == '1':
                    _tk2 = True
                    _tk2_t0 = t
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
                # vyaw 一阶低通：CTE 纠偏符号翻转导致的蛇形过冲
                # （wp5 台顶实测 1.56->2.5 rad 过冲后下台侧翻）
                _vyaw_f += (vyaw - _vyaw_f) * float(os.environ.get(
                    'S10_LINE_VYAW_LP', '0.4'))
                vyaw = _vyaw_f
                # 2026-08-19: 只保留“加速 + 到点刹车”两个能力；
                # head-err 降速/锐角预刹/MIN_VX 地板全部删除（不过弯）。
                # 终点速度由 SMppi 终点代价（dx=0/ref_v=0）二次兜底。
                _bd = max(float(os.environ.get(
                    'S10_LINE_BRAKE_DIST', '3.5')), 1e-3)
                _brk = float(np.clip(
                    (dist_wp - float(os.environ.get(
                        'S10_WP_ARRIVE_R', '0.2'))) / _bd, 0.0, 1.0))
                # 物理一致刹车剖面 v∝sqrt(剩余距离)：配合 A_MAX 保证
                # 到点前可刹停（线性剖面短段刹不住，wp3 过冲侧翻实测）
                vx = float(os.environ.get('S10_LINE_VMAX', '4.0')) * float(np.sqrt(_brk))

            # TK1：CRUISE 中检测到前方楼梯，只做“对准”，不改模式。
            # 减速由 SMppi 终点代价 + decel_request 负责；TK1 只在进入
            # 交付圈（S10_STAIR_ENTER_DIST）后补 1.5m/s 交付速度上限。
            if (os.environ.get('S10_TK1', '0') == '1'
                    and os.environ.get('S10_RL_ELEV', '0') == '1'
                    and stair.mode == 'CRUISE'
                    and abs(_cte) <= float(os.environ.get(
                        'S10_TK1_CTE_MAX', '0.8'))):
                th = _ph
                if th is not None:
                    if _tk1_t0 is None:
                        _tk1_t0 = t
                    ey = float(np.arctan2(np.sin(th - yaw),
                                          np.cos(th - yaw)))
                    _ad = stair.stair_ahead_dist
                    if (_ad is not None and _ad <= float(
                            os.environ.get('S10_STAIR_ENTER_DIST', '2.0'))):
                        vx = min(vx, float(os.environ.get('S10_TK1_VX', '1.5')))
                    _correction += 'TK1'
                    if abs(ey) > float(os.environ.get('S10_TK1_YAW_DB', '0.20')):
                        if (_tk1_align_t0 is None and (_ad is None
                                or _ad <= float(os.environ.get(
                                    'S10_STAIR_ENTER_DIST', '2.0')))):
                            _tk1_align_t0 = t
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
                    if _tk2_t0 is not None:
                        print('[TK2] 上顶->对准 %.2fs (预算1.0s)'
                              % (t - _tk2_t0), flush=True)
                        _tk2_t0 = None
                    _tk2 = False

            # 刚离开楼梯时先保持短时间直线慢行，避免平台边缘 yaw 过冲侧翻
            if (_post_stair_t is not None
                    and t - _post_stair_t < float(os.environ.get(
                        'S10_POST_STAIR_HOLD_T', '0.6'))):
                vx = min(vx, 0.5)
                vyaw = 0.0
            # 楼梯后慢速瞄准超时清除：一直追不上下一 wp 时
            # 恢复全速（此前永不清除，wp5 实测 0.6m/s 永久爬行）
            if (_post_stair_t is not None
                    and t - _post_stair_t > float(os.environ.get(
                        'S10_POST_STAIR_MAX_T', '5.0'))):
                _post_stair_xy = None
                _post_stair_t = None

            v_ref = vx
            # STAIR 扫描是权威台阶源：riser 逼近到 1.5m 内时
            # 压 0.6 并锁存——贴沿后扫描会变空（s_proj 越过 riser）
            # 但轮子还在坎上，解锁会 3.6m/s 冲坎侧翻（wp5 实测）。
            # 锁存到 min(terr) 比锁存时高 0.10（后轮上台）才释放。
            # 骑坎锁存兜底：前轮已上台、后轮还在台下（跨骑状态）
            # 也触发锁存——路径扫描/EDGE 都有盲区，跨骑是最后信号
            if (os.environ.get('S10_LIP_LATCH', '1') == '1'
                    and next_idx >= 4
                    and float(np.max(terr[0:2])) - float(np.min(terr))
                    >= 0.08):
                if not _lip_hold:
                    _lip_g0 = float(np.min(terr))
                _lip_hold = True
            if (os.environ.get('S10_LIP_LATCH', '1') == '1'
                    and next_idx >= 4
                    and stair.decel_request > 0.5
                    and stair.stair_ahead_dist is not None
                    and stair.stair_ahead_dist <= 1.5):
                if not _lip_hold:
                    _lip_g0 = float(np.min(terr))
                    if os.environ.get('S10_LIP_DEBUG', '0') == '1':
                        print('[LATCH] set t=%.2f pos=(%.2f,%.2f) g0=%.2f'
                              % (t, pos2[0], pos2[1], _lip_g0), flush=True)
                _lip_hold = True
            if _lip_hold:
                vx = float(os.environ.get('S10_STAIR_APPROACH_VX', '0.6'))
                v_ref = min(v_ref, vx)   # MPPI 的上限是 v_ref，必须同步
                # 锁存期间保持正对路径航向 + 前轮抬升：斜贴台沿会
                # 单轮骑坎沿坎侧滑（wp5 实测东滑 3m 卡死）
                if line is not None:
                    _ehl = float(np.arctan2(
                        np.sin(line_head - yaw),
                        np.cos(line_head - yaw)))
                    vyaw = float(np.clip(1.0 * _ehl, -0.35, 0.35))
                    _clf = np.array([np.cos(line_head),
                                     np.sin(line_head)])
                    _hli = perc.height(
                        body_pos[0] + _clf[0] * 0.8,
                        body_pos[1] + _clf[1] * 0.8, t,
                        float(body_pos[2]),
                        float(body_pos[2]) - 0.55)
                    _rli = float(_hli - float(np.min(terr)))
                    # 抬轮与推力冲突（抬轮减轮压致打滑），锁存期不抬，
                    # 靠 1.2m/s 冲量 + 轮半径滚上 0.064 立面
                    _edge_lift[0:2] = 0.0
                if float(np.min(terr)) >= _lip_g0 + 0.10:
                    if os.environ.get('S10_LIP_DEBUG', '0') == '1':
                        print('[LATCH] release t=%.2f pos=(%.2f,%.2f)'
                              % (t, pos2[0], pos2[1]), flush=True)
                    _lip_hold = False
                    _edge_lift[0:2] = 0.0
                    _lip_grind_since = None
            if stair.decel_request > 0.0:
                if _tk1_t0 is None:
                    _tk1_t0 = t
                dv = float(os.environ.get('S10_ELEV_DECEL_VX', '2.0'))
                vx = vx * (1.0 - stair.decel_request) + dv * stair.decel_request
                v_ref = min(v_ref, vx)
            # 下行落差保护：前方检测到 >=0.08m 跌落沿时强制低速直行，
            # 避免高速下台栽头（下行不交 RL，先慢速爬行兜底）。
            if (stair.drop_ahead_dist is not None
                    and stair.drop_ahead_dist < float(os.environ.get(
                        'S10_DROP_LOOKAHEAD', '2.0'))):
                vx = min(vx, float(os.environ.get('S10_DROP_VX', '0.3')))
                vyaw = float(np.clip(vyaw, -0.5, 0.5))
                v_ref = min(v_ref, vx)
                _correction += 'DROP'
            # 机器人相对 riser 检测（路径扫描盲区：偏离路径时前方台阶）
            # 前方 0.3~1.2m 升高 0.08~0.25m => 正对直行低速骑上（不交 RL）
            _rf = float(os.environ.get('S10_EDGE_LOOKAHEAD', '1.2'))
            if (_rf > 0.0 and stair.mode == 'CRUISE'
                    and abs(_cte) <= float(os.environ.get(
                        'S10_EDGE_CTE_MAX', '0.8'))):
                # 探针沿路径航向（机器人实际航向偏移时也能看到前方台阶）
                _ehd = line_head if line is not None else yaw
                _cf = np.array([np.cos(_ehd), np.sin(_ehd)])
                _ch0e = float(body_pos[2]) - 0.55
                _hfn = perc.height(body_pos[0] + _cf[0] * 0.3,
                                    body_pos[1] + _cf[1] * 0.3, t,
                                    float(body_pos[2]), _ch0e)
                _hff = perc.height(body_pos[0] + _cf[0] * _rf,
                                    body_pos[1] + _cf[1] * _rf, t,
                                    float(body_pos[2]), _ch0e)
                _h05 = perc.height(body_pos[0] + _cf[0] * 0.5,
                                    body_pos[1] + _cf[1] * 0.5, t,
                                    float(body_pos[2]), _ch0e)
                # 基准用 min(terr)（后轮未上台前保持地面高）：
                # 前轮贴台阶时 max(terr) 已升高导致抬轮失效
                _rise_edge = float(_hff - float(np.min(terr)))
                # 双探针：rise 取 2.5m（提前刹停距离），平顶判别
                # 1.5m vs 2.5m 差 <=0.05（台面 2.1m 长，2.6m 探针
                # 会越过远沿致平顶误判；缓坡 1m 内升 0.07 排除）
                _hff2 = perc.height(body_pos[0] + _cf[0] * 2.5,
                                     body_pos[1] + _cf[1] * 2.5, t,
                                     float(body_pos[2]), _ch0e)
                # rise 取 1.5m 探针；允许 2.5m 探针更低（台面只有
                # 2.1m 长，贴近沿时远探针越过对侧沿）但不允许更高
                # （缓坡 1m 升 0.07 > 0.05 => 排除坡道误触发）
                _flat_top = (float(_hff2) <= float(_hff) + 0.05)
                _rise_edge = float(_hff - float(np.min(terr)))
                # 台阶判别：1.5m 与 0.5m 探针高差 >=0.08（坡面 1m
                # 升 0.07、坡顶 0.03 均排除，wp0-1 坡顶误触发实测）
                _step_rise = float(_hff - _h05)
                if (0.05 <= _rise_edge <= 0.25 and _flat_top
                        and _step_rise >= 0.08):
                    vx = min(vx, float(os.environ.get('S10_EDGE_VX', '0.6')))
                    v_ref = min(v_ref, vx)
                    _correction += 'EDGE'
                    # EDGE 锁存只认 >=10cm 的真台阶：坡顶 0.05 的
                    # 边界误锁 + om 强制正对 = 坡顶自旋侧翻（round46）
                    if (os.environ.get('S10_LIP_LATCH', '1') == '1'
                            and next_idx >= 4
                            and _rise_edge >= 0.10):
                        if not _lip_hold:
                            _lip_g0 = float(np.min(terr))
                        _lip_hold = True
                    # 前轮抬轮前馈：仅 >=10cm 的台阶（小台阶
                    # 抬轮会失牵引，wp0-1 坡底实测 12s 卡死）
                    if _rise_edge >= 0.10:
                        _edge_lift[0:2] = float(np.clip(
                            (_rise_edge - 0.06) / 0.06, 0.0, 1.0))
                    else:
                        _edge_lift[0:2] = 0.0
                else:
                    _edge_lift[0:2] = 0.0
            # 终点刹车直入 v_ref（不依赖软代价）：STOP_DX 内目标
            # 速度按剩余距离线性归零——实测软终端代价刹不住，
            # wp3 以 3.2m/s 冲点过冲 1.6m。
            if line is not None:
                _stdx = float(os.environ.get('S10_SMppi_STOP_DX', '4.0'))
                if dist_wp <= _stdx:
                    _vstop = float(os.environ.get(
                        'S10_AUTO_VMAX', '4.0')) * float(np.clip(
                        dist_wp / _stdx, 0.0, 1.0))
                    v_ref = min(v_ref, _vstop)
            ref_pts = []
            _wp_dx = None
            if line is not None:
                u = np.asarray(line['end'] - line['start'], dtype=np.float64)
                _len = float(np.linalg.norm(u))
                u = u / max(_len, 1e-9)
                _wp_dx = float(line['dist_to_wp'] or 0.0)
                # 参考路径锚定在真实航线上（机器人投影点 -> wp）：
                # 从 pos2 出发会让横向偏移对距离成本不可见，
                # wp0-1 实测整段平行漂移 +2m 失控侧翻。
                _proj = float(np.dot(pos2 - np.asarray(line['start']), u))
                _proj = float(np.clip(_proj, 0.0, _len))
                _base = np.asarray(line['start']) + u * _proj
                for ds in np.arange(0.0, min(_wp_dx, 12.0) + 1e-6, 1.0):
                    pt = _base + u * ds
                    ref_pts.append([pt[0], pt[1], float(line['heading'])])
                ref_pts.append([float(line['end'][0]), float(line['end'][1]),
                                float(line['heading'])])
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
                                        float(vyaw), wp_dx=_wp_dx)
            latmax = float(os.environ.get('S10_AUTO_LAT_MAX', '1.8'))
            if used_turn:
                omcap = min(float(os.environ.get('S10_TURN_OM_MAX', '3.0')),
                            latmax / max(abs(vx_c), 0.5))
            else:
                omcap = min(float(os.environ.get('S10_VMC_OM_CAP', '2.0')),
                            latmax / max(abs(vx_c), 0.5))
            # 侧倾过大时停止转向并减速，防止 roll 正反馈翻车。
            # 滞回门控（0.30 触发 / 0.15 释放）：防坡面抖振；
            # 释放后经 sync_applied 从真实指令慢升，不 0.4->4.0 阶跃。
            _body_roll = float(np.arctan2(
                2.0 * (d.xquat[1][0] * d.xquat[1][1]
                       + d.xquat[1][2] * d.xquat[1][3]),
                1.0 - 2.0 * (d.xquat[1][1] ** 2 + d.xquat[1][2] ** 2)))
            if abs(_body_roll) > 0.30:
                if not _roll_gate:
                    _roll_gate_since = t
                _roll_gate = True
            elif abs(_body_roll) < 0.20:
                _roll_gate = False
                _roll_gate_since = None
            if _roll_gate:
                # 允许 +/-0.3 慢转向脱困：此前 om=0 让机器人在
                # 台沿侧倾死锁（wp5 实测 30s 卡死）
                om_c = float(np.clip(om_c, -0.3, 0.3))
                vx_c = min(float(vx_c), 0.3)
                # 死锁脱困：门控持续 >2s => 直线倒车离开台阶边缘。
                # 台沿锁存期不倒车：跨骑台阶时 roll 恒定超阈值，
                # 倒车轮子打滑卡死（wp5 实测）；交给抬轮把后轮
                # 拉上台面恢复水平。
                if (_roll_gate_since is not None
                        and t - _roll_gate_since > 2.0
                        and not _lip_hold):
                    vx_c = -0.4
                    om_c = 0.0
            # 楼梯后前 1.2m：只直线低速前进，禁止转向，避免平台边缘 yaw 反冲侧翻
            # 楼梯后：低倍率直接瞄当前目标 wp，直到距其足够近才放开
            if (_post_stair_xy is not None and next_idx < len(wp)
                    and float(np.linalg.norm(
                        body_pos[:2] - wp[next_idx, :2]))
                    > float(os.environ.get('S10_POSTSTAIR_HOLD_DIST', '0.7'))):
                vx_c = min(float(vx_c), 0.6)
                _ph = float(np.arctan2(wp[next_idx, 1] - body_pos[1],
                                       wp[next_idx, 0] - body_pos[0]))
                _pe = float(np.arctan2(np.sin(_ph - yaw),
                                       np.cos(_ph - yaw)))
                om_c = float(np.clip(0.5 * _pe, -0.2, 0.2))
            # 高台弱抓地：进一步限制速度与转向率
            if float(body_pos[2]) > 1.0:
                vx_c = min(float(vx_c), 0.8)
                omcap = min(omcap, 0.3)
            om_c = float(np.clip(om_c, -omcap, omcap))
            # 台沿锁存期：cmd 级强制正对路径航向（经 MPPI 的弱 guide
            # 被其它代价压过，wp5 实测贴沿航向漂 0.3rad 侧滑）
            if _lip_hold and not _roll_gate:
                _ehl = float(np.arctan2(np.sin(line_head - yaw),
                                         np.cos(line_head - yaw)))
                om_c = float(np.clip(2.0 * _ehl, -1.0, 1.0))
                vx_c = min(float(vx_c), 0.6)
            # TK 阶段（TK1 对准 / TK2 转出）直接执行对准转向：
            # MPPI 的 om 上限被 VMC_OM_CAP=1.0 压住，<1s 预算不够；
            # 低 vx 时 car_omega_limit≈3.0，TK 专用上限 2.0 安全。
            # 2026-08-19 修复：必须让位 roll 门控（此前 TK 直接给 om
            # 会覆盖门控的 om=0，平台边 TK2 侧倾正反馈翻车实测）。
            if ('TK1' in _correction or 'TK2' in _correction) \
                    and not _roll_gate:
                _om_tk = float(os.environ.get('S10_TK_OM_MAX', '2.0'))
                om_c = float(np.clip(vyaw, -_om_tk, _om_tk))
                vx_c = min(float(vx_c), float(os.environ.get(
                    'S10_TK_VX', '1.5')))
            prev_u = np.asarray([vx_c, om_c])
            smppi.sync_applied(prev_u)
        else:
            vx_c, om_c = prev_u

        # 全局 lidar 轮下地形
        terr = np.asarray([perc.height(float(wheel_xyz[i, 0]),
                                       float(wheel_xyz[i, 1]), t,
                                       float(body_pos[2]),
                                       float(wheel_xyz[i, 2]) - 0.081)
                           for i in range(4)], dtype=np.float64)
        _terr_raw = terr.copy()  # 抬轮基准用原始地形（LP/blend 会拉高）
        # 地形低通：坡顶过渡/栅格噪声防腿阻抗踢振（riser 不在此处理）
        _tlp = float(os.environ.get('S10_VMC_TERRAIN_LP', '0.0'))
        if _tlp > 0.0:
            if '_terr_f' not in globals():
                globals()['_terr_f'] = terr.copy()
            _terr_f = globals()['_terr_f']
            _terr_f = _tlp * terr + (1.0 - _tlp) * _terr_f
            globals()['_terr_f'] = _terr_f
            terr = _terr_f
        # 坡顶前瞻平滑（仅 CRUISE）：前方 0.05~0.25m 升高时把前轮
        # 地形参考预伸，防坡顶 pitch/roll 踢振。非抬轮前馈；
        # riser(>=8cm) 在 2m 外已交 STAIR，STAIR 模式内不生效。
        _lk = float(os.environ.get('S10_VMC_TERRAIN_LOOKAHEAD', '0.0'))
        if _lk > 0.0 and stair.mode == 'CRUISE':
            _fwd = np.asarray(d.xmat[1][[0, 3]], dtype=np.float64)
            _fwd = _fwd / max(float(np.linalg.norm(_fwd)), 1e-9)
            _hx = body_pos[0] + _fwd[0] * _lk
            _hy = body_pos[1] + _fwd[1] * _lk
            _ahead = perc.height(_hx, _hy, t, float(body_pos[2]),
                                 float(body_pos[2]) - 0.55)
            _ahead = min(_ahead, float(np.max(terr)) + 0.25)
            _rise_ahead = float(_ahead - np.max(terr))
            if 0.05 <= _rise_ahead <= 0.25:
                _w = float(os.environ.get('S10_VMC_TERRAIN_AHEAD_W', '0.6'))
                terr = (1.0 - _w) * terr + _w * _ahead
        roll_tar = float(np.clip(
            -float(os.environ.get('S10_CAR_ROLL_K', '0.06')) * om_c
            * abs(vx_c),
            -float(os.environ.get('S10_CAR_ROLL_AMP', '0.06')),
            float(os.environ.get('S10_CAR_ROLL_AMP', '0.06'))))
        # 高台/弱抓地地形关闭压弯，优先防侧翻
        if float(body_pos[2]) > 1.0:
            roll_tar = 0.0
        # 逐轮抬轮前馈（tune4 版恢复）：台阶/台沿由 CarVMC 巡航爬升。
        # RL 单级 12.5cm 未训练（T8 实测 policy 直接摔倒），
        # STAIR 只接管 6 级楼梯。
        _fwd_lift = np.array([np.cos(yaw), np.sin(yaw)])
        _step_lift = np.zeros(4, dtype=np.float64)
        # 抬轮只在台沿锁存期生效：坡顶/横脊全速段不抬
        # （round44/45 坡顶单后轮误抬侧翻实测）
        if stair.mode == 'CRUISE' and _post_stair_xy is None and _lip_hold:
            _front_on_step = (float(np.max(terr[0:2]))
                              > float(np.max(terr[2:4])) + 0.05)
            for _li in range(4):
                # 后轴只有在前轴骑上台面后才允许抬
                if _li >= 2 and not _front_on_step:
                    continue
                if _terr_raw[_li] <= 0.01:
                    continue  # 无效地形（fallback 0）不抬轮
                _lx = float(wheel_xyz[_li, 0] + _fwd_lift[0] * 0.40)
                _ly = float(wheel_xyz[_li, 1] + _fwd_lift[1] * 0.40)
                _ha = perc.height(_lx, _ly, t, float(body_pos[2]),
                                  fallback_h=float(np.max(terr)))
                _rise = float(_ha - _terr_raw[_li])
                if 0.06 <= _rise <= 0.25:
                    _step_lift[_li] = float(np.clip(
                        (_rise - 0.04) / 0.08, 0.0, 1.0))
        _max_lift = float(np.max(_step_lift))
        _last_lift = _step_lift.copy()
        if _max_lift > 0.05:
            _last_lift_t = t
        _lift_hold = (t - _last_lift_t
                      < float(os.environ.get('S10_LIFT_HOLD_T', '1.0')))
        if _max_lift > 0.05 or _lift_hold:
            os.environ['S10_CAR_WHEEL_GF'] = '0.5'
        else:
            os.environ['S10_CAR_WHEEL_GF'] = '1.0'
        cmd = dict(vx=(0.6 if (_max_lift > 0.05 or _lift_hold) else vx_c),
                   omega=(0.0 if _max_lift > 0.05 else
                          (0.3 if _lift_hold else om_c)),
                   roll_tar=roll_tar, pitch_tar=0.0,
                   step_lift=_step_lift, lift_swing=1.2,
                   yaw_scale=1.0 - 0.6 * _max_lift,
                   ridge_dist=99.0,
                   lift_f_scale=(0.3 if _max_lift > 0.05 else 1.0),
                   wheel_press=(0.1 if _max_lift > 0.05 else 0.0))

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
            if _tk1_t0 is not None:
                print('[TK1] 减速+对准 %.2fs / 对准 %.2fs (预算 2.0/1.0s)'
                      % (t - _tk1_t0,
                         t - _tk1_align_t0 if _tk1_align_t0 is not None
                         else -1.0),
                      flush=True)
                _tk1_t0 = None
                _tk1_align_t0 = None
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
                  'spd=%.2f roll=%.2f cmd=(%.2f,%.2f) mode=%s corr=%s plan=%s '
                  'terr=%s'
                  % (t, next_idx, body_pos[0], body_pos[1], body_pos[2],
                     yaw, spd, roll, vx_c, om_c, stair.mode,
                     (_correction or 'NONE'), _planner,
                     np.round(terr, 2)), flush=True)
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
                         float(next_idx),
                         float(np.hypot(d.cvel[1][3], d.cvel[1][4])),
                         1.0 if stair.mode == 'STAIR' else 0.0])

        # 航点推进：只按原始折线水平距离；到点判（未越过）要求
        # 对准下一段 + 角速度收敛才推点（防原地转完成前抢跑）；
        # 过点兜底：刚越过（<=len+0.8，仍可能原地转中）同样要求
        # 对准；只有明显越过后才跳过对准直接推点。
        if next_idx < len(wp):
            _arr = nav.reached(next_idx, d.xpos[1][:2])
            _align_ok = True
            if _arr and next_idx > 0 and next_idx + 1 < len(wp):
                _segv0 = wp[next_idx, :2] - wp[next_idx - 1, :2]
                _len0 = float(np.linalg.norm(_segv0)) + 1e-9
                _proj = float(np.dot(pos2 - wp[next_idx - 1, :2],
                                     _segv0 / _len0))
                if _proj <= _len0 + 0.8:
                    _segv = wp[next_idx + 1, :2] - wp[next_idx, :2]
                    _hdr = float(np.arctan2(_segv[1], _segv[0]))
                    _yerr = abs(float(np.arctan2(np.sin(_hdr - yaw),
                                                 np.cos(_hdr - yaw))))
                    _align_ok = (_yerr <= float(os.environ.get(
                                     'S10_WP_ALIGN_DB', '0.25'))
                                 and abs(float(qvel[5]))
                                 <= float(os.environ.get(
                                     'S10_WP_ALIGN_OM', '0.3')))
            if _arr and _align_ok:
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
    _p_ema, _p_max, _p_n = smppi.plan_stats
    print('规划器: 40Hz 目标 plan=%d次 avg=%.1fms max=%.1fms | '
          '实际控制拍=%.1fHz'
          % (_p_n, _p_ema, _p_max, _n_nav / max(t - 0.5, 1e-3)), flush=True)
    if os.environ.get('VMC_TRAJ'):
        np.save(os.environ['VMC_TRAJ'], np.asarray(traj))


if __name__ == '__main__':
    main()
