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
from s10_mpc.vmc_legs import WHEEL_QV_IDX

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
    _lip_xy = None
    _lip_rel_t = None
    _lip_wp_idx = None
    _lip_t0 = None
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
        # 线性速度（机体系）：cvel[3:6] 是角速度！此前把 roll 率
        # 当车体速度喂给 TK1 门/M PPI 状态（round164 刹车点 vx=-0.4
        # 实为角速度投影，MPPI 状态系统性错误）
        body_vel = _Rbm.T @ np.asarray(d.cvel[1][0:3], dtype=np.float64)

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
            _bp = float(np.arctan2(
                2.0 * (d.xquat[1][0] * d.xquat[1][2]
                       - d.xquat[1][3] * d.xquat[1][1]),
                1.0 - 2.0 * (d.xquat[1][1] ** 2 + d.xquat[1][2] ** 2)))
            _br = float(np.arctan2(
                2.0 * (d.xquat[1][0] * d.xquat[1][1]
                       + d.xquat[1][2] * d.xquat[1][3]),
                1.0 - 2.0 * (d.xquat[1][1] ** 2 + d.xquat[1][2] ** 2)))
            stair.update(pos2, next_idx, yaw, local_map,
                         float(body_vel[0]), wheel_z, pitch=_bp, roll=_br,
                         vy=float(body_vel[1]))
            if os.environ.get('S10_LIP_DEBUG', '0') == '1' and next_idx in (4, 5):
                print('[LIPDBG] t=%.2f pos=(%.2f,%.2f) rises=%s ahead=%s '
                      'drops=%s dropahead=%s ch=%.2f sc=%.2f'
                      % (t, pos2[0], pos2[1],
                         [round(v, 2) for v in stair.fol.stair_rises_s[:2]],
                         None if stair.stair_ahead_dist is None
                         else round(stair.stair_ahead_dist, 2),
                         [round(v, 2) for v in getattr(stair.fol, '_elev_drops', [])],
                         None if stair.drop_ahead_dist is None
                         else round(stair.drop_ahead_dist, 2),
                         stair.climb_heading if stair.climb_heading is not None else -9.0,
                         stair.s_cur), flush=True)

            if stair.mode != 'STAIR' and _tk2_was_stair:
                _post_stair_xy = np.asarray(pos2, dtype=np.float64)
                _post_stair_t = t
                if os.environ.get('S10_TK2', '0') == '1':
                    _tk2 = True
                    _tk2_t0 = t
            _tk2_was_stair = (stair.mode == 'STAIR')

            if os.environ.get('S10_RL_ELEV', '0') == '1':
                rxy, rtops = perc.riser_table(stair)
                if (rxy is not None and len(rxy) == 1
                        and stair.drop_ahead_dist is not None):
                    # 单级台面：把远侧跌落沿补成第二级 riser——
                    # RL 训练分布是连续台阶，单 riser 观测外分布
                    # 起跳摔倒（T8 实测），补第二级后正常爬升
                    _s2 = float(stair.s_cur) + float(stair.drop_ahead_dist)
                    _k2 = int(np.searchsorted(stair.path_cum, _s2,
                                                side='right') - 1)
                    _k2 = min(max(_k2, 0), len(stair.path_pts) - 1)
                    _xy2 = np.asarray(stair.path_pts[_k2][:2])
                    rxy = np.vstack([rxy, _xy2])
                    rtops = np.append(rtops, rtops[0])
                if rxy is not None and len(rxy) >= 2:
                    h = stair.climb_heading
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
                # RL 漂移强退：策略在两节台阶上西漂 6m、6.9m/s
                # 侧翻（round177 实测），离航线 >1.2m 时强制交还
                # CRUISE 自救（MPPI 拉回航线）
                if (stair.mode == 'STAIR' and abs(_cte) > 1.2
                        and t - float(getattr(stair.fol, '_stair_enter_s', t))
                        > 1.0):
                    stair.fol.mode = 'CRUISE'
                    stair.fol.decel_request = 0.0
                    stair.fol._stair_exit_xy = np.asarray(
                        pos2, dtype=np.float64).copy()
                    stair.fol._stair_exit_s = float(stair.s_cur)
                    print('[RL-DIAG] drift-abort cte=%.2f pos=(%.1f,%.1f)'
                          % (_cte, pos2[0], pos2[1]), flush=True)
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

            # 航线夹角门（用户指示）：只对爬升轴与航线夹角小的台阶
            # 生效——平台东角在 wp4 前 0.44m、爬升轴与航线差 60°+，
            # 属路径外障碍，EDGE/锁存/TK1/decel 不得把它当台阶骑
            # （round91 骑上角台 0.44m 窄条西沿侧翻、round98 角部
            # 减速对准 44s 死循环实测）
            _edge_route_ok = True
            if line is not None:
                _chh = stair.climb_heading
                if _chh is not None:
                    _raa = float(np.arctan2(np.sin(_chh - line_head),
                                            np.cos(_chh - line_head)))
                    if abs(_raa) > float(os.environ.get(
                            'S10_TK1_ROUTE_ANGLE', '0.45')):
                        _edge_route_ok = False
            # TK1：CRUISE 中检测到前方楼梯，只做“对准”，不改模式。
            # 减速由 SMppi 终点代价 + decel_request 负责；TK1 只在进入
            # 交付圈（S10_STAIR_ENTER_DIST）后补 1.5m/s 交付速度上限。
            if (os.environ.get('S10_TK1', '0') == '1'
                    and os.environ.get('S10_RL_ELEV', '0') == '1'
                    and stair.mode == 'CRUISE'
                    and _post_stair_xy is None
                    and float(body_pos[2]) <= 1.1
                    and dist_wp <= float(os.environ.get(
                        'S10_TK1_WP_MAX', '2.5'))
                    and abs(_cte) <= float(os.environ.get(
                        'S10_TK1_CTE_MAX', '0.8'))):
                # 用户流：SMppi 快到 wp -> TMppi 转向 -> 前进一点点后
                # TK1。所以 TK1 只在当前 wp 2.5m 内（转完向之后）
                # 才对准，不再提前 4m 把直线段蛇形爬行。
                th = stair.climb_heading
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

            # 前方航点：当前目标 wp 已被越过（投影过段末+0.5）时
            # 瞄下一航点——RL 斜向爬台后 next_idx 可能还是身后的
            # wp4，TK2 瞄它会把机器人拉回西行（round92 西漂 3.5m）
            _ahead = next_idx
            if next_idx > 0 and next_idx + 1 < len(wp):
                # 弧长判定：路径投影已越过当前 wp（RL 斜向爬台时
                # 直线段投影不前进——exit 点投影仅 4.56<len，
                # 而 s_cur 已越过 wp4 2m，投影法把 wp 判成在前方
                # （round95 TK2 瞄 wp4 拉机器人西转侧翻实测）
                _wsc = stair.wp_s(next_idx)
                if (_wsc is not None and stair.s_cur > _wsc + 0.2):
                    _ahead = next_idx + 1
            # TK2：STAIR→CRUISE 后立即对准下一航点；对齐后交回 TMppi/SMppi
            # 平顶（z>1）出口延迟 2s 让高站姿下蹲过渡先完成：
            # 台顶原地转+下蹲激起 roll 2.5rad/s 冲量卡死侧翻
            # （round210 实测）；低台出口（wp4/5 平台、两级台阶）
            # 立即对准不受影响（round211 全延迟破坏 wp5→6 实测）
            if (os.environ.get('S10_TK2', '0') == '1' and _tk2
                    and stair.mode == 'CRUISE'
                    and (_post_stair_t is None
                         or t - _post_stair_t > 2.0
                         or float(body_pos[2]) <= 1.15)):
                _correction += 'TK2'
                th2 = float(np.arctan2(wp[_ahead, 1] - body_pos[1],
                                       wp[_ahead, 0] - body_pos[0]))
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
                # RL 退出带 2.3m/s 动量且航向随机（实测西漂），
                # 交接先近停再对准，不给动量续命
                vx = min(vx, 0.2)
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
                    and _edge_route_ok
                    and float(body_pos[2]) <= 1.1
                    and float(np.max(terr[0:2])) - float(np.min(terr))
                    >= 0.08):
                if not _lip_hold:
                    _lip_g0 = float(np.min(terr))
                    _lip_xy = np.asarray(pos2, dtype=np.float64).copy()
                    _lip_wp_idx = next_idx
                    _lip_t0 = t
                _lip_hold = True
            _past_wp = (stair.wp_s(next_idx - 1) is not None
                        and stair.s_cur
                        > stair.wp_s(next_idx - 1) + 0.3)
            if (os.environ.get('S10_LIP_LATCH', '1') == '1'
                    and next_idx >= 4
                    and _edge_route_ok
                    and float(body_pos[2]) <= 1.1
                    and _past_wp
                    and stair.decel_request > 0.5
                    and stair.stair_ahead_dist is not None
                    and stair.stair_ahead_dist <= 1.2):
                if not _lip_hold:
                    _lip_g0 = float(np.min(terr))
                    _lip_wp_idx = next_idx
                    _lip_t0 = t
                    if os.environ.get('S10_LIP_DEBUG', '0') == '1':
                        print('[LATCH] set t=%.2f pos=(%.2f,%.2f) g0=%.2f'
                              % (t, pos2[0], pos2[1], _lip_g0), flush=True)
                _lip_hold = True
            if _lip_hold:
                # 骑坎滞留是台沿扭振侧滑的根源：前轮一上台就给
                # 1.2m/s 冲量把后轮拉过立面，骑坎时间压到 1-2s
                if (float(np.max(terr[0:2])) - float(np.min(terr)) >= 0.04
                        and (stair.drop_ahead_dist is None
                             or stair.drop_ahead_dist > 1.0)):
                    _lv = float(os.environ.get('S10_LIP_BURST_VX', '1.2'))
                    vx = _lv
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
                # 释放锚定 wp 推进点：推进过锁存时的 wp 且过点 0.3m
                # 才放（距离制释放过早，台顶压速磨对侧沿 wp5 实测）
                _rel = False
                if (_lip_wp_idx is not None
                        and _lip_wp_idx < len(wp)
                        and next_idx > _lip_wp_idx):
                    _dwp = float(np.linalg.norm(
                        np.asarray(pos2) - wp[_lip_wp_idx, :2]))
                    if _dwp >= 0.3:
                        _rel = True
                if _lip_t0 is not None and t - _lip_t0 > 25.0:
                    _rel = True   # 兜底：长时间未推进强放
                if _rel:
                    if os.environ.get('S10_LIP_DEBUG', '0') == '1':
                        print('[LATCH] release t=%.2f pos=(%.2f,%.2f)'
                              % (t, pos2[0], pos2[1]), flush=True)
                    _lip_hold = False
                    _lip_xy = None
                    _lip_wp_idx = None
                    _lip_t0 = None
                    _lip_rel_t = t
                    _edge_lift[0:2] = 0.0
                    _lip_grind_since = None
            if stair.decel_request > 0.0 and float(body_pos[2]) <= 1.1:
                if _tk1_t0 is None:
                    _tk1_t0 = t
                dv = float(os.environ.get('S10_ELEV_DECEL_VX', '2.0'))
                # 交付圈内压到 TK1 交付速度（1.5）：wp5-6 两级台阶
                # 前 decel 目标 2.0 高于交付门，机器人 1.67m/s 冲进
                # 台阶面 tip 后 STAIR 才接手（round103 实测）；
                # 圈外保持 2.0 供角部绕行速度
                _ad = stair.stair_ahead_dist
                if (_ad is not None and _ad <= float(os.environ.get(
                        'S10_STAIR_ENTER_DIST', '2.0'))
                        and _edge_route_ok):
                    # 压到交付门以下 0.3：贴门限交付时 RL 带转向动量
                    # 接手，六级楼梯西漂侧翻（round112 实测）；
                    # 只对航线对齐的楼梯压速——角部（路径外障碍）
                    # 压 1.2 会再现 round98 慢速绕角死循环
                    dv = min(dv, float(os.environ.get('S10_TK1_VX', '1.5'))
                             - 0.3)
                vx = vx * (1.0 - stair.decel_request) + dv * stair.decel_request
                v_ref = min(v_ref, vx)
            # 下行落差保护：前方检测到 >=0.08m 跌落沿时强制低速直行，
            # 避免高速下台栽头（下行不交 RL，先慢速爬行兜底）。
            # 轮下兜底：s 投影越过跌落后扫描变空（round93 台沿前
            # drops=[] 失保、0.6m/s 下台栽头实测）——后轮上台前轮
            # 已下的跨骑状态也强制低速直行
            # 下沿双窗口：om 锁止用大窗口（0.8m，防转弯动量带进
            # 沿口，round142 顶沿 om0.98 侧翻实测），vx 爬行用小窗口
            # （0.5m，1.2m 大窗口在平台角提前爬行致入口劣化 round144）
            _drop_om_w = float(os.environ.get(
                'S10_DROP_OM_LOOKAHEAD', '0.8'))
            _drop_straddle = (float(np.max(terr[2:4]))
                             - float(np.min(terr)) >= 0.08)
            # 平顶（z>1.2）s 投影假 drop 连续触发 18s（round199 实测
            # 平顶爬行 0.3 + roll 累积侧翻）——只保留轮下跨骑兜底，
            # 真下行台阶的跨骑保护不受影响
            _drop_s_ok = (float(body_pos[2]) <= 1.2)
            _drop_active = False
            if ((stair.drop_ahead_dist is not None
                    and _drop_s_ok
                    and stair.drop_ahead_dist < _drop_om_w)
                    or _drop_straddle):
                _drop_active = True
                if ((stair.drop_ahead_dist is not None
                        and _drop_s_ok
                        and stair.drop_ahead_dist < float(os.environ.get(
                            'S10_DROP_LOOKAHEAD', '2.0')))
                        or _drop_straddle):
                    vx = min(vx, float(os.environ.get(
                        'S10_DROP_VX', '0.3')))
                    vyaw = float(np.clip(vyaw, -0.5, 0.5))
                    v_ref = min(v_ref, vx)
                    _correction += 'DROP'
            # 机器人相对 riser 检测（路径扫描盲区：偏离路径时前方台阶）
            # 前方 0.3~1.2m 升高 0.08~0.25m => 正对直行低速骑上（不交 RL）
            _rf = float(os.environ.get('S10_EDGE_LOOKAHEAD', '1.2'))
            if (_rf > 0.0 and stair.mode == 'CRUISE'
                    and _edge_route_ok
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
                if (0.08 <= _rise_edge <= 0.25 and _flat_top
                        and _step_rise >= 0.08):
                    vx = min(vx, float(os.environ.get('S10_EDGE_VX', '0.6')))
                    v_ref = min(v_ref, vx)
                    _correction += 'EDGE'
                    # EDGE 锁存只认 >=10cm 的真台阶：坡顶 0.05 的
                    # 边界误锁 + om 强制正对 = 坡顶自旋侧翻（round46）
                    if (os.environ.get('S10_LIP_LATCH', '1') == '1'
                            and next_idx >= 4
                            and _past_wp
                            and _rise_edge >= 0.10):
                        if not _lip_hold:
                            _lip_g0 = float(np.min(terr))
                            _lip_xy = np.asarray(pos2, dtype=np.float64).copy()
                            _lip_wp_idx = next_idx
                            _lip_t0 = t
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

            # MPPI 状态用机体前向/侧向速度：此前直接喂世界系
            # body_vel[0]（x 分量）——wp0-1 航向 1.57 时世界 x 速度
            # 恒≈0，航向过 ±90° 时变负，规划器把前向速度看成
            # 0/负值，坡顶/凸包处计划莫名刹到 1.1（round157 实测
            # vref=4.0 但 cmd=1.12）
            state = np.asarray([pos2[0], pos2[1], yaw,
                                float(body_vel[0]), float(body_vel[1]),
                                qvel[5]])
            # 巡航规划器二选一：
            # TMppi：已在当前 wp 0.2m 内、实际速度<0.2、且下一段方向误差>10°
            # SMppi：其余所有 CRUISE 时间
            wp_next = wp[next_idx + 1] if next_idx + 1 < len(wp) else None
            used_turn, vx_c, om_c = tmppi.try_plan(
                pos2, yaw, float(np.linalg.norm(d.cvel[1][0:3])),
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
                # 平顶点转限速：om1.5 的原地转在 1.166 平顶激起
                # roll 0.61+（round193 wp7 出点转向实测），
                # 高台转用 0.6 慢转
                if float(body_pos[2]) > 1.2:
                    omcap = min(omcap, 0.6)
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
            _rg_hi = float(os.environ.get('S10_ROLL_GATE_HI', '0.34'))
            _rg_lo = float(os.environ.get('S10_ROLL_GATE_LO', '0.28'))
            # wp7->8 是 1.235 窄脊骑行（脊宽~0.7m，两侧 1.166），
            # 轮沿磕碰激起 roll 冲量（round221 实测 -4.2rad/s 侧翻）；
            # 0.22 提前门控在冲量不可逆前爬行保位（round219 该段
            # 0.22 下 19.6s 通过）；开敞平顶巡航 roll 摆动 ±0.2 会
            # 误触 0.22 成 drive-crawl 极限环（round219 wp10 后 90s
            # 不进点），故只在窄脊段收紧
            if float(body_pos[2]) > 1.0 and next_idx == 8:
                _rg_hi = min(_rg_hi, 0.22)
                _rg_lo = min(_rg_lo, 0.18)
            if abs(_body_roll) > _rg_hi:
                if not _roll_gate:
                    _roll_gate_since = t
                _roll_gate = True
            elif abs(_body_roll) < _rg_lo:
                _roll_gate = False
                _roll_gate_since = None
            if _roll_gate:
                # 允许 +/-0.3 慢转向脱困：此前 om=0 让机器人在
                # 台沿侧倾死锁（wp5 实测 30s 卡死）；roll 力矩
                # 主要靠 counter-roll 扶正（round121 六级楼梯底
                # om=0 时原地自旋 roll 2.23 侧翻实测）
                om_c = float(np.clip(om_c, -0.3, 0.3))
                vx_c = min(float(vx_c), 0.3)
                # 高台门控期不转向：慢转持续喂侧倾（round204 门控
                # 期 om-0.3 右转 yaw 漂 0.5rad、roll 卡 0.7 九秒不
                # 恢复、溜下台沿侧翻实测）；低台脱困转向保留
                if float(body_pos[2]) > 1.0:
                    om_c = 0.0
                # 死锁脱困：门控持续 >2s => 直线倒车离开台阶边缘。
                # 台沿锁存期不倒车：跨骑台阶时 roll 恒定超阈值，
                # 倒车轮子打滑卡死（wp5 实测）；交给抬轮把后轮
                # 拉上台面恢复水平。
                if (_roll_gate_since is not None
                        and t - _roll_gate_since > 2.0
                        and not _lip_hold
                        and float(body_pos[2]) <= 1.0
                        and float(np.max(terr)) - float(np.min(terr))
                        < 0.08
                        and stair.stair_ahead_dist is not None
                        and stair.stair_ahead_dist <= 1.5):
                    # 高台（平顶 z>1.0）不倒车：倒车在台沿反复
                    # 打滑把稳态侧倾推成侧翻（round137 平顶实测）；
                    # 跨骑（轮下高差>=0.08，如平台角窄条）也不倒车
                    # ——round167 角部倒车-回冲 50s 死循环实测；
                    # 平直段转向侧倾不倒车（round171 wp3 后倒车 7s
                    # 浪费——ad=2.4 无台阶可脱困）
                    vx_c = -0.4
                    om_c = 0.0
            # 楼梯后前 1.2m：只直线低速前进，禁止转向，避免平台边缘 yaw 反冲侧翻
            # 楼梯后：低倍率直接瞄当前目标 wp，直到距其足够近才放开
            if (_post_stair_xy is not None and _ahead < len(wp)
                    and (t - _post_stair_t < float(os.environ.get(
                        'S10_POSTSTAIR_HOLD_T', '2.5'))
                         or float(np.linalg.norm(
                            body_pos[:2] - wp[_ahead, :2]))
                         > float(os.environ.get(
                            'S10_POSTSTAIR_HOLD_DIST', '1.2')))):
                # 交接瞬态限速：RL 快走交还时 MPPI 平滑刹车太弱
                # （round175 台面 cmd0.2 实际 2.18->4.08m/s 加速），
                # 前 1.0s 直接零指令硬停 + 禁止转向（台面 om-0.96
                # 带角动量侧翻实测），之后 0.6 缓行过台面
                _hv = float(os.environ.get('S10_POSTSTAIR_HOLD_VX', '0.6'))
                if t - _post_stair_t < 1.0:
                    vx_c = 0.0
                    om_c = 0.0
                else:
                    vx_c = min(float(vx_c), _hv)
                _ph = float(np.arctan2(wp[_ahead, 1] - body_pos[1],
                                       wp[_ahead, 0] - body_pos[0]))
                _pe = float(np.arctan2(np.sin(_ph - yaw),
                                       np.cos(_ph - yaw)))
                # roll 门控期让位：hold 的 om 覆盖在门控之后，
                # 门控 om=0 被顶掉持续喂转向（round207 台顶
                # 交还 roll 0.76 卡 8.5s 实测根因）
                if not _roll_gate:
                    om_c = float(np.clip(0.5 * _pe, -0.2, 0.2))
            # 高台分档：1.166 平顶限 1.8m/s+om0.6（round190 平顶
            # 2.4-4.0m/s roll 累积 -0.95 侧翻实测）；2.0+ 的高台
            # 保持旧弱抓地限速
            if float(body_pos[2]) > 2.0:
                vx_c = min(float(vx_c), 0.8)
                omcap = min(omcap, 0.3)
            elif float(body_pos[2]) > 1.2:
                vx_c = min(float(vx_c), 1.8)
                omcap = min(omcap, 0.6)
            om_c = float(np.clip(om_c, -omcap, omcap))
            # 下沿/跨骑期 MPPI 的航向代价仍会给大 om（round142 两级
            # 台阶顶沿 om0.98 转向 roll-1.12 侧翻实测）：DROP 期
            # cmd 级限 ±0.5
            if _drop_active:
                om_c = float(np.clip(om_c, -0.5, 0.5))
            # 台沿锁存期：cmd 级强制正对路径航向（经 MPPI 的弱 guide
            # 被其它代价压过，wp5 实测贴沿航向漂 0.3rad 侧滑）
            # 平顶出口后 2s 内锁存转向也让位（round212 台顶锁存
            # om0.8 转向激起 roll-1.12 侧翻实测）；低台锁存照旧
            if (_lip_hold and not _roll_gate
                    and (_post_stair_t is None
                         or t - _post_stair_t > 2.0
                         or float(body_pos[2]) <= 1.15)):
                # 锁存期瞄当前 wp（不是航线航向）：爬升中若漂离航线，
                # 朝 wp 的方向自然把机器人拉回线上（round51 西漂 3.6m
                # 掉西沿实测）
                _thw = float(np.arctan2(wp[next_idx, 1] - body_pos[1],
                                          wp[next_idx, 0] - body_pos[0]))
                _ehl = float(np.arctan2(np.sin(_thw - yaw),
                                         np.cos(_thw - yaw)))
                om_c = float(np.clip(2.0 * _ehl, -1.0, 1.0))
                # 不再提前压速：锁存期 vx 交给 MPPI（跨骑冲量在上游）
            # 锁存释放后 1s 航向保持：上台面瞬间 MPPI 会立刻拉偏
            # 航向（台面顶自旋侧翻实测），先稳 1s 再交还
            if (_lip_rel_t is not None and not _roll_gate
                    and t - _lip_rel_t < 1.0
                    and (_post_stair_t is None
                         or t - _post_stair_t > 2.0
                         or float(body_pos[2]) <= 1.15)):
                _thr = float(np.arctan2(wp[next_idx, 1] - body_pos[1],
                                          wp[next_idx, 0] - body_pos[0]))
                _err = float(np.arctan2(np.sin(_thr - yaw),
                                         np.cos(_thr - yaw)))
                om_c = float(np.clip(1.5 * _err, -0.8, 0.8))
                vx_c = min(float(vx_c), 1.2)
            elif _lip_rel_t is not None and t - _lip_rel_t >= 1.0:
                _lip_rel_t = None
            # 大偏航纠偏：偏离航线 >1.5m 时 cmd 级直指当前 wp
            # （爬升后沿台壁西滑 90s 死锁实测——MPPI 弱 guide
            # 压不过平台边腿阻抗扭振）
            if (line is not None and abs(_cte) > 1.2
                    and stair.mode == 'CRUISE'
                    and not _roll_gate):
                _thc = float(np.arctan2(wp[next_idx, 1] - body_pos[1],
                                          wp[next_idx, 0] - body_pos[0]))
                _ecc = float(np.arctan2(np.sin(_thc - yaw),
                                         np.cos(_thc - yaw)))
                om_c = float(np.clip(2.0 * _ecc, -1.2, 1.2))
                vx_c = min(float(vx_c), 1.0)
            # TK 阶段（TK1 对准 / TK2 转出）直接执行对准转向：
            # MPPI 的 om 上限被 VMC_OM_CAP=1.0 压住，<1s 预算不够；
            # 低 vx 时 car_omega_limit≈3.0，TK 专用上限 2.0 安全。
            # 2026-08-19 修复：必须让位 roll 门控（此前 TK 直接给 om
            # 会覆盖门控的 om=0，平台边 TK2 侧倾正反馈翻车实测）。
            if ('TK1' in _correction or 'TK2' in _correction) \
                    and not _roll_gate:
                _om_tk = float(os.environ.get('S10_TK_OM_MAX', '2.0'))
                # TK 转向也守侧向加速度上限：2m/s 时 om1.5=3.0>1.8
                # 打滑致 roll 正反馈（round100 下台后 TK1 转侧翻）
                _om_tk = min(_om_tk, latmax / max(abs(vx_c), 0.5))
                # 平顶 TK 转向也守 0.6：round206 台顶交还 TK2
                # om1.5 原地快转激起 roll 0.65 卡死 4.7s 溜下台沿
                # 侧翻实测；0.6 慢转在 vx1.0 下侧向 0.6m/s^2 安全
                if float(body_pos[2]) > 1.2:
                    _om_tk = min(_om_tk, 0.4)
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
            # 机身高度钳制：保持 0.25（平顶 terr 统一 1.11，站高稳定；
            # round202 试 0.15：terr 1.15~1.24 跳动+台顶交还 roll 摇振
            # 不衰减侧翻）。ground_f 腾空误判已在 vmc_legs 高台分支修掉，
            # 这里不需要放大地形误差。
            if float(body_pos[2]) > 1.0:
                terr = np.minimum(terr, float(body_pos[2]) - 0.25)
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
        # v595 骑坎找平：任意轮间地形差 >=0.04 时把所有低轮参考抬到
        # 最高轮附近——消除骑坎对角扭振（平台沿逆指令左旋的根因），
        # 同时把后轮拉过立面。
        _tmax = float(np.max(terr))
        if (_tmax - float(np.min(terr)) >= 0.04
                and stair.mode == 'CRUISE'):
            terr = np.maximum(terr, _tmax - 0.02)
        _roll_tar_c = float(np.clip(
            -float(os.environ.get('S10_CAR_ROLL_K', '0.06')) * om_c
            * abs(vx_c),
            -float(os.environ.get('S10_CAR_ROLL_AMP', '0.06')),
            float(os.environ.get('S10_CAR_ROLL_AMP', '0.06'))))
        # 高台/弱抓地地形关闭压弯，优先防侧翻
        if float(body_pos[2]) > 1.0:
            _roll_tar_c = 0.0
        # roll 门控期主动反向压弯：门控只限 om/vx，roll 动量仍会
        # 把机器人推过侧翻点（round116 平顶转向 roll -0.86 实测）；
        # 目标反向偏置把 CarVMC 内部 roll 环推向扶正方向
        if _roll_gate:
            _roll_tar_c = float(np.clip(-2.0 * _body_roll, -0.25, 0.25))
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
                # 后轴不抬轮：摆腿只推车身不上轮，
                # 后轮登台交给 v595 地形参考抬升
                if _li >= 2:
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
                   roll_tar=_roll_tar_c, pitch_tar=0.0,
                   # round205 实测 70N 反力反而卡死：卸载侧轮被抬起
                   # 离地后 roll 力矩绕接触线失效（roll 0.66 悬停 4.7s），
                   # 回 40N（round201 同值可恢复）
                   step_lift=_step_lift, lift_swing=1.2,
                   yaw_scale=1.0 - 0.6 * _max_lift,
                   ridge_dist=99.0,
                   lift_f_scale=(0.3 if _max_lift > 0.05 else 1.0),
                   # 常驻 1.5cm 压轮：terr 轮下兜底（wheel_z-0.081）
                   # 自指引用——高度控制永远判轮在目标高、轮实际悬空 2mm
                   # 空转无牵引（round234 wp10 后卡死根因，探针实测 ncon=0）；
                   # 1.5cm 下压保证贴地（增法向 ~4.5N/轮，无副作用）
                   wheel_press=(0.1 if _max_lift > 0.05 else 0.015),
                   # 摇振抑制/轮滑钳制只在 wp10 后开敞平顶启用（卡死区）；
                   # 窄脊段(wp7-8)与 wp8-9 段保持基线（round227/229 实测
                   # 全域启用改变平顶动力学、窄脊骑偏掉台沿）
                   rock_kill=(1.0 if (float(body_pos[2]) > 1.2
                                       and next_idx >= 10) else 0.0))

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
        if stair.mode == 'STAIR' and stair.stair_first_heading is not None:
            # 每次 STAIR 入口更新航向目标：此前只在首次入口设置，
            # 两级台阶沿用平台的 1.65（差 0.13），策略爬完第一级
            # 西转（round177 实测）
            rl.set_heading(float(stair.stair_first_heading))
        if stair.mode == 'STAIR' and not _rl_diag_done:
            _lip_hold = False   # STAIR 接管，巡航抬轮锁存让位
            if _tk1_t0 is not None:
                print('[TK1] 减速+对准 %.2fs / 对准 %.2fs (预算 2.0/1.0s)'
                      % (t - _tk1_t0,
                         t - _tk1_align_t0 if _tk1_align_t0 is not None
                         else -1.0),
                      flush=True)
                _tk1_t0 = None
                _tk1_align_t0 = None
            _rl_diag_done = True
            # 六级楼梯给 RL 1.2m/s 速度目标：策略自由跑 2.85 时
            # 越顶退出 vx 门永远不满足、兜底交还 roll 动量带翻
            # （round139/148 平顶实测）；短台阶保持默认
            _nr0 = len(getattr(stair.fol, 'stair_rises_s', []) or [])
            if _nr0 >= 3:
                rl.set_cmd(1.2)
            # RL 速度目标保持策略默认（set_cmd 改写观测后两级台阶
            # 入口处策略南转 roll 1.30 侧翻 round134；round128 的
            # 两级台阶 wheel-clear 交还 vx 1.02 干净）
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
            spd = float(np.hypot(d.cvel[1][0], d.cvel[1][1]))
            print('[T] t=%.0f wp=%d pos=(%.1f,%.1f,%.2f) yaw=%.2f '
                  'spd=%.2f roll=%.2f cmd=(%.2f,%.2f) mode=%s corr=%s plan=%s '
                  'vref=%.2f dec=%.2f ad=%s '
                  'terr=%s'
                  % (t, next_idx, body_pos[0], body_pos[1], body_pos[2],
                     yaw, spd, roll, vx_c, om_c, stair.mode,
                     (_correction or 'NONE'), _planner,
                     v_ref, round(stair.decel_request, 2),
                     None if stair.stair_ahead_dist is None
                     else round(stair.stair_ahead_dist, 2),
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
            _dbg_roll = float(np.arctan2(
                2.0 * (d.xquat[1][0] * d.xquat[1][1]
                       + d.xquat[1][2] * d.xquat[1][3]),
                1.0 - 2.0 * (d.xquat[1][1] ** 2 + d.xquat[1][2] ** 2)))
            _f2d = np.array([
                1.0 - 2.0 * (d.xquat[1][2] ** 2 + d.xquat[1][3] ** 2),
                2.0 * (d.xquat[1][1] * d.xquat[1][2]
                       + d.xquat[1][0] * d.xquat[1][3]), 0.0])
            _vx_b_debug = float(np.dot(d.qvel[0:3], _f2d))
            traj.append([t, body_pos[0], body_pos[1], body_pos[2], yaw,
                         float(next_idx),
                         float(np.hypot(d.cvel[1][0], d.cvel[1][1])),
                         1.0 if stair.mode == 'STAIR' else 0.0,
                         _dbg_roll, float(d.qvel[3]), om_c, vx_c,
                         float(np.median(terr)), _roll_tar_c,
                         float(wheel_xyz[0, 2]), float(wheel_xyz[1, 2]),
                         float(wheel_xyz[2, 2]), float(wheel_xyz[3, 2]),
                         float(d.qvel[WHEEL_QV_IDX[0]]),
                         float(d.qvel[WHEEL_QV_IDX[2]]),
                         float(tau[3]), float(tau[7]),
                         float(vx_c), float(_vx_b_debug)])

        # 航点推进：只按原始折线水平距离；到点判（未越过）要求
        # 对准下一段 + 角速度收敛才推点（防原地转完成前抢跑）；
        # 过点兜底：刚越过（<=len+0.8，仍可能原地转中）同样要求
        # 对准；只有明显越过后才跳过对准直接推点。
        if next_idx < len(wp):
            # 平顶判点半径放大：1.166 平顶机器人东偏 1.2m 绕圈
            # 20s+（0.5m 判点圆够不着、s 投影差 1.9m，round197/198
            # 实测），z>1.2 用 1.5m 判点圆
            _adv_r = 2.5 if float(body_pos[2]) > 1.2 else None
            _arr = nav.reached(next_idx, d.xpos[1][:2], radius=_adv_r)
            # s 弧长兜底删除：路径跟随器 s_cur 会跑飞（round218 实测
            # s_cur=96.6 而机器人真实路径位置 60——绕圈爬行期间 s 持续
            # 积分），配合 4m 门仍把 wp8/wp9 在 4m 外推掉；现代代码
            # 楼梯出口都在 wp 2.5m 内（六梯顶 wp7 出口 0.6m），
            # 判点半径覆盖即可，不再需要弧长兜底
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
            # 楼梯顶刚交还时跳过对准门：wp7 就在六级楼梯顶沿上
            # （距顶沿 0.15m），原地转对 wp8 方向（西 2.81）会在
            # 台沿原地转 10s+，roll 门反复触发后翻（round111 实测）；
            # 先推点让机器人在平顶上边开边转
            if _post_stair_xy is not None:
                _align_ok = True
            # 平顶判点也跳对准门：1.5m 判点圆内 MPPI 终点代价
            # 拉着绕圈（round215 wp9 圈 130s 不推点实测——到点
            # 判点圆需对齐下一段航向，但 MPPI 目标仍是当前 wp，
            # 无理由转向，极限环死锁）；平顶边开边转（同楼梯
            # 顶 round111 逻辑）
            if float(body_pos[2]) > 1.2:
                _align_ok = True
            if _arr and _align_ok:
                if next_idx == 0 and t_start is None:
                    t_start = t
                wp_times[next_idx] = t
                last_adv_t = t
                print('[T] wp%d @ t=%.2f' % (next_idx, t), flush=True)
                print('[ADV] wp%d dist=%.2f s_cur=%.2f wp_s=%s radius=%s'
                      % (next_idx,
                         float(np.linalg.norm(pos2 - wp[next_idx, :2])),
                         float(stair.s_cur),
                         (round(float(stair.wp_s(next_idx)), 2)
                          if stair.wp_s(next_idx) is not None else None),
                         _adv_r), flush=True)
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
