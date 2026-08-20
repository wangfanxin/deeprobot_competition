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
    _terr_raw = np.zeros(4, dtype=np.float64)
    _prev_bz = None
    _edge_lift = np.zeros(4)
    _vyaw_f = 0.0
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

            if stair.mode != 'STAIR' and _tk2_was_stair:
                _post_stair_xy = np.asarray(pos2, dtype=np.float64)
                _post_stair_t = t
                if os.environ.get('S10_TK2', '0') == '1':
                    _tk2 = True
                    _tk2_t0 = t
            _tk2_was_stair = (stair.mode == 'STAIR')

            if os.environ.get('S10_RL_ELEV', '0') == '1':
                rxy, rtops = perc.riser_table(stair)
                if (rxy is not None and len(rxy) >= 1
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
                    rtops = np.append(rtops, rtops[-1])
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
                # 航段后方（投影 <0，如 wp11 推进后机器人仍在段起点北侧
                # 3.9m）时直接瞄段起点 wp——此前只按段航向+cte 饱和
                # 拉西，机器人斜切走廊撞上 x=-4.79 柱（round239 wp11→12
                # 侧翻实测）；瞄起点先南下进入走廊再沿线西行
                _proj_cur = float(np.dot(_rel, _seg)) / _segl
                # 仅平顶启用：低台交还/弯道推进后的瞬时后方态瞄起点会
                # 改变各段进近动力学（round240 实测 wp1-3 全面劣化+
                # wp3→4 侧翻）。条件放宽到段首 1m 内且横向 >0.8：
                # wp12 推进后机器人东偏 2.1m（proj=0.13），只按段航向
                # +cte 饱和会斜切台角掉下南沿（round241 实测侧翻），
                # 瞄起点先横移回台阶顶再下行
                if ((_proj_cur < -1.0
                     or (0.0 <= _proj_cur < 1.0 and abs(_cte) > 0.8))
                        and float(body_pos[2]) > 1.2
                        and float(np.linalg.norm(
                            pos2 - np.asarray(line['start'])[:2])) > 2.0):
                    _des = float(np.arctan2(line['start'][1] - pos2[1],
                                            line['start'][0] - pos2[0]))
                if dist_wp < 0.5:
                    _des = float(np.arctan2(line['end'][1] - pos2[1],
                                            line['end'][0] - pos2[0]))
                # over-point swing: past wp by 0.2m, dist<1.5m -> aim at next wp
                # so the nose turns before the advance. Gated on no-stair-ahead:
                # with the wp-clip the platform stair is invisible pre-advance,
                # so this fires at wp2/wp3 and aligns the next leg; once the stair
                # is visible (post-advance) TK1 owns the approach.
                if (stair.mode == 'CRUISE'
                        and stair.stair_ahead_dist is None
                        and next_idx + 1 < len(wp)
                        and _proj_cur > _segl + 0.2
                        and dist_wp < 1.5):
                    _des = float(np.arctan2(wp[next_idx + 1, 1] - pos2[1],
                                            wp[next_idx + 1, 0] - pos2[0]))
                    vx = min(vx, float(os.environ.get('S10_WP_SWING_VX',
                                                     '1.2')))
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
            # TMppi 触发镜像（用户指示）：TK1/TK2 只在 SMppi 模式下
            # 进入；TMppi 只与 SMppi 互切，不切 TK1/TK2
            _tmppi_will = False
            if next_idx + 1 < len(wp):
                _tmppi_will = tmppi.will_fire(
                    pos2, yaw, float(np.linalg.norm(d.cvel[1][0:3])),
                    wp[next_idx], wp[next_idx + 1], wide=_tk2)
            if (os.environ.get('S10_TK1', '0') == '1'
                    and os.environ.get('S10_RL_ELEV', '0') == '1'
                    and stair.mode == 'CRUISE'
                    and _post_stair_xy is None
                    and not _tmppi_will
                    and _edge_route_ok
                    and dist_wp <= float(os.environ.get(
                        'S10_TK1_WP_MAX', '2.5'))
                    and float(body_pos[2]) <= 1.1
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
            # 平顶（z>1）出口延迟让高站姿下蹲过渡先完成：
            # 台顶原地转+下蹲激起 roll 2.5rad/s 冲量卡死侧翻
            # （round210 实测）；低台出口立即对准（1.5 提速实验
            # 致窄脊骑偏掉脊 round263/265 实测，回 2.0）
            if (os.environ.get('S10_TK2', '0') == '1' and _tk2
                    and stair.mode == 'CRUISE'
                    and not _tmppi_will
                    and (_post_stair_t is None
                         or t - _post_stair_t > 0.5)):
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

            # 楼梯交还状态超时清除：保留状态生命周期供 TK1 门/TK2 延迟/
            # PRETRANS 退出混合使用（hold 动作已按用户指示删除）
            if (_post_stair_t is not None
                    and t - _post_stair_t > float(os.environ.get(
                        'S10_POST_STAIR_MAX_T', '5.0'))):
                _post_stair_xy = None
                _post_stair_t = None

            v_ref = vx
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
            # 楼梯逼近速度全局有界（统一化替代原高台分档）：前方 2m 内
            # 有楼梯且未跨骑时交付 ≤1.5，STAIR 接管不超速（round314 窄脊顶
            # cmd 3.94 → RL 冲 6.94m/s 侧翻）；跨骑=正在登阶入口，保持
            # 275 式动量（round322 入口 2.26m/s 时 RL yaw 突跳 2.43 侧翻）
            if (stair.stair_ahead_dist is not None
                    and stair.stair_ahead_dist <= 2.0):
                vx = min(vx, float(os.environ.get(
                    'S10_STAIR_APPROACH_VX', '1.5')))
                v_ref = min(v_ref, vx)
            # 机器人相对 riser 检测（路径扫描盲区：偏离路径时前方台阶）
            # 前方 0.3~1.2m 升高 0.08~0.25m => 正对直行低速骑上（不交 RL）
            _rf = float(os.environ.get('S10_EDGE_LOOKAHEAD', '1.2'))
            if (_rf > 0.0 and stair.mode == 'CRUISE'
                    and _edge_route_ok
                    and float(body_pos[2]) <= 1.1
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
                float(qvel[5]), wp[next_idx], wp_next, wide=_tk2)
            if used_turn:
                _planner = 'TMppi'
            else:
                _planner = 'SMppi'
                vx_c, om_c = smppi.plan(state, ref_path, v_ref, prev_u,
                                        float(vyaw), wp_dx=_wp_dx)
                _om_raw_plan = float(om_c)
            latmax = float(os.environ.get('S10_AUTO_LAT_MAX', '1.8'))
            if used_turn:
                omcap = min(float(os.environ.get('S10_TURN_OM_MAX', '3.0')),
                            latmax / max(
                                float(np.linalg.norm(d.cvel[1][0:3])), 0.5))
            else:
                omcap = min(float(os.environ.get('S10_VMC_OM_CAP', '2.0')),
                            latmax / max(abs(vx_c), 0.5))
            _roll_gate = False  # roll门控已删除（用户指示：速度够快可落地，CarVMC边界待探索）

            om_c = float(np.clip(om_c, -omcap, omcap))
            # TK 阶段（TK1 对准 / TK2 转出）直接执行对准转向：
            # MPPI 的 om 上限被 VMC_OM_CAP=1.0 压住，<1s 预算不够；
            # 低 vx 时 car_omega_limit≈3.0，TK 专用上限 2.0 安全。
            # 2026-08-19 修复：必须让位 roll 门控（此前 TK 直接给 om
            # 会覆盖门控的 om=0，平台边 TK2 侧倾正反馈翻车实测）。
            if ('TK1' in _correction or 'TK2' in _correction) \
                    and not _roll_gate and not used_turn:
                _om_tk = float(os.environ.get('S10_TK_OM_MAX', '2.0'))
                # TK 转向也守侧向加速度上限：2m/s 时 om1.5=3.0>1.8
                # 打滑致 roll 正反馈（round100 下台后 TK1 转侧翻）
                _om_tk = min(_om_tk, latmax / max(
                    float(np.linalg.norm(d.cvel[1][0:3])), 0.5))
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
        _step_lift = np.zeros(4, dtype=np.float64)
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
        if os.environ.get('S10_DUMP_TICKS', '0') == '1' and t < 7.0:
            print('[DUMP] t=%.4f x=%.6f y=%.6f z=%.6f yaw=%.6f bvx=%.6f bvy=%.6f qv5=%.6f vyaw=%.6f vxl=%.6f vref=%.6f dist=%.6f cte=%.6f des=%.6f herr=%.6f omc=%.6f vxc=%.6f omraw=%.6f prevu=%.6f,%.6f roll=%.6f gate=%d corr=%s ut=%d'
                  % (t, float(pos2[0]), float(pos2[1]), float(body_pos[2]),
                     float(yaw), float(body_vel[0]), float(body_vel[1]),
                     float(qvel[5]), float(vyaw), float(vx), float(v_ref),
                     float(dist_wp), float(_cte), float(_des), float(head_err),
                     float(om_c), float(vx_c), float(_om_raw_plan), float(prev_u[0]), float(prev_u[1]),
                     float(_body_roll), int(_roll_gate), _correction, int(used_turn)),
                  flush=True)
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
                   wheel_press=(0.1 if _max_lift > 0.05 else 0.0),
                   rock_kill=0.0)

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
                if _tk2:
                    # 用户指示：TK2 转体期间姿势原地切回巡航半蹲
                    tpr = 0.0
                else:
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
            if _tk1_t0 is not None:
                print('[TK1] 减速+对准 %.2fs / 对准 %.2fs (预算 2.0/1.0s)'
                      % (t - _tk1_t0,
                         t - _tk1_align_t0 if _tk1_align_t0 is not None
                         else -1.0),
                      flush=True)
                _tk1_t0 = None
                _tk1_align_t0 = None
            if os.environ.get('S10_RISER_DEBUG', '0') == '1':
                _rxy, _rtops = perc.riser_table(stair)
                print('[RISER-ENTRY] n_s=%d tops=%s table_n=%d tops_t=%s'
                      % (len(getattr(stair.fol, 'stair_rises_s', []) or []),
                         [round(float(h), 2) for h in
                          getattr(stair.fol, '_elev_rises_tops', []) or []],
                         len(_rxy) if _rxy is not None else 0,
                         None if _rtops is None else
                         [round(float(h), 2) for h in _rtops]), flush=True)
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
        if (stair.mode == 'STAIR'
                and os.environ.get('S10_RISER_DEBUG', '0') == '1'
                and int(t * 2) % 2 == 0):
            _rxy, _rtops = perc.riser_table(stair)
            print('[RISERS] t=%.2f n_s=%d tops=%s table_n=%d tops_t=%s'
                  % (t, len(getattr(stair.fol, 'stair_rises_s', []) or []),
                     [round(float(h), 2) for h in
                      getattr(stair.fol, '_elev_rises_tops', []) or []],
                     len(_rxy) if _rxy is not None else 0,
                     None if _rtops is None else
                     [round(float(h), 2) for h in _rtops]), flush=True)
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
        if os.environ.get('S10_DUMP_STOP', '0') == '1' and t > 6.5:
            print('[T] dump-stop t=%.2f' % t, flush=True)
            break
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
                     np.round(terr, 2)),
                  ' dA=%s dS=%s dD=%s'
                  % (None if stair.drop_ahead_dist is None
                     else round(stair.drop_ahead_dist, 2),
                     [round(float(x), 1) for x in
                      getattr(stair.fol, '_elev_drops', [])][:3],
                     [round(float(dh), 2) for d, dh in
                      getattr(stair.fol, '_elev_drop_ds', [])][:3]),
                  flush=True)
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

        # 航点推进：纯半径判点（S10_WP_ADVANCE_DIST=0.2），无对准门/
        # 过点兜底/悬停豁免/post-stair 豁免（用户指示：判点由
        # SMppi+TMppi 到点能力达成，只保留 0.2m 半径判定）
        if next_idx < len(wp):
            if nav.reached(next_idx, pos2, radius=None):
                if next_idx == 0 and t_start is None:
                    t_start = t
                wp_times[next_idx] = t
                last_adv_t = t
                print('[T] wp%d @ t=%.2f' % (next_idx, t), flush=True)
                print('[ADV] wp%d dist=%.2f'
                      % (next_idx,
                         float(np.linalg.norm(
                             pos2 - wp[next_idx, :2]))), flush=True)
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
