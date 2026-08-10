"""cruise_vmc_noros.py — v218 方案无 ROS 独立测试（wp0→MAX_WP，原始赛道）。

结构：导航（AutoNavFollower pursuit/vlim）→ 身体层 MPPI [vx,ω]（20Hz）
     → VMC/阻抗腿层（200Hz）→ mujoco。
已知地图：地形高 mj_ray 逐轮查询；横脊预扫描 dh>0.12 → 弧长表 → 抬轮前馈。
"""
import os, sys, time
import numpy as np
import mujoco
try:
    import mujoco.viewer
    _HAS_VIEWER = True
except Exception:
    _HAS_VIEWER = False

PKG = '/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy'
sys.path.insert(0, PKG)
from s10_mpc.auto_nav import AutoNavFollower
from s10_mpc.body_mppi import BodyMPPI
from s10_mpc.vmc_legs import (VMCController, CarVMC, LEG_ATTACH, WHEEL_BODY,
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

    _viewer = None
    if os.environ.get('S10_USE_VIEWER', '0') == '1' and _HAS_VIEWER:
        try:
            _tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
            _viewer = mujoco.viewer.launch_passive(m, d)
            with _viewer.lock():
                _viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                _viewer.cam.trackbodyid = _tb if _tb >= 0 else 1
            print('[VMC] viewer 已打开（关窗口即停止）', flush=True)
        except Exception as _e:
            print('[VMC] viewer 打开失败，无头运行:', _e, flush=True)
            _viewer = None

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
    # v236: 台阶几何预扫描——wp6->7 楼梯区 riser 弧长表（已知地图，供
    # 相位步态）。wp5->6 台阶间距 2m 与相位窗不匹配（v447 卡第一级），
    # 仍由连续抬轮处理。
    stair_risers = []
    try:
        _s6 = float(fol.path_wp_s[6]); _s7 = float(fol.path_wp_s[7])
        stair_risers = [(sr, dhv) for (sr, dhv) in ridge_arcs
                        if _s6 <= sr <= _s7 and dhv >= 0.09]
        print(f'[VMC] 楼梯区 riser {len(stair_risers)} 处', flush=True)
    except Exception as e:
        print('[VMC] 楼梯预扫描失败', e, flush=True)
    # v247: 横脊世界坐标表（供物理距离触发，替代 s_cur 投影——转向时 s_cur
    # 滞后导致步态漏触发，wp4→5 撞脊翻车实测）
    ridge_world = []
    try:
        for (sr, dhv) in ridge_arcs:
            _k = int(np.searchsorted(fol.path_cum, sr, side='right') - 1)
            _k = min(max(_k, 0), len(fol.path_pts) - 1)
            _pt = fol.path_pts[_k, :2].copy()
            # v259: 存脊点处路径切线（脊近似垂直路径；距离沿切线投影，
            # 忽略横向偏移——狗偏西 0.5m 时欧氏距离误判 0.44m 不触发抬轮）
            _tng = np.zeros(2)
            if _k < len(fol.path_heading):
                _th = float(fol.path_heading[_k])
                _tng = np.array([np.cos(_th), np.sin(_th)])
            ridge_world.append((_pt, _tng, sr, dhv))
    except Exception as e:
        print('[VMC] 横脊坐标表失败', e, flush=True)

    mppi = BodyMPPI(
        N=int(os.environ.get('VMC_MPPI_N', '4096')),
        H=int(os.environ.get('VMC_MPPI_H', '40')),
        vx_max=float(os.environ.get('S10_AUTO_VMAX', '5.0')))
    if os.environ.get('S10_VMC_MODE', 'wbc') == 'pd':
        from s10_mpc.vmc_legs import LegPDDrive
        vmc = LegPDDrive()
        print('[VMC] LegPDDrive 模式（腿锁蹲姿+轮驱动）', flush=True)
    elif os.environ.get('S10_VMC_MODE', 'wbc') == 'car':
        vmc = CarVMC()
        print('[VMC] CarVMC 模式（车化：轮驱动/差速，腿=主动悬架姿态）', flush=True)
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
        _lupd = -1.0
        def terrain_at(x, y):
            nonlocal _lupd
            if t - _lupd >= 0.1:
                lterr.update()
                _lupd = t
            _h = lterr.height(x, y)
            # v275: 高架伪影抑制——lidar 在起步坡看到上方高架盒底面
            # （2.16m 读数，实测抬轮误触发/腿阻抗过激侧翻）；读数高于
            # 机体+1.0m 视为伪影，用运动学地面（机高-0.55）兜底
            if _h > body_pos[2] + 1.0:
                return float(body_pos[2] - 0.55)
            return _h
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
    _terr_f = None
    # v292: 力矩合规统计（腿 ±50 Nm / 轮 ±14 Nm，连续超限>0.5s 不合格）
    _max_tau_leg = 0.0
    _max_tau_wh = 0.0
    _over_run = 0.0
    _over_worst = 0.0
    _over_total = 0.0
    while t < MAX_SIM:
        qpos = np.asarray(d.qpos, dtype=np.float64)
        qvel = np.asarray(d.qvel, dtype=np.float64)
        body_pos = d.xpos[1]
        quat = d.xquat[1]
        yaw = float(np.arctan2(
            2.0*(quat[3]*quat[0]+quat[1]*quat[2]),
            1.0-2.0*(quat[2]**2+quat[3]**2)))
        # v291d: continuous roll safety envelope for crest-driven rear
        # lift boost (banked approach must not get extra rear lift).
        _roll_now = float(np.arctan2(
            2.0*(quat[0]*quat[1]+quat[2]*quat[3]),
            1.0-2.0*(quat[1]**2+quat[2]**2)))
        _roll_env = float(np.clip((0.35 - abs(_roll_now)) / 0.15, 0.0, 1.0))
        wheel_xyz = np.asarray([d.xpos[WHEEL_BODY[i]] for i in range(4)])
        wheel_vel = np.asarray([d.cvel[WHEEL_BODY[i]][0:3] for i in range(4)])

        # 导航 + MPPI 更新率（S10_NAV_HZ，默认 2）
        # v437: 原 int(t*20)%10==0 实为 2Hz（0.5s 一次）——方向翻转后 0.5s
        # 才更新，2.5m/s 下已转过 0.75rad，过弯振荡的直接放大器。
        # MPPI N=4096/H=40 单次 89ms → 实际上限 ~10Hz；默认 2 保 v410。
        _nav_period = max(1, int(round(200.0 / float(os.environ.get(
            'S10_NAV_HZ', '2')))))
        if int(t * 200) % _nav_period == 0 and (
                dbg == 0 or t - last_log >= 0.05):
            pos2 = body_pos[:2]
            # v462: 双模式判定（此前从未调用——STAIR 技能从不激活，wp6→7
            # 楼梯全程用巡航参数）
            try:
                fol.update_mode(pos2, next_idx, yaw=yaw)
            except Exception:
                pass
            vx, vyaw = fol.compute_cmd(
                pos2, yaw, next_idx,
                robot_z=float(body_pos[2]), yaw_rate=float(qvel[5]))
            v_ref = fol._last_vlim
            # 路径参考轨迹（弧长采样：当前位置起 8m，步长 0.5m）
            _ref = []
            _s0 = float(fol._s_cur)
            for _ds in np.arange(0.0, 12.0, 0.5):   # v340: H=40 视界 7m，ref 采到 12m
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
            # v316: 瞄航点期间 MPPI 参考 = 指向航点的光束（狗→wp→出口延长）。
            # 纯路径参考的航向成本会把狗往路径航向（过弯后朝西）拉，错过
            # wp1 北侧 0.6m（实测）；瞄航点（_last_tgt==当前 wp）时让 MPPI
            # 以 wp 为终点规划，过点后恢复路径参考。
            try:
                _aim_tgt = np.asarray(getattr(fol, '_last_tgt', [0, 0, 0])[:2])
                _aim_wp = np.asarray(fol.wp[next_idx][:2])
                _beam_on = (np.abs(_aim_tgt[0] - _aim_wp[0]) < 0.05
                            and np.abs(_aim_tgt[1] - _aim_wp[1]) < 0.05)
                # v327: 过点后 3m 内保持"上一航点出口光束"——过点瞬间导航
                # 目标切到前视点，err 突跳 -> 出弯蛇形（wp1->wp2 7s 实测）。
                # 用 wp[i-1] 的出口方向（=wp[i]-wp[i-1]）拉直出弯段。
                _aim_idx = next_idx
                _after_len = float(os.environ.get(
                    'S10_AUTO_BEAM_AFTER', '3.0'))
                # v329: 过点后光束只对真实弯道（wp1 之后）生效——起点
                # wp0 也在 3m 内会误触发，改变全链路 ref 导致混沌发散。
                if (not _beam_on and next_idx >= 2):
                    _pw = np.asarray(fol.wp[next_idx - 1][:2])
                    _dd = float(np.hypot(pos2[0] - _pw[0],
                                         pos2[1] - _pw[1]))
                    if _dd < _after_len:
                        _beam_on = True
                        _aim_wp = _pw
                        _aim_idx = next_idx - 1
                if _beam_on:
                    _dvec = _aim_wp - pos2
                    _L = float(np.hypot(_dvec[0], _dvec[1]))
                    # v322: 出口方向 = 过点后的路径方向（wp[i+1]-wp[i]），
                    # 不是狗→wp 的接近方向——wp1 处接近方向指向西南（南坡）。
                    _ex = np.asarray(fol.wp[_aim_idx + 1][:2]) - _aim_wp
                    _Le = float(np.hypot(_ex[0], _ex[1]))
                    _uv = (_ex / _Le) if _Le > 1e-3 else (
                        _dvec / _L if _L > 1e-3 else np.array([1.0, 0.0]))
                    _hd = float(np.arctan2(_uv[1], _uv[0]))
                    # 局部零代价陷阱：不能放狗当前位置/航向点（最近点距离=0、
                    # 航向=当前 yaw -> MPPI 视为已在目标上，om 恒 0）。只给
                    # wp + 出口延长两点。过点后光束用更长出口拉直出弯。
                    if _aim_idx == next_idx - 1:
                        _exit_len = float(os.environ.get(
                            'S10_AUTO_BEAM_AFTER_EXIT', '2.5'))
                    else:
                        _exit_len = float(os.environ.get(
                            'S10_AUTO_BEAM_EXIT', '1.0'))
                    # v434: 光束起点投影（S10_AUTO_BEAM_PROJ 默认开）——狗沿
                    # 出口方向已越过 wp 时（如 wp3→4 狗偏北 1.5m），原光束从
                    # wp 起算，最近点=身后 wp → MPPI 距离成本往回拉（绕圈
                    # 打转 wp3→4 实测）；把起点平移到狗在出口线上的投影，
                    # 距离成本只剩垂直拉线（走廊效应），转向交给 guide 主导。
                    _bproj = float(os.environ.get('S10_AUTO_BEAM_PROJ', '0'))
                    _bs = np.array([_aim_wp[0], _aim_wp[1]])
                    if _bproj > 0.0:
                        _pp = float(np.dot(pos2 - _bs, _uv))
                        if _pp > 0.0:
                            _bs = _bs + _pp * _uv
                    # v439: 过点后光束指向下一航点（S10_AUTO_BEAM_AIM_NEXT
                    # 默认 0）——wp4→5 实测：出口光束沿 wp4 出口线向北
                    # (x=-15.02)，狗在 x=-15.3 偏西 0.28m，距离成本把狗往东
                    # 拉，与"右转去 wp5"的 guide 打架 → MPPI 输出 om≈0 直线
                    # 西行错过 wp5。改为从狗前方 0.5m 指向 wp[next_idx]
                    # （方向=guide 方向，距离成本无横向偏置；0.5m 前移避免
                    # v318 局部零代价陷阱）。连续量，仅过点后分支。
                    if (_aim_idx == next_idx - 1 and float(os.environ.get(
                            'S10_AUTO_BEAM_AIM_NEXT', '0')) > 0):
                        _nv = np.asarray(fol.wp[next_idx][:2]) - pos2
                        _Lv = float(np.hypot(_nv[0], _nv[1]))
                        if _Lv > 1e-3:
                            _uv = _nv / _Lv
                        _hd = float(np.arctan2(_uv[1], _uv[0]))
                        _bs = pos2 + 0.5 * _uv
                        _exit_len = float(os.environ.get(
                            'S10_AUTO_BEAM_AFTER_EXIT', '2.5'))
                    _ref = np.array([
                        [_bs[0], _bs[1], _hd],
                        [_bs[0] + _exit_len * _uv[0],
                         _bs[1] + _exit_len * _uv[1], _hd]])
            except Exception:
                pass
            st = np.array([pos2[0], pos2[1], yaw,
                           float(d.cvel[1][3]), float(d.cvel[1][4]), float(qvel[5])])
            if os.environ.get('S10_NAV_DEBUG', '0') == '1' and next_idx <= 6:
                print('[NAV] t=%.1f pos=(%.2f,%.2f) yaw=%.2f err=%.2f '
                      'tgt=(%.2f,%.2f) s_cur=%.2f vyaw=%.2f cte=%.2f'
                      % (t, body_pos[0], body_pos[1], yaw,
                         getattr(fol, '_last_err', 0.0),
                         getattr(fol, '_last_tgt', [0, 0, 0])[0],
                         getattr(fol, '_last_tgt', [0, 0, 0])[1],
                         getattr(fol, '_s_cur', 0.0), vyaw,
                         getattr(fol, '_last_cte', 0.0)), flush=True)
            if os.environ.get('S10_VMC_USE_NAV', '0') == '1':
                vx_c, om_c = vx, vyaw   # 直接导航指令（无 MPPI 随机性）
            else:
                # v270: MPPI 采样中心加曲率前馈 κ·v_ref（导航放开、MPPI
                # 约束兜底；样本围绕正确转向率，约束仍在摩擦锥内）
                # v315: MPPI 采样中心 = 导航完整转向指令（err + 曲率FF + cte）
                # ——纯路径跟踪会切内弯错过航点（wp1 最近 0.6m 实测）；导航的
                # 瞄航点逻辑保证 0.3m 判点，MPPI 负责平滑 + 摩擦锥约束兜底。
                _g_om = float(vyaw) if vyaw is not None else 0.0
                vx_c, om_c = mppi.plan(
                    st, _ref, v_ref, prev_u, guide_om=_g_om)
                if (os.environ.get('S10_NAV_DEBUG', '0') == '1'
                        and next_idx <= 6):
                    print('[MPPI] g_om=%.2f out=(%.2f,%.2f) vref=%.2f'
                          % (_g_om, vx_c, om_c, v_ref), flush=True)
            # v218p: omega 上限匹配 VMC yaw 能力（防指令远超执行导致振荡）
            # v245: 速度相关上限——横向加速度包线 a_lat=ω·v 防高速大 ω 侧翻
            # （实测 YAW_TMAX 滑移权威下 ω 可达 3.6+，v=1.9 时 a_lat 7m/s2 翻车）
            _omcap = float(os.environ.get("S10_VMC_OM_CAP", "0.5"))
            _latmax = float(os.environ.get("S10_AUTO_LAT_MAX", "5.0"))
            _omcap = min(_omcap, _latmax / max(abs(vx_c), 0.5))
            om_c = float(np.clip(om_c, -_omcap, _omcap))
            # v263: 回退 v261/v262 脊前对准（起步被扰动 + wp4→5 仍不稳，
            # 脆）。过脊靠地形前瞻（ERR_GATE 提高后转弯中保持生效）+ 慢速。
            # 近脊仅保留速度相关 omcap（v245）。
            prev_u = np.array([vx_c, om_c])
            last_log = t
        else:
            vx_c, om_c = prev_u

        # 地形 + 前瞻（v222: 连续地形响应，无门控）——
        # 每轮高程取"轮前方 _lk 处"，腿垂直阻抗连续跟随前方地形：
        # 接近横脊时高程升 → 腿伸 → 轮抬，过脊自然回落
        _lk = float(os.environ.get('S10_VMC_TERRAIN_LOOKAHEAD', '0.0'))
        if _lk > 0.0:
            _bx = d.xmat[1][0]
            _by = d.xmat[1][3]
            _bn = float(np.hypot(_bx, _by)) + 1e-9
            _bx, _by = _bx / _bn, _by / _bn
            # v223g: 前瞻权重随导航 yaw 误差连续衰减——弯道未完成(err大)
            # 前瞻≈0(防转向中抬轮失控)，直线对准(err小)前瞻满(过脊抬轮)
            _w = float(os.environ.get('S10_VMC_TERRAIN_AHEAD_W', '1.0'))
            _err = float(getattr(fol, '_last_err', 0.0))
            _w_eff = _w * float(np.clip(
                1.0 - abs(_err) / float(os.environ.get(
                    'S10_VMC_TERRAIN_ERR_GATE', '0.7')), 0.0, 1.0))
            terr_foot = np.array([terrain_at(wheel_xyz[i, 0],
                                             wheel_xyz[i, 1])
                                  for i in range(4)])
            # v231: 运动学 fallback 默认关（改变 wp4→5 部分格值致混沌翻车）；
            # 需要时 S10_VMC_TERRAIN_KIN=1（wp5→6 缓坡无数据区）
            if (os.environ.get('S10_VMC_TERRAIN_KIN', '0') == '1'
                    and os.environ.get('S10_VMC_TERRAIN', 'ray') == 'lidar'):
                for _i in range(4):
                    if not lterr.has(wheel_xyz[_i, 0], wheel_xyz[_i, 1]):
                        terr_foot[_i] = float(wheel_xyz[_i, 2] - 0.081)
            # v223h: 前瞻用车体中心前方单值，四腿共用——逐腿前瞻在狗斜向
            # 接近脊时右轮先到→单侧抬轮侧翻（wz 0.85/0.84 实测）
            _hx = body_pos[0] + _bx * _lk
            _hy = body_pos[1] + _by * _lk
            terr_ahead = np.full(4, terrain_at(_hx, _hy))
            terr = (1.0 - _w_eff) * terr_foot + _w_eff * terr_ahead
        else:
            terr = np.array([terrain_at(wheel_xyz[i, 0], wheel_xyz[i, 1])
                             for i in range(4)])
        # v223b: 地形低通（lidar 栅格稀疏/噪声，防腿抖）
        _tlp = float(os.environ.get('S10_VMC_TERRAIN_LP', '0.0'))
        if _tlp > 0.0:
            if _terr_f is None:
                _terr_f = terr.copy()
            else:
                _terr_f = _tlp * terr + (1.0 - _tlp) * _terr_f
            terr = _terr_f
        s_cur = float(getattr(fol, '_s_cur', 0.0))
        # v292: 台阶窗 vx 连续插值到 STAIR_WIN_VX（默认 1.8，不归零）——
        # 窗内只换腿控制（几何相位）与轮力矩模式，vx 参考保持连续
        if stair_risers:
            _sw0 = float(stair_risers[0][0]) - 1.0
            _sw1 = float(stair_risers[-1][0]) + 2.0
            _sramp = float(os.environ.get('S10_STAIR_VX_RAMP', '1.0'))
            _win_vx = float(os.environ.get('S10_STAIR_WIN_VX', '1.8'))
            if _sw0 - _sramp <= s_cur <= _sw1 + _sramp:
                if s_cur < _sw0:
                    _f = (s_cur - (_sw0 - _sramp)) / _sramp
                elif s_cur > _sw1:
                    _f = (_sw1 + _sramp - s_cur) / _sramp
                else:
                    _f = 1.0
                _f = float(np.clip(_f, 0.0, 1.0))
                vx_c = _win_vx * _f + vx_c * (1.0 - _f)

        # v236: 台阶相位步态——按弧长对每级 riser 调度前轴/后轴抬放
        # （几何已知，无硬模式：仅当台阶 riser 在前方窗口内才产生抬放量）。
        # 前轴窗口 = 棱边前 0.40m -> 棱边后 0.30m；后轴按半轴距 0.228m 延后。
        step_lift = np.zeros(4)
        stair_lift_flag = 0.0
        # v264: 连续前瞻抬轮（无门控）——按"轴前 0.35m 地形高 - 轴下地形高"
        # 连续抬放（比例 clamp 0.15m）。纯几何连续量，替代 hop 冲量/横脊
        # 步态等离散触发（用户原则：除 cruise/stair 切换外无门控）。
        # v266: 抬轮前瞻独立于 terr 前瞻（LOOKAHEAD=0.5 在起步坡上使地形
        # 阻抗过激翻车实测；抬轮用自身 0.35m 窗口）
        # v276: 已知横脊连续抬放（替代 lidar rise——lidar 在起步坡误触发、
        # 近场盲区噪声；已知地图连续响应，与台阶 skill 同类）。0.5m 前起抬、
        # 过脊 0.3m 释放，前/后轴分别。
        if (float(os.environ.get('S10_VMC_RIDGE_LIFT_CONT', '1')) > 0
                and ridge_world):
            _fwd5 = np.array([d.xmat[1][0], d.xmat[1][3]])
            _fn5 = float(np.hypot(_fwd5[0], _fwd5[1])) + 1e-9
            _fx5, _fy5 = _fwd5[0] / _fn5, _fwd5[1] / _fn5
            # v291d: crest state (front axle already above rear axle) selects
            # the long rear release window; otherwise rear keeps the proven
            # 0.30m window (v291b regression at the wp2->3 turn). Crest boost
            # also fades with body roll (continuous safety envelope).
            _wzc = np.asarray([d.xpos[WHEEL_BODY[i], 2] for i in range(4)])
            _crest_pre = (float(np.clip((float(np.max(_wzc[0:2]))
                                          - float(np.min(_wzc[2:4])) - 0.03)
                                         / 0.06, 0.0, 1.0)) * _roll_env)
            for _ai in range(2):
                _sgn = 1.0 if _ai == 0 else -1.0
                _ax = np.array([body_pos[0] + _fx5 * 0.228 * _sgn,
                                body_pos[1] + _fy5 * 0.228 * _sgn])
                _dmin_r = 1e9
                for (_rp, _tng, _sr, _dh) in ridge_world:
                    # v285: 只对 dh>=0.08 的脊抬轮（<0.08 微脊=轮滚微起伏，
                    # 腿阻抗吸收，不抬）
                    if _dh < 0.08:
                        continue
                    _dd = float(np.dot(_ax - _rp, _tng))
                    if -0.3 <= _dd <= 0.8 and _dd < _dmin_r:
                        _dmin_r = _dd
                if _dmin_r < 0.8:
                    # v278: 紧窗口满幅——棱边前 0.25m 起抬、0.1m 前满幅、
                    # 过棱 0.1m 后保持、0.3m 后释放。0.141m 够清 0.12m 脊；
                    # v277 提前 0.5m 满幅→前轮长时间离地失驱动侧翻
                    # v291c: front axle keeps the proven 0.30m release
                    # (v277/v288: long front lift during turns loses drive);
                    # rear axle gets the 0.50m window only at crest state
                    # (front already on the ridge top).
                    if _ai == 0 or _crest_pre <= 0.0:
                        _release, _span = 0.30, 0.20
                    else:
                        _release, _span = 0.50, 0.25
                    _lstart = float(os.environ.get(
                        'S10_VMC_LIFT_START', '0.25'))
                    _lramp = float(os.environ.get(
                        'S10_VMC_LIFT_RAMP', '0.15'))
                    _lift_r = float(
                        np.clip((_lstart - _dmin_r) / _lramp, 0.0, 1.0)
                        * np.clip((_release - _dmin_r) / _span, 0.0, 1.0))
                    if _lift_r > 0.02:
                        if _ai == 0:
                            step_lift[0:2] = np.maximum(
                                step_lift[0:2], _lift_r)
                        else:
                            step_lift[2:4] = np.maximum(
                                step_lift[2:4], _lift_r)
            # v406: 前后轴防同抬——前轴抬升中连续抑制后轴（至少保持后轮
            # 接地保牵引），防 wp4->5 横脊上四轮全抬（sl=[1,1,1,1] 实测
            # 无抓地卡死/漂东翻车）。连续量：前轴 lift>0.3 起后轴按比例衰减。
            # v279: 抬轮幅度乘航向对准系数——转弯（|err| 大）时抬轮衰减，
            # 防"转弯+抬轮"叠加侧翻（wp2→3 后轮满抬转弯中翻车实测）。
            _err_l = abs(float(getattr(fol, '_last_err', 0.0)))
            _lift_align = float(np.clip(
                1.0 - _err_l / float(os.environ.get(
                    'S10_VMC_LIFT_ERR_GATE', '0.8')), 0.0, 1.0))
            # v288: 回退 v287——前轮抬轮转弯中同样不稳（wp1→2 sl满幅+ω4.08
            # 翻车）；wp4→5 真问题=偏航处导航命令反向，需 NAV_DEBUG 定位
            if _lift_align < 1.0:
                step_lift *= _lift_align
            if float(np.max(step_lift)) > 0.02:
                stair_lift_flag = 1.0
        # v264: lidar rise 抬轮（仅 S10_VMC_RIDGE_LIFT_CONT=0 时兜底）
        _lk_c = float(os.environ.get('S10_VMC_LIFT_LOOKAHEAD', '0.35'))
        if (float(os.environ.get('S10_VMC_RIDGE_LIFT_CONT', '1')) <= 0
                and _lk_c > 0.05):
            _fwd4 = np.array([d.xmat[1][0], d.xmat[1][3]])
            _fn4 = float(np.hypot(_fwd4[0], _fwd4[1])) + 1e-9
            _fx4, _fy4 = _fwd4[0] / _fn4, _fwd4[1] / _fn4
            for _ai in range(2):
                _sgn = 1.0 if _ai == 0 else -1.0
                _ax = np.array([body_pos[0] + _fx4 * 0.228 * _sgn,
                                body_pos[1] + _fy4 * 0.228 * _sgn])
                _ha = terrain_at(_ax[0] + _fx4 * _lk_c,
                                 _ax[1] + _fy4 * _lk_c)
                _hb = terrain_at(_ax[0], _ax[1])
                if (os.environ.get('S10_LIFT_DEBUG', '0') == '1'
                        and next_idx == 5 and 20.5 < t < 24.0):
                    print('[LIFT] t=%.1f ai=%d ha=%.3f hb=%.3f ax=(%.2f,%.2f)'
                          % (t, _ai, _ha, _hb, _ax[0], _ax[1]), flush=True)
                # v265/v274: 上升量带通——只对 0.02~0.35m 可爬台阶响应
                # （0.12m 脊→0.8；起步坡 0.5m 伪尖峰→0；v273 前向修正后
                # 抬轮在起步坡误触发侧翻，收紧上限）
                _rise0 = float(_ha - _hb)
                _rise = 0.0
                if _rise0 > 0.08 and _rise0 < 0.35:
                    _rise = (float(np.clip(_rise0 / 0.15, 0.0, 1.0))
                             * float(np.clip((0.35 - _rise0) / 0.15, 0.0, 1.0)))
                if _rise > 0.02:
                    if _ai == 0:
                        step_lift[0:2] = np.maximum(step_lift[0:2], _rise)
                    else:
                        step_lift[2:4] = np.maximum(step_lift[2:4], _rise)
            if float(np.max(step_lift)) > 0.02:
                stair_lift_flag = 1.0
        if (float(os.environ.get('S10_VMC_STAIR_GAIT', '0')) > 0
                and stair_risers
                and stair_risers[0][0] - 1.0 <= s_cur
                <= stair_risers[-1][0] + 2.0):
            _fl_max, _rl_max = 0.0, 0.0
            # v238: 收窄抬放窗口（梯面0.4m≈轴距0.456m，宽窗口致前轮持续
            # 抬起不落地）——前轴 df∈[-0.08,0.18]、后轴 dr∈[-0.06,0.20]，
            # 每级 riser 前后轴快速衔接抬放；可选 FRONT_HOLD 连续挂前轮
            for (sr, dhv) in stair_risers:
                _df = s_cur - (sr - 0.228)   # 前轴到棱边
                _dr = s_cur - (sr + 0.228)   # 后轴到棱边
                _fl = (float(np.clip((0.18 - _df) / 0.06, 0.0, 1.0))
                       * float(np.clip((_df + 0.08) / 0.06, 0.0, 1.0)))
                _rl = (float(np.clip((0.20 - _dr) / 0.06, 0.0, 1.0))
                       * float(np.clip((_dr + 0.06) / 0.06, 0.0, 1.0)))
                _fl_max = max(_fl_max, _fl); _rl_max = max(_rl_max, _rl)
            if (float(os.environ.get('S10_VMC_STAIR_FRONT_HOLD', '0')) > 0
                    and stair_risers[0][0] - 0.6 <= s_cur
                    <= stair_risers[-1][0] + 1.2):
                _fl_max = max(_fl_max, 1.0)
            if _fl_max + _rl_max > 0.02:
                step_lift[:] = [_fl_max, _fl_max, _rl_max, _rl_max]
                stair_lift_flag = 1.0
        elif float(os.environ.get('S10_VMC_STEP_OVER', '0')) > 0:
            # v247: 横脊连续抬放（替代 v220a 状态机）——按前/后轴到横脊的
            # 世界坐标物理距离触发（s_cur 投影在转向时滞后会漏触发）：
            # 前轴 0.9m 前起抬、过脊即放；后轴同窗口。与台阶步态同模式。
            _fwdv = np.array([d.xmat[1][0], d.xmat[1][3]])
            _fn2 = float(np.hypot(_fwdv[0], _fwdv[1])) + 1e-9
            _fx2, _fy2 = _fwdv[0] / _fn2, _fwdv[1] / _fn2
            _fax = np.array([body_pos[0] + _fx2 * 0.228,
                             body_pos[1] + _fy2 * 0.228])
            _rax = np.array([body_pos[0] - _fx2 * 0.228,
                             body_pos[1] - _fy2 * 0.228])
            _fl2, _rl2 = 0.0, 0.0
            for (_rp, _tng, _sr, _dh) in ridge_world:
                # v259: 沿路径切线投影的距离（横向偏移不计入）
                _dfr = float(abs(np.dot(_fax - _rp, _tng)))
                _drr = float(abs(np.dot(_rax - _rp, _tng)))
                # v254/v258: 0.45m 前起抬；**满值保持到轮子越过棱边 0.25m**
                # （v254 过脊即放，失速时窗口掉到 0.3~0.4 部分抬升→后轮
                # 爬不上脊死锁，wp4→5 卡 90s 实测）
                _fl2 = max(_fl2, float(
                    np.clip((0.45 - _dfr) / 0.15, 0.0, 1.0)
                    * np.clip((_dfr + 0.25) / 0.20, 0.0, 1.0)))
                _rl2 = max(_rl2, float(
                    np.clip((0.45 - _drr) / 0.15, 0.0, 1.0)
                    * np.clip((_drr + 0.25) / 0.20, 0.0, 1.0)))
            if _fl2 + _rl2 > 0.02:
                # v256: 横脊步态激活时用轮下地形（禁前瞻）——前瞻抬轮 +
                # step_lift 抬轮双重叠加致腿饱和侧翻（wp4→5 实测）
                if 'terr_foot' in locals():
                    terr = terr_foot.copy()
                # v255: 横脊只抬前轮，后轮滚动/蹬过（斜过脊时后轮单侧先触
                # 脊，双侧同抬反而侧翻；"前挂后蹬"）
                if float(os.environ.get(
                        "S10_VMC_RIDGE_REAR_LIFT", "1.0")) > 0:
                    step_lift[:] = [_fl2, _fl2, _rl2, _rl2]
                else:
                    step_lift[:] = [_fl2, _fl2, 0.0, 0.0]

        # v221e: 车身抬升（过脊）——渐变：脊前 0.6m 起、0.2m 处满，
        # 过脊后 0.4m 内回落（防阶跃弹跳侧翻）
        _body_lift = 0.0
        # v221g: 物理检测——前轮前方 0.5m 真实地形高差>0.07 才抬身
        # （s_cur 投影推进快于物理位置，弯道出口误触发侧翻实测）
        if os.environ.get('S10_VMC_BODY_LIFT', '1') == '1':
            _fwd3 = np.array([d.xmat[1][0], d.xmat[1][3]])
            _fl = float(np.hypot(_fwd3[0], _fwd3[1])) + 1e-9
            _fx3, _fy3 = _fwd3[0] / _fl, _fwd3[1] / _fl
            _hf = terrain_at(body_pos[0] + _fx3 * 0.5,
                             body_pos[1] + _fy3 * 0.5)
            _h0 = terrain_at(body_pos[0], body_pos[1])
            if _hf - _h0 > 0.07:
                _body_lift = 1.0
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
        # v291: crest = continuous geometry (front axle already on ridge top,
        # rear axle at the face). Boosts rear terrain bump to ~1.4x _lift so
        # the rear wheel center target >= ridge top, and restores rear
        # step_lift to full (overrides yaw-alignment decay only for the rear).
        _crest = (float(np.clip((_lf - _lr - 0.03) / 0.06, 0.0, 1.0))
                  if (_in_ridge and _lf > _lr + 0.03) else 0.0) * _roll_env
        # v220b: 迈步期间禁用后轮跟抬——前轮被迈步抬起时 wz 天然高于后轮，
        # 会误把后轮也抬离地（全轮无推力死锁）
        if _step_state == 0 and _crest > 0.0:
            _rear_x = float(os.environ.get('S10_VMC_REAR_CREST_EXTRA', '0.6'))
            terr[2:] = np.maximum(
                terr[2:], terr[2:] + _lift * (0.8 + _rear_x * _crest))
            step_lift[2:4] = np.maximum(step_lift[2:4], _crest)


        # v234: 巡航转弯=差速为主 + hip 微 roll ±3.5°(0.06 rad)协调——
        # 不采用腿足式压弯（轮足深压弯→后轮打滑/无悬架抖动，用户总结）
        # v290: 压弯幅度可调（S10_CAR_ROLL_AMP，默认 0.06；测试 0.12 强化
        # 高速转弯向心/减侧翻）
        _ramp = float(os.environ.get("S10_CAR_ROLL_AMP", "0.06"))
        roll_tar = float(np.clip(-0.06 * om_c * abs(vx_c), -_ramp, _ramp))
        pitch_tar = 0.0
        # v285: 脊前速度**微缩**（用户"横脊=连续扰动非障碍"）——±0.15m 内
        # 线性从 vlim 缩到 min(vlim,2.5)，脊后即恢复；严禁硬砍 1.0 制造
        # 速度凹坑
        if ridge_world:
            _fwd6 = np.array([d.xmat[1][0], d.xmat[1][3]])
            _fn6 = float(np.hypot(_fwd6[0], _fwd6[1])) + 1e-9
            _fx6, _fy6 = _fwd6[0] / _fn6, _fwd6[1] / _fn6
            _dd6_min = 1e9
            for _sgn6 in (1.0, -1.0):
                _ax6 = np.array([body_pos[0] + _fx6 * 0.228 * _sgn6,
                                 body_pos[1] + _fy6 * 0.228 * _sgn6])
                for (_rp, _tng, _sr, _dh) in ridge_world:
                    _dd6 = float(np.dot(_ax6 - _rp, _tng))
                    if abs(_dd6) < abs(_dd6_min):
                        _dd6_min = _dd6
            _d6 = abs(_dd6_min)
            if _d6 < 0.15:
                _soft = float(os.environ.get("S10_RIDGE_SOFT_VX", "2.5"))
                vx_c = min(vx_c, _soft + max(vx_c - _soft, 0.0) * _d6 / 0.15)
        try:
            fwd = np.array([d.xmat[1][0], d.xmat[1][3]])
            fx, fy = fwd[0], fwd[1]
            h_a = terrain_at(body_pos[0] + fx*0.6, body_pos[1] + fy*0.6)
            h_b = terrain_at(body_pos[0] - fx*0.6, body_pos[1] - fy*0.6)
            # v225: 爬坡前倾匹配坡度（借鉴 dial-MPC stair_pitch_tar）——
            # 缓坡 0.8°~15° 时车身前倾，轮推力方向对准
            _slope = (h_a - h_b) / 1.2
            pitch_tar = -float(np.clip(np.arctan(_slope),
                                       -0.30, 0.30))
            pitch_tar = float(np.clip(np.arctan2(h_a - h_b, 1.2), -0.35, 0.35))
            # v218o: 横脊抬前轮时顺坡仰头（防 pitch 控制器对抗抬升导致腿饱和）
            if _lift_act > 0.05:
                pitch_tar = max(pitch_tar, 0.25 * _lift_act)
            # v285: 撤 v282（静态后轮抗抬头会顶起车身）——横脊靠 kd_pitch
            # 俯仰率阻尼吸收（削峰不硬顶）
        except Exception:
            pass

        if os.environ.get('VMC_STAND', '0') == '1':
            cmd = dict(vx=0.0, omega=0.0, roll_tar=0.0, pitch_tar=0.0)
        else:
            # v220n: 前轮地形前瞻 hop——只在横脊区触发（防平地毛刺误触发，
            # wp3→4 高差毛刺曾导致 hop=300 侧翻）
            hop = np.zeros(4, dtype=np.float64)
            _in_hop_zone = any(
                float(abs(np.dot(
                    np.array([body_pos[0] + d.xmat[1][0] * 0.228,
                              body_pos[1] + d.xmat[1][3] * 0.228])
                    - _rp, _tng))) < 1.5 for (_rp, _tng, _sr, _dh) in ridge_world)
            if _in_hop_zone and stair_lift_flag <= 0.0:
                _fwd2 = np.array([d.xmat[1][0], d.xmat[1][3]])
                _fx, _fy = _fwd2[0], _fwd2[1]
                _fn = float(np.hypot(_fx, _fy)) + 1e-9
                _fx, _fy = _fx / _fn, _fy / _fn
                for _wi in (0, 1):
                    _w0 = wheel_xyz[_wi]
                    # v220l: hop 检测用原始地形（terr 已被 bump 抬高，差值<0.08）
                    _h0 = terrain_at(_w0[0], _w0[1])
                    _ha = terrain_at(_w0[0] + _fx * 0.80,
                                     _w0[1] + _fy * 0.80)
                    if _ha - _h0 > 0.08:
                        hop[_wi] = float(os.environ.get('S10_VMC_HOP_F', '180.0'))
            # v449: 软抬轮技能划分——台阶/陡升段（step_zone/stair_zone，
            # 永久升面需保牵引爬升）用 S10_VMC_LIFT_F_SCALE_STEP；巡航平脊
            # 段（z 不升，动量冲过）用默认 1.0 硬抬轮。地形属性=技能切换。
            _lfs = 1.0
            _lsw = float(os.environ.get('S10_VMC_LIFT_SWING', '0.66'))
            if (next_idx >= 2 and next_idx - 1 < len(fol.step_zone)
                    and fol.step_zone[next_idx - 1]):
                _lfs = float(os.environ.get(
                    'S10_VMC_LIFT_F_SCALE_STEP', '0.3'))
                # v457: 台阶区加大抬轮摆动（0.66rad 只抬 0.05m < 0.13m 台阶）
                _lsw = float(os.environ.get(
                    'S10_VMC_LIFT_SWING_STEP', '1.5'))
            cmd = dict(vx=vx_c, omega=om_c, roll_tar=roll_tar,
                      pitch_tar=pitch_tar,
                      yaw_scale=1.0 - _lift_act, hop=hop,
                      step_lift=step_lift,
                      body_lift=_body_lift,
                      stair_lift=stair_lift_flag,
                      lift_f_scale=_lfs,
                      lift_swing=_lsw)
        tau = vmc.compute_tau(qpos, qvel, wheel_xyz, wheel_vel, cmd, terr, DT)
        _tleg = float(np.abs(tau[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]]).max())
        _twh = float(np.abs(tau[[3, 7, 11, 15]]).max())
        _max_tau_leg = max(_max_tau_leg, _tleg)
        _max_tau_wh = max(_max_tau_wh, _twh)
        if _tleg >= 50.0 or _twh >= 14.0:
            _over_run += DT
            _over_total += DT
            _over_worst = max(_over_worst, _over_run)
        else:
            _over_run = 0.0
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        t += DT
        if _viewer is not None:
            if not _viewer.is_running():
                print('[VMC] viewer 已关闭，结束', flush=True)
                break
            _viewer.sync()

        # 航点推进（0.5m + v204 捷径）
        if next_idx < len(wp):
            rp = d.xpos[1][:2]
            dist = float(np.linalg.norm(rp - wp[next_idx][:2]))
            # v294: 判据=质心投影 xy 进入航点 0.3m（机器狗任意一点经过的等效简化）
            _adv = float(os.environ.get('S10_WP_ADVANCE_DIST', '0.3'))
            reached = dist <= _adv
            if reached:
                if next_idx == 0 and t_start is None:
                    t_start = t
                wp_times[next_idx] = t
                print(f'[VMC-T] wp{next_idx} @ t={t:.2f}s', flush=True)
                next_idx += 1
                # v253: 过航点后强制弧长游标越过该航点——切弯时 s_cur 投影
                # 滞后会令前视目标回指后方（wp4→5 指令反向侧翻实测）
                try:
                    _s_min = float(fol.path_wp_s[next_idx - 1]) - 0.05
                    if fol._s_cur < _s_min:
                        fol._s_cur = _s_min
                except Exception:
                    pass
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
    print('[VMC] 力矩合规: 腿max|tau|=%.1fNm(限50) 轮max|tau|=%.1fNm(限14) '
          '连续超限最长 %.2fs / 累计 %.2fs%s' % (
              _max_tau_leg, _max_tau_wh, _over_worst, _over_total,
              '  [超0.5s不合格!]' if _over_worst > 0.5 else ''), flush=True)
    if os.environ.get('VMC_TRAJ'):
        np.save(os.environ['VMC_TRAJ'], np.array(traj))
    if _viewer is not None:
        try:
            _viewer.close()
        except Exception:
            pass
        # v293: mujoco.viewer 在 WSLg 下正常退出会段错误，改用 os._exit
        sys.stdout.flush()
        os._exit(0)


if __name__ == '__main__':
    main()
