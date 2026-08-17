"""cruise 无 ROS 独立测试：wp0→MAX_WP，直接 mujoco + MPC + 导航。
不初始化 rclpy，可与 stair 会话并行（无 DDS 冲突）。"""
import os, sys, time
from collections import deque
import numpy as np
import mujoco
try:
    import mujoco.viewer
    _HAS_VIEWER = True
except Exception:
    _HAS_VIEWER = False

PKG = '/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy'
sys.path.insert(0, PKG)
sys.path.insert(0, '/home/wfx/DR_competition/0810new/deeprobot_competition')
sys.path.insert(0, '/home/wfx/DR_competition/0810new/deeprobot_competition/dial-mpc')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',
                      os.path.expanduser('~/.cache/s10_dial_mpc'))

from s10_mpc.mpc_controller import MPCController
from s10_mpc.stair_auto_nav import AutoNavFollower
from s10_mpc.stair_contact_planner import StairContactPlanner
from s10_mpc.stair_stance_guard import StairStanceGuard
from s10_mpc.body_mppi import BodyMPPI
from s10_mpc.vmc_legs import WHEEL_BODY
from rl_stair.deploy.rlstair_ctrl import RLStairCtrl

DT = 0.005
STAND_TIME = float(os.environ.get('S10_STAND_TIME', '0.6'))
STAND_KP = float(os.environ.get('S10_STAND_KP', '120.0'))
STAND_KD = float(os.environ.get('S10_STAND_KD', '3.0'))
MAX_SIM = float(os.environ.get('S10_TEST_MAX_SIM', '120'))
MAX_WP = int(os.environ.get('S10_AUTO_MAX_WP', '5'))
WP_TIMEOUT = float(os.environ.get('S10_WP_TIMEOUT', '60.0'))
XML = os.environ.get('S10_XML',
    f'{PKG}/S10_description/s10_mjcf/mjcf/S10_track.xml')
MPC_YAML = os.environ.get('S10_MPC_YAML',
    '/home/wfx/DR_competition/0810new/deeprobot_competition/doc/s10_mpc_deploy.yaml')

JOINT_INIT = np.array([-0.438, -1.16, 2.45, 0.0,
                        0.438, -1.16, 2.45, 0.0,
                       -0.438,  1.16, -2.45, 0.0,
                        0.438,  1.16, -2.45, 0.0], dtype=np.float64)
STAND_TARGET = np.array([-0.05, -1.10, 1.90, 0.0,
                          0.05, -1.10, 1.90, 0.0,
                         -0.05,  1.10, -1.90, 0.0,
                          0.05,  1.10, -1.90, 0.0], dtype=np.float64)
STAIR_STAND_TARGET = np.array([-0.05, -1.10, 1.90, 0.0,
                             0.05, -1.10, 1.90, 0.0,
                            -0.05,  1.10, -1.90, 0.0,
                             0.05,  1.10, -1.90, 0.0], dtype=np.float64)


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    d.qpos[7:23] = JOINT_INIT
    _sx = os.environ.get('S10_START_XY')
    if _sx:
        _s0, _s1 = [float(v) for v in _sx.split(',')]
        d.qpos[0:3] = [_s0, _s1, 0.2]
    else:
        d.qpos[0:3] = [0.0, -2.5, 0.2]
    _init_yaw = float(os.environ.get('S10_INIT_YAW', '0.0'))
    if abs(_init_yaw) > 1e-3:
        d.qpos[3:7] = [np.cos(_init_yaw / 2.0), 0, 0,
                       np.sin(_init_yaw / 2.0)]
    else:
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
    START_WP = int(os.environ.get('S10_START_WP', '0'))
    if START_WP > 0 and START_WP < len(wps):
        _sbk = float(os.environ.get('S10_START_BACK', '1.0'))
        _sz0 = float(wps[START_WP][2]) + 0.21
        try:
            _g0 = np.array([-1], dtype=np.int32)
            _dist0 = np.zeros(1)
            _nrm0 = np.zeros(3)
            _hit0 = mujoco.mj_ray(m, d,
                                  [float(wps[START_WP][0]),
                                   float(wps[START_WP][1]) - _sbk, 8.0],
                                  [0, 0, -1], None, True, -1, _g0, _nrm0)
            if _hit0 > 0:
                _sz0 = (8.0 - _hit0) + 0.24
        except Exception:
            pass
        if _sbk <= 0.0:
            d.qpos[0:3] = [float(wps[START_WP][0]),
                           float(wps[START_WP][1]), _sz0]
        else:
            d.qpos[0:3] = [float(wps[START_WP][0]),
                           float(wps[START_WP][1]) - _sbk, _sz0]
        if START_WP + 1 < len(wps):
            _dy = wps[START_WP + 1][1] - wps[START_WP][1]
            _dx = wps[START_WP + 1][0] - wps[START_WP][0]
            _iy = float(np.arctan2(_dy, _dx))
        else:
            _iy = 1.5708
        if os.environ.get('S10_INIT_YAW'):
            _iy = float(os.environ.get('S10_INIT_YAW'))
        d.qpos[3:7] = [np.cos(_iy / 2), 0, 0, np.sin(_iy / 2)]
        d.qpos[7:23] = STAIR_STAND_TARGET.copy()
        mujoco.mj_forward(m, d)
        print(f'[NOROS] 从 wp{START_WP} 起跑', flush=True)
    track_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    assert track_body >= 0
    # S10_USE_VIEWER=1：被动 viewer（同 mujoco_simulation_ros2.py 方案）
    _viewer = None
    if os.environ.get('S10_USE_VIEWER', '0') == '1' and _HAS_VIEWER:
        _viewer = mujoco.viewer.launch_passive(m, d)
        with _viewer.lock():
            _viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            _viewer.cam.trackbodyid = track_body
        print('[NOROS] viewer 已打开（关窗口即停止）', flush=True)

    t0 = time.time()
    mpc = MPCController(MPC_YAML)
    mpc.init_state(np.asarray(d.qpos[:23], dtype=np.float32),
                   np.asarray(d.qvel[:22], dtype=np.float32))
    mpc.set_cmd(0.0, 0.0, 0.0)
    rl_ctrl = None
    if os.environ.get('S10_DIAL_RL_STAIR', '0') == '1':
        rl_ctrl = RLStairCtrl(m, vx=float(os.environ.get('S10_RL_VX', '1.5')))
    print(f'[NOROS] MPC ready ({time.time()-t0:.1f}s)', flush=True)

    fol = AutoNavFollower(
        wps,
        max_speed=float(os.environ.get('S10_AUTO_VMAX', '6.0')),
        vyaw_max=float(os.environ.get('S10_AUTO_VYAW_MAX', '1.5')),
        yaw_gain=float(os.environ.get('S10_AUTO_YAW_GAIN', '3.0')),
        lookahead=float(os.environ.get('S10_AUTO_LOOKAHEAD', '1.5')),
    )
    planner = StairContactPlanner(m, d, fol)

    def _lidar_stair_heading():
        _sc = float(getattr(fol, '_s_cur', 0.0))
        _lo = _sc + 0.5
        _hi = _sc + float(os.environ.get('S10_TK1_LOOKAHEAD', '6.0'))
        _cum = fol.path_cum
        _k0 = int(np.searchsorted(_cum, _lo))
        _k1 = int(np.searchsorted(_cum, _hi))
        if _k1 <= _k0 + 3:
            return None
        try:
            _rs = planner.lidar.detect_risers(
                fol.path_pts[_k0:_k1], _cum[_k0:_k1], _lo, _hi,
                rise=0.05, max_dh=0.16)
        except Exception:
            _rs = []
        if _rs:
            _sm = float(np.mean([float(r[0]) for r in _rs]))
            _ki = int(np.searchsorted(_cum, _sm, side='right')) - 1
            _ki = max(0, min(_ki, len(fol.path_heading) - 1))
            return float(fol.path_heading[_ki])
        _wpts = fol.path_pts[_k0:_k1]
        _x0 = float(_wpts[:, 0].min() - 1.0)
        _x1 = float(_wpts[:, 0].max() + 1.0)
        _y0 = float(_wpts[:, 1].min() - 1.0)
        _y1 = float(_wpts[:, 1].max() + 1.0)
        _res = planner.lidar.res
        _wi0 = max(int(np.floor((_y0 - planner.lidar.oy) / _res)), 0)
        _wi1 = min(int(np.ceil((_y1 - planner.lidar.oy) / _res)), planner.lidar.ny - 1)
        _wj0 = max(int(np.floor((_x0 - planner.lidar.ox) / _res)), 0)
        _wj1 = min(int(np.ceil((_x1 - planner.lidar.ox) / _res)), planner.lidar.nx - 1)
        if _wi1 <= _wi0 or _wj1 <= _wj0:
            return None
        _wv = planner.lidar.wall_valid[_wi0:_wi1, _wj0:_wj1]
        _iy, _ix = np.where(_wv > 0)
        if len(_ix) == 0:
            return None
        _wx = planner.lidar.ox + (_wj0 + _ix) * _res
        _wy = planner.lidar.oy + (_wi0 + _iy) * _res
        _lat_min = float(os.environ.get('S10_OBST_LAT_MIN', '0.5'))
        _d2 = ((_wpts[None, :, 0] - _wx[:, None]) ** 2
               + (_wpts[None, :, 1] - _wy[:, None]) ** 2)
        _lat = np.sqrt(_d2.min(axis=1))
        _op = _lat < _lat_min
        if int(_op.sum()) < int(os.environ.get('S10_TK1_MIN_CELLS', '8')):
            return None
        _scs = []
        for _ii in np.where(_op)[0]:
            _dd = ((_wpts[:, 0] - _wx[_ii]) ** 2
                   + (_wpts[:, 1] - _wy[_ii]) ** 2)
            _scs.append(_cum[_k0 + int(np.argmin(_dd))])
        _sm = float(np.mean(_scs))
        _ki = int(np.searchsorted(_cum, _sm, side='right')) - 1
        _ki = max(0, min(_ki, len(fol.path_heading) - 1))
        return float(fol.path_heading[_ki])
    guard = StairStanceGuard(m, d)
    mpc.set_yaw_gain_lo(float(os.environ.get('S10_AUTO_YAW_FF_GAIN', '20.0')))
    # 已知地图横脊预扫描（与节点 _scan_ridge_zones 同法）：段内 0.12m+ 棱限速。
    # 这是已知地图（上帝视角）捷径；目标要求禁上帝视角 → S10_RIDGE_PRESCAN=0 关闭。
    if os.environ.get('S10_RIDGE_PRESCAN', '1') == '1':
        try:
            _pts = fol.path_pts
            _hs = np.empty(len(_pts))
            for _k, _p in enumerate(_pts):
                _g = np.array([-1], dtype=np.int32)
                _dist = np.zeros(1); _nrm = np.zeros(3)
                _hit = mujoco.mj_ray(
                    m, d, [_p[0], _p[1], 8.0], [0, 0, -1],
                    None, False, -1, _g, _nrm)
                _hs[_k] = (8.0 - _hit) if _g[0] >= 0 else float(_p[2])
            _dh = np.abs(np.diff(_hs))
            _skip = float(fol.path_wp_s[1]) - 2.0
            _ri = np.where((_dh > 0.12)
                           & (fol.path_cum[:len(_dh)] > _skip))[0]
            _rv = float(os.environ.get('S10_RIDGE_VX', '2.5'))
            for _k in _ri:
                _lo = max(0, _k - int(2.0 / fol.path_res))
                _hi = min(len(fol.path_vlim), _k + int(1.2 / fol.path_res))
                fol.path_vlim[_lo:_hi] = np.minimum(
                    fol.path_vlim[_lo:_hi], _rv)
            print(f'[NOROS] 横脊预扫描 {len(_ri)} 处，限速 {_rv}', flush=True)
        except Exception as _e:
            print('[NOROS] 横脊预扫描失败', _e, flush=True)

    # v218: 身体层 MPPI（S10_BODY_MPPI=1 启用）——替代 compute_cmd 直出，输出 [vx,ω]
    _bmpi = None
    if os.environ.get('S10_BODY_MPPI', '0') == '1':
        from s10_mpc.body_mppi import BodyMPPI as _B
        _bmpi = _B(N=int(os.environ.get('S10_BODY_MPPI_N', '256')),
                   H=int(os.environ.get('S10_BODY_MPPI_H', '20')))
        print('[NOROS] 身体层 MPPI 启用', flush=True)

    # 进程内 JIT 预热（2026-08-08 修复“卡在起点”）：本机 JAX 持久化编译缓存
    # 跨进程不生效（实测首次 plan_once 仍编译 17.5s），单独预编译对真跑无帮助。
    # 这里在仿真主循环前把 CRUISE/STAIR 两套 MBDPI 先编译完（约 20-40s），
    # 机器人保持趴姿；否则 t=3s 起步瞬间才触发编译，视觉上就是“站起后不动”。
    mpc._first = False
    _w0 = time.time()
    _q0 = np.asarray(d.qpos[:23], dtype=np.float32)
    _qd0 = np.asarray(d.qvel[:22], dtype=np.float32)
    for _k in range(4):
        _t1 = time.time()
        mpc.plan_once(_q0, _qd0, 0.02 * _k)
        if time.time() - _t1 < 0.5:
            break
    try:
        if hasattr(mpc, 'mbdpi_h20'):
            mpc.set_mode('STAIR')
            # 预热同时编译 hard-mode 摆动激活态的 STAIR trace：运行中首次
            # 激活 gait_swing 会触发 ~25s 重编译（2026-08-14 实测 max
            # plan_ms 6.7~26s 尖峰），吃掉仿真时间并扰动爬梯。
            _gsw_bak = getattr(mpc, '_gait_swing', None)
            mpc._gait_swing = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
            for _k in range(3):
                _t1 = time.time()
                mpc.plan_once(_q0, _qd0, 0.02 * _k)
                if time.time() - _t1 < 0.5:
                    break
            if _gsw_bak is not None:
                mpc._gait_swing = _gsw_bak
            else:
                mpc._gait_swing = np.zeros(4, dtype=np.float32)
            mpc.set_mode('CRUISE')
    except Exception:
        pass
    print(f'[NOROS] MPC JIT 预热完成（{time.time()-_w0:.1f}s），即将开始', flush=True)

    next_idx = START_WP if 'START_WP' in dir() else 0
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
    plan_times = []
    plan_recent = deque(maxlen=4)
    plan_cnt = 0
    traj_file = os.environ.get('S10_TRAJ_FILE')
    if traj_file:
        _tf = open(traj_file, 'w')
        _tf.write('t,x,y,yaw,next_idx,err,d_wp,vx,vyaw,cte,s_cur,tgt_x,tgt_y,mode,tk1,tk2\n')

    _rl_was_stair = False
    _rl_trans_t0 = None
    _tk2 = False
    _tk1_active = False
    _tk2_active = False
    while t < MAX_SIM:
        step = int(t / DT)
        if not auto_active:
            # 站起 PD（3s）
            q = d.qpos[7:23].reshape(-1, 1)
            dq = d.qvel[6:22].reshape(-1, 1)
            tau = (STAND_KP * (STAND_TARGET.reshape(-1, 1) - q) - STAND_KD * dq).flatten()
            tau[3::4] = -0.3 * dq[3::4].flatten()
            # v191：站起阶段 base yaw 预转向（轮子差速，可移植真机）
            if os.environ.get('S10_STAND_TURN', '0') == '1':
                _q = d.xquat[track_body]
                _yaw = float(np.arctan2(
                    2.0*(_q[3]*_q[0]+_q[1]*_q[2]),
                    1.0-2.0*(_q[2]**2+_q[3]**2)))
                _yt = float(os.environ.get('S10_STAND_TURN_YAW', '1.624'))
                _err = float(np.arctan2(np.sin(_yt-_yaw), np.cos(_yt-_yaw)))
                _k = float(os.environ.get('S10_STAND_TURN_K', '5.0'))
                _turn = float(np.clip(_k*_err, -40.0, 40.0))
                tau[3::4] += np.array([_turn, _turn, -_turn, -_turn])
            d.ctrl[:] = tau
            if t >= STAND_TIME:
                qq = np.asarray(d.qpos[:23], dtype=np.float32)
                qqd = np.asarray(d.qvel[:22], dtype=np.float32)
                last_act = mpc.plan_once(qq, qqd, t)
                t_cmd0 = t
                # v186: JIT 预热后原地等 GPU 空闲再起步（把干净窗口留给行驶段）
                if os.environ.get('S10_GPU_HOLD', '0') == '1':
                    import subprocess as _sp
                    _hold_max = float(os.environ.get('S10_GPU_HOLD_MAX', '300'))
                    _h0 = time.time()
                    _hfree = 0
                    _phase = 0   # 0=等 busy（确保对齐到空隙起点），1=等 free
                    while time.time() - _h0 < _hold_max:
                        _busy = True
                        try:
                            _r1 = _sp.run(
                                ['bash', '-lc',
                                 "pgrep -f 'test_auto_[n]av.py' | wc -l"],
                                capture_output=True, text=True, timeout=5)
                            _r2 = _sp.run(
                                ['nvidia-smi', '--query-gpu=memory.used',
                                 '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5)
                            _n = int(_r1.stdout.strip() or 0)
                            _mem = int(_r2.stdout.strip().split()[0])
                            _busy = (_n > 0 or _mem >= 1500)
                        except Exception:
                            pass
                        if _phase == 0:
                            if _busy:
                                _phase = 1
                                _hfree = 0
                            else:
                                # 修复（2026-08-08）：GPU 本来空闲时原逻辑
                                # 会一直等“忙→闲”转变，干等满 S10_GPU_HOLD_MAX
                                # （默认 300s）——起步卡住主因。空闲连续 2 次放行。
                                _hfree += 1
                                if _hfree >= 2:
                                    break
                        else:
                            if not _busy:
                                _hfree += 1
                                if _hfree >= 2:   # 空隙起点连续 2 次确认
                                    break
                            else:
                                _hfree = 0
                        time.sleep(3)
                    print(f'[NOROS] GPU_HOLD 释放 (t={t:.1f}s) wall={time.strftime("%H:%M:%S")}', flush=True)
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
                # DiAL STAIR: perception is the only source for riser geometry.
                if step % 20 == 0:
                    planner.update_perception(t)
                    planner.update_risers(float(getattr(fol, '_s_cur', 0.0)))
                local_tile = planner.get_tile(pos, t)
                mpc.set_elevation_map(local_tile)
                _pc = planner.stair_confirmed(pos, yaw)
                fol.update_mode(pos, next_idx, yaw=yaw, local_map=local_tile,
                                percept_confirmed=_pc)
                mpc.set_mode(fol.mode)
                if rl_ctrl is not None and fol.mode == 'STAIR':
                    if not _rl_was_stair:
                        _rl_trans_t0 = float(t)
                        if os.environ.get('S10_DIAL_RL_DEBUG', '0') == '1':
                            _rs, _ts = fol._stair_tables()
                            print('[RLRISERS]', [round(float(v),2) for v in _rs],
                                  [round(float(v),2) for v in _ts], flush=True)
                    _rs, _ts = fol._stair_tables()
                    rl_ctrl.set_risers(np.asarray(_rs), np.asarray(_ts))
                if _rl_was_stair and fol.mode != 'STAIR':
                    _tk2 = True
                _rl_was_stair = (fol.mode == 'STAIR')
                if fol.mode == 'STAIR':
                    _wy = np.asarray([d.xpos[_wb, 1] for _wb in (5, 9, 13, 17)], dtype=np.float64)
                    _wz = np.asarray([d.xpos[_wb, 2] for _wb in (5, 9, 13, 17)], dtype=np.float64)
                    planner.apply_contact(mpc, _wy, _wz, t, float(pos[1]))
                # v199: pass measured body yaw rate so nav yaw damping
                # (- S10_YAW_DAMP * yaw_rate) actually activates -> less
                # heading overshoot on straights
                _wyaw_real = float(np.asarray(
                    d.cvel[track_body, :3]).dot(
                    np.asarray(d.xmat[track_body]).reshape(3, 3).T)[2])
                vx, vyaw = fol.compute_cmd(
                    pos, yaw, next_idx,
                    robot_z=float(d.xpos[track_body][2]), yaw_rate=_wyaw_real)
                # TK1: CRUISE + lidar stair heading -> align before handoff
                if (os.environ.get('S10_TK1', '0') == '1'
                        and fol.mode == 'CRUISE'):
                    _th = _lidar_stair_heading()
                    if os.environ.get('S10_TK1_DEBUG', '0') == '1':
                        print('[TK1DBG] t=%.1f pos=(%.2f,%.2f) yaw=%.2f th=%s mode=%s'
                              % (t, float(pos[0]), float(pos[1]), yaw, _th, fol.mode), flush=True)
                    if _th is not None:
                        _ey = float(np.arctan2(np.sin(_th - yaw), np.cos(_th - yaw)))
                        _db = float(os.environ.get('S10_TK1_YAW_DB', '0.20'))
                        if abs(_ey) > _db:
                            _tk1_active = True
                            vx = min(vx, float(os.environ.get('S10_TK1_VX', '2.2')))
                            _ky = float(os.environ.get('S10_TK1_YAW_K', '2.5'))
                            _ymax = float(os.environ.get('S10_TK1_YAW_MAX', '1.5'))
                            vyaw = float(np.clip(_ky * _ey, -_ymax, _ymax))
                        else:
                            _tk1_active = False
                # TK2: after stair handback, align to path heading then release
                if (os.environ.get('S10_TK2', '0') == '1' and _tk2
                        and fol.mode == 'CRUISE'):
                    _s2 = float(getattr(fol, '_s_cur', 0.0))
                    _ki2 = int(np.searchsorted(fol.path_cum, _s2, side='right')) - 1
                    _ki2 = max(0, min(_ki2, len(fol.path_heading) - 1))
                    _th2 = float(fol.path_heading[_ki2])
                    _ey2 = float(np.arctan2(np.sin(_th2 - yaw), np.cos(_th2 - yaw)))
                    _db2 = float(os.environ.get('S10_TK2_YAW_DB', '0.15'))
                    if abs(_ey2) > _db2:
                        _tk2_active = True
                        _k2 = float(os.environ.get('S10_TK2_YAW_K', '2.5'))
                        _ymax2 = float(os.environ.get('S10_TK2_YAW_MAX', '1.5'))
                        vyaw = float(np.clip(_k2 * _ey2, -_ymax2, _ymax2))
                        vx = min(vx, float(os.environ.get('S10_TK2_VX', '1.5')))
                    else:
                        _tk2 = False
                        _tk2_active = False

                if _bmpi is not None:
                    # 路径参考轨迹（弧长采样）
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
                        _ref.append([fol.path_pts[_k, 0] + _t * (fol.path_pts[_k + 1, 0] - fol.path_pts[_k, 0]),
                                     fol.path_pts[_k, 1] + _t * (fol.path_pts[_k + 1, 1] - fol.path_pts[_k, 1]),
                                     fol.path_heading[min(_k, len(fol.path_heading) - 1)]])
                    _ref = np.array(_ref) if len(_ref) else np.array([[pos[0], pos[1], yaw]])
                    _Rm = np.asarray(d.xmat[track_body]).reshape(3, 3)
                    _vw = _Rm.T @ np.asarray(d.cvel[track_body][3:6])
                    _st = np.array([pos[0], pos[1], yaw,
                                    float(_vw[0]), float(_vw[1]),
                                    float(d.cvel[track_body][5])])
                    _vx_c, _om_c = _bmpi.plan(_st, _ref, float(fol._last_vlim))
                    mpc.set_cmd(float(_vx_c), 0.0, float(_om_c))
                else:
                    mpc.set_cmd(vx, 0.0, vyaw)
                if os.environ.get('S10_CURVE_DEBUG') == '1':
                    _q = d.xquat[track_body]
                    _roll = float(np.arctan2(
                        2.0*(_q[0]*_q[1]+_q[2]*_q[3]),
                        1.0-2.0*(_q[1]*_q[1]+_q[2]*_q[2])))
                    _wyaw = float(d.cvel[track_body, 0])  # world yaw rate? use body ang
                    _vyaw_real = float(np.asarray(
                        d.cvel[track_body, :3]).dot(
                        np.asarray(d.xmat[track_body]).reshape(3, 3).T)[2])
                    if next_idx >= 2 and next_idx <= 5:
                        print(f"[CURVE] t={t:.1f} next={next_idx} "
                              f"x={pos[0]:.1f} y={pos[1]:.1f} "
                              f"vx_cmd={vx:.2f} vyaw_cmd={vyaw:.2f} "
                              f"vyaw_real={_vyaw_real:.2f} roll={_roll:.2f}",
                              flush=True)
                if os.environ.get('S10_USE_REF_PATH', '0') == '1':
                    ref = fol.ref_path_3d(pos, next_idx, local_map=local_tile)
                    mpc.set_ref_path(ref if ref is not None else [],
                                     valid=ref is not None)
                dbg_cnt += 1
                if traj_file and dbg_cnt % 2 == 0:
                    _tf.write(f'{t:.2f},{pos[0]:.3f},{pos[1]:.3f},'
                              f'{yaw:.3f},{next_idx},{fol._last_err:.3f},'
                              f'{fol._last_dwp:.3f},{vx:.3f},{vyaw:.3f},'
                              f'{getattr(fol,"_last_cte",0.0):.3f},'
                              f'{fol._s_cur:.3f},'
                              f'{getattr(fol,"_last_tgt",[0,0])[0]:.3f},'
                              f'{getattr(fol,"_last_tgt",[0,0])[1]:.3f},'
                              f'{fol.mode},{1 if _tk1_active else 0},{1 if _tk2_active else 0}\n')
                    _tf.flush()
                if os.environ.get('S10_AUTO_DEBUG') == '1' and dbg_cnt % 40 == 1:
                    print(f'[NAVDBG] next={next_idx} vx={vx:.2f} '
                          f'vyaw={vyaw:.2f} err={fol._last_err:.2f} '
                          f'd_wp={fol._last_dwp:.2f} mode={fol.mode}',
                          flush=True)
            if step % plan_interval == 0:
                _pt0 = time.time()
                last_act = mpc.plan_once(qq, qqd, t)
                _pt1 = time.time() - _pt0
                plan_cnt += 1
                if os.environ.get('S10_PLAN_DEBUG') == '1' and plan_cnt <= 70:
                    print(f'[PLANDBG] plan#{plan_cnt} ms={_pt1*1000:.0f} wp={next_idx} t={t:.2f}', flush=True)
                if plan_cnt > 12:   # 前 12 个 plan 仍含剩余 JIT，不计频率/守卫
                    plan_times.append(_pt1)
                    plan_recent.append(_pt1)
                    if (len(plan_recent) == 4
                            and all(float(x) > 0.13 for x in plan_recent)):
                        crashed = 'gpu_busy'
                        print(f'[NOROS] GPU 争抢中止: plans_ms={[float(x)*1000 for x in plan_recent]} wall={time.strftime("%H:%M:%S")}', flush=True)
                        break
                if t_cmd0 is None:
                    t_cmd0 = t
            if rl_ctrl is not None and fol.mode == 'STAIR':
                wheel_xyz = np.asarray([d.xpos[_wb] for _wb in WHEEL_BODY])
                wheel_vel = np.asarray([d.cvel[_wb][0:3] for _wb in WHEEL_BODY])
                terr = np.asarray(fol.stair_terrain(wheel_xyz[:, 1]))
                _cmd_rl = dict(vx=0.0, omega=0.0, roll_tar=0.0, pitch_tar=0.0)
                mpc.latest_tau = rl_ctrl.compute_tau(
                    qq, qqd, wheel_xyz, wheel_vel, _cmd_rl, terr, DT)
                _pretrans = float(os.environ.get('S10_DIAL_RL_PRETRANS_TIME', '1.0'))
                if _rl_trans_t0 is not None and (float(t) - _rl_trans_t0) < _pretrans:
                    _li = rl_ctrl.idx['leg_idx']
                    _lj = rl_ctrl.idx['act2jnt'][_li]
                    _lv = rl_ctrl.idx['act2vel'][_li]
                    _tau_leg = np.clip(
                        60.0 * (rl_ctrl.default_dof[_li] - qq[_lj])
                        - 4.0 * qqd[_lv], -48.0, 48.0)
                    mpc.latest_tau[_li] = _tau_leg
            else:
                mpc.latest_tau = mpc.compute_tau(last_act, qq, qqd)
            if rl_ctrl is None and fol.mode == 'STAIR':
                _gsw_now = np.asarray(getattr(mpc, '_gait_swing', np.zeros(4)), dtype=np.float64)
                _com_xy = d.xpos[track_body][:2]
                _wy = np.asarray([d.xpos[_wb, 1] for _wb in (5, 9, 13, 17)], dtype=np.float64)
                _wz = np.asarray([d.xpos[_wb, 2] for _wb in (5, 9, 13, 17)], dtype=np.float64)
                _terr = np.asarray(fol.stair_terrain(_wy), dtype=np.float64)
                _prox = np.asarray(getattr(mpc, '_stair_prox', np.full(4, 1e9)), dtype=np.float64)
                mpc.latest_tau = guard.apply(mpc.latest_tau, _gsw_now, _com_xy, wheel_y=_wy, wheel_z=_wz, terrain_z=_terr, prox=_prox)
            d.ctrl[:] = np.asarray(mpc.latest_tau, dtype=np.float64)
            if os.environ.get('S10_STAIR_JOINT_DEBUG', '0') == '1' and step % 20 == 0:
                _qleg = np.asarray(d.qpos[7:23]).reshape(-1,1).flatten()
                _tleg = np.asarray(mpc.latest_tau)
                _qfr = _qleg[4:8]
                _qhl = _qleg[8:12]
                _qhr = _qleg[12:16]
                _tfr = _tleg[4:8]
                _thl = _tleg[8:12]
                _thr = _tleg[12:16]
                _qb = d.xquat[track_body]
                _proll = float(np.arctan2(
                    2.0*(_qb[0]*_qb[1]+_qb[2]*_qb[3]),
                    1.0-2.0*(_qb[1]*_qb[1]+_qb[2]*_qb[2])))
                _ppitch = float(np.arcsin(np.clip(
                    2.0*(_qb[0]*_qb[2]-_qb[3]*_qb[1]), -1.0, 1.0)))
                _bvx = float(np.asarray(d.cvel[track_body, 3:6]).dot(
                    np.asarray(d.xmat[track_body]).reshape(3, 3).T)[0])
                print(f'[JOINT] t={t:.1f} roll={_proll:.2f} pitch={_ppitch:.2f} vx={_bvx:.2f} FLq={[round(float(v),2) for v in _qleg[0:4]]} FRq={[round(float(v),2) for v in _qfr]} HLq={[round(float(v),2) for v in _qhl]} HRq={[round(float(v),2) for v in _qhr]} FLt={[round(float(v),1) for v in _tleg[0:4]]} FRt={[round(float(v),1) for v in _tfr]} HLt={[round(float(v),1) for v in _thl]} HRt={[round(float(v),1) for v in _thr]} mode={fol.mode}', flush=True)

        mujoco.mj_step(m, d)
        t += DT
        if _viewer is not None:
            if not _viewer.is_running():
                print('[NOROS] viewer 已关闭，结束', flush=True)
                break
            _viewer.sync()

        # 航点推进（0.5m 半径，与节点一致）
        if next_idx < len(wps):
            rp = d.xpos[track_body][:2]
            dist = float(np.linalg.norm(rp - wps[next_idx][:2]))
            _reached = dist <= 0.5
            if not _reached:
                # v199: 弧长已越过航点且物理距离在容差内 → 视为通过，
                # 避免“错过目标点反复绕圈”（S10_WP_ADVANCE_DIST=0 关闭）
                _adv = float(os.environ.get('S10_WP_ADVANCE_DIST', '0.0'))
                if (next_idx >= 1 and _adv > 0.0
                        and hasattr(fol, '_s_cur')
                        and next_idx < len(fol.path_wp_s)):
                    _passed = (fol._s_cur
                               > fol.path_wp_s[next_idx] - 0.05)
                    if _passed and dist <= _adv:
                        _reached = True
                # v204: 越过航点平面（沿 wp->wp_next 方向的投影已过 wp）且
                # 侧向在容差内 -> 视为通过，防"切弯/外漂后回头蹭 0.5m 圆门绕圈"。
                # S10_WP_ADVANCE_PLANE=0 关闭（默认）。
                if not _reached:
                    _pdist = float(os.environ.get(
                        'S10_WP_ADVANCE_PLANE', '0.0'))
                    if (_pdist > 0.0 and next_idx >= 1
                            and next_idx + 1 < len(wps)
                            and dist <= _pdist):
                        _d = wps[next_idx + 1][:2] - wps[next_idx][:2]
                        _L = float(np.linalg.norm(_d))
                        if _L > 1e-6:
                            _proj = float(np.dot(
                                rp - wps[next_idx][:2], _d) / _L)
                            if _proj >= 0.0:
                                _reached = True
            if _reached:
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
        if t_start is not None and t - t_start > WP_TIMEOUT:
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
    if plan_times:
        _avg_ms = float(np.mean(plan_times)) * 1000.0
        _med_ms = float(np.median(plan_times)) * 1000.0
        _max_ms = float(np.max(plan_times)) * 1000.0
        _hz = 1000.0 / max(_med_ms, 1e-3)
    else:
        _avg_ms = _med_ms = _max_ms = _hz = 0.0
    _brk = getattr(mpc, '_last_plan_times', None)
    if _brk:
        _bavg = {k: float(_brk.get(k, 0.0))
                 for k in ('upd_ms', 'scan_ms', 'shift_ms', 'sync_ms')}
    else:
        _bavg = {}
    print(f'[NOROS] RESULT version={os.environ.get("S10_VER","?")} '
          f'wp_times={ {k: round(v,1) for k,v in wp_times.items()} } '
          f'crashed={crashed} final_wp={next_idx} route_len={route_len:.1f}m '
          f'run_t={run_t:.1f}s avg_speed={avg:.2f}m/s '
          f'plan_ms={_avg_ms:.0f}(med{_med_ms:.0f},max{_max_ms:.0f}) ctrl_hz={_hz:.1f} '
          f'brk=upd{_bavg.get("upd_ms",0):.0f}/scan{_bavg.get("scan_ms",0):.0f}/'
          f'shift{_bavg.get("shift_ms",0):.0f}/sync{_bavg.get("sync_ms",0):.0f}', flush=True)


if __name__ == '__main__':
    main()
