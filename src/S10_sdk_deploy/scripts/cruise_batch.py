# -*- coding: utf-8 -*-
"""cruise batch 测试：一个进程复用 MPC，连续跑多个 nav 参数版本。
JIT 编译只一次，大幅缩短多版本迭代时间。"""
import os, sys, time
import numpy as np
import mujoco

PKG = '/home/wfx/DR_competition/deeprobot_competition/src/S10_sdk_deploy'
sys.path.insert(0, PKG)
sys.path.insert(0, '/home/wfx/DR_competition/dial-mpc')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',
                      os.path.expanduser('~/.cache/s10_dial_mpc'))
# 持久化缓存目录（JAX 原生环境变量，jax import 前生效）

from s10_mpc.mpc_controller import MPCController
from s10_mpc.auto_nav import AutoNavFollower

DT = 0.005
MAX_SIM = float(os.environ.get('S10_TEST_MAX_SIM', '90'))
MAX_WP = 5
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

def _stair_busy():
    """stair 会话在跑或 GPU 显存被占 → True。"""
    try:
        import subprocess
        r = subprocess.run(
            ['bash', '-lc', "ps aux | grep test_auto_nav | grep -v grep | wc -l"],
            capture_output=True, text=True, timeout=5)
        n = int(r.stdout.strip())
        r2 = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        mem = int(r2.stdout.strip().split()[0])
        return n > 0 or mem >= 5000
    except Exception:
        return True

def wait_gpu_free(max_wait=60):
    """最多等 max_wait 秒；等不到也返回（用户授权并行一个测试）。"""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if not _stair_busy():
            return True
        time.sleep(5)
    return False

# 每个版本的 nav 参数（env 覆盖）。MPC 层参数固定（真实一致）。
VERSIONS = [
    dict(ver='nr126b_r1', env={'S10_AUTO_VMAX': '5.0', 'S10_AUTO_VYAW_MAX': '3.0', 'S10_AUTO_YAW_GAIN': '2.5', 'S10_AUTO_LOOKAHEAD': '1.5', 'S10_CURVE_DECEL_AHEAD': '5.0', 'S10_CURVE_SWING_WINDOW': '6.0', 'S10_CURVE_SWING_VX': '3.2', 'S10_AUTO_BIGERR_VX': '1.8', 'S10_AUTO_TURN_VX': '2.2', 'S10_MPC_ANG_W': '60', 'S10_GLOBAL_TANGENT_K': '0.7', 'S10_MPC_H_CRUISE': '22', 'S10_USE_REF_PATH': '1', 'S10_MPC_W_PATH': '30', 'S10_MPC_W_PROG': '5'}),
    dict(ver='nr126b_r2', env={'S10_AUTO_VMAX': '5.0', 'S10_AUTO_VYAW_MAX': '3.0', 'S10_AUTO_YAW_GAIN': '2.5', 'S10_AUTO_LOOKAHEAD': '1.5', 'S10_CURVE_DECEL_AHEAD': '5.0', 'S10_CURVE_SWING_WINDOW': '6.0', 'S10_CURVE_SWING_VX': '3.2', 'S10_AUTO_BIGERR_VX': '1.8', 'S10_AUTO_TURN_VX': '2.2', 'S10_MPC_ANG_W': '60', 'S10_GLOBAL_TANGENT_K': '0.7', 'S10_MPC_H_CRUISE': '22', 'S10_USE_REF_PATH': '1', 'S10_MPC_W_PATH': '30', 'S10_MPC_W_PROG': '5'}),
    dict(ver='nr126b_r3', env={'S10_AUTO_VMAX': '5.0', 'S10_AUTO_VYAW_MAX': '3.0', 'S10_AUTO_YAW_GAIN': '2.5', 'S10_AUTO_LOOKAHEAD': '1.5', 'S10_CURVE_DECEL_AHEAD': '5.0', 'S10_CURVE_SWING_WINDOW': '6.0', 'S10_CURVE_SWING_VX': '3.2', 'S10_AUTO_BIGERR_VX': '1.8', 'S10_AUTO_TURN_VX': '2.2', 'S10_MPC_ANG_W': '60', 'S10_GLOBAL_TANGENT_K': '0.7', 'S10_MPC_H_CRUISE': '22', 'S10_USE_REF_PATH': '1', 'S10_MPC_W_PATH': '30', 'S10_MPC_W_PROG': '5'}),
    dict(ver='nr126b_r4', env={'S10_AUTO_VMAX': '5.0', 'S10_AUTO_VYAW_MAX': '3.0', 'S10_AUTO_YAW_GAIN': '2.5', 'S10_AUTO_LOOKAHEAD': '1.5', 'S10_CURVE_DECEL_AHEAD': '5.0', 'S10_CURVE_SWING_WINDOW': '6.0', 'S10_CURVE_SWING_VX': '3.2', 'S10_AUTO_BIGERR_VX': '1.8', 'S10_AUTO_TURN_VX': '2.2', 'S10_MPC_ANG_W': '60', 'S10_GLOBAL_TANGENT_K': '0.7', 'S10_MPC_H_CRUISE': '22', 'S10_USE_REF_PATH': '1', 'S10_MPC_W_PATH': '30', 'S10_MPC_W_PROG': '5'}),
    dict(ver='nr128', env={'S10_AUTO_VMAX': '5.0', 'S10_AUTO_VYAW_MAX': '3.0', 'S10_AUTO_YAW_GAIN': '2.5', 'S10_AUTO_LOOKAHEAD': '1.5', 'S10_CURVE_DECEL_AHEAD': '5.0', 'S10_CURVE_SWING_WINDOW': '6.0', 'S10_CURVE_SWING_VX': '3.4', 'S10_AUTO_BIGERR_VX': '1.8', 'S10_AUTO_TURN_VX': '2.2', 'S10_MPC_ANG_W': '60', 'S10_GLOBAL_TANGENT_K': '0.7', 'S10_MPC_H_CRUISE': '22', 'S10_USE_REF_PATH': '1', 'S10_MPC_W_PATH': '30', 'S10_MPC_W_PROG': '5'}),
    dict(ver='nr128b', env={'S10_AUTO_VMAX': '5.0', 'S10_AUTO_VYAW_MAX': '3.0', 'S10_AUTO_YAW_GAIN': '2.5', 'S10_AUTO_LOOKAHEAD': '1.5', 'S10_CURVE_DECEL_AHEAD': '5.0', 'S10_CURVE_SWING_WINDOW': '6.0', 'S10_CURVE_SWING_VX': '3.2', 'S10_AUTO_BIGERR_VX': '1.8', 'S10_AUTO_TURN_VX': '2.2', 'S10_MPC_ANG_W': '60', 'S10_GLOBAL_TANGENT_K': '0.7', 'S10_MPC_H_CRUISE': '22', 'S10_USE_REF_PATH': '1', 'S10_MPC_W_PATH': '40', 'S10_MPC_W_PROG': '5'}),
]

_mpc_ref = {}
def run_version(m, d, mpc, ver, env, waypoints):
    for k, v in env.items():
        os.environ[k] = v
    h_cruise = env.get('S10_MPC_H_CRUISE')
    ndiff = env.get('S10_MPC_NDIFFUSE')
    leg_sig = env.get('S10_LEG_SIGMA_SCALE')
    wh_sig = env.get('S10_WHEEL_SIGMA_SCALE')
    ang_w = env.get('S10_MPC_ANG_W')
    vel_scale = env.get('S10_MPC_VEL_SCALE')
    # 每版本重建 MPC：保证干净初始状态（rng/Y 不延续），否则多版本
    # 复测被 MPC 状态污染（batch13 假波动根因）。构建 ~2s + 缓存 JIT ~5s。
    need_rebuild = True
    if need_rebuild:
        # MPC 重建（构建期参数 H/NDIFFUSE/sigma/ANG_W 均从 env 读取）
        import time as _t
        t0 = _t.time()
        mpc = MPCController(MPC_YAML)
        d.qpos[7:23] = JOINT_INIT
        d.qpos[0:3] = [0.0, -2.5, 0.2]
        init_yaw = float(os.environ.get('S10_INIT_YAW', '0.0'))
        if abs(init_yaw) > 1e-3:
            # 模拟"站起时已转向 wp0→1 方向"（验证起步转向收益）
            d.qpos[3:7] = [np.cos(init_yaw / 2.0), 0, 0,
                           np.sin(init_yaw / 2.0)]
        else:
            d.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        mpc.init_state(np.asarray(d.qpos[:23], dtype=np.float32),
                       np.asarray(d.qvel[:22], dtype=np.float32))
        mpc.set_cmd(0.0, 0.0, 0.0)
        _mpc_ref['mpc'] = mpc
        for k in ('S10_MPC_H_CRUISE_ACTIVE', 'S10_MPC_NDIFFUSE_ACTIVE',
                  'S10_LEG_SIGMA_ACTIVE', 'S10_WHEEL_SIGMA_ACTIVE',
                  'S10_MPC_ANG_W_ACTIVE', 'S10_MPC_VEL_SCALE_ACTIVE'):
            os.environ.pop(k, None)
        if h_cruise:
            os.environ['S10_MPC_H_CRUISE_ACTIVE'] = h_cruise
        if ndiff:
            os.environ['S10_MPC_NDIFFUSE_ACTIVE'] = ndiff
        if leg_sig:
            os.environ['S10_LEG_SIGMA_ACTIVE'] = leg_sig
        if wh_sig:
            os.environ['S10_WHEEL_SIGMA_ACTIVE'] = wh_sig
        if ang_w:
            os.environ['S10_MPC_ANG_W_ACTIVE'] = ang_w
        if vel_scale:
            os.environ['S10_MPC_VEL_SCALE_ACTIVE'] = vel_scale
        print(f'[BATCH] {ver}: MPC rebuilt (H={h_cruise} ndiff={ndiff} '
              f'angw={ang_w} sig={leg_sig}/{wh_sig}) '
              f'({_t.time()-t0:.0f}s)', flush=True)
    # 重置仿真
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.qpos[7:23] = JOINT_INIT
    d.qpos[0:3] = [0.0, -2.5, 0.2]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    wps = waypoints
    fol = AutoNavFollower(
        wps,
        max_speed=float(env.get('S10_AUTO_VMAX', '5.0')),
        vyaw_max=float(env.get('S10_AUTO_VYAW_MAX', '2.0')),
        yaw_gain=float(env.get('S10_AUTO_YAW_GAIN', '2.5')),
        lookahead=float(env.get('S10_AUTO_LOOKAHEAD', '1.5')),
    )
    mpc.set_yaw_gain_lo(20.0)
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
    traj_file = f'/tmp/traj_{ver}.csv'
    _tf = open(traj_file, 'w')
    _tf.write('t,x,y,yaw,next_idx,err,d_wp,vx,vyaw,cte,s_cur\n')
    dbg_cnt = 0
    plan_times = []
    t0 = time.time()
    while t < MAX_SIM:
        step = int(t / DT)
        if not auto_active:
            q = d.qpos[7:23].reshape(-1, 1)
            dq = d.qvel[6:22].reshape(-1, 1)
            tau = (80.0 * (STAND_TARGET.reshape(-1, 1) - q) - 2.0 * dq).flatten()
            tau[3::4] = -0.3 * dq[3::4].flatten()
            # 站起阶段 base yaw 预转向（S10_STAND_TURN=1，真实力矩驱动：
            # 轮子差速让身体转到 wp0→1 方向，可移植到真实节点）
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
            if t >= 3.0:
                qq = np.asarray(d.qpos[:23], dtype=np.float32)
                qqd = np.asarray(d.qvel[:22], dtype=np.float32)
                last_act = mpc.plan_once(qq, qqd, t)
                t_cmd0 = t
                auto_active = True
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
                # ref_path 注入（与真实节点一致）：MPC reward 跟踪路径，
                # 减少横向偏差 → err 小 → 弯道减速少。
                if os.environ.get('S10_USE_REF_PATH', '0') == '1':
                    ref = fol.ref_path_3d(pos, next_idx, local_map=None)
                    mpc.set_ref_path(ref if ref is not None else [],
                                     valid=ref is not None)
                dbg_cnt += 1
                if dbg_cnt % 10 == 0:
                    _tf.write(f'{t:.2f},{pos[0]:.3f},{pos[1]:.3f},'
                              f'{yaw:.3f},{next_idx},{fol._last_err:.3f},'
                              f'{fol._last_dwp:.3f},{vx:.3f},{vyaw:.3f},'
                              f'{getattr(fol,"_last_cte",0.0):.3f},'
                              f'{fol._s_cur:.3f}\n')
            if step % plan_interval == 0:
                _pt0 = time.time()
                last_act = mpc.plan_once(qq, qqd, t)
                plan_times.append(time.time() - _pt0)
            mpc.latest_tau = mpc.compute_tau(last_act, qq, qqd)
            d.ctrl[:] = np.asarray(mpc.latest_tau, dtype=np.float64)
        mujoco.mj_step(m, d)
        t += DT
        if next_idx < len(wps):
            rp = d.xpos[track_body][:2]
            dist = float(np.linalg.norm(rp - wps[next_idx][:2]))
            if dist <= 0.5:
                if next_idx == 0 and t_start is None:
                    t_start = t
                else:
                    wp_times[next_idx] = t - (t_start or 0.0)
                next_idx += 1
                if next_idx >= MAX_WP:
                    break
        quat = d.xquat[track_body]
        w, x, y, z = quat
        roll = float(np.arctan2(2.0*(w*x + y*z), 1.0-2.0*(x*x + y*y)))
        xyz = d.xpos[track_body]
        if abs(roll) > 0.7 or xyz[2] < 0.12:
            crashed = f'roll={roll:.2f} z={xyz[2]:.3f}'
            break
        if t_start is not None and t - t_start > 60.0:
            crashed = 'wp_timeout60'
            break
        if next_idx != last_progress_idx:
            last_progress_idx = next_idx
            last_progress_t = t
        if last_progress_t is not None and t - last_progress_t > 15.0:
            crashed = 'stuck15'
            break
    _tf.close()
    route_len = float(np.sum(np.linalg.norm(
        np.diff(wps[:MAX_WP + 1], axis=0), axis=1)))
    run_t = (t - t_cmd0) if t_cmd0 else (t - (t_start or 0.0))
    avg = route_len / run_t if run_t > 0 else 0.0
    if plan_times:
        _avg_ms = float(np.mean(plan_times)) * 1000.0
        _hz = 1000.0 / max(_avg_ms, 1e-3)
        _max_ms = float(np.max(plan_times)) * 1000.0
    else:
        _avg_ms, _hz, _max_ms = 0.0, 0.0, 0.0
    print(f'[BATCH] {ver}: wall={time.time()-t0:.0f}s '
          f'wp_times={ {k:round(v,1) for k,v in wp_times.items()} } '
          f'crashed={crashed} final_wp={next_idx} avg={avg:.2f}m/s '
          f'plan_ms={_avg_ms:.0f}(max{_max_ms:.0f}) ctrl_hz={_hz:.1f}',
          flush=True)
    return wp_times, crashed, next_idx, avg

def main():
    m = mujoco.MjModel.from_xml_path(XML)
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    # 先设初始位姿并 forward，geom_xpos 才有有效值
    d.qpos[7:23] = JOINT_INIT
    d.qpos[0:3] = [0.0, -2.5, 0.2]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    waypoints = []
    for gid in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if name and name.startswith('track_waypoint_'):
            idx = int(name[len('track_waypoint_'):].split('_')[0])
            waypoints.append((idx, d.geom_xpos[gid].copy()))
    waypoints.sort()
    wps = np.array([w for _, w in waypoints], dtype=np.float64)
    global track_body
    track_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    # stair 感知：最多等 60s，等不到就并行跑（用户授权）
    if wait_gpu_free(60):
        print('[BATCH] GPU 空闲，独占运行', flush=True)
    else:
        print('[BATCH] GPU 被 stair 占用，并行运行', flush=True)
    t0 = time.time()
    mpc = MPCController(MPC_YAML)
    d.qpos[7:23] = JOINT_INIT
    d.qpos[0:3] = [0.0, -2.5, 0.2]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    mpc.init_state(np.asarray(d.qpos[:23], dtype=np.float32),
                   np.asarray(d.qvel[:22], dtype=np.float32))
    mpc.set_cmd(0.0, 0.0, 0.0)
    print(f'[BATCH] MPC ready ({time.time()-t0:.1f}s)', flush=True)
    for v in VERSIONS:
        wait_gpu_free(45)
        run_version(m, d, mpc, v['ver'], v['env'], wps)
        if 'mpc' in _mpc_ref:
            mpc = _mpc_ref.pop('mpc')

if __name__ == '__main__':
    main()
