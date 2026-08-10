#!/usr/bin/env python3
"""CarVMC limit calibration (fixed stand-up, matches cruise_vmc_noros start)."""
import os, sys, json
import numpy as np
import mujoco

PKG = '/home/wfx/DR_competition/deeprobot_competition/src/S10_sdk_deploy'
sys.path.insert(0, PKG)
from s10_mpc.vmc_legs import CarVMC, WHEEL_BODY

# 简化平地 XML（仅 S10.xml，100x100 平地）：自动生成到 /tmp，不污染仓库。
# 注意：这是简化 XML；全赛道验证请用原版 S10_track.xml（scene+track_overlay）。
FLAT_XML = '/tmp/flat_calib.xml'
if not os.path.exists(FLAT_XML):
    with open(FLAT_XML, 'w') as f:
        f.write('<mujoco model="flat_calib">\n'
                '  <include file="%s/S10_description/s10_mjcf/mjcf/S10.xml"/>\n'
                '</mujoco>\n' % PKG)
XML = os.environ.get('S10_CALIB_XML', FLAT_XML)
DT = 0.005

m = mujoco.MjModel.from_xml_path(XML)
m.opt.timestep = DT
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

START_POS = np.array([0.0, -2.5, 0.2])
START_YAW = np.pi / 8.0
LEGS_START = np.array([-0.438, -1.16, 2.45, 0.0,
                        0.438, -1.16, 2.45, 0.0,
                       -0.438,  1.16, -2.45, 0.0,
                        0.438,  1.16, -2.45, 0.0])
STAND_TARGET = np.array([-0.05, -1.16, 2.30, 0.0,
                          0.05, -1.16, 2.30, 0.0,
                         -0.05,  1.16, -2.30, 0.0,
                          0.05,  1.16, -2.30, 0.0])
t_global = [0.0]

def reset():
    d.qpos[0:3] = START_POS
    d.qpos[3:7] = [np.cos(START_YAW / 2), 0, 0, np.sin(START_YAW / 2)]
    d.qpos[7:23] = LEGS_START
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

def stand_up(dur=2.0):
    t0 = t_global[0]
    while t_global[0] - t0 < dur:
        q = d.qpos[7:23].reshape(-1, 1); dq = d.qvel[6:22].reshape(-1, 1)
        tau = (80.0 * (STAND_TARGET.reshape(-1, 1) - q) - 2.0 * dq).flatten()
        tau[[3, 7, 11, 15]] = -0.3 * dq[[3, 7, 11, 15]].flatten()
        d.ctrl[:] = tau; mujoco.mj_step(m, d); t_global[0] += DT

def body_state():
    fx, fy = d.xmat[1][0], d.xmat[1][1]
    vx = float(np.dot(d.qvel[0:3], [fx, fy, 0.0]))
    q = d.xquat[1]
    yaw = float(np.arctan2(2 * (q[3] * q[0] + q[1] * q[2]),
                           1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    roll = float(np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]),
                            1 - 2 * (q[1] ** 2 + q[2] ** 2)))
    wz = np.array([d.xpos[WHEEL_BODY[i], 2] for i in range(4)])
    return vx, float(d.qvel[5]), yaw, roll, wz, float(d.xpos[1][2])

def step_cmd(drv, vx_cmd, om_cmd, roll_k=1.0):
    w = np.array([d.xpos[WHEEL_BODY[i]] for i in range(4)])
    wv = np.array([d.cvel[WHEEL_BODY[i]][0:3] for i in range(4)])
    roll_tar = float(np.clip(-0.06 * om_cmd * abs(vx_cmd), -0.06, 0.06)) * roll_k
    cmd = dict(vx=vx_cmd, omega=om_cmd, roll_tar=roll_tar, pitch_tar=0.0,
               hop=np.zeros(4), step_lift=np.zeros(4), yaw_scale=1.0)
    tau = drv.compute_tau(np.asarray(d.qpos), np.asarray(d.qvel),
                          w, wv, cmd, np.zeros(4), DT)
    d.ctrl[:] = tau; mujoco.mj_step(m, d); t_global[0] += DT

def run_phase(drv, vx_cmd, om_cmd, dur, settle=0.0, roll_k=1.0,
              abort_on_fail=False):
    t0 = t_global[0]
    rows = []
    last_pos = None; last_yaw = None; last_tt = None
    while t_global[0] - t0 < dur + settle:
        step_cmd(drv, vx_cmd, om_cmd, roll_k)
        if int((t_global[0] - t0) * 20) % 2 == 0:
            vx, om, yaw, roll, wz, bz = body_state()
            tt = t_global[0] - t0
            sp = 0.0; yw = float(om)
            if last_pos is not None:
                dp = np.linalg.norm(np.array(d.xpos[1][0:2]) - last_pos)
                sp = dp / max(tt - last_tt, 1e-4)
            if last_yaw is not None:
                dy = yaw - last_yaw
                dy = (dy + np.pi) % (2 * np.pi) - np.pi
                yw = dy / max(tt - last_tt, 1e-4)
            rows.append((tt, sp, yw, roll, wz[0], wz[1], wz[2], wz[3], bz))
            last_pos = np.array(d.xpos[1][0:2]); last_yaw = yaw; last_tt = tt
        if abort_on_fail:
            vx, om, yaw, roll, wz, bz = body_state()
            lift = float(np.mean(wz[[0, 2]]) - np.mean(wz[[1, 3]]))
            if abs(roll) > 0.32 or abs(lift) > 0.11 or bz < 0.35 or abs(om) > 3.0:
                break
    rows = np.array(rows) if rows else np.zeros((0, 9))
    if len(rows) == 0:
        return dict(vx_med=0.0, om_med=0.0, om_std=99.0, roll_med=0.0,
                    roll_max=0.0, lift_med=0.0, base_z=0.0, t=t_global[0]-t0)
    tail = rows[int(len(rows) * 0.5):]
    om_med = float(np.median(tail[:, 2]))
    om_std = float(np.std(rows[int(len(rows) * 0.3):, 2]))
    zl = (tail[:, 4] + tail[:, 6]) / 2.0 - (tail[:, 5] + tail[:, 7]) / 2.0
    return dict(
        vx_med=float(np.median(tail[:, 1])),
        om_med=om_med, om_std=om_std,
        roll_med=float(np.median(tail[:, 3])),
        roll_max=float(np.max(np.abs(rows[:, 3]))),
        lift_med=float(np.median(zl)),
        base_z=float(np.median(tail[:, 8])), t=t_global[0]-t0)

def stable(r, cmd_om):
    return (abs(r['roll_max']) < 0.32 and abs(r['lift_med']) < 0.11
            and r['base_z'] > 0.35 and abs(r['om_med']) > 0.5 * abs(cmd_om)
            and r['om_std'] < 0.5)

def sample_speed(drv, vx_cmd, dur):
    t0 = t_global[0]; last_t = t0
    ts, vxs = [], []
    last_pos = None
    while t_global[0] - t0 < dur:
        step_cmd(drv, vx_cmd, 0.0)
        vx, om, yaw, roll, wz, bz = body_state()
        if t_global[0] - last_t >= 0.1:
            sp = 0.0
            if last_pos is not None:
                dp = np.linalg.norm(np.array(d.xpos[1][0:2]) - last_pos)
                sp = dp / (t_global[0] - last_t)
            ts.append(t_global[0] - t0); vxs.append(sp); last_t = t_global[0]
            last_pos = np.array(d.xpos[1][0:2])
    return np.array(ts), np.array(vxs)

def t_at(vxs, ts, v):
    if len(vxs) == 0:
        return None
    i = int(np.searchsorted(vxs, v))
    if i <= 0:
        return float(ts[0])
    if i >= len(vxs):
        return float(ts[-1])
    return float(ts[i-1] + (v - vxs[i-1]) / (vxs[i] - vxs[i-1]) * (ts[i] - ts[i-1]))

def calib_accel(drv):
    print('\n=== Longitudinal accel/brake limits ===', flush=True)
    out = {}
    reset(); stand_up()
    ts, vxs = sample_speed(drv, 3.5, 5.0)
    acc = np.diff(vxs) / np.diff(ts)
    out['acc_max'] = float(np.max(acc))
    out['acc_med'] = float(np.median(acc[acc > 0.1])) if np.any(acc > 0.1) else 0.0
    t10 = t_at(vxs, ts, 0.35); t90 = t_at(vxs, ts, 3.15); t35 = t_at(vxs, ts, 3.5)
    out['t_10_90'] = round(t90 - t10, 2) if (t10 is not None and t90 is not None) else None
    out['t_to_3.5'] = round(t35, 2) if t35 is not None else None
    out['v_end'] = float(vxs[-1]) if len(vxs) else 0.0
    print('  accel 0->3.5: ax_max=%.2f m/s2 t10-90=%s t_to_3.5=%s v_end=%.2f' % (
        out['acc_max'], out['t_10_90'], out['t_to_3.5'], out['v_end']), flush=True)

    reset(); stand_up()
    run_phase(drv, 3.5, 0.0, 2.0)
    t0 = t_global[0]; last_t = t0
    ts, vxs = [], []
    last_pos = np.array(d.xpos[1][0:2])
    while t_global[0] - t0 < 3.0:
        step_cmd(drv, 0.0, 0.0)
        vx, om, yaw, roll, wz, bz = body_state()
        if t_global[0] - last_t >= 0.1:
            dp = np.linalg.norm(np.array(d.xpos[1][0:2]) - last_pos)
            sp = dp / (t_global[0] - last_t)
            ts.append(t_global[0] - t0); vxs.append(sp); last_t = t_global[0]
            last_pos = np.array(d.xpos[1][0:2])
    ts, vxs = np.array(ts), np.array(vxs)
    dec = -np.diff(vxs) / np.diff(ts)
    out['dec_max'] = float(np.max(dec)) if len(dec) else 0.0
    out['dec_med'] = float(np.median(dec[dec > 0.1])) if np.any(dec > 0.1) else 0.0
    out['dist_stop'] = float(np.sum(vxs[:-1] * np.diff(ts)))
    print('  brake 3.5->0: ax_max=%.2f m/s2 dist=%.2f m v_start=%.2f' % (
        out['dec_max'], out['dist_stop'], float(vxs[0]) if len(vxs) else 0.0), flush=True)
    return out

def calib_turn(drv, speeds):
    print('\n=== Turn envelope (omega_max per vx) ===', flush=True)
    om_cands = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    out = {}
    for vx in speeds:
        print('--- vx=%.1f ---' % vx, flush=True)
        reset(); stand_up()
        r0 = run_phase(drv, vx, 0.0, 2.2, settle=0.4)
        print('  straight: vx=%.2f base_z=%.2f om=%.2f' % (
            r0['vx_med'], r0['base_z'], r0['om_med']), flush=True)
        best = None
        for om in om_cands:
            r = run_phase(drv, vx, om, 1.2, roll_k=1.0, abort_on_fail=True)
            ok = stable(r, om)
            print('  om_cmd=%.1f om_act=%.2f roll_max=%.2f lift=%.3f '
                  'std=%.2f vx=%.2f base_z=%.2f -> %s' % (
                      om, r['om_med'], r['roll_max'], r['lift_med'],
                      r['om_std'], r['vx_med'], r['base_z'],
                      'OK' if ok else 'FAIL'), flush=True)
            if ok:
                best = (om, r)
            else:
                break
        if best is None:
            out[str(vx)] = dict(om_max=0.0, om_act=0.0, a_lat=0.0, vx_act=0.0,
                                roll=0.0, note='even 0.4 fails')
        else:
            om_c, r = best
            out[str(vx)] = dict(om_max=om_c, om_act=round(r['om_med'], 3),
                                a_lat=round(r['om_med'] * r['vx_med'], 2),
                                vx_act=round(r['vx_med'], 2),
                                roll=round(r['roll_max'], 3))
        reset()
    return out

def main():
    drv = CarVMC()
    res = {}
    res['accel'] = calib_accel(drv)
    res['turn'] = calib_turn(drv, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    print('\n=== RESULT ===', flush=True)
    print(json.dumps(res, indent=2), flush=True)
    with open('/tmp/calib_limits.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('saved /tmp/calib_limits.json', flush=True)

if __name__ == '__main__':
    main()