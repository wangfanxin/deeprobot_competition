"""plot_traj_speed.py -- 画 wp0-33 轨迹 xy 图，颜色表示速度。"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PKG = os.path.join(REPO, 'src', 'S10_sdk_deploy')
for _p in (HERE, PKG, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

XML = os.path.join(PKG, 'S10_description/s10_mjcf/mjcf/S10_track.xml')


def load_waypoints():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    wps = []
    for gid in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if name.startswith('track_waypoint_'):
            idx = int(name[len('track_waypoint_'):].split('_')[0])
            wps.append((idx, d.geom_xpos[gid].copy()))
    wps.sort()
    return np.asarray([w for _, w in wps])


def main():
    traj_path = sys.argv[1] if len(sys.argv) > 1 else 'tmp_cruise_traj.npy'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'tmp/traj_xy_speed.png'
    a = np.load(traj_path)
    x, y, speed = a[:, 1], a[:, 2], a[:, 6]
    wp = load_waypoints()

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(x, y, c=speed, cmap='turbo', s=3, linewidths=0,
                    vmin=0.0, vmax=max(float(speed.max()), 1.0))
    ax.plot(wp[:, 0], wp[:, 1], 'k.--', markersize=6, lw=0.8, alpha=0.7)
    for i, p in enumerate(wp):
        ax.annotate(str(i), (p[0], p[1]), fontsize=6, color='black',
                    ha='center', va='bottom')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('wp0-33 trajectory (color = speed [m/s])')
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(sc, ax=ax, label='speed [m/s]')
    fig.tight_layout()
    _od = os.path.dirname(out_path)
    if _od:
        os.makedirs(_od, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print('saved', out_path)


if __name__ == '__main__':
    main()
