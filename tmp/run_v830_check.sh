#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
S10_REF_DUMP=tmp/ref_path_v830.npz S10_DUMP_ONLY=1 bash tmp/run_v780.sh >/dev/null 2>&1; echo "fillet rc=$?"
S10_CORNER_FILLET=0 S10_REF_DUMP=tmp/ref_path_v830_old.npz S10_DUMP_ONLY=1 bash tmp/run_v780.sh >/dev/null 2>&1; echo "old rc=$?"
/home/wfx/DR_competition/.venv/bin/python - <<'PYEOF'
import numpy as np
def stats(f, tag):
    z = np.load(f)
    pts = z['path_pts']; curv = z['path_curv']; vlim = z['path_vlim']; wp_s = z['path_wp_s']; cum = z['path_cum']; wp = z['wp']
    k = np.abs(curv); R = np.where(k > 1e-4, 1.0/np.maximum(k,1e-4), 99.0)
    rmin_wp = []; dpass = []
    for i in range(len(wp)):
        a = int(np.searchsorted(cum, wp_s[i], side='right')-1)
        lo = max(0, a-int(1.2/0.05)); hi = min(len(R)-1, a+int(1.2/0.05))
        rmin_wp.append(R[lo:hi+1].min())
        dpass.append(float(np.min(np.linalg.norm(pts[:, :2]-wp[i,:2], axis=1))))
    rmin_wp = np.array(rmin_wp); dpass = np.array(dpass)
    a = int(np.searchsorted(cum, wp_s[1], side='right')-1); b = int(np.searchsorted(cum, wp_s[2], side='right')-1)
    d = wp[2,:2]-wp[1,:2]; L = np.linalg.norm(d); u = d/L
    rel = pts[a:b+1,:2]-wp[1,:2]
    bow = float(np.max(np.abs(rel[:,0]*(-u[1])+rel[:,1]*u[0])))
    print('%s: 路径长=%.1f | 航点R_med=%.2f R_min=%.2f | R<1m=%d | wp距离max=%.3f (>0.3超限=%d) | vlim_min=%.2f mean=%.2f | wp1->2弓=%.3f' % (
        tag, cum[-1], np.median(rmin_wp[rmin_wp<99]), R.min(), int(np.sum(rmin_wp<1.0)), dpass.max(), int(np.sum(dpass>0.3)), vlim.min(), vlim.mean(), bow))
stats('tmp/ref_path_v830_old.npz', 'Catmull-Rom(旧)')
stats('tmp/ref_path_v830.npz', '圆弧圆角(v830)')
PYEOF