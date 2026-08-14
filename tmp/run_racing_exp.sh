#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
for cr in 1.0 1.5 2.0 2.5; do
  S10_GLOBAL_TANGENT_K=0.5 S10_RACING_CUT_R=$cr S10_REF_DUMP=tmp/ref_path_cr${cr}.npz S10_DUMP_ONLY=1 bash tmp/run_v780.sh >/dev/null 2>&1
  echo "cut_r=$cr rc=$?"
done
/home/wfx/DR_competition/.venv/bin/python - <<'PYEOF'
import numpy as np
for cr in (0.0, 1.0, 1.5, 2.0, 2.5):
    z = np.load('tmp/ref_path_cr%s.npz' % (cr if cr else 'base'))
    pts = z['path_pts']; curv = z['path_curv']; vlim = z['path_vlim']; wp_s = z['path_wp_s']; cum = z['path_cum']; wp = z['wp']
    k = np.abs(curv); R = np.where(k > 1e-4, 1.0/np.maximum(k,1e-4), 99.0)
    rmin_wp = []; dpass = []
    for i in range(len(wp)):
        a = int(np.searchsorted(cum, wp_s[i], side='right')-1)
        lo = max(0, a-int(1.2/0.05)); hi = min(len(R)-1, a+int(1.2/0.05))
        rmin_wp.append(R[lo:hi+1].min())
        dpass.append(float(np.min(np.linalg.norm(pts[:, :2]-wp[i,:2], axis=1))))
    rmin_wp = np.array(rmin_wp); dpass = np.array(dpass)
    # wp1->2 弓形
    a = int(np.searchsorted(cum, wp_s[1], side='right')-1); b = int(np.searchsorted(cum, wp_s[2], side='right')-1)
    d = wp[2,:2]-wp[1,:2]; L = np.linalg.norm(d); u = d/L
    rel = pts[a:b+1,:2]-wp[1,:2]
    bow = float(np.max(np.abs(rel[:,0]*(-u[1])+rel[:,1]*u[0])))
    print('cut_r=%-4s | R_min=%.2f 航点R_med=%.2f | wp距离max=%.3f | R<1m=%d | vlim_min=%.2f mean=%.2f | wp1->2弓=%.3f' % (
        cr, R.min(), np.median(rmin_wp[rmin_wp<99]), dpass.max(), int(np.sum(rmin_wp<1.0)), vlim.min(), vlim.mean(), bow))
PYEOF