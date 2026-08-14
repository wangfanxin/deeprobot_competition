#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
S10_GLOBAL_TANGENT_K=0.5 S10_REF_DUMP=tmp/ref_path_cfg_base.npz S10_DUMP_ONLY=1 bash tmp/run_v780.sh >/dev/null 2>&1; echo "base rc=$?"
S10_GLOBAL_TANGENT_K=0.5 S10_PATH_STRAIGHT_LEN=0 S10_REF_DUMP=tmp/ref_path_cfg_nostraight.npz S10_DUMP_ONLY=1 bash tmp/run_v780.sh >/dev/null 2>&1; echo "nostraight rc=$?"
S10_GLOBAL_TANGENT_K=0.5 S10_PATH_PERP_K=0.5 S10_REF_DUMP=tmp/ref_path_cfg_perp05.npz S10_DUMP_ONLY=1 bash tmp/run_v780.sh >/dev/null 2>&1; echo "perp05 rc=$?"
S10_GLOBAL_TANGENT_K=0.5 S10_PATH_PERP_K=1.0 S10_REF_DUMP=tmp/ref_path_cfg_perp10.npz S10_DUMP_ONLY=1 bash tmp/run_v780.sh >/dev/null 2>&1; echo "perp10 rc=$?"
/home/wfx/DR_competition/.venv/bin/python - <<'PYEOF'
import numpy as np
for tag in ('base','nostraight','perp05','perp10'):
    z = np.load('tmp/ref_path_cfg_%s.npz' % tag)
    pts = z['path_pts']; curv = z['path_curv']; vlim = z['path_vlim']; wp_s = z['path_wp_s']; cum = z['path_cum']
    k = np.abs(curv); R = np.where(k > 1e-4, 1.0/np.maximum(k,1e-4), 99.0)
    rmin_wp = []
    for i in range(len(wp_s)):
        a = int(np.searchsorted(cum, wp_s[i], side='right')-1)
        lo = max(0, a-int(1.2/0.05)); hi = min(len(R)-1, a+int(1.2/0.05))
        rmin_wp.append(R[lo:hi+1].min())
    rmin_wp = np.array(rmin_wp)
    print('%-10s | R_global_min=%.2f | 航点R_median=%.2f | vlim_min=%.2f mean=%.2f | 航点R<1m=%d' % (
        tag, R.min(), np.median(rmin_wp[rmin_wp<99]), vlim.min(), vlim.mean(), int(np.sum(rmin_wp<1.0))))
PYEOF