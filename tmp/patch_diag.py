#!/usr/bin/env python3
import io

# ---------- stair_wbc_qp.py: expose per-wheel d/top/done for diagnostics ----------
p1 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
s1 = io.open(p1, "r", encoding="utf-8").read()
old = """        d = np.zeros(4); top = np.zeros(4)
        for i in range(4):
            d[i], top[i] = self._nearest_riser(wheel_xyz[i, :2])
        wz = wheel_xyz[:, 2]"""
new = """        d = np.zeros(4); top = np.zeros(4)
        for i in range(4):
            d[i], top[i] = self._nearest_riser(wheel_xyz[i, :2])
        wz = wheel_xyz[:, 2]
        # v965: 诊断——每轮到棱距离/台面顶/完成标志（STAIRDBG 用）
        self._dbg_d = d.copy()
        self._dbg_top = top.copy()
        self._dbg_done = self._done.copy() if hasattr(self, "_done") else np.zeros(4)"""
assert old in s1, "qp anchor missing"
s1 = s1.replace(old, new)
io.open(p1, "w", encoding="utf-8").write(s1)

# ---------- stair_vmc_noros.py: extend STAIRDBG with per-wheel d ----------
p2 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py"
s2 = io.open(p2, "r", encoding="utf-8").read()
old = """                    print('[STAIRDBG] t=%.1f pos=(%.2f,%.2f) pitch=%.2f roll=%.2f '
                          'bz=%.3f wz=%s sl=%s terrF=%.3f terrR=%.3f fn=%s '
                          'cmd=(%.2f,%.2f)' % (t, body_pos[0], body_pos[1],
                             _stpitch, _stroll, body_pos[2],
                             np.round([d.xpos[WHEEL_BODY[i], 2]
                                       for i in range(4)], 2),
                             np.round(step_lift, 1), terr[0], terr[2],
                             np.round(_fn9, 0), vx_c, om_c), flush=True)"""
new = """                    _dbgd = getattr(vmc, "_dbg_d", None)
                    _dbgdn = getattr(vmc, "_dbg_done", None)
                    _dbgs = ''
                    if _dbgd is not None:
                        _dbgs = ' d=%s dn=%s' % (
                            np.round(np.asarray(_dbgd), 2).tolist(),
                            np.round(np.asarray(_dbgdn), 0).tolist())
                    print('[STAIRDBG] t=%.1f pos=(%.2f,%.2f) pitch=%.2f roll=%.2f '
                          'bz=%.3f wz=%s sl=%s terrF=%.3f terrR=%.3f fn=%s '
                          'cmd=(%.2f,%.2f)%s' % (t, body_pos[0], body_pos[1],
                             _stpitch, _stroll, body_pos[2],
                             np.round([d.xpos[WHEEL_BODY[i], 2]
                                       for i in range(4)], 2),
                             np.round(step_lift, 1), terr[0], terr[2],
                             np.round(_fn9, 0), vx_c, om_c, _dbgs), flush=True)"""
assert old in s2, "main anchor missing"
s2 = s2.replace(old, new)
io.open(p2, "w", encoding="utf-8").write(s2)
print("patched diagnostics OK")