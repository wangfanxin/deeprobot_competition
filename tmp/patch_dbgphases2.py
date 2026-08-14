#!/usr/bin/env python3
import io

# StairWBC: expose internal phases
p1 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
s1 = io.open(p1, "r", encoding="utf-8").read()
old = """        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)
        place_z = self._face_place_z(wheel_xyz, step_lift)"""
new = """        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)
        place_z = self._face_place_z(wheel_xyz, step_lift)
        self._dbg_phases = (self._sp_f, self._sp_r)"""
assert old in s1, "stw anchor"
s1 = s1.replace(old, new)
io.open(p1, "w", encoding="utf-8").write(s1)

# main script: print sp in STAIRDBG (add if missing)
p2 = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py"
s2 = io.open(p2, "r", encoding="utf-8").read()
if "_dbgp" not in s2:
    old = """                    _dbgd = getattr(vmc, "_dbg_d", None)
                    _dbgdn = getattr(vmc, "_dbg_done", None)
                    _dbgs = ''
                    if _dbgd is not None:
                        _dbgs = ' d=%s dn=%s' % (
                            np.round(np.asarray(_dbgd), 2).tolist(),
                            np.round(np.asarray(_dbgdn), 0).tolist())"""
    new = """                    _dbgd = getattr(vmc, "_dbg_d", None)
                    _dbgdn = getattr(vmc, "_dbg_done", None)
                    _dbgp = getattr(vmc, "_dbg_phases", None)
                    _dbgs = ''
                    if _dbgd is not None:
                        _dbgs = ' d=%s dn=%s' % (
                            np.round(np.asarray(_dbgd), 2).tolist(),
                            np.round(np.asarray(_dbgdn), 0).tolist())
                    if _dbgp is not None:
                        _dbgs += ' sp=%s' % (np.round(
                            np.asarray(_dbgp), 1).tolist())"""
    assert old in s2, "main anchor"
    s2 = s2.replace(old, new)
    io.open(p2, "w", encoding="utf-8").write(s2)
print("patched")