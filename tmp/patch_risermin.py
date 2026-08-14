#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# replace hardcoded 0.085 thresholds with env var
src = src.replace("""            if _dhv <= 0.085:
                continue""",
"""            _dhmin = float(os.environ.get("S10_STAIR_RISER_MIN", "0.085"))
            if _dhv <= _dhmin:
                continue""")

# face_place_z selection
src = src.replace("""            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                if _dhv <= 0.085:
                    continue
                _dd = float(np.dot(_ax_xy - _rp, _tng))
                if abs(_dd) < abs(_best_d):""",
"""            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                _dhmin2 = float(os.environ.get("S10_STAIR_RISER_MIN", "0.085"))
                if _dhv <= _dhmin2:
                    continue
                _dd = float(np.dot(_ax_xy - _rp, _tng))
                if abs(_dd) < abs(_best_d):""")

# _qp_solve wrench riser selection uses the passed stance; no dhv there. Check z_ref loop:
src = src.replace("""                    for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                        if _dhv <= 0.085:
                            continue
                        _ddz = float(np.dot(wheel_xyz[_i, :2] - _rp, _tng))""",
"""                    for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                        _dhmin3 = float(os.environ.get("S10_STAIR_RISER_MIN", "0.085"))
                        if _dhv <= _dhmin3:
                            continue
                        _ddz = float(np.dot(wheel_xyz[_i, :2] - _rp, _tng))""")

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")