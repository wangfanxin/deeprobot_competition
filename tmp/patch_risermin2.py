#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# RISER_MIN parameterization (all 0.085 thresholds -> env var)
n0 = src.count("_dhv <= 0.085")
src = src.replace("""            if _dhv <= 0.085:
                continue""",
"""            _dhmin = float(os.environ.get("S10_STAIR_RISER_MIN", "0.085"))
            if _dhv <= _dhmin:
                continue""")
src = src.replace("""            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                if _dhv <= 0.085:
                    continue
                _dd = float(np.dot(_ax_xy - _rp, _tng))""",
"""            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                _dhmin2 = float(os.environ.get("S10_STAIR_RISER_MIN", "0.085"))
                if _dhv <= _dhmin2:
                    continue
                _dd = float(np.dot(_ax_xy - _rp, _tng))""")
src = src.replace("""                    for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                        if _dhv <= 0.085:
                            continue
                        _ddz = float(np.dot(wheel_xyz[_i, :2] - _rp, _tng))""",
"""                    for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                        _dhmin3 = float(os.environ.get("S10_STAIR_RISER_MIN", "0.085"))
                        if _dhv <= _dhmin3:
                            continue
                        _ddz = float(np.dot(wheel_xyz[_i, :2] - _rp, _tng))""")
n1 = src.count("S10_STAIR_RISER_MIN")
io.open(path, "w", encoding="utf-8").write(src)
print("thresholds 0.085 ->", n0, "env refs ->", n1)