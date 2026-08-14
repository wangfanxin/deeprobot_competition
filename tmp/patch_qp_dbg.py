# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
anchor = "            res = prob.solve()\n            if res.info.status in (\"solved\", \"solved inaccurate\"):"
assert anchor in src, "anchor missing"
debug_block = """            res = prob.solve()
            if float(os.environ.get('S10_QP_DEBUG', '0')) > 0:
                _st_ok = res.info.status in ('solved', 'solved inaccurate')
                _lam_s = (np.round(np.asarray(res.x, dtype=np.float64).reshape(4, 3), 2)
                          if _st_ok else np.zeros((4, 3)))
                print('[QP] t=%.2f st=%s ad=[%.2f %.2f %.2f %.2f %.2f %.2f] lam=%s st=%s'
                      % (self._t, str(stance_mask), a_des[0], a_des[1], a_des[2],
                         a_des[3], a_des[4], a_des[5], np.round(_lam_s, 2).tolist(),
                         res.info.status), flush=True)
            if res.info.status in (\"solved\", \"solved inaccurate\"):"""
src = src.replace(anchor, debug_block, 1)
p.write_text(src, encoding="utf-8")
print("patched OK")