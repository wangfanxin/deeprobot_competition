# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = "        except Exception:\n            return lam_ref"
assert old in src
new = "        except Exception as _e:\n            print('QPTEST EXC:', repr(_e), flush=True)\n            import traceback; traceback.print_exc()\n            return lam_ref"
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")