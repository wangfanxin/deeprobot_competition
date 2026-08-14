# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                    q1t += float(dq[0]); q2t += float(dq[1])
                    q1t = float(np.clip(q1t, -0.35, 0.9))
                    q2t = float(np.clip(q2t, 0.5, 3.0))"""
assert old in src
new = """                    q1t += float(dq[0]); q2t += float(dq[1])
                    # v929: SWING 腿 q1 正常分支（防镜像折叠过伸，同 FP v925）
                    q1t = float(np.clip(q1t, -1.1, -0.3))
                    q2t = float(np.clip(q2t, 0.5, 3.0))"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched qp ik branch")