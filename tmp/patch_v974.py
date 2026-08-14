#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# stance branch q1 clip
old = """                            _q1t += float(_dq[0]); _q2t += float(_dq[1])
                            _q1t = float(np.clip(_q1t, -1.7, -0.35))
                            _q2t = float(np.clip(_q2t, -0.2, 3.0))"""
new = """                            _q1t += float(_dq[0]); _q2t += float(_dq[1])
                            # v974: q1 上界 -0.35 挡住台面正确解(需 q1≈0)，
                            # 轮悬空 4cm 无法落地；物理限位 ±2.53，放宽到 0.3
                            _q1t = float(np.clip(_q1t, -1.7, 0.3))
                            _q2t = float(np.clip(_q2t, -0.2, 3.0))"""
assert old in src, "edit1"
src = src.replace(old, new)

# swing branch q1 clip
old = """                    q1t += float(dq[0]); q2t += float(dq[1])
                    # v929: SWING 腿 q1 正常分支（防镜像折叠过伸，同 FP v925）
                    q1t = float(np.clip(q1t, -1.1, -0.3))
                    q2t = float(np.clip(q2t, 0.5, 3.0))"""
new = """                    q1t += float(dq[0]); q2t += float(dq[1])
                    # v929: SWING 腿 q1 正常分支（防镜像折叠过伸，同 FP v925）
                    # v974: q1 上界放宽到 0.2(台面贴面解需要)，下界 -1.1 保留
                    q1t = float(np.clip(q1t, -1.1, 0.2))
                    q2t = float(np.clip(q2t, 0.5, 3.0))"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")