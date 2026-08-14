#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                if _gt_hi > 0.4:
                    try:
                        # v959: 支撑腿目标压入台面 2mm（原 +0.01 余量让轮
                        # 悬空 0.01-0.06 无抓地、狗不推进实测）
                        _wzt = min(float(terrain_h[leg]) + self.fk.r,
                                   _gt_hi + self.fk.r - 0.002)"""
new = """                if _gt_hi > 0.4:
                    try:
                        # v959: 支撑腿目标压入台面 2mm（原 +0.01 余量让轮
                        # 悬空 0.01-0.06 无抓地、狗不推进实测）
                        _wzt = min(float(terrain_h[leg]) + self.fk.r,
                                   _gt_hi + self.fk.r - 0.002)
                        if float(os.environ.get("S10_QP_DEBUG", "0")) > 1:
                            print('[STIK] t=%.2f leg=%d bz=%.3f wz=%.3f '
                                  'wzt=%.3f q1=%.2f q2=%.2f q1t=%.2f q2t=%.2f'
                                  % (self._t, leg, body["pos"][2],
                                     wheel_xyz[leg, 2], _wzt, q1, q2,
                                     _q1t, _q2t), flush=True)"""
assert old in src, "edit1"
src = src.replace(old, new)

# also print the final target after IK iterations — need to place after the loop
old = """                        th += (_kpp * (_q1t - q1)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                        tk += (_kpp * (_q2t - q2)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                    except Exception:
                        th += (_kpp * (self.pose_target[b + 1] - q1)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                        tk += (_kpp * (self.pose_target[b + 2] - q2)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))"""
new = """                        th += (_kpp * (_q1t - q1)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                        tk += (_kpp * (_q2t - q2)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                        if float(os.environ.get("S10_QP_DEBUG", "0")) > 1:
                            p2 = self.fk.wheel_pos(_q1t, _q2t)
                            print('[STIK2] t=%.2f leg=%d q1t=%.2f q2t=%.2f '
                                  'errx=%.3f errz=%.3f th=%.1f tk=%.1f'
                                  % (self._t, leg, _q1t, _q2t,
                                     _relx - p2[0], _relz + p2[1], th, tk),
                                  flush=True)
                    except Exception:
                        th += (_kpp * (self.pose_target[b + 1] - q1)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                        tk += (_kpp * (self.pose_target[b + 2] - q2)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))"""
assert old in src, "edit2"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched")