#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                        th += (_kpp * (_q1t - q1)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                        tk += (_kpp * (_q2t - q2)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                    except Exception:"""
new = """                        th += (_kpp * (_q1t - q1)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 1]]))
                        tk += (_kpp * (_q2t - q2)
                               - _kdp * float(qvel[6 + LEG_QV_LEG[b + 2]]))
                        if float(os.environ.get("S10_QP_DEBUG", "0")) > 2:
                            p2 = self.fk.wheel_pos(_q1t, _q2t)
                            print('[STIK2] t=%.2f leg=%d q1t=%.2f q2t=%.2f '
                                  'errx=%.3f errz=%.3f th=%.1f tk=%.1f'
                                  % (self._t, leg, _q1t, _q2t,
                                     _relx - p2[0], _relz + p2[1], th, tk),
                                  flush=True)
                    except Exception:"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched")