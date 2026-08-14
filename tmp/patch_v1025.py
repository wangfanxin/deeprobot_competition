# -*- coding: utf-8 -*-
import io
p = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
s = io.open(p, encoding="utf-8").read()
start_marker = "        # v1021/v1023/v1024:"
end_marker = "        except Exception:\n            pass\n"
i0 = s.find(start_marker)
assert i0 >= 0, "start marker not found"
i1 = s.find(end_marker, i0)
assert i1 >= 0, "end marker not found"
i1 += len(end_marker)
new_block = '''        # v1021/v1023/v1024: stance-wheel forward-drive floor (all time) + yaw-rate diff
        # v1025: FIX yaw-rate diff to LEFT/RIGHT split (old front/rear split was pitch
        # modulation, not heading correction -> unbraked spin at takeover, ex04 flip)
        # + smooth yaw-rate brake (continuous ramp above gate, overrides drive floor).
        try:
            _dfx = -float(os.environ.get("S10_FP_DRIVE_FLOOR", "6.0"))
            _kd_yx = float(os.environ.get("S10_FP_YAW_DIFF", "2.0"))
            _om_yx = float(qvel[5])
            _brake_g = float(os.environ.get("S10_FP_YAW_BRAKE_GATE", "1.2"))
            _brake_k = float(os.environ.get("S10_FP_YAW_BRAKE_K", "8.0"))
            _brake_w = float(np.clip(
                (abs(_om_yx) - _brake_g) / 1.0, 0.0, 1.0))
            for _leg in range(4):
                if step_lift[_leg] <= 0.5:
                    # left wheels(0,2)=+1 right(1,3)=-1: positive yaw rate (left spin)
                    # -> left more forward / right reduced, generating right-turn moment
                    _sx = 1.0 if _leg in (0, 2) else -1.0
                    _corr = _sx * _om_yx * _kd_yx * self.track_half
                    _tw = min(float(tau[WHEEL_Q_IDX[_leg]]), _dfx) - _corr
                    if _brake_w > 0.0:
                        _tw = min(float(tau[WHEEL_Q_IDX[_leg]]), _dfx) - (
                            _corr + _sx * _om_yx * _brake_k
                            * self.track_half * _brake_w)
                    tau[WHEEL_Q_IDX[_leg]] = _tw
        except Exception:
            pass
'''
s = s[:i0] + new_block + s[i1:]
io.open(p, "w", encoding="utf-8").write(s)
print("PATCHED OK, old block len", i1 - i0)
