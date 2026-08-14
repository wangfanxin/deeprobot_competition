from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/stair_contact_planner.py')
s=p.read_text(encoding='utf-8')
needle='''        mpc.set_stair_ref(pitch, base_z)\n\n        # Soft time-varying action bias'''
add='''        mpc.set_stair_ref(pitch, base_z)\n\n        wr_now = np.asarray(fol.stair_wheel_ref(np.asarray(wheel_y, dtype=np.float64)), dtype=np.float64)\n        lift_now = np.clip(wr_now - np.asarray(wheel_z, dtype=np.float64), 0.0, 0.25)\n        lneed = max(float(lift_now[0]), float(lift_now[2]))\n        rneed = max(float(lift_now[1]), float(lift_now[3]))\n        _imb = float(np.clip((lneed - rneed) * float(os.environ.get("S10_ROLL_IMB_GAIN", "0.8")), -0.15, 0.15))\n        mpc._stair_roll_override = _imb\n\n        # Soft time-varying action bias'''
if needle not in s:
    raise SystemExit('roll needle not found')
s=s.replace(needle,add)
p.write_text(s, encoding='utf-8')
print('patched roll override')
