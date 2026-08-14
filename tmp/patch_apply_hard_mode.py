from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/stair_contact_planner.py')
s=p.read_text(encoding='utf-8')
old="""        if (os.environ.get("S10_GAIT", "0") == "1"\n                or os.environ.get("S10_GAIT_UTIL", "0") == "1"):\n            mpc._gait_swing = np.asarray(\n                fol.gait_schedule(wheel_y, wheel_z, t), dtype=np.float32)\n"""
new="""        if os.environ.get('S10_STAIR_HARD_MODE', '1') == '1':\n            _mode, _fz = self.compute_hard_mode(wheel_y, wheel_z)\n            mpc._gait_swing = _mode\n            mpc._hard_foothold_z = _fz\n        elif (os.environ.get("S10_GAIT", "0") == "1"\n                or os.environ.get("S10_GAIT_UTIL", "0") == "1"):\n            mpc._gait_swing = np.asarray(\n                fol.gait_schedule(wheel_y, wheel_z, t), dtype=np.float32)\n"""
s=s.replace(old,new)
p.write_text(s, encoding='utf-8')
print('patched apply hard mode')
