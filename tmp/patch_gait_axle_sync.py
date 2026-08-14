from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/stair_auto_nav.py')
s=p.read_text(encoding='utf-8')
needle='''        if _axle == 1:\n            _fneed = max(float(util[0]), float(util[1]))\n            _rneed = max(float(util[2]), float(util[3]))\n            if _fneed >= need_thr and _fneed >= _rneed * 0.75:\n                _fpw = float(os.environ.get("S10_GAIT_AXLE_W", "0.8"))\n                swing[0] = _fpw if util[0] >= need_thr else 0.0\n                swing[1] = _fpw if util[1] >= need_thr else 0.0\n            elif _rneed >= need_thr:\n                _fpw = float(os.environ.get("S10_GAIT_AXLE_W", "0.8"))\n                swing[2] = _fpw if util[2] >= need_thr else 0.0\n                swing[3] = _fpw if util[3] >= need_thr else 0.0\n'''
add='''        if _axle == 1:\n            _fneed = max(float(util[0]), float(util[1]))\n            _rneed = max(float(util[2]), float(util[3]))\n            _fpw = float(os.environ.get("S10_GAIT_AXLE_W", "0.8"))\n            if _fneed >= need_thr and _fneed >= _rneed * 0.75:\n                # Synchronize the front axle: lift both front wheels together.\n                swing[0] = _fpw\n                swing[1] = _fpw\n            elif _rneed >= need_thr:\n                # Synchronize the rear axle when the rear phase is active.\n                swing[2] = _fpw\n                swing[3] = _fpw\n'''
if needle not in s:
    raise SystemExit('gait axle needle not found')
s=s.replace(needle,add)
p.write_text(s, encoding='utf-8')
print('patched gait axle sync')
