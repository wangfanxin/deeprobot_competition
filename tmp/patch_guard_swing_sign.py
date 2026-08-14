from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/stair_stance_guard.py')
s=p.read_text(encoding='utf-8')
old="""                # Empirical S10 direction: positive hipy/knee raises the wheel.\n                tau[hip_idx] += kp_z * err\n                tau[knee_idx] += kp_z * err\n"""
new="""                # Empirical S10 direction: front knee flexes to raise wheel,\n                # rear knee extends. Hipy positive raises both axles.\n                knee_sign = -1.0 if i in (0, 1) else 1.0\n                tau[hip_idx] += kp_z * err\n                tau[knee_idx] += knee_sign * kp_z * err\n"""
s=s.replace(old,new)
p.write_text(s, encoding='utf-8')
print('patched swing sign')
