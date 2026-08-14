from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/stair_contact_planner.py')
s=p.read_text(encoding='utf-8')
old="""        return mode, foothold_z.astype(np.float32)\n"""
new="""        if os.environ.get('S10_HARD_MODE_DEBUG', '0') == '1':\n            print(f'[HARD] wy={[round(float(v),2) for v in wheel_y]} wz={[round(float(v),2) for v in wheel_z]} mode={[int(v) for v in mode]} fz={[round(float(v),2) for v in foothold_z]}', flush=True)\n        return mode, foothold_z.astype(np.float32)\n"""
s=s.replace(old,new)
p.write_text(s, encoding='utf-8')
print('patched hard debug')
