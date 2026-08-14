from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/stair_stance_guard.py')
s=p.read_text(encoding='utf-8')
old="""            tau_roll = kp_roll * roll_err - kd_roll * float(ang[0])\n            tau_pitch = kp_pitch * pitch_err - kd_pitch * float(ang[1])\n"""
new="""            tau_roll = kp_roll * roll_err - kd_roll * float(ang[0])\n            tau_pitch = kp_pitch * pitch_err - kd_pitch * float(ang[1])\n            if __import__('os').environ.get('S10_STANCE_BODY_DEBUG', '0') == '1':\n                print(f'[BODY] roll={roll:.3f} pitch={pitch:.3f} ang={[round(float(v),3) for v in ang[:2]]} tau_roll={tau_roll:.1f} tau_pitch={tau_pitch:.1f}', flush=True)\n"""
s=s.replace(old,new)
p.write_text(s, encoding='utf-8')
print('patched body debug')
