from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/stair_stance_guard.py')
s=p.read_text(encoding='utf-8')
needle='        return tau\n'
add='''        if __import__('os').environ.get('S10_STANCE_GUARD_DEBUG', '0') == '1':\n            print(f'[GUARD] fn={[round(float(v),1) for v in fn]} contact={[bool(v) for v in contact]} swing={[bool(v) for v in request_swing]}', flush=True)\n        return tau\n'''
if needle not in s:
    raise SystemExit('guard return not found')
s=s.replace(needle,add,1)
p.write_text(s, encoding='utf-8')
print('patched guard debug')
