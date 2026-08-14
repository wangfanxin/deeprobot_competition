from pathlib import Path
p=Path('src/S10_sdk_deploy/scripts/stair_dial_noros.py')
s=p.read_text(encoding='utf-8')
old="""                print(f'[JOINT] t={t:.1f} q={[round(float(v),2) for v in _qleg[0:3]]}/{round(float(_qleg[3]),2)} tau={[round(float(v),1) for v in _tleg[0:4]]} mode={fol.mode}', flush=True)\n"""
new="""                _qhl = _qleg[8:12]\n                _qhr = _qleg[12:16]\n                _thl = _tleg[8:12]\n                print(f'[JOINT] t={t:.1f} FLq={[round(float(v),2) for v in _qleg[0:4]]} FLt={[round(float(v),1) for v in _tleg[0:4]]} HLq={[round(float(v),2) for v in _qhl]} HLt={[round(float(v),1) for v in _thl]} mode={fol.mode}', flush=True)\n"""
if old not in s:
    raise SystemExit('joint debug line not found')
s=s.replace(old,new)
p.write_text(s, encoding='utf-8')
print('patched joint debug 2')
