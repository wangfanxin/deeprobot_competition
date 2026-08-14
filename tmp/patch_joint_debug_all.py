from pathlib import Path
p=Path('src/S10_sdk_deploy/scripts/stair_dial_noros.py')
s=p.read_text(encoding='utf-8')
old="""                _qhl = _qleg[8:12]\n                _qhr = _qleg[12:16]\n                _thl = _tleg[8:12]\n                print(f'[JOINT] t={t:.1f} FLq={[round(float(v),2) for v in _qleg[0:4]]} FLt={[round(float(v),1) for v in _tleg[0:4]]} HLq={[round(float(v),2) for v in _qhl]} HLt={[round(float(v),1) for v in _thl]} mode={fol.mode}', flush=True)\n"""
new="""                _qfr = _qleg[4:8]\n                _qhl = _qleg[8:12]\n                _qhr = _qleg[12:16]\n                _tfr = _tleg[4:8]\n                _thl = _tleg[8:12]\n                _thr = _tleg[12:16]\n                print(f'[JOINT] t={t:.1f} FLq={[round(float(v),2) for v in _qleg[0:4]]} FRq={[round(float(v),2) for v in _qfr]} HLq={[round(float(v),2) for v in _qhl]} HRq={[round(float(v),2) for v in _qhr]} FLt={[round(float(v),1) for v in _tleg[0:4]]} FRt={[round(float(v),1) for v in _tfr]} HLt={[round(float(v),1) for v in _thl]} HRt={[round(float(v),1) for v in _thr]} mode={fol.mode}', flush=True)\n"""
s=s.replace(old,new)
p.write_text(s, encoding='utf-8')
print('patched joint debug all')
