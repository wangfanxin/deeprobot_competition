from pathlib import Path
p=Path('src/S10_sdk_deploy/scripts/stair_dial_noros.py')
s=p.read_text(encoding='utf-8')
needle='''            mpc.latest_tau = mpc.compute_tau(last_act, qq, qqd)\n            d.ctrl[:] = np.asarray(mpc.latest_tau, dtype=np.float64)\n'''
add='''            mpc.latest_tau = mpc.compute_tau(last_act, qq, qqd)\n            d.ctrl[:] = np.asarray(mpc.latest_tau, dtype=np.float64)\n            if os.environ.get('S10_STAIR_JOINT_DEBUG', '0') == '1' and step % 20 == 0:\n                _qleg = np.asarray(d.qpos[7:23]).reshape(-1,1).flatten()\n                _tleg = np.asarray(mpc.latest_tau)\n                print(f'[JOINT] t={t:.1f} q={[round(float(v),2) for v in _qleg[0:3]]}/{round(float(_qleg[3]),2)} tau={[round(float(v),1) for v in _tleg[0:4]]} mode={fol.mode}', flush=True)\n'''
if needle not in s:
    raise SystemExit('joint debug needle not found')
s=s.replace(needle,add)
p.write_text(s, encoding='utf-8')
print('patched joint debug')
