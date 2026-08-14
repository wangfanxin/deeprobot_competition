from pathlib import Path
p=Path('src/S10_sdk_deploy/scripts/stair_dial_noros.py')
s=p.read_text(encoding='utf-8')
old="""                _terr = np.asarray(fol.stair_terrain(_wy), dtype=np.float64)\n                _prox = np.asarray(getattr(mpc, '_stair_prox', np.full(4, 1e9)), dtype=np.float64)\n                mpc.latest_tau = guard.apply(mpc.latest_tau, _gsw_now, _com_xy, wheel_y=_wy, wheel_z=_wz, terrain_z=_terr, prox=_prox)\n"""
new="""                _terr = np.asarray(fol.stair_terrain(_wy), dtype=np.float64)\n                _prox = np.asarray(getattr(mpc, '_stair_prox', np.full(4, 1e9)), dtype=np.float64)\n                _wref = np.asarray(fol.stair_wheel_ref(_wy), dtype=np.float64)\n                mpc.latest_tau = guard.apply(mpc.latest_tau, _gsw_now, _com_xy, wheel_y=_wy, wheel_z=_wz, terrain_z=_terr, prox=_prox, wheel_ref_z=_wref)\n"""
s=s.replace(old,new)
p.write_text(s, encoding='utf-8')
print('patched script guard wheel_ref')
