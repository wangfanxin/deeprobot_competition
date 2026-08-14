from pathlib import Path
p=Path('src/S10_sdk_deploy/scripts/stair_dial_noros.py')
s=p.read_text(encoding='utf-8')
old="""                _wref = np.asarray(fol.stair_wheel_ref(_wy), dtype=np.float64)\n                mpc.latest_tau = guard.apply(mpc.latest_tau, _gsw_now, _com_xy, wheel_y=_wy, wheel_z=_wz, terrain_z=_terr, prox=_prox, wheel_ref_z=_wref)\n"""
new="""                _wref = np.asarray(fol.stair_wheel_ref(_wy), dtype=np.float64)\n                _hard_fz = getattr(mpc, '_hard_foothold_z', None)\n                if _hard_fz is not None:\n                    _wref = np.asarray(_hard_fz, dtype=np.float64)\n                mpc.latest_tau = guard.apply(mpc.latest_tau, _gsw_now, _com_xy, wheel_y=_wy, wheel_z=_wz, terrain_z=_terr, prox=_prox, wheel_ref_z=_wref)\n"""
s=s.replace(old,new)
p.write_text(s, encoding='utf-8')
print('patched hard z in script')
