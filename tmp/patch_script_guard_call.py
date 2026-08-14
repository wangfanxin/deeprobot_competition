from pathlib import Path
p=Path('src/S10_sdk_deploy/scripts/stair_dial_noros.py')
s=p.read_text(encoding='utf-8')
old="""                mpc.latest_tau = guard.apply(mpc.latest_tau, _gsw_now, _com_xy)\n"""
new="""                _wy = np.asarray([d.xpos[_wb, 1] for _wb in (5, 9, 13, 17)], dtype=np.float64)\n                _wz = np.asarray([d.xpos[_wb, 2] for _wb in (5, 9, 13, 17)], dtype=np.float64)\n                _terr = np.asarray(fol.stair_terrain(_wy), dtype=np.float64)\n                mpc.latest_tau = guard.apply(mpc.latest_tau, _gsw_now, _com_xy, wheel_y=_wy, wheel_z=_wz, terrain_z=_terr)\n"""
s=s.replace(old,new)
p.write_text(s, encoding='utf-8')
print('patched script guard call')
