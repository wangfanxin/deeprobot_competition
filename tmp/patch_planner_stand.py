from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/stair_contact_planner.py')
s=p.read_text(encoding='utf-8')
old="""        base_z = float(fol.stair_base_z_ref(y_arr)[0])\n"""
new="""        base_z = float(fol.stair_base_z_ref(y_arr, stand=float(os.environ.get('S10_STAIR_STAND', '0.205')))[0])\n"""
s=s.replace(old,new)
p.write_text(s, encoding='utf-8')
print('patched stand')
