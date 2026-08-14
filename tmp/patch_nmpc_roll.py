from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/s10_nmpc_wbc.py')
s=p.read_text(encoding='utf-8')
old="""        al_des[0] = -18.0 * body['roll']\n        al_des[0] -= 12.0 * float(np.dot(R[:, 0], body['omega']))\n"""
new="""        al_des[0] = -35.0 * body['roll']\n        al_des[0] -= 18.0 * float(np.dot(R[:, 0], body['omega']))\n"""
s=s.replace(old,new)
p.write_text(s, encoding='utf-8')
print('patched nmpc roll gain')
