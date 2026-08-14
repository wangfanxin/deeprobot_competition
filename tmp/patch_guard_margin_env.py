from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/stair_stance_guard.py')
s=p.read_text(encoding='utf-8')
old="""        self.support_margin = float(support_margin)\n"""
new="""        self.support_margin = float(__import__('os').environ.get('S10_STANCE_SUPPORT_MARGIN', str(support_margin)))\n"""
s=s.replace(old,new)
p.write_text(s, encoding='utf-8')
print('patched guard margin env')
