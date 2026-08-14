#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
python3 - <<'PYEOF'
import io
p='src/S10_sdk_deploy/s10_mpc/s10_nmpc_wbc.py'
s=io.open(p,encoding='utf-8').read()
old="""            F_ref[3*i+2] = m * g / _n_cont * (
                1.0 if swing[i] <= 0.5 else 0.2)"""
new="""            F_ref[3*i+2] = m * g / _n_cont * (
                1.0 if swing[i] <= 0.5 else 0.0)"""
assert old in s
s=s.replace(old,new,1)
io.open(p,'w',encoding='utf-8').write(s)
print('patched')
PYEOF
/home/wfx/DR_competition/.venv/bin/python -m py_compile src/S10_sdk_deploy/s10_mpc/s10_nmpc_wbc.py && echo COMPILE_OK
sed -e 's|tmp/log_nmpc_real10.txt|tmp/log_nmpc_real11.txt|' \
    -e 's/S10_NMPC_KP_Z=300/S10_NMPC_KP_Z=200/' \
    tmp/run_nmpc_real10.sh > tmp/run_nmpc_real11.sh
bash tmp/run_nmpc_real11.sh > tmp/log_nmpc_real11.txt 2>&1
grep 'VMC-T\|侧翻\|卡死\|wp7 @' tmp/log_nmpc_real11.txt | tail -12