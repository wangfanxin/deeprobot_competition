#!/bin/bash
cd /home/wfx/DR_competition/0810new/deeprobot_competition
python3 - <<'PYEOF'
import io
p='src/S10_sdk_deploy/s10_mpc/s10_nmpc_wbc.py'
s=io.open(p,encoding='utf-8').read()
old="""                # v1080(方向3最小子集): SWING 轮 F 完全置 0——接触矩阵按
                # mode 删列（抬升轮不进 SRBD 接触力变量），NMPC 真按离散
                # 接触态解；pitch 由 v1072 前馈处理（v1070 的 46N 半支撑
                # 让接触连续松弛，非离散 mode）
                l[3 + len(rows) + 3*i + 2] = 0.0
                u[3 + len(rows) + 3*i + 2] = 0.0"""
new="""                # v1081: 前轴 SWING F=0（滚动越阶，v1080 实测前轮能滚过
                # riser2 y=38.3）、后轴 SWING F_z≥46（滚爬，v1070 实测后轮
                # 需要力才能爬）——不对称接触界
                if i in (0, 1):
                    l[3 + len(rows) + 3*i + 2] = 0.0
                    u[3 + len(rows) + 3*i + 2] = 0.0
                else:
                    l[3 + len(rows) + 3*i + 2] = float(os.environ.get(
                        'S10_NMPC_SWING_FZ_MIN', '46.0'))
                    u[3 + len(rows) + 3*i + 2] = self.fz_max"""
assert old in s, 'fz bound not found'
s=s.replace(old,new,1)
io.open(p,'w',encoding='utf-8').write(s)
print('patched')
PYEOF
/home/wfx/DR_competition/.venv/bin/python -m py_compile src/S10_sdk_deploy/s10_mpc/s10_nmpc_wbc.py && echo COMPILE_OK
sed -e 's|tmp/log_nmpc_real11.txt|tmp/log_nmpc_real12.txt|' tmp/run_nmpc_real11.sh > tmp/run_nmpc_real12.sh
bash tmp/run_nmpc_real12.sh > tmp/log_nmpc_real12.txt 2>&1
grep 'VMC-T\|侧翻\|卡死\|wp7 @' tmp/log_nmpc_real12.txt | tail -12