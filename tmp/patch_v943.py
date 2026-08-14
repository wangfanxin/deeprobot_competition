# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            # v935: 释放阈值恢复 0.10（滞回）——v930 改 0.05 与触发上限
            # 相同 → 无滞回 swing 反复翻动 → 前轮过伸 1.12 实测。触发
            # <-0.30..0.05，释放 >0.10，中间带稳定
            if _df > 0.10 and _wz_f >= self._sp_f_top + r + 0.005:"""
assert old in src
new = """            # v943: 释放阈值 0.08（与触发 0.05 保持 0.03 滞回）——v935 的
            # 0.10 让前轮 SWING 卡住（悬空无抓地→不前进→d 到不了 0.10→
            # 不释放死循环，QP 内部 st=[0,0,1,1] 持续 698 步实测）；0.08
            # 让前轮早点释放转 STANCE 压台面、狗能推进
            if _df > 0.08 and _wz_f >= self._sp_f_top + r + 0.005:"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v943")