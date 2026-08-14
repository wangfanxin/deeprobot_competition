# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
# 确认第一处替换已应用（v946 trigger latch）
if "v946: 爬完锁存" not in src:
    old = """        if self._sp_f <= 0.0:
            if -_swd < _df < 0.05 and self._sp_r <= 0.0:
                self._sp_f = 1.0
                self._sp_f_top = _tf
                self._rel_f_t = None
                self._sw_f_t0 = self._t"""
    assert old in src
    new = """        if self._sp_f <= 0.0:
            # v946: 爬完锁存——前轮一旦成功到台面高度(top+r+0.005)且过棱，
            # 该 riser 不再重触发 SWING（狗在棱口晃动 d 振荡反复重触发、
            # 轮一直悬空不推进实测）；远离该 riser(d<-0.5)才复位
            if not hasattr(self, "_sp_f_done"):
                self._sp_f_done = False
            if _df < -0.5:
                self._sp_f_done = False
            if (not self._sp_f_done and -_swd < _df < 0.05
                    and self._sp_r <= 0.0):
                self._sp_f = 1.0
                self._sp_f_top = _tf
                self._rel_f_t = None
                self._sw_f_t0 = self._t"""
    src = src.replace(old, new, 1)
# 释放分支设置锁存
old2 = """            if _df > 0.10 and _wz_f >= self._sp_f_top + r + 0.005:"""
assert old2 in src
new2 = """            if (_df > 0.10 and _wz_f >= self._sp_f_top + r + 0.005
                    and not getattr(self, '_sp_f_done', False)):
                self._sp_f_done = True"""
src = src.replace(old2, new2, 1)
p.write_text(src, encoding="utf-8")
print("patched v946")