# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                _ok = (_win_lo < d[i] < _win_hi
                       and not self._done[i]
                       and (not _lead or _front_done)
                       and (_lead or _opp_done))"""
assert old in src
new = """                # v953: 对侧轮(FR/RR)触发要求 body 水平(|roll|<0.08)——FL
                # 爬完 body 侧滚、对侧髋变低必须折叠才能到台面 → 过伸实测
                _roll_gate = True
                if not _lead:
                    _roll_gate = abs(body["roll"]) < 0.08
                _ok = (_win_lo < d[i] < _win_hi
                       and not self._done[i]
                       and (not _lead or _front_done)
                       and (_lead or _opp_done)
                       and _roll_gate)"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v953")