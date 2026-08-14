# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """                _wz_t = float(terrain_h[leg]) + self.fk.r
                if _pz > 0.01 and _sl > 0.02:
                    _wz_t = min(_pz + self.fk.r + 0.04,
                                float(body["pos"][2]) + 0.25)"""
assert old in src
new = """                _wz_t = float(terrain_h[leg]) + self.fk.r
                if _pz > 0.01 and _sl > 0.02:
                    # v950: swing 目标严格封顶到台面顶+r+0.005——原
                    # body_z+0.25 松上限允许轮抬到 1.08（FR/RL 过伸实测）；
                    # 台面顶来自 place_z 反解（pz = 顶-r-margin）
                    _wz_t = min(_pz + self.fk.r + 0.04,
                                _pz + self.fk.r + 0.045)"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v950")