# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            # v944: 爬升期 z 参考跟随几何——固定 0.78 在 body 0.85-0.95 时
            # 让 QP 把前轮往下压(悬空)来压低 body，和爬升姿势打架（前轮
            # 悬空 0.03-0.24 无抓地、卡死不推进实测）。爬升期(任一轴 swing)
            # z 参考=当前高度（几何决定），QP 只保支撑+姿态；平地恢复 0.78。
            _z_ref = 0.78
            if float(getattr(self, '_any_swing', 0.0)) > 0.5:
                _z_ref = float(body["pos"][2])
            a_des[2] = float(os.environ.get("S10_QP_AZ_K", "-30.0")) * (
                float(body["pos"][2]) - _z_ref)"""
assert old in src
new = """            # v945: 爬升期 z 目标固定 0.88（爬升体位：前轮台面 0.747+折叠
            # 腿）+ 增益降到 -10——v944 跟随当前去掉支撑(body 失高 roll 崩
            # 实测)；固定 0.78 会把前轮往下压悬空（卡死不推进实测）。
            _z_ref = 0.88 if float(getattr(self, '_any_swing', 0.0)) > 0.5 \
                else 0.78
            _az_k = float(os.environ.get("S10_QP_AZ_K", "-30.0"))
            if float(getattr(self, '_any_swing', 0.0)) > 0.5:
                _az_k = float(os.environ.get("S10_QP_AZ_K_CLIMB", "-10.0"))
            a_des[2] = _az_k * (float(body["pos"][2]) - _z_ref)"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v945")