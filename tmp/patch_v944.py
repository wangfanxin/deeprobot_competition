# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            a_des[2] = float(os.environ.get("S10_QP_AZ_K", "-30.0")) * (
                float(body["pos"][2]) - 0.78)"""
assert old in src
new = """            # v944: 爬升期 z 参考跟随几何——固定 0.78 在 body 0.85-0.95 时
            # 让 QP 把前轮往下压(悬空)来压低 body，和爬升姿势打架（前轮
            # 悬空 0.03-0.24 无抓地、卡死不推进实测）。爬升期(任一轴 swing)
            # z 参考=当前高度（几何决定），QP 只保支撑+姿态；平地恢复 0.78。
            _z_ref = 0.78
            _any_sw = float(np.max(step_lift)) > 0.5
            if _any_sw:
                _z_ref = float(body["pos"][2])
            a_des[2] = float(os.environ.get("S10_QP_AZ_K", "-30.0")) * (
                float(body["pos"][2]) - _z_ref)"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v944")