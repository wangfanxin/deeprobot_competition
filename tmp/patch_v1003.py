#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """                # v997: 贴面软阻尼跟随——前轮贴面时 KP 降到 15、KD 提到 100，
                # 目标跟轮高+3mm，轮靠前驱滚上立面，腿只吸收冲击。
                if _face_drive[leg] and leg in (0, 1):
                    _kps_d = float(os.environ.get("S10_QP_KP_SW_SOFT", "15.0"))
                    _kds_d = float(os.environ.get("S10_QP_KD_SW_SOFT", "100.0"))
                    _wz_t = float(wheel_xyz[leg, 2]) + float(os.environ.get(
                        "S10_QP_FOLLOW_GAP", "0.003"))"""
new = """                # v1003: 贴面软弹簧——前轮贴面时 KP 降到 15、KD 提到 100，
                # 目标用贴面轮廓(v976 几何面)，软弹簧把轮轻压到面上(悬空
                # 时向下压、接触后随面滚)，前驱负责滚上。原 v997 用"轮高
                # +3mm"跟随目标把轮往上托→悬空(实测)。
                if _face_drive[leg] and leg in (0, 1):
                    _kps_d = float(os.environ.get("S10_QP_KP_SW_SOFT", "15.0"))
                    _kds_d = float(os.environ.get("S10_QP_KD_SW_SOFT", "100.0"))"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")