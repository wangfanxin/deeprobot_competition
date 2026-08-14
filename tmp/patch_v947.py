# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")
old = """            # v907: yaw 率阻尼——QP 侧向力耦合出 yaw 自旋(om±2.9 实测)
            a_des[5] = float(os.environ.get("S10_QP_AY_K", "-20.0")) * getattr(self, '_yaw_rate', 0.0)"""
assert old in src
new = """            # v947: yaw 率阻尼只在后轮爬顶期启用——前轮 SWING 期(双后腿
            # 支撑)QP 用侧向力阻尼 yaw 造成载荷极端不对称(RL 161N vs RR
            # 10N, 横向到摩擦锥极限 → roll 崩实测)；后轮爬顶期(双前腿)
            # 才有支撑做侧向力
            _ay_k = (float(os.environ.get("S10_QP_AY_K", "-20.0"))
                     if float(getattr(self, '_rear_swing', 0.0)) > 0.5
                     else 0.0)
            a_des[5] = _ay_k * getattr(self, '_yaw_rate', 0.0)"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched v947")