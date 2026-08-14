#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)
        place_z = self._face_place_z(wheel_xyz, step_lift)
        self._dbg_phases = (self._sp_f, self._sp_r)"""
new = """        step_lift, place_z = self._update_phases(body["pos"], fwd, wheel_xyz)
        place_z = self._face_place_z(wheel_xyz, step_lift)
        self._dbg_phases = (self._sp_f, self._sp_r)
        # v1019: swing 目标升速限制——贴面轮廓随前速变化可达 3m/s，腿 PD
        # 追不上 → 轮被贴面滚动带飞折叠过伸(1.0-1.2 vs 目标 0.747 实测)。
        # 目标只允许升 SW_TGT_RATE(1.2m/s)，轮靠贴面接触自然上升。
        _sw_rate = float(os.environ.get("S10_FP_SW_TGT_RATE", "1.2"))
        _margin_f = self.lift_margin
        if not hasattr(self, "_sw_z_ref"):
            self._sw_z_ref = np.zeros(4)
        for _leg in range(4):
            if step_lift[_leg] > 0.5 and place_z[_leg] > 0.01:
                _wzt = float(place_z[_leg]) + self.fk.r + _margin_f
                _wzt = min(_wzt, self._sw_z_ref[_leg] + _sw_rate * dt)
                self._sw_z_ref[_leg] = _wzt
                place_z[_leg] = _wzt - self.fk.r - _margin_f
            else:
                self._sw_z_ref[_leg] = (float(place_z[_leg]) + self.fk.r
                                        + _margin_f)"""
assert old in src, "edit1"
src = src.replace(old, new)
io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")