#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """    def _face_place_z(self, wheel_xyz, step_lift):
        \"\"\"SWING 腿 place_z（沿 riser 立面 smoothstep，窗=SWING_D）。\"\"\"
        pz = np.zeros(4)
        _r = self.fk.r
        _cl = float(os.environ.get("S10_STAIR_SWING_D", "0.30"))
        _margin = float(os.environ.get("S10_STAIR_LIFT_MARGIN", "0.04"))
        for _leg in range(4):
            if step_lift[_leg] <= 0.02:
                continue
            # v949: 单轮序列——贴面目标用该轮自身位置（不再用轴均值，
            # 否则左右轮同相抬升破坏序列）
            _ax_xy = wheel_xyz[_leg, :2]
            _best_d = 1e9
            _best = None
            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                if _dhv <= 0.085:
                    continue
                _dd = float(np.dot(_ax_xy - _rp, _tng))
                if -_cl < _dd < 0.05 and abs(_dd) < abs(_best_d):
                    _best_d = _dd
                    _best = (_rp, _tng, _dhv, _top)
            if _best is None:
                continue
            (_rp, _tng, _dhv, _top) = _best
            _z_bot = float(_top - _dhv)
            _d_w = float(np.dot(_ax_xy - _rp, _tng))
            # v928: 贴面轮廓（v901 同款，符号修正）——d<0=棱前：
            # d∈[-cl,0] 平滑 ramp flat→top（d=0 到台面顶），d>0 台面顶+r。
            # 此前 v920/v924 把符号写反（d>0 过棱给平地，d<-0.05 反而给
            # 台面顶），前轮抬不上去/过伸实测。
            # v939: 抬升窗收紧到 [-0.08, 0]（轮半径 0.081，d=-0.08 时轮
            # 正好贴棱）——v901 的 [-cl,0] 窗在轮还在地上 0.3m 时就开始抬
            # → 轮推不动反顶 body（0.96 泵高、后腿够不到、roll 崩实测）。
            # 棱口才抬，动量+贴面把轮带上去。
            if _d_w <= 0.0:
                if _d_w >= -0.08:
                    _t = float(np.clip((_d_w + 0.08) / 0.08, 0.0, 1.0))
                    _ss = _t * _t * (3.0 - 2.0 * _t)
                    _z_face = min(_z_bot + _r + _dhv * _ss, _top + _r + 0.005)
                else:
                    _z_face = _z_bot + _r
            else:
                _z_face = min(_top + _r, _top + _r + 0.005)
            pz[_leg] = _z_face - _r - _margin
        return pz"""
new = """    def _face_place_z(self, wheel_xyz, step_lift):
        \"\"\"SWING 腿 place_z（沿 riser 立面 smoothstep，窗=SWING_D）。

        v976: 目标全部用 stair_world 几何，绝不回退到 lidar terrain_h——
        原选择窗 (-cl, 0.05) 之外(d<-0.12 或 d>0.05)pz=0 → swing 目标回退
        到 terrain_h+r，lidar 在棱口读 0.775 → 目标 0.856 把前轮抬到
        0.86-0.89 悬空、狗后仰翻车实测。
        \"\"\"
        pz = np.zeros(4)
        _r = self.fk.r
        _cl = float(os.environ.get("S10_STAIR_SWING_D", "0.30"))
        _margin = float(os.environ.get("S10_STAIR_LIFT_MARGIN", "0.04"))
        for _leg in range(4):
            if step_lift[_leg] <= 0.02:
                continue
            # v949: 单轮序列——贴面目标用该轮自身位置（不再用轴均值，
            # 否则左右轮同相抬升破坏序列）
            _ax_xy = wheel_xyz[_leg, :2]
            # v976: 找最近高 riser（无窗口限制）——任何 d 都有几何目标
            _best_d = 1e9
            _best = None
            for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                if _dhv <= 0.085:
                    continue
                _dd = float(np.dot(_ax_xy - _rp, _tng))
                if abs(_dd) < abs(_best_d):
                    _best_d = _dd
                    _best = (_rp, _tng, _dhv, _top)
            if _best is None:
                continue
            (_rp, _tng, _dhv, _top) = _best
            _z_bot = float(_top - _dhv)
            _d_w = _best_d
            # v928/v939: 贴面轮廓——d∈[-0.08,0] ramp(棱口才抬)，d<0.08
            # 几何地面，d>0 台面顶+r。
            if _d_w <= 0.0:
                if _d_w >= -0.08:
                    _t = float(np.clip((_d_w + 0.08) / 0.08, 0.0, 1.0))
                    _ss = _t * _t * (3.0 - 2.0 * _t)
                    _z_face = min(_z_bot + _r + _dhv * _ss, _top + _r + 0.005)
                else:
                    _z_face = _z_bot + _r
            else:
                _z_face = min(_top + _r, _top + _r + 0.005)
            pz[_leg] = _z_face - _r - _margin
        return pz"""
assert old in src, "edit1 anchor missing"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")