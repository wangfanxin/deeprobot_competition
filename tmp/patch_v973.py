#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

old = """            # a_des：z 高度阻尼 + roll/pitch 回零（世界系角加速度近似）
            a_des = np.zeros(6)
            # v939 原状: z 参考固定 0.78（v944/v945 试验 z 跟随/0.88 均引发
            # roll 崩回退）——最佳状态为前轮上台面稳定 15s 卡死。
            a_des[2] = float(os.environ.get("S10_QP_AZ_K", "-30.0")) * (
                float(body["pos"][2]) - 0.78)"""
new = """            # a_des：z 高度阻尼 + roll/pitch 回零（世界系角加速度近似）
            a_des = np.zeros(6)
            # v973: z 参考自适应支撑高度——前轮上平台后轮心应在台面顶+r，
            # body 固定 0.78 时前腿只剩 3cm 垂距够不到台面，轮悬空 2-4cm
            # 无抓地、狗不推进(v971/972 实测)。z_ref = 各轮支撑轮心均值
            # + 腿垂距(drop)，支撑轮心用 stair_world 几何台面封顶(不用
            # 噪声 lidar)。全平:0.62+0.16=0.78；前上台面:0.844；全上台面:
            # 0.907。
            _z_ref = 0.78
            try:
                _sup = np.zeros(4)
                for _i in range(4):
                    _sup[_i] = float(terrain_h[_i]) + self.fk.r
                    _gtz = 0.0
                    for (_rp, _tng, _sr, _dhv, _top) in self.stair_world:
                        if _dhv <= 0.085:
                            continue
                        _ddz = float(np.dot(wheel_xyz[_i, :2] - _rp, _tng))
                        if _ddz > 0.0:
                            _gtz = max(_gtz, float(_top))
                    if _gtz > 0.4:
                        _sup[_i] = min(_sup[_i], _gtz + self.fk.r)
                _z_ref = float(np.mean(_sup)) + float(os.environ.get(
                    "S10_QP_Z_DROP", "0.16"))
            except Exception:
                pass
            self._z_ref = _z_ref
            a_des[2] = float(os.environ.get("S10_QP_AZ_K", "-30.0")) * (
                float(body["pos"][2]) - _z_ref)"""
assert old in src, "edit1"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")