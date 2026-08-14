#!/usr/bin/env python3
import io

path = "/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py"
src = io.open(path, "r", encoding="utf-8").read()

# ---------- Edit 1: contact-aware stance mask for QP ----------
old = """        # λ_ref：支撑载荷基准——单轮抬升时用支撑三角形重心解(CoM 在三角
        # 形内的静态分配)，其余 mg/4 均载。3 点支撑下均载是错误目标：
        # RR 抬起时静解 FR≈RL≈mg/2、FL≈0，均载会让 QP 的载荷分布远离
        # 物理可行域(配合 b0 外力矩让姿态修正不饱和)。
        lam_ref = np.zeros((4, 3))
        stance_mask = 1.0 - step_lift
        _sw_l = [i for i in range(4) if stance_mask[i] <= 0.5]"""
new = """        # λ_ref：支撑载荷基准——单轮真正离地时用支撑三角形重心解(CoM 在
        # 三角形内的静态分配)，其余 mg/4 均载。3 点支撑下均载是错误目标：
        # RR 抬起时静解 FR≈RL≈mg/2、FL≈0，均载会让 QP 的载荷分布远离
        # 物理可行域(配合 b0 外力矩让姿态修正不饱和)。
        lam_ref = np.zeros((4, 3))
        stance_mask = 1.0 - step_lift
        # v967: QP 支撑掩码改为接触感知——SWING 轮只要还在地上(未离地)就
        # 继续算支撑(mg/4 均载)，只有真正离地才移除 QP 支撑并切换重心解。
        # 原因：SWING 触发(棱前 0.12m)时轮还在平地上，立即移除支撑让 QP 按
        # 3 点支撑分配(FR≈93/RL≈89/RR=10)在平地上把狗顶歪→向西漂移实测。
        qp_stance = stance_mask.copy()
        for _i in range(4):
            if step_lift[_i] > 0.5:
                _lo_z = float(terrain_h[_i]) + self.fk.r + 0.015
                if float(wheel_xyz[_i, 2]) > _lo_z:
                    qp_stance[_i] = 0.0
        _sw_l = [i for i in range(4) if qp_stance[i] <= 0.5]"""
assert old in src, "edit1 anchor missing"
src = src.replace(old, new)

# ---------- Edit 2: barycentric uses qp_stance ----------
old = """        if len(_sw_l) == 1:
            _st_l = [i for i in range(4) if stance_mask[i] > 0.5]"""
new = """        if len(_sw_l) == 1:
            _st_l = [i for i in range(4) if qp_stance[i] > 0.5]"""
assert old in src, "edit2 anchor missing"
src = src.replace(old, new)

# ---------- Edit 3: else branch uses qp_stance ----------
old = """        else:
            for i in range(4):
                if stance_mask[i] > 0.5:
                    lam_ref[i, 2] = self.m * self.g / 4.0"""
new = """        else:
            for i in range(4):
                if qp_stance[i] > 0.5:
                    lam_ref[i, 2] = self.m * self.g / 4.0"""
assert old in src, "edit3 anchor missing"
src = src.replace(old, new)

# ---------- Edit 4: _qp_solve gets qp_stance; wrench uses it ----------
old = """            _any_sw_q = float(np.min(stance_mask)) < 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0"))
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0"))
            # v964: 抬升腿动态反作用力矩建模"""
new = """            _any_sw_q = float(np.min(stance_mask)) < 0.5
            _ar_k = float(os.environ.get("S10_QP_AR_K", "-20.0"))
            _ap_k = float(os.environ.get("S10_QP_AP_K", "-20.0"))
            # v964: 抬升腿动态反作用力矩建模"""
assert old in src, "edit4 anchor missing"
src = src.replace(old, new)

# Note: _qp_solve already receives stance_mask param; pass qp_stance at call site
old = """        lam = self._qp_solve(body, wheel_xyz, stance_mask, lam_ref)"""
new = """        lam = self._qp_solve(body, wheel_xyz, qp_stance, lam_ref)"""
assert old in src, "edit5 anchor missing"
src = src.replace(old, new)

io.open(path, "w", encoding="utf-8").write(src)
print("patched OK")