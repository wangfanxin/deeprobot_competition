# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/s10_mpc/stair_wbc_qp.py")
src = p.read_text(encoding="utf-8-sig")

# 1) a_des 增益环境可调（默认温和：roll/pitch -20）
old = """            a_des[2] = float(os.environ.get("S10_QP_AZ_K", "-30.0")) * (
                float(body["pos"][2]) - 0.78)
            a_des[3] = float(os.environ.get("S10_QP_AR_K", "-50.0")) * body["roll"]
            a_des[4] = float(os.environ.get("S10_QP_AP_K", "-50.0")) * body["pitch"]
            # v907: yaw 率阻尼——QP 侧向力耦合出 yaw 自旋(om±2.9 实测)
            a_des[5] = float(os.environ.get("S10_QP_AY_K", "-20.0")) * getattr(self, '_yaw_rate', 0.0)
            W1 = np.diag([0.0, 0.0, 400.0, 50.0, 50.0, 20.0])"""
assert old in src
new = """            a_des[2] = float(os.environ.get("S10_QP_AZ_K", "-30.0")) * (
                float(body["pos"][2]) - 0.78)
            a_des[3] = float(os.environ.get("S10_QP_AR_K", "-20.0")) * body["roll"]
            a_des[4] = float(os.environ.get("S10_QP_AP_K", "-20.0")) * body["pitch"]
            # v907: yaw 率阻尼——QP 侧向力耦合出 yaw 自旋(om±2.9 实测)
            a_des[5] = float(os.environ.get("S10_QP_AY_K", "-20.0")) * getattr(self, '_yaw_rate', 0.0)
            W1 = np.diag([0.0, 0.0,
                          float(os.environ.get("S10_QP_W_Z", "400.0")),
                          float(os.environ.get("S10_QP_W_R", "20.0")),
                          float(os.environ.get("S10_QP_W_P", "20.0")),
                          float(os.environ.get("S10_QP_W_Y", "10.0"))])"""
src = src.replace(old, new, 1)

# 2) 总法向力硬约束：支撑腿 Σλ_z ≥ mg*0.92（防 roll 项牺牲支撑力）
old2 = """            A_sp = sparse.csc_matrix(
                (np.asarray(vals, dtype=np.float64),
                 (np.asarray(rows, dtype=np.int64),
                  np.asarray(cols, dtype=np.int64))),
                shape=(len(lo), n))"""
assert old2 in src
new2 = """            # v909: 总法向力支撑约束——QP 若只追姿态会把 λ_z 压到下限
            # 10N，狗失支撑坠落翻车（台架实测 λ_z=10 全轮）。硬约束
            # Σ λ_z(stance) ≥ 0.92mg，姿态修正只能在保支撑前提下做。
            rows.append(len(lo)); cols.append(-1); vals.append(0.0)
            _sup_w = 0.0
            for i in range(4):
                if stance_mask[i] > 0.5:
                    rows.append(len(lo)); cols.append(i * 3 + 2); vals.append(1.0)
                    _sup_w += 1.0
            lo.append(self.m * self.g * 0.92)
            hi.append(np.inf)
            A_sp = sparse.csc_matrix(
                (np.asarray(vals, dtype=np.float64),
                 (np.asarray(rows, dtype=np.int64),
                  np.asarray(cols, dtype=np.int64))),
                shape=(len(lo), n))"""
src = src.replace(old2, new2, 1)

p.write_text(src, encoding="utf-8")
print("patched")