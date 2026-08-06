"""minisnap 2D 路径优化（Richter et al. 2011 风格，纯 numpy）。

用途（2026-08-06，用户指示 2）：把"航点折线"优化为平滑轨迹——
- 每段为 5 阶多项式，最小化全程 snap（4 阶导数）平方和；
- 边界：起点 p0/v0=0/a0=0，终点 pn/vn=0/an=0；
- 内点：位置固定（航点），速度/加速度为自由优化变量；
- 输出：按时间均匀重采样的稠密路径 (M,2)。
"""
import numpy as np
import math


def _seg_cost_matrix(T):
    """Q = int_0^T phi''''(t) phi''''(t)^T dt，phi=[1,t,t2,t3,t4,t5]。"""
    Q = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i >= 4 and j >= 4:
                pi = i - 4
                pj = j - 4
                ci = math.factorial(i) / math.factorial(pi)
                cj = math.factorial(j) / math.factorial(pj)
                Q[i, j] = ci * cj * T ** (pi + pj + 1) / (pi + pj + 1)
    return Q


def _seg_matrix(T):
    """A(T)：[p0 v0 a0 p1 v1 a1] = A @ c（p(t)=sum c_i t^i）。"""
    A = np.zeros((6, 6))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 2.0
    for i in range(6):
        A[3, i] = T ** i
    for i in range(1, 6):
        A[4, i] = i * T ** (i - 1)
    for i in range(2, 6):
        A[5, i] = i * (i - 1) * T ** (i - 2)
    return A


def _solve_1d(p, T, v_fixed, free_acc=True):
    """一维 min-snap：p=(N,) 航点，T=(N-1,) 段时长，v_fixed=(N,) 内点速度。

    内点速度固定为路径切线（防外摆），内点加速度为自由优化变量。
    返回 (N-1,6) 段系数。
    """
    n = p.shape[0]
    n_seg = n - 1
    M = [np.linalg.inv(_seg_matrix(t)) for t in T]
    Q = [_seg_cost_matrix(t) for t in T]
    n_e = 6 * n_seg
    n_free = max(n - 2, 0) if free_acc else 0
    # e 布局：段 i = [p_i, v_i, a_i, p_{i+1}, v_{i+1}, a_{i+1}]（6 值）
    E_fix = np.zeros(n_e)
    E_free = np.zeros((n_e, n_free))
    for i in range(n_seg):
        E_fix[6 * i] = p[i]
        E_fix[6 * i + 3] = p[i + 1]
        # 段 i 左端速度（v_i）
        E_fix[6 * i + 1] = v_fixed[i]
    for j in range(1, n - 1):
        col = j - 1
        E_fix[6 * (j - 1) + 4] = v_fixed[j]     # 段 j-1 右端 v_j
        E_fix[6 * j + 1] = v_fixed[j]           # 段 j 左端 v_j（连续）
        if free_acc:
            E_free[6 * (j - 1) + 5, col] = 1.0  # 段 j-1 右端 a_j
            E_free[6 * j + 2, col] = 1.0        # 段 j 左端 a_j（连续）
    H = np.zeros((n_e, n_e))
    for i in range(n_seg):
        Mi = M[i]
        Hi = Mi.T @ Q[i] @ Mi
        s = 6 * i
        H[s:s + 6, s:s + 6] += Hi
    if n_free > 0:
        Hff = E_free.T @ H @ E_free
        Hfx = E_free.T @ H @ E_fix
        y = -np.linalg.solve(Hff + 1e-9 * np.eye(n_free), Hfx)
        e = E_fix + E_free @ y
    else:
        e = E_fix
    return [M[i] @ e[6 * i:6 * i + 6] for i in range(n_seg)]


def minisnap_2d(waypoints, speed=2.0, free_acc=True, corridor=0.5):
    """2D min-snap：waypoints (N,2) -> (M,2) 稠密路径（每段 ~0.2m 采样）。"""
    wp = np.asarray(waypoints, dtype=np.float64)
    n = wp.shape[0]
    if n < 2:
        return wp
    seg_len = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    T = np.maximum(seg_len / max(speed, 1e-3), 0.25)
    # 内点切线速度：p_{j+1} - p_{j-1} 方向 × speed
    v2 = np.zeros((n, 2))
    for j in range(1, n - 1):
        d = wp[j + 1] - wp[j - 1]
        dd = np.linalg.norm(d)
        if dd > 1e-6:
            v2[j] = d / dd * speed
    cx = _solve_1d(wp[:, 0], T, v2[:, 0], free_acc)
    cy = _solve_1d(wp[:, 1], T, v2[:, 1], free_acc)
    pts = [wp[0]]
    for i in range(n - 1):
        ti = T[i]
        ns = max(int(ti * speed / 0.2), 4)
        for k in range(1, ns + 1):
            t = ti * k / ns
            px = sum(cx[i][d] * t ** d for d in range(6))
            py = sum(cy[i][d] * t ** d for d in range(6))
            pts.append(np.array([px, py]))
    path = np.array(pts)
    # 走廊约束：偏离折线 > corridor 的点投影回折线（剪掉大外摆；
    # 2026-08-06 实测 minisnap 在急弯外摆可达 4.8m，必须约束）
    if corridor > 0:
        out = path.copy()
        for k, p in enumerate(path):
            best_d = 1e9
            best_q = p
            for i in range(n - 1):
                a, b = wp[i], wp[i + 1]
                ab = b - a
                L2 = float(ab @ ab)
                if L2 < 1e-12:
                    continue
                t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
                q = a + t * ab
                d = float(np.linalg.norm(p - q))
                if d < best_d:
                    best_d = d
                    best_q = q
            if best_d > corridor:
                # 拉回：保持超出方向 10% 平滑余量
                out[k] = p + (best_q - p) * (1.0 - corridor / max(best_d, 1e-9))
        path = out
    return path
