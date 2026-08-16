"""模式 A：已知地图自动导航（纯追踪 + 弯道限速）。

v829 架构铁律（用户指令）：AutoNavFollower 只做 xy 路径规划——
输入仅有航点 xy 坐标与起点 xy 坐标；禁止任何 z/高程/地形先验
（坡度限速、台阶/楼梯区限速等一律删除）。地形响应由感知层
（lidar 高程图）与 stair 技能接管。类内保留的 stair_zone/
stair_* 几何表属于 stair 技能地图数据（非路径规划），最终须
迁移为由高程图感知提供。

输入：全局航点路径（track_overlay 的 track_waypoint_*，已知地图）。
输出：每控制周期生成 (vx, vyaw) 指令，交给 dial-mpc set_cmd 执行。

限速策略（基于已知地图离线预计算，无需传感器）：
- 弯道：按相邻航点转角估算曲率半径，v = sqrt(a_lat * R)
- 坡度：按相邻航点高差/距离估算坡度，上坡/下坡都限速
- 前瞻窗口：取前方 2~3 个航点限速的最小值，提前减速
"""
import os
import numpy as np


class AutoNavFollower:
    def __init__(self, waypoints, max_speed=5.0, vyaw_max=2.0,
                 yaw_gain=2.5, lookahead=4.0, lat_accel_max=6.0,
                 climb_max_speed=1.5, grade_scale=5.0, speed_window=3,
                 lat_gain=1.5, max_accel=5.0, yaw_damp=0.6, cte_gain=4.0):
        """
        waypoints: (N,3) 全局航点 [x, y, z]
        """
        self.wp = np.asarray(waypoints, dtype=np.float64)
        self.max_speed = float(max_speed)
        # v220q: 爬坡/台阶限速可覆盖（提速调试）
        self.climb_max_speed = float(os.environ.get(
            "S10_AUTO_CLIMB_VX", str(climb_max_speed)))
        self.vyaw_max = float(vyaw_max)
        self.yaw_gain = float(yaw_gain)
        # pursuit 前瞻 4m（2026-08-06 用户 1.1）：2.5m 在弯道切内圈 →
        # CTE 0.8m+ → 拉回侧翻；4m 瞄准更远，弯道走线更贴路径。
        self.lookahead = float(os.environ.get(
            "S10_AUTO_LOOKAHEAD", str(lookahead)))
        self.lat_accel_max = float(os.environ.get(
            "S10_AUTO_LAT_ACCEL", str(lat_accel_max)))

        self.grade_scale = float(grade_scale)
        # v219d: 限速前瞻窗口可覆盖（S10_AUTO_SPEED_WINDOW）。
        # 高架限速只应看当前段+下一段：过长则 wp4→5
        # 横脊前被远处 wp7(z=1.16)拖累到 1.37m/s，低于过脊需的 1.57m/s
        self.speed_window = int(os.environ.get(
            "S10_AUTO_SPEED_WINDOW", str(speed_window)))
        self.lat_gain = float(lat_gain)
        self.max_accel = float(os.environ.get(
            "S10_AUTO_MAX_ACCEL", str(max_accel)))
        self.yaw_damp = float(os.environ.get(
            "S10_YAW_DAMP", str(yaw_damp)))   # yaw rate 阻尼，防航向过冲
        # 横向偏差修正增益 2.0（2026-08-06）：1.2 在横脊/弯道处横向漂移
        # 修正不足 → 西漂侧翻；2.0 更强纠偏。
        self.cte_gain = float(os.environ.get(
            "S10_CTE_GAIN", str(cte_gain)))
        # 台阶区限速（2026-08-05 重新实现，见 0806 §3.7"航点 z 兜底"）：
        # 相邻航点 z 上升 >0.08m → 该航段视为台阶/陡升区，接近时降到 step_vx。
        # 区分"0.12m 横脊"（wp4→5 z 不变 → 不限速，靠动量 2m/s 冲过）与
        # "0.13m riser"（wp5→6 z +0.13 → 限速 1.5，避免 3.1m/s 撞面翻车）。
        self.step_vx = float(os.environ.get("S10_AUTO_STEP_VX", "1.5"))
        self.step_dist = float(os.environ.get("S10_AUTO_STEP_DIST", "6.0"))
        # 连续楼梯段限速（链 52，用户"楼梯也要快"）：z 上升 >0.25m 的航段
        # 判定为多级楼梯（wp6→7 +0.57），用 stair_vx（可快）；
        # 单级横脊（wp5→6 +0.13，step_zone 但非 stair_zone）仍用保守
        # step_vx——3.0 m/s 撞 0.13m 脊实测侧翻（chain 51）。
        self.stair_vx = float(os.environ.get("S10_AUTO_STAIR_VX", "2.5"))
        # 双模式状态机（用户方案 2.3）：CRUISE / STAIR_SEQUENCE。
        # STAIR 触发 = 当前航段是连续楼梯（z 升 >0.25）且距段末航点 <
        # stair_mode_dist（默认 3m，接近楼梯才切换，避免平地提前放开屈膝）。
        # 到达段末航点（wp7）后 next 推进到平地段 → 自动回 CRUISE。
        self.mode = "CRUISE"
        self.stair_mode_dist = float(os.environ.get(
            "S10_STAIR_MODE_DIST", "3.0"))
        # 2026-08-15 23:15 (USER acceptance): elevation-map STAIR entry/exit.
        # decel_request [0,1]: cruise_vmc reads it to ramp vx toward stair_vx when
        # a stair/ridge is detected AHEAD on the elevation map.
        self.decel_request = 0.0
        self._elev_last_steps = None   # GOAL #3 raw step edges (perception turn+ridge assist)
        # 全局平滑路径（2026-08-06 用户方向 1.1/1.2）：航点折线 → 圆角
        # 折线（弯道圆弧过渡）→ 密集弧长参数化路径 + 曲率/速度剖面。
        # dial-mpc 只做 locomotion，导航层（本类）负责"平滑全局路径 +
        # 局部滚动 ref_path + 速度参考"。圆角外偏 < 判点半径（0.5m 模拟器）。
        self.fillet_r = float(os.environ.get("S10_GLOBAL_FILLET_R", "1.0"))
        self.path_res = float(os.environ.get("S10_GLOBAL_PATH_RES", "0.05"))
        # 赛车线圆弧切弯（2026-08-07）：弯道用 R 更大的圆弧替代样条，
        # 增大有效转弯半径 → 提高弯速。0=关闭（默认，保持原样条）。
        # vyaw 变化率限制（rad/s 每 0.05s 更新，S10_AUTO_VYAW_SLEW 默认 0.8）：
        # 反馈 ±1.28 瞬跳 + 轮 FF 放大 → yaw 打转（全航点 #7 wp2 处 ±3rad 振荡），
        # 限制指令变化率可消振荡（2026-08-05）。
        self.vyaw_slew = float(os.environ.get("S10_AUTO_VYAW_SLEW", "0.8"))
        self._last_vx = 0.0
        self._last_vyaw_out = 0.0
        self._ref_dbg_cnt = 0
        self._s_cur = 0.0
        # v219h: 横脊弧长列表（cruise 预扫描填充），
        # 用于脊前强制加速冲脊
        self.ridge_s = []
        self._precompute()
        self._build_smooth_path()

    def _stair_corridor_xy(self, xy):
        """v132 台阶段走廊偏移：wp6→7 直线路径（y≈36-41 处 x≈-15.0）骑在
        中央隔脊上（地形射线实测：x∈[-15.05,-14.95]、y∈[34.5,40.25]、高出
        两侧 0.3-0.6m）→ 狗车身骑脊、左右轮悬在两侧台阶 → 卡死/侧翻
        （v90-v131 统一失败模式根因，2026-08-08 扫描实锤）。
        把 y∈[33,41.2] 的路径点沿 +x 平移 S10_STAIR_CORRIDOR_X（默认 0.6m，
        半正弦平滑进出），让整条狗（轮距±0.24m）落在东侧台阶带 x≈-14.5
        （安全走廊 x∈[-14.6,-13.9]，两侧余量 0.25m+）。0=关闭。
        """
        out = np.asarray(xy, dtype=np.float64).copy()
        # v371: 起步门架走廊偏移——wp0->wp1 段在 45% 弧长处（实测门架墙
        # y~4.95，开口 x∈(-1.0,+0.7)）西移 S10_START_CORRIDOR_X，两端
        # 高斯归零（wp0/wp1 位置不动，0.3m 判点不受影响）。狗稳定走开口
        # 中线，避免漂到 x>=0.7 撞墙（此前"起步坡自旋"实为撞门架扰动）。
        _sx = float(os.environ.get("S10_START_CORRIDOR_X", "0.0"))
        if _sx > 0.0 and len(self.cum_len) > 1:
            _s_end = float(self.cum_len[1])
            for _i in range(len(out)):
                _s = float(self.cum_len[_i])
                if _s < _s_end:
                    _u = _s / max(_s_end, 1e-3)
                    # v373: 峰 42%、σ0.20——门架前达峰、y~9.4 前回归名义线；
                    # 偏移量 0.5（狗东漂 0.4-0.65，路径需 <=-0.1 才稳开口内）
                    _w = float(np.exp(-((_u - 0.42) / 0.20) ** 2))
                    out[_i, 0] -= _sx * _w
        amp = float(os.environ.get("S10_STAIR_CORRIDOR_X", "0.6"))
        if amp <= 0.0:
            return out
        # v198 地图无关化：走廊范围由 stair_zone（z 升>0.25 的航段）自动推导，
        # 替代硬编码 y∈[33,41.2]。取第一个非起始台阶区（index>=2，排除 wp0→1
        # 起步缓坡），范围=该台阶段首尾航点。新地图台阶区在哪就作用在哪；
        # 无台阶（如 new_wp30 平面版）自动不触发，无需手动 S10_STAIR_CORRIDOR_X=0。
        idxs = np.where(self.stair_zone)[0]
        idxs = idxs[idxs >= 2]
        if len(idxs) == 0:
            return out
        i0 = int(idxs[0])
        i1 = min(int(idxs[0]) + 1, len(self.cum_len) - 1)
        s0, s1 = self.cum_len[i0], self.cum_len[i1]
        if s1 <= s0:
            return out
        for i in range(len(out)):
            _s = self.cum_len[i]
            if s0 <= _s <= s1:
                _t = (_s - s0) / (s1 - s0)
                # v469: 走廊偏移剖面可调（S10_STAIR_CORRIDOR_FAST 默认 1）。
                # 原半正弦在段中才达峰——wp6→7 中央脊 y≥34.4 起点时只移
                # 0.35m，狗滞后 0.3m 到 x≈-15.0 被脊钉死西侧（左轮撞 0.4m
                # 脊面）。快速梯形：前 30% 段内完成全偏移（y≈34.3 前狗已
                # 完全到东侧走廊），中段保持，末 30% 退回。
                if float(os.environ.get("S10_STAIR_CORRIDOR_FAST", "1")) > 0:
                    if _t < 0.30:
                        _w = float(np.sin(0.5 * np.pi * _t / 0.30) ** 2)
                    elif _t > 0.70:
                        _te = (1.0 - _t) / 0.30
                        _w = float(np.sin(0.5 * np.pi * _te) ** 2)
                    else:
                        _w = 1.0
                else:
                    _w = float(np.sin(np.pi * _t) ** 2)
                out[i, 0] += amp * _w
        return out

    def _stair_diag_bump(self, xy):
        """v133: shared diagonal bump for dense path points (y in [37.8,40.6]).
        Amp from S10_STAIR_DIAG_AMP (0=off). Returns copy."""
        out = np.asarray(xy, dtype=np.float64).copy()
        a = float(os.environ.get("S10_STAIR_DIAG_AMP", "0.0"))
        if a <= 0.0:
            return out
        # v198 地图无关化：与走廊偏移同一推导（第一个非起始台阶段首尾航点）。
        idxs = np.where(self.stair_zone)[0]
        idxs = idxs[idxs >= 2]
        if len(idxs) == 0:
            return out
        i0 = int(idxs[0])
        i1 = min(int(idxs[0]) + 1, len(self.cum_len) - 1)
        s0, s1 = self.cum_len[i0], self.cum_len[i1]
        if s1 <= s0:
            return out
        for i in range(len(out)):
            _s = self.cum_len[i]
            if s0 <= _s <= s1:
                _t = (_s - s0) / (s1 - s0)
                out[i, 0] += a * np.sin(np.pi * _t)
        return out

    def _precompute(self):
        wp = self.wp
        n = len(wp)
        self.seg_len = np.zeros(max(n - 1, 0))
        self.heading = np.zeros(max(n - 1, 0))
        self.cum_len = np.zeros(n)
        for i in range(n - 1):
            d = wp[i + 1, :2] - wp[i, :2]
            self.seg_len[i] = np.linalg.norm(d)
            self.heading[i] = np.arctan2(d[1], d[0])
            self.cum_len[i + 1] = self.cum_len[i] + self.seg_len[i]

        # 每航点限速 = min(弯道限速, 坡度限速)
        self.speed_limit = np.full(n, self.max_speed)
        self.step_zone = np.zeros(n, dtype=bool)
        self.stair_zone = np.zeros(n, dtype=bool)
        for i in range(n):
            # 弯道：前后航段转角 -> 曲率半径 R ≈ L / (2 sin(Δθ/2))
            if 0 < i < n - 1:
                dtheta = np.arctan2(
                    np.sin(self.heading[i] - self.heading[i - 1]),
                    np.cos(self.heading[i] - self.heading[i - 1]))
                L = max(self.seg_len[i - 1], self.seg_len[i], 0.5)
                R = L / (2.0 * abs(np.sin(dtheta / 2.0)) + 1e-6)
                v_curve = np.sqrt(self.lat_accel_max * R)
            else:
                v_curve = self.max_speed
            # v829: 删除坡度限速（z 先验作弊，用户指令——路径规划只用 xy）
            if (i < n - 1
                    and wp[i + 1, 2] - wp[i, 2] > 0.08):
                self.step_zone[i] = True     # 本航段终点是台阶/陡升
            if (i < n - 1
                    and wp[i + 1, 2] - wp[i, 2] > 0.25):
                self.stair_zone[i] = True    # 连续楼梯（多级台阶）
            self.speed_limit[i] = min(self.max_speed, v_curve)

    def _path_point_at(self, dist):
        """沿**平滑路径**取距起点 dist 处的点（线性插值，弧长参数化）。"""
        if hasattr(self, "path_cum") and self.path_cum is not None:
            cum = self.path_cum
            pts = self.path_pts
            if dist <= 0:
                return pts[0].copy()
            if dist >= cum[-1]:
                return pts[-1].copy()
            k = int(np.searchsorted(cum, dist, side="right") - 1)
            k = max(0, min(k, len(pts) - 2))
            t = (dist - cum[k]) / max(cum[k + 1] - cum[k], 1e-6)
            return pts[k] + t * (pts[k + 1] - pts[k])
        if dist <= 0:
            return self.wp[0].copy()
        if dist >= self.cum_len[-1]:
            return self.wp[-1].copy()
        k = int(np.searchsorted(self.cum_len, dist, side="right") - 1)
        k = max(0, min(k, len(self.seg_len) - 1))
        t = (dist - self.cum_len[k]) / max(self.seg_len[k], 1e-6)
        return self.wp[k] + t * (self.wp[k + 1] - self.wp[k])

    def _biarc_path(self, xy):
        """v850: 每 wp 一段圆弧（wp 精确在弧上，硬指标）+ 弧间切线链连接。

        用户规格（2026-08-11）：
        - 半径 R_i = min(3, min(相邻两段长)/2)；
        - 每 wp 圆弧转满该弯转角、apex=wp（wp 在弧上，精确）；
        - 相邻圆弧用直线连接（切线链，**不严格**：允许小折角，不强求相切）。
        构造确定性、无需优化器：圆心 = wp + R·rot(转角平分线, s·90°)，
        弧 S_i→wp→E_i（切线 u_{i-1}→u_i），段间直接连线。
        """
        import numpy as _np
        n = len(xy)
        if n < 3:
            return _np.asarray(xy, dtype=_np.float64)
        r_cap = float(os.environ.get("S10_CORNER_R_MAX", "3.0"))
        segs = []
        for i in range(n - 1):
            d = xy[i + 1] - xy[i]
            L = float(_np.linalg.norm(d))
            if L < 1e-9:
                return None
            segs.append((d / L, L))
        m = n - 2
        R = _np.zeros(m)
        sgn = _np.zeros(m)
        bis = _np.zeros((m, 2))
        for k in range(m):
            i = k + 1
            R[k] = min(r_cap, min(segs[i - 1][1], segs[i][1]) / 2.0)
            u1, u2 = segs[i - 1][0], segs[i][0]
            cr = float(u1[0] * u2[1] - u1[1] * u2[0])
            sgn[k] = 1.0 if cr >= 0 else -1.0
            b = u1 + u2
            nb = float(_np.linalg.norm(b))
            bis[k] = b / nb if nb > 1e-9 else u1
        out = [xy[0].copy()]
        prev = xy[0].copy()
        for k in range(m):
            i = k + 1
            s = sgn[k]
            r = R[k]
            b = bis[k]
            c = xy[i] + r * _np.array([-s * b[1], s * b[0]])
            def rot(v):
                return _np.array([-s * v[1], s * v[0]])
            S = c - r * rot(segs[i - 1][0])
            E = c - r * rot(segs[i][0])
            # 弧 S→wp→E（短弧，含 wp）
            a_s = _np.arctan2(S[1] - c[1], S[0] - c[0])
            a_e = _np.arctan2(E[1] - c[1], E[0] - c[0])
            a_w = _np.arctan2(xy[i, 1] - c[1], xy[i, 0] - c[0])
            dlt = (a_e - a_s + _np.pi) % (2.0 * _np.pi) - _np.pi
            e = (a_w - a_s + _np.pi) % (2.0 * _np.pi) - _np.pi
            if not (e * dlt >= 0 and abs(e) <= abs(dlt) + 1e-6):
                dlt = -dlt
            if abs(dlt) > _np.pi - 1e-3:
                dlt = _np.sign(dlt) * (_np.pi - 1e-3)
            # 段间连接（切线链，不严格）：prev → S
            # v876: inter-arc joint - straight if E->S already follows the
            # shared tangent (no kink <5deg, keeps wp0-1 straight); else
            # spline (C1) removes v850 kink (heading jump 15deg, vlim R~0.7).
            # Arcs keep formula R.
            if _np.linalg.norm(S - prev) > 1e-6:
                _dj = (S - prev) / float(_np.linalg.norm(S - prev))
                _ut0 = segs[i - 1][0]
                _angj = abs(float(_np.arctan2(
                    _dj[0] * _ut0[1] - _dj[1] * _ut0[0],
                    _dj[0] * _ut0[0] + _dj[1] * _ut0[1])))
                if _angj < _np.radians(5.0):
                    out.append(S.copy())
                else:
                    from scipy.interpolate import CubicSpline as _CS
                    _ut = segs[i - 1][0]
                    _len = float(_np.linalg.norm(S - prev))
                    _csx = _CS(_np.array([0.0, _len]),
                               _np.array([prev[0], S[0]]),
                               bc_type=((1, float(_ut[0])), (1, float(_ut[0]))))
                    _csy = _CS(_np.array([0.0, _len]),
                               _np.array([prev[1], S[1]]),
                               bc_type=((1, float(_ut[1])), (1, float(_ut[1]))))
                    _nseg = max(int(_len / 0.08), 4)
                    for _kk in range(1, _nseg + 1):
                        _ss = _len * _kk / _nseg
                        out.append(_np.array(
                            [float(_csx(_ss)), float(_csy(_ss))]))
            else:
                out.append(S.copy())
            npt = max(int(abs(dlt) * r / 0.08), 6)
            for kk in range(1, npt + 1):
                ang = a_s + dlt * kk / npt
                out.append(c + r * _np.array([_np.cos(ang), _np.sin(ang)]))
            prev = E
        if _np.linalg.norm(xy[-1] - prev) > 1e-6:
            out.append(xy[-1].copy())
        return _np.asarray(out, dtype=_np.float64)

    def _build_smooth_path(self):
        """v836+: 唯一路径规划 = biarc（全弧双圆弧样条）——航点精确在圆弧上、
        全程 C1 相切、段中可由外公切线直线化。其他路径规划方法已删除
        （用户指令 2026-08-11：只有 biarc）。
        """
        wp = self.wp
        n = len(wp)
        res = self.path_res
        xy = self._stair_corridor_xy(wp[:, :2])
        raw = self._biarc_path(xy)
        # v133: navigation smooth path also gets the diagonal bump so that
        # pursuit/yaw-FF and MPC r_path use the SAME path (v132 r1 drifted
        # back onto the ridge partly because the two layers diverged).
        raw = self._stair_diag_bump(raw)

        # 均匀弧长重采样（path_res）
        cum_raw = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(raw, axis=0), axis=1))])
        n_out = max(int(cum_raw[-1] / res), 2)
        s_uniform = np.linspace(0.0, cum_raw[-1], n_out)
        pts = np.column_stack([
            np.interp(s_uniform, cum_raw, raw[:, 0]),
            np.interp(s_uniform, cum_raw, raw[:, 1]),
            np.interp(s_uniform, self.cum_len, self.wp[:, 2]),
        ])

        # v199 全局路径轻量平滑（滑动平均窗口 S10_PATH_SMOOTH_W，默认 7 点
        # ≈0.35m）：Catmull-Rom tangent=0.7 在长直线段有 Hermite 过冲弓形
        # （实测 wp0→1 直线 x 弓 0.38m、wp13→14 直线 y 弓 0.67m），ref_path
        # 直接采样这条弓形 → MPC 追弓形切向 → 轨迹画龙（cte std ~0.47m）。
        # 平滑后直线归直、弯角削尖；端点保持原值防判点偏移。
        _sw = int(os.environ.get("S10_PATH_SMOOTH_W", "1"))   # 默认关：弓形为低频弯曲，非画龙主因
        if _sw > 1:
            _k = np.ones(_sw) / _sw
            _xs = np.convolve(pts[:, 0], _k, mode="same")
            _ys = np.convolve(pts[:, 1], _k, mode="same")
            _h = _sw // 2
            _xs[:_h] = pts[:_h, 0]; _xs[-_h:] = pts[-_h:, 0]
            _ys[:_h] = pts[:_h, 1]; _ys[-_h:] = pts[-_h:, 1]
            pts = np.column_stack([_xs, _ys, pts[:, 2]])

        # v825: 删除墙绕行（用户指令：AutoNavFollower 只看 xy 平滑轨迹）
        # 弧长累积 + 数值曲率（|dθ/ds|，平滑 5 点）
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        heading = np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0]))
        self.path_pts = pts
        self.path_cum = cum
        self.path_heading = np.append(heading, heading[-1])
        dh = np.abs(np.diff(heading)) % (2.0 * np.pi)
        dh = np.minimum(dh, 2.0 * np.pi - dh)
        dh_signed = (np.diff(heading) + np.pi) % (2.0 * np.pi) - np.pi
        ds = seg[:-1]
        kappa = np.append(dh / np.maximum(ds, 1e-4), 0.0)
        kappa = np.convolve(kappa, np.ones(5) / 5.0, mode="same")
        # 带符号曲率（左转+，右转-）：弯道 yaw 前馈用（赛用摩托恒定转向率）
        kappa_s = np.append(dh_signed / np.maximum(ds, 1e-4), 0.0)
        kappa_s = np.convolve(kappa_s, np.ones(5) / 5.0, mode="same")
        # 曲率数组补齐到 pts 长度（diff 少一个点，否则 S 弯窗口小/0 时
        # path_curv_signed[k:k+1] 越界 → zero-size）
        if len(kappa) < len(pts):
            kappa = np.concatenate([kappa, np.zeros(len(pts) - len(kappa))])
        if len(kappa_s) < len(pts):
            kappa_s = np.concatenate(
                [kappa_s, np.zeros(len(pts) - len(kappa_s))])
        # 曲率 clamp：R_min 可配（默认 2.0m，2026-08-07 巡航提速：
        # 原 1.0m + curve_accel=4.0 -> 限速上限仅 2.0 m/s，把整段压慢；
        # 2.0m + 6.0 -> 上限 3.46 m/s，弯道仍保守但直线段不被拖累）
        r_min = float(os.environ.get("S10_CURVE_R_MIN", "2.0"))
        kappa = np.minimum(kappa, 1.0 / r_min)
        self.path_curv = np.asarray(kappa, dtype=np.float64)
        self.path_curv_signed = np.asarray(kappa_s, dtype=np.float64)

        # v340: 速度剖面极简——导航只保留几何/任务限速（台阶/楼梯区映射在下方），
        # 弯道动力学（a_lat/om·R/急弯/S弯/前向传播）全部移交 MPPI（摩擦锥 + 2.0s
        # 视界自行决定进弯速度）。path_curv 仍保留供 pursuit yaw_ff / MPPI guide。
        vlim = np.full(len(pts), self.max_speed, dtype=np.float64)
        # v389: S 弯组合限速（几何，局部曲率门控）——wp2->3->4 连续反向弯
        # （右66°->左55°->右71°，段长仅4.6-4.8m）2.5m/s 下振荡翻车（全程
        # 测试 wp3 出弯朝东实测）。仅当**该点本身**曲率显著（|kappa|>0.1，
        # 即在弯上）且 4m 窗口内同时存在正负大曲率时，该点限速
        # S10_CURVE_SWING_VX。直道点（wp1->2）不进入窗口判定，避免
        # v388 的耦合回翻。
        # v563: 急弯曲率限速——vmax 6 时 wp1 90° 弯翻车，直线可快、急弯
        # 限速（v=sqrt(a/κ)）。阈值 S10_CURVE_VLIM_K 默认 0.2、横向加速度
        # S10_CURVE_VLIM_A 默认 8.0（wp1 κ0.5 → 4.0m/s）。
        _cvk = float(os.environ.get("S10_CURVE_VLIM_K", "0.2"))
        _cva = float(os.environ.get("S10_CURVE_VLIM_A", "4.3"))
        _cext = float(os.environ.get("S10_CURVE_EXTEND", "0.0"))
        for k in range(len(vlim)):
            _kk = abs(float(self.path_curv_signed[k]))
            if _kk > _cvk:
                # v749: 发卡弯(κ>2.5)限速固定用 min(_cva,8.0)——wp4→5
                # 发卡+0.125m台阶复合段，VLIM_A 提到 9.5 时入弯速度高
                # 西飘 0.3m 斜撞台阶卡死实测（751 vs 756 轨迹对比）；
                # 普通弯道(κ<=2.5)享受 VLIM_A 提速。
                _cva_eff = (min(_cva, 8.0) if _kk > 2.5 else _cva)
                _vl9 = float(np.sqrt(_cva_eff / _kk))
                # v673: 急弯限速向后延伸**默认关**（延伸会扰动 wp0→5 剖面致
                # wp5→6 翻车 v656-667）；wp8→15 段（wp9 急弯+高架直道）需要
                # 延伸 0.5m 才稳（v657）——S10_CURVE_EXTEND 分赛段启用
                if _cext > 0.0 and _kk > 1.0:
                    # v681: 分级延伸——κ>4.2（wp12 发卡弯）用满额 _cext；
                    # 其余 κ>1 只延伸 0.5m（wp9 弯延伸 2m 把 wp9→10 拖到
                    # 10.71s 超限）
                    _ext9 = (_cext if _kk > 4.2 else min(_cext, 0.5))
                    _end9 = min(len(vlim), k + int(_ext9 / res) + 1)
                    vlim[k:_end9] = np.minimum(vlim[k:_end9], _vl9)
                else:
                    vlim[k] = min(vlim[k], _vl9)
        # v825: 删除 S 弯抑制（用户指令；交 MPPI）
        # v829: 删除台阶/楼梯区限速（z 先验作弊，用户指令）——路径规划
        # 只用 xy；台阶减速由 stair 技能（感知高程图）接管。速度剖面只
        # 剩曲率几何限速。
        # v825: 删除墙区限速（用户指令）
        self.path_vlim = vlim
        self.path_total = float(cum[-1])
        # 航点在平滑路径上的弧长（统一 pursuit 的 passed 判断标尺：
        # 平滑弧长 ≠ 折线弧长，直接用折线 cum_len 会错位 10m+）
        # v825: 删除起步限速（用户指令）
        self.path_wp_s = np.array([
            float(self.path_cum[np.argmin(np.sum(
                (self.path_pts[:, :2] - self.wp[i, :2]) ** 2, axis=1))])
            for i in range(len(self.wp))])

    def compute_cmd(self, robot_xy, yaw, next_idx, robot_z=None, yaw_rate=0.0):
        """返回 (vx, vyaw)。robot_xy: (2,) 全局位置；next_idx: 下一个未到达航点。"""
        if next_idx >= len(self.wp):
            return 0.0, 0.0
        wp_next = self.wp[next_idx]
        # v217: 模式化参数（巡航/台阶各自调优的前视/转向增益/最大偏航率）
        _lk = float(os.environ.get(
            "S10_AUTO_LOOKAHEAD_STAIR"
            if self.mode == "STAIR" else "S10_AUTO_LOOKAHEAD",
            str(self.lookahead)))
        # v825: 删除墙区前视收紧（用户指令）
        _yg = float(os.environ.get(
            "S10_AUTO_YAW_GAIN_STAIR"
            if self.mode == "STAIR" else "S10_AUTO_YAW_GAIN",
            str(self.yaw_gain)))
        _vm = float(os.environ.get(
            "S10_AUTO_VYAW_MAX_STAIR"
            if self.mode == "STAIR" else "S10_AUTO_VYAW_MAX",
            str(self.vyaw_max)))
        d_wp = float(np.linalg.norm(robot_xy - wp_next[:2]))

        # 纯 pursuit（全局导航层，2026-08-06）：机器人到**平滑圆角路径**
        # 最近点的弧长（避免折线切弦绕路）——dial-mpc 只做 locomotion，
        # 本层输出平滑全局路径 + 速度剖面 + 局部滚动 ref_path。
        _d2 = np.sum(
            (self.path_pts[:, :2] - robot_xy[None, :]) ** 2, axis=1)
        self._k_near = int(np.argmin(_d2))
        s_proj = float(self.path_cum[self._k_near])
        # 单调弧长游标（2026-08-06）：**路径切线投影 + 最近点兜底**——
        # 纯最近点法在弯道外侧滞后（狗偏 1m 时最近点回退 → s_cur 卡在弯道
        # 前 → 目标仍朝 wp3→4 方向 → 西漂，full_course_21 复现）。切线
        # 投影：狗位置沿路径切线方向推进，侧面偏离不推进；s_cur 只前进。
        _k0 = int(np.searchsorted(
            self.path_cum, self._s_cur, side="right") - 1)
        _k0 = max(0, min(_k0, len(self.path_pts) - 2))
        _tang = self.path_heading[_k0]
        _rel = robot_xy - self.path_pts[_k0, :2]
        _proj = float(_rel[0] * np.cos(_tang) + _rel[1] * np.sin(_tang))
        # v219p: 最近点兑底每拍最多推进 1m，防狗偏离路径时
        # s_cur 瞬移到航点后（导航提前切目标）
        self._s_cur = max(
            getattr(self, "_s_cur", 0.0) + _proj * 0.5,
            min(s_proj, getattr(self, "_s_cur", 0.0) + 1.0))
        # 上限：不超前当前航点后 2m（2026-08-06 修复：切线投影在弯道过冲
        # → target 指向 wp5 后（偏西）→ 斜撞横脊侧翻，full_course_25 复现）。
        _s_max = (self.path_wp_s[next_idx] + 2.0
                  if next_idx < len(self.path_wp_s) else self.path_total)
        self._s_cur = min(self._s_cur, _s_max)

        # 目标点：视距内或已越过航点平面 → 瞄准航点本身（保证 0.2m 判点）；
        # 否则 → 路径前视点（平滑跟线，不切弦离轨）
        passed = self._s_cur > self.path_wp_s[next_idx] - 0.05
        # 提前入弯（2026-08-07，赛用摩托 out-in-out 参考）：仅当真正接近
        # 航点（默认 0.8m，保证 0.5m 判点）或已越过时才瞄准航点本身；
        # 其余时间沿平滑路径前视点瞄准——路径在航点处转向，前视点越过
        # 航点后即提前给出转向指令，避免弯道处 err 突跳触发龟速。
        if (self.mode == "STAIR"
                or (d_wp < float(os.environ.get("S10_AUTO_WP_AIM", "2.5"))
                    and os.environ.get("S10_AUTO_WP_AIM_ON", "1") == "1")):
            target = wp_next
            if self.mode != "STAIR":
                # v886: 曲率连续混合——前视点权重随前方曲率增大（连续量，
                # 非门控）：wp1 小弧少混合防过转，wp3 大弯满混合提前入弯，
                # 消除过点 err 突跳。
                _kfut = min(
                    self._k_near + int(float(os.environ.get(
                        "S10_AUTO_YAW_FF_DIST", "1.5")) / self.path_res),
                    len(self.path_curv) - 1)
                _kahead = float(self.path_curv[_kfut])
                _lk_eff = float(np.clip(
                    _lk * (1.0 + _kahead * float(os.environ.get(
                        "S10_AUTO_LOOKAHEAD_CURVE_K", "2.0"))),
                    _lk, float(os.environ.get(
                        "S10_AUTO_LOOKAHEAD_MAX", "3.2"))))
                _s_t2 = min(self._s_cur + _lk_eff, self.path_total)
                if next_idx < len(self.path_wp_s):
                    _s_t2 = min(
                        _s_t2, float(self.path_wp_s[next_idx]) + 0.8)
                _pp2 = self._path_point_at(_s_t2)
                _w_p = float(np.clip((abs(_kahead) - 0.20) / 0.35, 0.0, 1.0))
                target = ((1.0 - _w_p) * wp_next[:2]
                          + _w_p * np.asarray(_pp2)[:2])
        else:
            # v267: 已越过航点（passed）或未接近 → 一律瞄**路径前视点**
            # ——修复"passed 后瞄身后当前航点→err 饱和振荡→漂西卡脊"
            # （wp4→5 实测）；也不切弦（AIM_AHEAD=1 在 S 弯切弦翻车实测）。
            if (passed
                    and os.environ.get("S10_AUTO_AIM_AHEAD", "0") == "1"
                    and next_idx + 1 < len(self.wp)):
                target = self.wp[next_idx + 1]
            else:
                # v217t: 曲率自适应前视——弯道提前转，防"过点后 err 突跳
                # 76° 硬转"翻车；直道保持 _lk 不切弯。
                _kfut = min(
                    self._k_near + int(float(os.environ.get(
                        "S10_AUTO_YAW_FF_DIST", "1.5")) / self.path_res),
                    len(self.path_curv) - 1)
                _kahead = float(self.path_curv[_kfut])
                _lk_eff = float(np.clip(
                    _lk * (1.0 + _kahead * float(os.environ.get(
                        "S10_AUTO_LOOKAHEAD_CURVE_K", "2.0"))),
                    _lk, float(os.environ.get(
                        "S10_AUTO_LOOKAHEAD_MAX", "3.2"))))
                s_target = min(self._s_cur + _lk_eff, self.path_total)
                # v819: pursuit 目标不跨段（封顶到下一航点+0.8m）——wp16→17
                # 段长 2.5m < 前视 3.5m，目标越段看到 wp17→18 东向弯导致
                # 提前东转绕 wp17 打转翻车实测；段内跟线保 0.3m 判点。
                if next_idx < len(self.path_wp_s):
                    s_target = min(
                        s_target, float(self.path_wp_s[next_idx]) + 0.8)
                target = self._path_point_at(s_target)
        err = np.arctan2(target[1] - robot_xy[1],
                         target[0] - robot_xy[0]) - yaw
        err = float(np.arctan2(np.sin(err), np.cos(err)))
        self._last_err = err
        self._last_dwp = d_wp
        self._last_tgt = target
        # v825/v829: 删除高架限速与 z 前视（z 先验作弊，用户指令——
        # 路径规划只用 xy 航点+起点）
        vyaw_max_eff = _vm
        # 弯道 yaw 前馈（2026-08-07 赛用摩托/MPPI 参考）：按前视点路径
        # 曲率给出恒定转向率 v/R，err 只做修正——弯道内不靠纯反馈追线，
        # 避免 err 突跳触发龟速。vx 在函数末尾才算出，用上一拍指令
        # （限幅 6m/s^2，近似足够）。
        _k_ff = min(self._k_near + int(float(os.environ.get(
            "S10_AUTO_YAW_FF_DIST", "1.5")) / self.path_res),
            len(self.path_curv_signed) - 1)
        # 前馈随 |err| 淡出（2026-08-07）：进弯对准阶段 err 大时前馈会与
        # err 修正反向打架（nr14 wp1 极限环复现）；|err|>0.8 时完全不用 FF。
        _ff_fade = float(np.clip(1.0 - abs(err) / 0.8, 0.0, 1.0))
        yaw_ff = float(self._last_vx * self.path_curv_signed[_k_ff] * _ff_fade)
        # 横向偏差修正（防漂移）：v143b cte 相对**平滑路径**（含走廊偏移+斜向
        # bump）计算，而非原始航点直线——爬坡直线锁用 f.heading[6] 锁方向，
        # 但 cte 若相对骑脊线（wp6->wp7 直线 x=-15.0）算，会把狗往脊上拉
        # （v136-v142 西漂根因，走廊路径在 x=-14.4）。纠偏目标=走廊。
        _kn = getattr(self, "_k_near", None)
        _smooth_ok = (_kn is not None
                      and 0 <= _kn < len(self.path_heading)
                      and hasattr(self, "path_pts"))
        cte = 0.0
        cte_corr = 0.0
        if _smooth_ok:
            _pk = self.path_pts[_kn]
            _tang = self.path_heading[_kn]
            _tx, _ty = float(np.cos(_tang)), float(np.sin(_tang))
            _rel = robot_xy - _pk[:2]
            # 与原始公式一致：cte = cross(路径切线, rel)，左为正
            cte = float(_tx * _rel[1] - _ty * _rel[0])
            self._last_cte = cte
        else:
            seg_a = self.wp[max(next_idx - 1, 0)]
            seg_b = self.wp[next_idx]
            d = seg_b[:2] - seg_a[:2]
            L = float(np.linalg.norm(d))
            if L > 1e-6:
                n = d / L
                rel = robot_xy - seg_a[:2]
                cte = float(n[0] * rel[1] - n[1] * rel[0])   # 左为正
                self._last_cte = cte
        if cte != 0.0 or _smooth_ok:
            # 修正方向：左偏→右转（vyaw 负）；与 pursuit 航向一致性检查
            # （err 符号），避免与航点航向打架形成极限环。cte 纠偏仅在 |err| 小时叠加。
            # v459: STAIR 模式独立 cte 增益——wp6→7 走廊东移段（x -15→-14.6）
            # 狗滞后 0.3m 西侧，右轮先撞台阶斜向越阶卡死；台阶区加强纠偏。
            _cte_g = float(os.environ.get(
                "S10_AUTO_CTE_GAIN_STAIR"
                if self.mode == "STAIR" else "S10_CTE_GAIN",
                str(self.cte_gain)))
            cte_corr = -_cte_g * float(np.clip(cte / 2.0, -1.0, 1.0))
            # v213: cte low-pass (S10_CTE_LP, 0=off/1=full, default 0.5) to
            # smooth lateral correction on straights (anti-snake); disabled on
            # curves where |err|>gate so turning is unaffected.
            _lp = float(os.environ.get("S10_CTE_LP", "0.5"))
            if _lp < 1.0:
                _prev = getattr(self, "_cte_corr_filt", 0.0)
                self._cte_corr_filt = _prev + _lp * (cte_corr - _prev)
                cte_corr = self._cte_corr_filt
            else:
                self._cte_corr_filt = cte_corr
            # v272: 离线（|cte|>S10_AUTO_CTE_MAX 默认0.5）时丢弃 cte——让
            # 前视转向主导（恢复路线，避免 cte 与 err 抵消致不转漂西，
            # wp4→5 实测）；在线小偏差保持 cte 精修。连续幅值条件，非门控。
            if (abs(cte) < float(os.environ.get("S10_AUTO_CTE_MAX", "0.5"))
                    and abs(err) < float(os.environ.get(
                        "S10_AUTO_CTE_ERR_GATE", "0.6"))
                    and cte_corr * err >= -0.5):
                vyaw = float(_yg * err + yaw_ff
                             - self.yaw_damp * yaw_rate + cte_corr)
            else:
                vyaw = float(self.yaw_gain * err + yaw_ff
                             - self.yaw_damp * yaw_rate)
        else:
            vyaw = float(self.yaw_gain * err + yaw_ff
                         - self.yaw_damp * yaw_rate)
        vyaw = float(np.clip(vyaw, -vyaw_max_eff, vyaw_max_eff))
        # 变化率限制（防反馈振荡；每次调用 = 0.05s）
        # v442: 每拍增量按真实更新周期缩放（原 0.05s 假设在 NAV_HZ=5 下
        # 实际加速度/转向率只有 1/4——wp4→5 到脊时 vx 只到 1.5m/s，0.12m
        # 脊需动量冲过，低速+前轮抬升=卡脊）。dt_nav=1/NAV_HZ。
        _dt_nav = 1.0 / max(float(os.environ.get('S10_NAV_HZ', '2')), 1e-3)
        _slew_eff = self.vyaw_slew * (_dt_nav / 0.05)
        vyaw = float(np.clip(
            vyaw, self._last_vyaw_out - _slew_eff,
            self._last_vyaw_out + _slew_eff))
        self._last_vyaw_out = vyaw

        # 限速：平滑路径速度剖面（全局导航层，曲率/坡度/台阶已编码；
        # 8m 前瞻 = 4m/s 下提前 2s 减速，弯道转向跟得上）
        # v617: STAIR 模式 vlim 前视收窄到 2m——楼梯是直道，5m 前视会
        # 看到 wp7 出口弯（κ2.66→1.73）把爬楼速度压到 1.74（实测）
        _vll_env = os.environ.get(
            "S10_AUTO_VLIM_LOOKAHEAD" + ("_STAIR" if self.mode == "STAIR"
                                         else ""), "")
        if _vll_env:
            _vll = float(_vll_env)
        elif self.mode == "STAIR":
            _vll = 2.0
        else:
            # v688: 速度自适应前视——刹车距离 v²/2a + 1.5m 余量
            # （低速 2.5→2.6m 不多爬，高速 5→4.6m 刹得住；固定 5m 让
            # wp12 弯前爬 3m 慢 1s，固定 2m 又刹不住翻车）
            _vll = float(np.clip(
                self._last_vx * self._last_vx / (2.0 * self.max_accel)
                + 1.5, 2.0, 5.0))
        # v743: vlim 前视用单调弧长 s_cur（非欧氏最近点）——弯道外侧时
        # 最近点已在弯后，vlim 提前恢复导致高速冲弯侧翻（wp1 90° 弯实测
        # vx=5.05/vlim=1.44 同帧）。s_cur 只随切线投影前进，弯中仍见弯心限速。
        _k_near_v = int(np.searchsorted(
            self.path_cum, self._s_cur, side="right") - 1)
        _k_near_v = max(0, min(_k_near_v, len(self.path_vlim) - 1))
        _k_far = min(_k_near_v + int(_vll / self.path_res),
                     len(self.path_vlim) - 1)
        v_lim = float(np.min(self.path_vlim[_k_near_v:_k_far + 1]))
        # v825: 删除航点转角制动（用户指令；交 MPPI）
        # v825: 删除 err 门控/出弯加速/到达制动/高架限速（用户指令，
        # 速度由曲率剖面+MPPI 决定）
        # v829: 删除台阶/楼梯区限速（z 先验作弊，用户指令；减速由
        # stair 技能感知接管）
        # v825: 删除横脊动量提升/墙区位置直判限速（用户指令；交 MPPI）
        # 速度限幅：避免转向后瞬间 0→4 m/s 的侧向冲击（侧翻风险）
        dv = self.max_accel * (_dt_nav)   # 每拍增量按真实更新周期缩放（v442）
        vx = float(np.clip(v_lim,
                           self._last_vx - dv,
                           self._last_vx + dv))
        vx = float(np.clip(vx, 0.0, self.max_speed))
        self._last_vlim = v_lim
        self._last_vx = vx
        self._last_vyaw = vyaw
        if os.environ.get("S10_AUTO_DEBUG") == "1":
            self._dbg_cnt = getattr(self, "_dbg_cnt", 0) + 1
            if self._dbg_cnt % 40 == 1:
                print(f"[NAVDBG] next={next_idx} vx={vx:.2f} vlim={v_lim:.2f} "
                      f"err={err:.2f} d_wp={d_wp:.2f} mode={self.mode} "
                      f"cte={getattr(self,'_last_cte',0.0):.2f} "
                      f"ff={yaw_ff:.2f}", flush=True)
        return vx, vyaw

    def speed_limit_at(self, idx):
        if idx >= len(self.wp):
            return 0.0
        return float(self.speed_limit[idx])

    def _seg_in_stair_band(self, i):
        """段 i (wp[i]->wp[i+1]) 是否穿越已知 riser 走廊带（x/y 窗口）。

        原始地图只有 wp6→7 是真楼梯；wp17→18/22→23/25→26/27→28 等
        z 升>0.25 航段是连续缓坡（无离散 riser），纯 dz 触发会把 STAIR
        权重误开在大坡上。用 riser 表 y 窗口 + STAIR_ZONE_X 走廊判定；
        S10_STAIR_BAND_GATE=0 关闭（回到纯 dz 触发）。
        """
        if os.environ.get("S10_STAIR_BAND_GATE", "1") != "1":
            return True
        if i < 0 or i >= len(self.wp) - 1:
            return False
        try:
            risers, _tops = self._stair_tables()
        except Exception:
            risers = self.STAIR_RISERS
        if len(risers) == 0:
            # v461: 无感知表（无头/建图未跑）时退回已知地图 z 升分类——
            # wp6→7(z+0.565) 等真楼梯仍切 STAIR；wp17→18 等缓坡段
            # stair_zone 也为 True，但本函数只被楼梯段调用（update_mode
            # 已用 stair_zone 门控），行为与感知确认等价。
            return bool(self.stair_zone[i]) if i < len(self.stair_zone) else False
        y0 = float(risers.min()) - 1.0
        y1 = float(risers.max()) + 1.2
        bx0, bx1 = self.STAIR_ZONE_X
        xa, ya = self.wp[i, :2]
        xb, yb = self.wp[i + 1, :2]
        for k in range(21):
            t = k / 20.0
            x = xa + (xb - xa) * t
            y = ya + (yb - ya) * t
            if bx0 <= x <= bx1 and y0 <= y <= y1:
                return True
        return False

    def update_mode(self, robot_xy, next_idx, yaw=None, local_map=None):
        """双模式判定：已知航段 z 大升（>0.25）**且感知确认离散台阶** →
        STAIR_SEQUENCE。2026-08-06 修复：仅按航点 z 升会把大坡度（wp0→1
        缓坡 +0.47m）误判为楼梯（STAIR σ=2.0 在坡上乱伸腿）；感知 step_flag
        只认离散台阶，陡坡/缓坡无 flag → 保持 CRUISE 轮子爬坡。
        滞回（2026-08-06 凌晨）：进入 STAIR 后**保持**直到离开楼梯区
        （y>40.5 或 next 推进）——狗在台阶底部来回时感知确认瞬时失败会
        反复切 CRUISE（权重突变 → MPC 行为突变 → 侧翻，batch v31 复现）。
        """
        if os.environ.get("S10_FORCE_MODE") in ("CRUISE", "STAIR"):
            if self.mode != os.environ["S10_FORCE_MODE"]:
                self.mode = os.environ["S10_FORCE_MODE"]
            return
        if self.mode == "STAIR":
            # BUGFIX 2026-08-16 (mode flapping -> top-platform fall): the sparse lidar
            # elevation map (_elev_region_passed) confirmed "crossed" MID-CLIMB (y~34-39,
            # flat tread between risers shows no step ahead) -> false STAIR exits -> the
            # global entry trigger re-entered STAIR -> STAIR<->CRUISE flap, control
            # switches destabilized and the robot fell. Elevation exit DISABLED when the
            # known riser table exists; the known-map exit (y > last riser + 0.45m) is
            # deterministic and sufficient for the fixed track.
            _exit_y = float(np.max(self.STAIR_RISERS)) + 0.15
            if robot_xy[1] > _exit_y:
                self.mode = "CRUISE"
                self.decel_request = 0.0
            return
        _dbg = os.environ.get("S10_MODE_DEBUG")
        _sz = None
        _d = None
        # USER-DIRECTED 2026-08-16: elevation-map gated SPEED control only
        # ("高程图中（在路径上）有横脊楼梯出现再控速"). Detect a stair/ridge RISE
        # AHEAD on the elevation map (path capsules only) and ramp decel_request by
        # proximity. Mode switching (STAIR) stays with the known stair_zone logic
        # below -- the 96-line lidar + smooth overlay capsules cannot reliably
        # distinguish stairs from long ramps, so perception must not force RL onto a
        # ramp. Runs before the known-map path (decel applies to ridges too).
        self.decel_request = 0.0
        _elook = float(os.environ.get("S10_ELEV_LOOKAHEAD", "6.0"))
        _eenter = float(os.environ.get("S10_ELEV_ENTER", "2.0"))
        if local_map is not None and yaw is not None:
            _ad = self._elev_stair_ridge_ahead(robot_xy, yaw, local_map, _elook)
            if os.environ.get("S10_ELEV_DEBUG"):
                print(f"[ELEV] t-pos={robot_xy[0]:.2f},{robot_xy[1]:.2f} ad={_ad} decel={self.decel_request:.2f}", flush=True)
            if _ad is not None:
                self.decel_request = float(np.clip(
                    1.0 - (_ad - _eenter) / max(_elook - _eenter, 1e-3), 0.0, 1.0))
        # USER-DIRECTED 2026-08-16: reliable backbone - KNOWN stair-zone proximity
        # decel (allowed wp-z info, same source as the approved stair_zone mode
        # switch). The 96-line lidar map is sparse (proven), so perception alone can
        # drop the signal 1-2m before the steps; this ramp keeps decel on through the
        # approach (y~33 -> ~38), max() with the perception term.
        # NOTE 2026-08-16: perception turn+ridge DECEL was tried (GOAL #3) and
        # REVERTED -- the wp4->5 hairpin+ridge needs MOMENTUM (v890 design), slowing
        # made the MPPI under-turn and drive off the path. wp4->5 remains a v890
        # boundary needing dedicated yaw-control stability work (om -3.51 overshoot).
        _seg_i = next_idx - 1
        # BUGFIX 2026-08-16: stair_zone (z-rise>0.25) also flags the wp0->1 START RAMP
        # (z 0->0.475) -> known-zone decel slowed the launch to 2.5 m/s. Gate with
        # _seg_in_stair_band (riser-corridor check): only REAL stair bands (wp6->7)
        # decel; smooth ramps stay fast.
        if (next_idx >= 1 and _seg_i < len(self.stair_zone) and self.stair_zone[_seg_i]
                and self._seg_in_stair_band(_seg_i)
                and next_idx < len(self.wp)):
            _sa = self.wp[_seg_i, :2]; _sb = self.wp[next_idx, :2]
            _dv = _sb - _sa
            _L2 = max(float(np.dot(_dv, _dv)), 1e-6)
            _t = float(np.clip(np.dot(np.asarray(robot_xy) - _sa, _dv) / _L2, 0.0, 1.0))
            _along = _t * np.sqrt(_L2)
            _off = float(os.environ.get("S10_ELEV_KNOWN_OFF", "1.0"))
            _ramp = float(os.environ.get("S10_ELEV_KNOWN_RAMP", "5.0"))
            if _along >= _off:
                _kd = float(np.clip((_along - _off) / max(_ramp, 1e-3), 0.0, 1.0))
                self.decel_request = max(self.decel_request, _kd)
                if os.environ.get("S10_ELEV_DEBUG"):
                    print(f"[ELEV-KNOWN] t-pos={robot_xy[0]:.2f},{robot_xy[1]:.2f} along={_along:.2f} decel={self.decel_request:.2f}", flush=True)
        if (next_idx >= 1 and next_idx - 1 < len(self.stair_zone)
                and self.stair_zone[next_idx - 1]):
            _sz = bool(self.stair_zone[next_idx - 1])
            d_wp = float(np.linalg.norm(
                robot_xy - self.wp[next_idx, :2]))
            _d = d_wp
            # v133: global known map first - if the segment is a stair zone
            # (z rise >0.25) and within stair_mode_dist of the segment end,
            # enter STAIR regardless of perception step_flag (v132 r2 stayed
            # CRUISE and rammed the riser because perception failed).
            # Perception confirmation only triggers EARLIER (S10_STAIR_CONFIRM_DIST).
            _confirm_dist = float(os.environ.get(
                "S10_STAIR_CONFIRM_DIST", "6.0"))
            # v460: 全局触发改为按**楼梯段起点**距离（原 d_wp<stair_mode_dist
            # 是到段末航点 3m 内——爬楼全程仍 CRUISE，wp6→7 卡 y=38）。
            # 进入段前 S10_STAIR_ENTER_DIST（默认 1.5m）即切 STAIR，保持到
            # 段末航点通过（update_mode 里 next_idx>7 回 CRUISE）。
            # v584: 切换用**到段首航点（wp6）的物理距离**——s_cur 投影
            # 比物理位置超前（>4m），按 s_cur 会提前到 wp5→6 第一级台阶
            # 就切 WBC，新抬升姿态在 0.06m 小台阶上翻车实测。
            # v699: STAIR 模式切换到**段末航点（wp7）前 4m**（真楼梯前）——
            # 原按段首 wp6 切换会在 wp5→6 第二级就接管，巡航能力被浪费
            _dseg1 = float(np.linalg.norm(
                robot_xy - self.wp[next_idx, :2]))
            # BUGFIX 2026-08-16 (mode flapping): after the robot passes the last riser
            # (y > exit_y), the global entry trigger kept re-entering STAIR right after
            # the exit (dist to wp7 < 8.5m while next_idx still 7) -> STAIR<->CRUISE flap
            # at the top, control switches destabilized, robot fell on the platform.
            _entry_y = float(np.max(self.STAIR_RISERS)) + 0.15
            _use_global = (_dseg1 <= float(os.environ.get(
                "S10_STAIR_ENTER_DIST", "1.5")) + 2.5
                           and self._seg_in_stair_band(next_idx - 1)
                           and robot_xy[1] < _entry_y)
            _use_percept = (d_wp < _confirm_dist
                            and self._stair_confirmed(
                                robot_xy, yaw, local_map))
            if _use_global or _use_percept:
                self.mode = "STAIR"
                if _dbg:
                    print(f"[MODE] STAIR next={next_idx} sz={_sz} "
                          f"d={d_wp:.1f} global={int(_use_global)} "
                          f"percept={int(_use_percept)}", flush=True)
                return
        if _dbg and int(robot_xy[1] * 2) % 10 == 0:
            print(f"[MODE] CRUISE next={next_idx} sz={_sz} d={_d} "
                  f"y={robot_xy[1]:.1f}", flush=True)
        self.mode = "CRUISE"

    def _stair_confirmed(self, robot_xy, yaw, local_map):
        """感知确认：机器人前方 0.3~1.5m 窗口内高程图 step_flag ≥1 处
        （离散台阶）才算连续楼梯；陡坡/缓坡无 step_flag → False。"""
        if local_map is None or yaw is None:
            return False
        hm = local_map.get("heightmap")
        valid = local_map.get("valid")
        stepf = (local_map.get("features") or {}).get("step_flag")
        if hm is None or valid is None or stepf is None:
            return False
        ox = float(local_map["origin"][0])
        oy = float(local_map["origin"][1])
        res = float(local_map["resolution"])
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        for d in (0.3, 0.6, 0.9, 1.2, 1.5):
            p = np.asarray(robot_xy) + fwd * d
            i = int(np.floor((p[1] - oy) / res))
            j = int(np.floor((p[0] - ox) / res))
            if (0 <= i < hm.shape[0] and 0 <= j < hm.shape[1]
                    and valid[i, j] and float(stepf[i, j]) > 0.3):
                return True
        return False

    def _elev_stair_ridge_ahead(self, robot_xy, yaw, local_map, lookahead=4.0, start=0.5,
                                 rise_th=0.30):
        """Elevation-map lookahead along the PATH (wp-derived): return distance (m) to the
        first stair/ridge RISE on the robot's path.

        USER-DIRECTED 2026-08-16: only control speed when the elevation map shows a
        ridge/stairs ON THE PATH (not off-path noise). Samples along self.path_pts
        (built from wp absolute coords, allowed) ahead of the robot's arc-length s_cur,
        and detects a height rise (far max - near min > rise_th)."""
        if local_map is None:
            return None
        hm = local_map.get("heightmap")
        valid = local_map.get("valid")
        if hm is None or valid is None:
            return None
        ox = float(local_map["origin"][0]); oy = float(local_map["origin"][1])
        res = float(local_map["resolution"])
        s_cur = float(getattr(self, "_s_cur", 0.0))
        if s_cur <= 0.0:
            k0 = int(np.argmin(np.sum((self.path_pts[:, :2] - np.asarray(robot_xy)) ** 2, axis=1)))
            s_cur = float(self.path_cum[k0])

        # USER-DIRECTED 2026-08-16 (GOAL #1): FAST corridor profile - sample the
        # elevation map ALONG the path (0.1m), max hmax over a lateral window per
        # sample. With the raised lidar the map is DENSE (25k cells), so per-sample
        # probing works (the old sparse-map cell loop cost 87ms/cycle; this is O(1500)
        # lookups, <5ms). Then SHARP-STEP + CLIMB-gate detection (see below).
        rise_th = float(os.environ.get("S10_ELEV_RISE_TH", str(rise_th)))
        lat_win = float(os.environ.get("S10_ELEV_LAT_WIN", "1.2"))
        lat_n = max(3, int(round(2 * lat_win / res)) + 1)
        lats = np.linspace(-lat_win, lat_win, lat_n)
        prof = []   # (ds, hmax)
        for ds in np.arange(start, lookahead + 1e-3, 0.1):
            sp = s_cur + ds
            if sp > self.path_total - 1e-3:
                break
            k = int(np.searchsorted(self.path_cum, sp, side="right") - 1)
            k = min(max(k, 0), len(self.path_pts) - 2)
            t = (sp - self.path_cum[k]) / max(self.path_cum[k + 1] - self.path_cum[k], 1e-6)
            x = self.path_pts[k, 0] + t * (self.path_pts[k + 1, 0] - self.path_pts[k, 0])
            y = self.path_pts[k, 1] + t * (self.path_pts[k + 1, 1] - self.path_pts[k, 1])
            dx = self.path_pts[k + 1, 0] - self.path_pts[k, 0]
            dy = self.path_pts[k + 1, 1] - self.path_pts[k, 1]
            L = max(np.hypot(dx, dy), 1e-6)
            nx, ny = -dy / L, dx / L
            best = None
            for lx in lats:
                px = x + nx * lx; py = y + ny * lx
                i = int(np.floor((py - oy) / res)); j = int(np.floor((px - ox) / res))
                if 0 <= i < hm.shape[0] and 0 <= j < hm.shape[1] and valid[i, j]:
                    hv = float(hm[i, j])
                    if best is None or hv > best:
                        best = hv
            if best is not None:
                prof.append((ds, best))
        if len(prof) < 3:
            return None
        if os.environ.get("S10_ELEV_STEP", "1") == "1":
            step_th = float(os.environ.get("S10_ELEV_STEP_TH", "0.10"))
            steps = []          # (ds, jump) sharp upward edges
            prev_h = prof[0][1]
            for (ds, h) in prof[1:]:
                if h - prev_h > step_th and ds >= 0.5:
                    # confirm sustained (plateau) over next 0.5m
                    _conf = any(h2 > prev_h + step_th
                                for (ds2, h2) in prof
                                if 0.0 < ds2 - ds <= 0.5)
                    if _conf:
                        steps.append((ds, h - prev_h))
                prev_h = max(prev_h, h)   # running max (staircase)
            self._elev_last_steps = [(float(d), float(j)) for d, j in steps]
            # Gate 1: stair SEQUENCE = >= S10_ELEV_MIN_STEPS sharp steps within
            # S10_ELEV_SEQ_SPAN (single ridges wp5->6 / wp4->5 do NOT decel).
            # Gate 2: net CLIMB to a high plateau (max_h - base >= S10_ELEV_CLIMB_TH)
            # so up-down bumps (0.48->0.73->0.48) don't decel; real stairs
            # (0.48->1.17) do.
            if len(steps) >= int(os.environ.get("S10_ELEV_MIN_STEPS", "2")):
                d0 = steps[0][0]
                span = float(os.environ.get("S10_ELEV_SEQ_SPAN", "3.0"))
                if all((d - d0) <= span for d, _ in steps):
                    base = float(min(h for _, h in prof[:20]))   # near field
                    max_h = float(max(h for _, h in prof))
                    if max_h - base >= float(os.environ.get("S10_ELEV_CLIMB_TH", "0.4")):
                        return d0
            return None
        # S10_ELEV_STEP=0: legacy cumulative-rise mode
        base = float(min(h for _, h in prof if _ <= 2.0))
        for (ds, h) in prof:
            if ds >= 0.5 and h > base + rise_th:
                return float(ds)
        return None

    def _elev_region_passed(self, robot_xy, yaw, local_map):
        """Elevation-map exit: no stair/ridge RISE within [0.0, 3.0]m ahead -> crossed."""
        _ahead = self._elev_stair_ridge_ahead(robot_xy, yaw, local_map,
                                              lookahead=3.0, start=0.0)
        return _ahead is None

    def ref_path(self, robot_xy, next_idx, n_pts=10, spacing=0.5,
                 smooth=2):
        """E4：沿路径取前方参考点（世界系 (n_pts,2)），弧长均匀采样 + 轻量平滑。

        - 起点：机器人在当前航段上的投影（保证参考路径从脚下开始）；
        - 采样：s_proj + 0.2 起，每 spacing 米一个点，最多 n_pts 个；
        - 平滑：3 点滑动平均（对折线弯角做轻量"切弯"，次数少、不越出走廊）。
        """
        if next_idx >= len(self.wp):
            return None
        seg_a = self.wp[max(next_idx - 1, 0)]
        seg_b = self.wp[next_idx]
        d = seg_b[:2] - seg_a[:2]
        L2 = max(float(np.dot(d, d)), 1e-6)
        t = float(np.clip(np.dot(robot_xy - seg_a[:2], d) / L2, 0.0, 1.0))
        s0 = self.cum_len[max(next_idx - 1, 0)] + t * np.sqrt(L2)
        pts = []
        s = s0 + 0.2
        while s <= self.cum_len[-1] and len(pts) < n_pts:
            pts.append(self._path_point_at(s)[:2])
            s += spacing
        if len(pts) < 2:
            return None
        pts = np.asarray(pts, dtype=np.float64)
        # 轻量滑动平均（窗口 2*smooth+1），端点保持
        if smooth > 0:
            out = pts.copy()
            for k in range(len(pts)):
                lo = max(0, k - smooth)
                hi = min(len(pts), k + smooth + 1)
                out[k] = pts[lo:hi].mean(axis=0)
            pts = out
        return pts.astype(np.float32)

    def ref_path_3d(self, robot_xy, next_idx, local_map=None,
                    n_wp=8, speed=4.0, corridor=0.5, z_gain=1.0,
                    spacing=0.3):
        """3D 参考路径（2026-08-06 改版，用户指示 2：去 minisnap）：

        1. 直接用**航点折线**（已知地图/astar 简化：逐点连线），按
           `speed`（默认 4 m/s，用户指示）滚动采样前方点；
        2. z 由**高程图决策**：质心 ref_z = 地形高 + 站姿高(0.205)；
           （轮中心 ref_z = 地形 + 轮半径由 r_ground reward 承担，见
           s10_env._reward）——形成 (N,3) 参考轨迹；
        3. dial-mpc 路径跟踪用（reward r_path/r_path_head/r_path_z）。
        实测：minisnap 让 wp0→wp6 从 28s 慢到 33s（1.6→1.2 m/s）且仍冲
        过头，用户决定去除，用航点折线直连。
        """
        if next_idx >= len(self.wp):
            return None
        # v183：ref_path 改从**平滑全局路径**采样（与导航 pursuit 同一曲线），
        # 不再用航点折线——折线尖角让 MPC 弯道里追锐角、与导航平滑线打架
        # （S 弯/55° 弯处速度方差 ±0.5 m/s、偶发卡弯的根因）。采样窗口
        # 覆盖到 next_idx 后 n_wp 个航点的弧长，保证弯道里 ref 完整。
        i1 = min(next_idx + n_wp, len(self.wp))
        s_end = self.path_wp_s[i1 - 1] if i1 > 0 else self.path_total
        s_end = min(s_end + 2.0, self.path_total)
        s_cur = getattr(self, "_s_cur", 0.0)
        if s_end - s_cur < 0.5:
            return None
        s_targets = np.arange(s_cur, s_end, spacing)
        if len(s_targets) < 3:
            return None
        pts = np.zeros((len(s_targets), 3))
        for k, s in enumerate(s_targets):
            i = int(np.searchsorted(self.path_cum, s, side="right") - 1)
            i = max(0, min(i, len(self.path_pts) - 2))
            t = ((s - self.path_cum[i])
                 / max(self.path_cum[i + 1] - self.path_cum[i], 1e-6))
            pts[k, :2] = (self.path_pts[i, :2]
                          + t * (self.path_pts[i + 1, :2]
                                 - self.path_pts[i, :2]))
        # z: elevation + stand height; v133 preview samples the path point
        # S10_REF_Z_PREVIEW (default 0.3m) ahead so the body lift command
        # arrives before the riser (complements r_ground lookahead lift).
        z_ref = np.full(len(pts), 0.205)
        last_ok = None
        if local_map is not None:
            hm = local_map.get("heightmap")
            valid = local_map.get("valid")
            if hm is not None:
                ox = float(local_map["origin"][0])
                oy = float(local_map["origin"][1])
                res = float(local_map["resolution"])
                _preview = float(os.environ.get("S10_REF_Z_PREVIEW", "0.3"))
                _nprev = max(0, int(round(_preview / max(spacing, 1e-3))))
                for k in range(len(pts)):
                    kk = min(k + _nprev, len(pts) - 1)
                    j = int(np.floor((pts[kk, 0] - ox) / res))
                    i = int(np.floor((pts[kk, 1] - oy) / res))
                    if (0 <= i < hm.shape[0] and 0 <= j < hm.shape[1]
                            and valid[i, j]):
                        last_ok = float(hm[i, j])
                    if last_ok is not None:
                        z_ref[k] = last_ok + 0.205
        # 平滑 z（3 点均值）
        zs = np.convolve(z_ref, np.ones(3) / 3.0, mode="same")
        zs[:1] = z_ref[:1]
        zs[-1:] = z_ref[-1:]
        pts[:, 2] = zs * z_gain
        if os.environ.get("S10_REF_DEBUG"):
            if int(self._ref_dbg_cnt % 20) == 0:
                n_valid = 0
                if local_map is not None:
                    hm = local_map.get("heightmap")
                    valid = local_map.get("valid")
                    if hm is not None:
                        ox = float(local_map["origin"][0])
                        oy = float(local_map["origin"][1])
                        res = float(local_map["resolution"])
                        j0 = int(np.floor((pts[0, 0] - ox) / res))
                        i0 = int(np.floor((pts[0, 1] - oy) / res))
                        if (0 <= i0 < hm.shape[0] and 0 <= j0 < hm.shape[1]
                                and valid[i0, j0]):
                            n_valid += 1
                        j1 = int(np.floor((pts[-1, 0] - ox) / res))
                        i1 = int(np.floor((pts[-1, 1] - oy) / res))
                        if (0 <= i1 < hm.shape[0] and 0 <= j1 < hm.shape[1]
                                and valid[i1, j1]):
                            n_valid += 2
                print(f"[REF] wp={next_idx} pos=({robot_xy[0]:.2f},{robot_xy[1]:.2f}) "
                      f"z_ref max={float(pts[:, 2].max()):.3f} "
                      f"z0={float(pts[0, 2]):.3f} z_last={float(pts[-1, 2]):.3f} "
                      f"npts={len(pts)} valid_sample={n_valid}", flush=True)
            self._ref_dbg_cnt += 1
        return pts.astype(np.float32)

    # ==================== 楼梯 3D 路径几何剖面（2026-08-08，路径规划层） ====================
    # wp6→wp7 楼梯段（已知地图实测 scan_wp7_fixed.py，x∈[-15.0,-14.5] 横向一致）：
    #   6 级 riser：y / 顶 z = 37.90/0.54, 38.375/0.67, 38.775/0.79,
    #                39.225/0.92, 39.625/1.04, 40.025/1.17；坡前地面 0.48；
    #   riser 0.12~0.13m > 轮半径 0.081m（静态滚不上，必须抬）；梯面 0.35~0.40m；
    #   前后轮轴距 0.455m > 梯面（后轮跟抬相位差 = 轴距，天然由逐轮位置场表达）。
    # 目标：把 cruise 的"路径+速度+3D 参考"机制延伸到楼梯段（软参考，无硬门控）：
    #   stair_wheel_ref(y)  ：轮心 z 参考场 = 地形+轮半径，riser 前 RAMP_A 起平滑抬到
    #                         顶+轮半径+MARGIN（提前抬轮，破解"顶到立面才接触抬"的根因）；
    #   stair_base_z_ref(y) ：机身 z 参考 = 地形+站姿高，滞后于轮 0.1m（车身跟轮上）；
    #   stair_pitch_ref(y)  ：riser 前仰头（前轮抬升窗口内），踏面回平；
    #   stair_v_ref(y)      ：riser 前减速 / 踏面提速。
    # 全部按世界 y 计算（走廊近似沿 y 轴），不受机体倾斜/局部地图遮挡影响。
    STAIR_GROUND = 0.48
    STAIR_RISERS = np.array([37.90, 38.375, 38.775, 39.225, 39.625, 40.025])
    STAIR_TOPS = np.array([0.54, 0.67, 0.79, 0.92, 1.04, 1.17])
    STAIR_ZONE_Y = (37.0, 41.5)      # wheel_ref 有效 y 带（含最后一级落地区）
    STAIR_ZONE_X = (-15.1, -13.7)    # wheel_ref 有效 x 带（走廊+余量）

    def _stair_tables(self):
        """返回 (riser_y, top_z) 数组；支持 S10_STAIR_RISERS/TOPS 逗号串覆盖。"""
        rs = os.environ.get("S10_STAIR_RISERS", "")
        ts = os.environ.get("S10_STAIR_TOPS", "")
        if rs and ts:
            r = np.array([float(v) for v in rs.split(",")], dtype=np.float64)
            t = np.array([float(v) for v in ts.split(",")], dtype=np.float64)
            if len(r) == len(t) and len(r) > 0:
                return r, t
        return self.STAIR_RISERS, self.STAIR_TOPS

    def stair_terrain(self, y):
        """楼梯段地形高 z(y)（阶梯函数，y 可数组；区外截断为最近已知级）。"""
        rs, ts = self._stair_tables()
        y = np.asarray(y, dtype=np.float64)
        out = np.full(y.shape, self.STAIR_GROUND, dtype=np.float64)
        for k, (y_r, z_top) in enumerate(zip(rs, ts)):
            out = np.where(y >= y_r, z_top, out)
        return out

    def stair_wheel_ref(self, y, radius=0.081):
        """轮心 z 参考场 w_c(y) = 地形+轮半径，riser 前 RAMP_A 平滑抬到 顶+半径+MARGIN。

        v207（S10_STAIR_REF_STEP=1，默认关）：接近段（riser 前
        S10_STAIR_REF_LEAD，默认 0.20m）直接给"下一级顶+半径+margin"满值
        （非 ramp 渐进）——卡点实测轮子满足 ramp 部分高度（0.638/0.78）后
        就停，到棱边仍差 0.125~0.136m；阶梯参考让 MPC 在视界内看到完整
        抬升需求并提前承诺抬腿。"""
        rs, ts = self._stair_tables()
        a = float(os.environ.get("S10_STAIR_RAMP_A", "0.14"))
        b = float(os.environ.get("S10_STAIR_RAMP_B", "0.10"))
        margin = float(os.environ.get("S10_STAIR_LIFT_MARGIN", "0.05"))
        y = np.asarray(y, dtype=np.float64)
        z = self.stair_terrain(y) + radius
        # v790: 贴面爬升参考（S10_STAIR_REF_ARC=1）——轮心目标在棱前最后
        # 轮半径 r 米内沿面线性爬升（0.081m 内升 h，轮贴垂直面滚上、腿随
        # 目标伸长，USC/Go2-W 机制）；替代提前 0.14m ramp（过早抬升让轮
        # 悬空失接触实测）。过棱后回台面+半径。
        if os.environ.get("S10_STAIR_REF_ARC", "0") == "1":
            radius = float(os.environ.get("S10_STAIR_R", "0.081"))
            # v793: 爬升段长可调（S10_STAIR_CLIMB_L，默认 0.15m）+ smoothstep
            # ——v792 线性 0.081m 内升 0.125m 太快（1.2m/s 下仅 0.067s）腿跟
            # 不上弹回；0.15m 平滑给 0.125s（1m/s 垂直），轮贴面滚上。
            _cl = float(os.environ.get("S10_STAIR_CLIMB_L", "0.15"))
            for k, (y_r, z_top) in enumerate(zip(rs, ts)):
                z_bottom = (self.STAIR_GROUND if k == 0
                            else float(ts[k - 1]))
                h = z_top - z_bottom
                if h <= radius:
                    continue
                d = y_r - y
                mask = (d > 0.0) & (d <= _cl)
                t = np.clip(1.0 - d / max(_cl, 1e-6), 0.0, 1.0)
                ss = t * t * (3.0 - 2.0 * t)
                z_climb = z_bottom + radius + h * ss
                z = np.where(mask, np.maximum(z, z_climb), z)
            return z
        if os.environ.get("S10_STAIR_REF_STEP", "0") == "1":
            lead = float(os.environ.get("S10_STAIR_REF_LEAD", "0.20"))
            # v209: 只对高 riser（>S10_STAIR_REF_MINH，默认 0.09m）用满值——
            # 0.062m 首级可滚过，强抬反而卡 riser1（v207 实测）。
            _minh = float(os.environ.get("S10_STAIR_REF_MINH", "0.09"))
            for k, (y_r, z_top) in enumerate(zip(rs, ts)):
                z_bottom = self.STAIR_GROUND if k == 0 else float(ts[k - 1])
                if (z_top - z_bottom) < _minh:
                    continue
                lo = y_r - lead
                hi = y_r + b
                z_full = z_top + radius + margin
                z = np.where((y >= lo) & (y <= hi),
                             np.maximum(z, z_full), z)
            return z
        for k, (y_r, z_top) in enumerate(zip(rs, ts)):
            z_bottom = self.STAIR_GROUND if k == 0 else float(ts[k - 1])
            lo = y_r - a
            hi = y_r + b
            t = np.clip((y - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            s = t * t * (3.0 - 2.0 * t)
            z_ramp = z_bottom + radius + s * (z_top - z_bottom + margin)
            z = np.where((y >= lo) & (y <= hi),
                         np.maximum(z, z_ramp), z)
        return z

    def stair_base_z_ref(self, y, stand=0.205):
        """机身 z 参考 = 地形+站姿高，riser 前滞后 0.1m 平滑抬（车随轮上）。"""
        rs, ts = self._stair_tables()
        a = float(os.environ.get("S10_STAIR_RAMP_A", "0.14"))
        b = float(os.environ.get("S10_STAIR_RAMP_B", "0.10"))
        lag = float(os.environ.get("S10_STAIR_BASE_LAG", "0.10"))
        y = np.asarray(y, dtype=np.float64)
        z = self.stair_terrain(y) + stand
        for k, (y_r, z_top) in enumerate(zip(rs, ts)):
            z_bottom = self.STAIR_GROUND if k == 0 else float(ts[k - 1])
            lo = y_r - a + lag
            hi = y_r + b + lag
            t = np.clip((y - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            s = t * t * (3.0 - 2.0 * t)
            z_ramp = z_bottom + stand + s * (z_top - z_bottom)
            z = np.where((y >= lo) & (y <= hi),
                         np.maximum(z, z_ramp), z)
        return z

    def stair_pitch_ref(self, y, amp=None):
        """riser 前仰头：窗口 [y_r-a, y_r+b] 内 tent 形（先升后平），踏面回 0。"""
        if amp is None:
            amp = float(os.environ.get("S10_STAIR_PITCH_AMP", "0.22"))
        rs, _ = self._stair_tables()
        a = float(os.environ.get("S10_STAIR_RAMP_A", "0.14"))
        b = float(os.environ.get("S10_STAIR_RAMP_B", "0.10"))
        y = np.asarray(y, dtype=np.float64)
        p = np.zeros(y.shape, dtype=np.float64)
        for y_r in rs:
            t1 = np.clip((y - (y_r - a)) / max(a, 1e-6), 0.0, 1.0)
            s1 = t1 * t1 * (3.0 - 2.0 * t1)
            t2 = np.clip((y - y_r) / max(b, 1e-6), 0.0, 1.0)
            s2 = t2 * t2 * (3.0 - 2.0 * t2)
            p = np.maximum(p, amp * s1 * (1.0 - s2))
        # 本工程约定：pitch 负值 = 仰头（抬前轮，见 s10_env r_pitch 注释）
        return -p

    def stair_v_ref(self, y, v_slow=None, v_fast=None):
        """riser 前 0.25m 减速到 v_slow、riser 后 0.15m 恢复到 v_fast（取各 riser 最小值）。"""
        if v_slow is None:
            v_slow = float(os.environ.get("S10_STAIR_V_SLOW", "1.2"))
        if v_fast is None:
            v_fast = float(os.environ.get("S10_STAIR_V_FAST", "1.8"))
        rs, _ = self._stair_tables()
        y = np.asarray(y, dtype=np.float64)
        v = np.full(y.shape, v_fast, dtype=np.float64)
        for y_r in rs:
            d_in = 0.10
            t_in = np.clip((y - (y_r - 0.25)) / d_in, 0.0, 1.0)
            s_in = t_in * t_in * (3.0 - 2.0 * t_in)
            t_out = np.clip((y - (y_r + 0.05)) / d_in, 0.0, 1.0)
            s_out = t_out * t_out * (3.0 - 2.0 * t_out)
            v_ramp = v_slow + (v_fast - v_slow) * s_in * (1.0 - s_out)
            v = np.minimum(v, np.where(
                (y >= y_r - 0.25) & (y <= y_r + 0.15), v_ramp, v_fast))
        return v

    def stair_known_tile(self, x0, y0, nx, ny, res):
        """P2.1（v203）：STAIR 区已知几何瓦片覆盖。

        楼梯带（STAIR_ZONE_X/Y）内用 stair_terrain 精确高度图；该区是已知
        可爬地形，slope/roughness/step 全部置 0（地形障碍惩罚=0，避免把必须
        翻越的 riser 当墙逼离走廊）；区外 valid=False 回退感知瓦片。返回 dict
        （heightmap/valid/slope/roughness/step/step_flag）或 None（无交集）。
        """
        ys = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * res
        xs = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * res
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        h = self.stair_terrain(yy).astype(np.float32)
        valid = ((yy >= self.STAIR_ZONE_Y[0]) & (yy <= self.STAIR_ZONE_Y[1])
                 & (xx >= self.STAIR_ZONE_X[0]) & (xx <= self.STAIR_ZONE_X[1]))
        if not bool(valid.any()):
            return None
        z = np.zeros((ny, nx), dtype=np.float32)
        return {
            "heightmap": h,
            "valid": valid,
            "slope": z,
            "roughness": z,
            "step": z,
            "step_flag": z,
        }

    def stair_foothold_y_grid(self, x0, y0, nx, ny, res):
        """v206：落脚点前拉场（foothold planning 软落地）。

        楼梯带内每格给出"下一级 riser 的 y + 踏面中距"作为该轮落脚点 y；
        摆动相（要抬的轮）被 cost 向前拉到落脚点，激励 hipy 前摆把轮子
        放到下一级踏面（卡点运动学分析：后轮上 riser 需 hipy 前摆 +1.2rad）。
        区外/末级之后 valid=False（无前拉）。返回 (foothold_y, valid)。
        """
        ys = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * res
        xs = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * res
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        rs = self.STAIR_RISERS
        step = float(os.environ.get("S10_FOOTHOLD_STEP", "0.12"))
        fy = np.full(yy.shape, np.nan, dtype=np.float64)
        # 每格取"下一个未过的 riser"作为落脚目标级
        for r in rs:
            fy = np.where((yy < r) & np.isnan(fy), r + step, fy)
        fy = np.where(np.isnan(fy), float(rs[-1]) + 0.4, fy)  # 末级后平地
        valid = ((yy >= self.STAIR_ZONE_Y[0]) & (yy <= self.STAIR_ZONE_Y[1])
                 & (xx >= self.STAIR_ZONE_X[0]) & (xx <= self.STAIR_ZONE_X[1])
                 & (yy < float(rs[-1]) + 0.25))
        return fy.astype(np.float32), valid

    def gait_schedule(self, wheel_y, wheel_z, t, dt=0.005):
        """v213: 顺序步态调度（按顺序踩楼梯，dial-MPC 自然涌现）。

        对角序列 FL(0)->HR(3)->FR(1)->HL(2)（S10_GAIT_SEQ 可配）。当前序列轮
        进入其下一级 riser 的接近区（y > next_riser - S10_GAIT_LEAD，默认
        0.30m）即摆动（swing=1，该轮得到完整抬升/落脚点 cost）；该轮轮心
        到位（>= 下一级顶+半径-0.02 持续 0.12s）或超时（S10_GAIT_TIMEOUT，
        默认 1.0s）-> 切换序列下一轮。区外/未进区不推进（等它滚到接近区）。
        返回 swing flags (4,) float32。
        """
        if os.environ.get("S10_GAIT_UTIL", "0") == "1":
            return self._gait_schedule_util(wheel_y, wheel_z, t)
        seq = [int(x) for x in os.environ.get(
            "S10_GAIT_SEQ", "0,3,1,2").replace(" ", "").split(",")]
        lead = float(os.environ.get("S10_GAIT_LEAD", "0.30"))
        timeout = float(os.environ.get("S10_GAIT_TIMEOUT", "1.0"))
        margin = float(os.environ.get("S10_STAIR_LIFT_MARGIN", "0.05"))
        rs, ts = self._stair_tables()
        if not hasattr(self, "_gait_pos"):
            self._gait_pos = 0
            self._gait_wheel = seq[0]
            self._gait_t0 = t
            self._gait_done_t = None
            self._gait_swing = np.zeros(4, dtype=np.float32)
        # 每轮下一个未过的 riser（y 方向）
        nxt = np.full(4, 1e9, dtype=np.float64)
        nxt_z = np.zeros(4, dtype=np.float64)
        for k in range(4):
            idx = int(np.searchsorted(rs, float(wheel_y[k])))
            if idx < len(rs):
                nxt[k] = float(rs[idx])
                nxt_z[k] = float(ts[idx]) + 0.081
            else:
                nxt_z[k] = float(wheel_z[k])
        active = int(self._gait_wheel)
        in_zone = (nxt[active] < 1e8
                   and float(wheel_y[active]) > nxt[active] - lead)
        if in_zone:
            # 完成判定：到位（轮心 >= 下一级顶+半径-0.02，持续 0.12s）或超时
            if float(wheel_z[active]) >= nxt_z[active] - 0.02:
                if self._gait_done_t is None:
                    self._gait_done_t = t
                elif t - self._gait_done_t > 0.12:
                    self._gait_pos = (self._gait_pos + 1) % len(seq)
                    self._gait_wheel = seq[self._gait_pos]
                    self._gait_t0 = t
                    self._gait_done_t = None
            elif t - self._gait_t0 > timeout:
                self._gait_pos = (self._gait_pos + 1) % len(seq)
                self._gait_wheel = seq[self._gait_pos]
                self._gait_t0 = t
                self._gait_done_t = None
            else:
                self._gait_done_t = None
        else:
            self._gait_done_t = None
        swing = np.zeros(4, dtype=np.float32)
        if in_zone:
            swing[active] = 1.0
        self._gait_swing = swing
        self._gait_seq = seq
        if os.environ.get("S10_GAIT_DEBUG", "0") == "1":
            print(f"[GAIT] t={t:.1f} seq={seq} pos={self._gait_pos} "
                  f"active={active} inzone={int(in_zone)} "
                  f"nxt={[round(float(v),2) if v<1e8 else -1 for v in nxt]} "
                  f"nxtz={[round(float(v),2) for v in nxt_z]} "
                  f"wz={[round(float(v),2) for v in wheel_z]} "
                  f"swing={[int(s) for s in swing]}",
                  flush=True)
        return swing


    def _gait_schedule_util(self, wheel_y, wheel_z, t):
        """v214: utility 选腿（Bjelonic IROS2021 运动学腿效用，软版）。

        每轮计算四腿 lift-need = clip((下一级 riser 轮心目标 - 当前轮心)
        / S10_GAIT_REACH, 0, 1)，作为连续 swing 权重（0~1）；同侧对
        （FL-HL / FR-HR）按 S10_GAIT_SS_SUPPRESS 软抑制防支撑线化；
        低于 S10_GAIT_NEED_THR 清零。只决定 cost 相位权重，无硬门控。
        """
        lead = float(os.environ.get("S10_GAIT_LEAD", "0.30"))
        reach = float(os.environ.get("S10_GAIT_REACH", "0.30"))
        need_thr = float(os.environ.get("S10_GAIT_NEED_THR", "0.05"))
        ss_supp = float(os.environ.get("S10_GAIT_SS_SUPPRESS", "0.6"))
        rs, ts = self._stair_tables()
        nxt = np.full(4, 1e9, dtype=np.float64)
        nxt_z = np.zeros(4, dtype=np.float64)
        for k in range(4):
            idx = int(np.searchsorted(rs, float(wheel_y[k])))
            if idx < len(rs):
                nxt[k] = float(rs[idx])
                nxt_z[k] = float(ts[idx]) + 0.081
            else:
                nxt_z[k] = float(wheel_z[k])
        in_zone = (nxt < 1e8) & (wheel_y > nxt - lead)
        util = np.where(
            in_zone, np.clip((nxt_z - wheel_z) / reach, 0.0, 1.0), 0.0)
        util = np.asarray(util, dtype=np.float64)
        # v214b: 对角双轮选择（Bjelonic 邻腿检查软版）——主选 lift-need
        # 最大的轮（全权重），次选与主选成对角的轮（按需权重）；同侧/同轴
        # 轮不同时摆动，避免双后轮齐摆把车身压塌（v214-D 实测 body 0.81→
        # 0.75）。对角：0<->3（FL-HR）、1<->2（FR-HL）。
        # v214c: S10_GAIT_FRONT_PRIO>1 时前轮在选择中加权（爬梯自然顺序：
        # 前轮先上 → 车身抬升卸载后轮 → 后轮再上），swing 权重仍按真实
        # lift-need（不放大 cost）。
        _fp = float(os.environ.get("S10_GAIT_FRONT_PRIO", "1.0"))
        _util_sel = util.copy()
        if _fp > 1.0:
            _util_sel[:2] = _util_sel[:2] * _fp
        _diag = {0: 3, 1: 2, 2: 1, 3: 0}
        _max_swing = int(os.environ.get("S10_GAIT_MAX_SWING", "2"))
        # v214g: 主选滞回——HL/HR need 几乎相等时每帧翻转（实测 HR 抬一次
        # 0.77 后回 0.63），任何轮都没有完整摆动窗口。S10_GAIT_HOLD_TIME
        # 内保持当前主选（只要其 need 仍有效），新候选需超过
        # S10_GAIT_SWITCH_MARGIN× 才切换（软时序，非门控）。
        _hold = float(os.environ.get("S10_GAIT_HOLD_TIME", "0.0"))
        _sw_m = float(os.environ.get("S10_GAIT_SWITCH_MARGIN", "1.0"))
        if not hasattr(self, "_gaitu_prim"):
            self._gaitu_prim = None
            self._gaitu_t0 = t
        # v214i: 前轴优先（S10_GAIT_AXLE=1）——爬梯自然顺序：双前轮齐抬
        # 越过 riser（后轮保持接地推身）→ 前进 → 后轴再抬。对角选择会把
        # 推身轮（前）和抬升轮（后）同时摆，无牵引（实测卡死）。
        _axle = int(os.environ.get("S10_GAIT_AXLE", "0"))
        swing = np.zeros(4, dtype=np.float32)
        if _axle == 1:
            _fneed = max(float(util[0]), float(util[1]))
            _rneed = max(float(util[2]), float(util[3]))
            if _fneed >= need_thr and _fneed >= _rneed * 0.75:
                _fpw = float(os.environ.get("S10_GAIT_AXLE_W", "0.8"))
                swing[0] = _fpw if util[0] >= need_thr else 0.0
                swing[1] = _fpw if util[1] >= need_thr else 0.0
            elif _rneed >= need_thr:
                _fpw = float(os.environ.get("S10_GAIT_AXLE_W", "0.8"))
                swing[2] = _fpw if util[2] >= need_thr else 0.0
                swing[3] = _fpw if util[3] >= need_thr else 0.0
            self._gaitu_prim = 0 if swing[:2].sum() > 0 else (
                2 if swing[2:].sum() > 0 else None)
            swing = np.asarray(swing, dtype=np.float32)
            self._gait_swing = swing
            if os.environ.get("S10_GAIT_DEBUG", "0") == "1":
                print(f"[GAITU] t={t:.1f} AXLE "
                      f"fneed={_fneed:.2f} rneed={_rneed:.2f} "
                      f"wz={[round(float(v), 2) for v in wheel_z]} "
                      f"swing={[round(float(v), 2) for v in swing]}",
                      flush=True)
            return swing
        _order = np.argsort(-_util_sel)
        _cand = int(_order[0])
        _prim = _cand
        if (self._gaitu_prim is not None and _hold > 0.0
                and t - self._gaitu_t0 < _hold
                and util[self._gaitu_prim] >= need_thr):
            # 严格保持：主选轮 need 仍有效就不切换（完成/失效才放行）
            _prim = self._gaitu_prim
        else:
            self._gaitu_prim = _prim
            self._gaitu_t0 = t
        if util[_prim] >= need_thr:
            swing[_prim] = 1.0
            if _max_swing >= 2:
                for _k in _order[1:]:
                    if util[_k] < need_thr:
                        break
                    if _k == _diag[_prim]:
                        swing[_k] = float(util[_k])
                        break
        # 同侧软抑制仍保留（若对角次选低于 need_thr，抑制同侧残余噪声）
        if ss_supp > 0.0:
            for a, b in ((0, 2), (1, 3)):
                if swing[a] > 0.0 and swing[b] > 0.0:
                    _m = min(swing[a], swing[b]) * ss_supp
                    if swing[a] <= swing[b]:
                        swing[a] = max(0.0, swing[a] - _m)
                    else:
                        swing[b] = max(0.0, swing[b] - _m)
        swing = np.asarray(swing, dtype=np.float32)
        self._gait_swing = swing
        if os.environ.get("S10_GAIT_DEBUG", "0") == "1":
            print(f"[GAITU] t={t:.1f} "
                  f"nxt={[round(float(v), 2) if v < 1e8 else -1 for v in nxt]} "
                  f"nxtz={[round(float(v), 2) for v in nxt_z]} "
                  f"wz={[round(float(v), 2) for v in wheel_z]} "
                  f"util={[round(float(v), 2) for v in swing]}",
                  flush=True)
        return swing

    def stair_next_riser_ref(self, y, radius=0.081):
        """v208：下一级 riser 的轮心满值参考（供 field-driven bias 用）。

        对每个 y，返回"下一个未过的 riser 顶 + 半径 + margin"（恒定满值，
        非 ramp 渐进）。卡点左轮距 ramp ref 仅 2.7-3.3cm（bias 按此缩放只发
        18% 力度），实际需抬 12.5cm；用满值参考让 bias 在接近段以全额幅度
        推动采样均值（cost 保持 ramp 不变，只在轮真抬起时奖励）。
        """
        rs, ts = self._stair_tables()
        margin = float(os.environ.get("S10_STAIR_LIFT_MARGIN", "0.05"))
        y = np.asarray(y, dtype=np.float64)
        z = self.stair_terrain(y) + radius
        _minh = float(os.environ.get("S10_STAIR_REF_MINH", "0.09"))
        for k, (y_r, z_top) in enumerate(zip(rs, ts)):
            # y 越过本级后（+0.05m），下一目标 = 本级顶满值（+margin）；
            # 只对高 riser 生效（0.062m 首级可滚过不强抬）
            z_bottom = self.STAIR_GROUND if k == 0 else float(ts[k - 1])
            if (z_top - z_bottom) < _minh:
                continue
            z = np.where(y > y_r + 0.05, z_top + radius + margin, z)
        return z

    def stair_wheel_ref_grid(self, x0, y0, nx, ny, res):
        """构建对齐感知瓦片 (origin=(x0,y0), res, shape=(ny,nx)) 的轮心 z 参考网格。
        有效区 = 楼梯带（STAIR_ZONE_X/Y 交集），区外 valid=False（回退感知/接触机制）。"""
        ys = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * res
        xs = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * res
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        wc = self.stair_wheel_ref(yy)
        valid = ((yy >= self.STAIR_ZONE_Y[0]) & (yy <= self.STAIR_ZONE_Y[1])
                 & (xx >= self.STAIR_ZONE_X[0]) & (xx <= self.STAIR_ZONE_X[1]))
        return wc.astype(np.float32), valid
