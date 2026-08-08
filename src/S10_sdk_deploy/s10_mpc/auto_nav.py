"""模式 A：已知地图自动导航（纯追踪 + 坡度/弯道限速）。

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
    def __init__(self, waypoints, max_speed=4.0, vyaw_max=2.0,
                 yaw_gain=2.5, lookahead=4.0, lat_accel_max=6.0,
                 climb_max_speed=1.5, grade_scale=5.0, speed_window=3,
                 lat_gain=1.5, max_accel=5.0, yaw_damp=0.6, cte_gain=2.0):
        """
        waypoints: (N,3) 全局航点 [x, y, z]
        """
        self.wp = np.asarray(waypoints, dtype=np.float64)
        self.max_speed = float(max_speed)
        self.vyaw_max = float(vyaw_max)
        self.yaw_gain = float(yaw_gain)
        # pursuit 前瞻 4m（2026-08-06 用户 1.1）：2.5m 在弯道切内圈 →
        # CTE 0.8m+ → 拉回侧翻；4m 瞄准更远，弯道走线更贴路径。
        self.lookahead = float(os.environ.get(
            "S10_AUTO_LOOKAHEAD", str(lookahead)))
        self.lat_accel_max = float(lat_accel_max)
        self.climb_max_speed = float(climb_max_speed)
        self.grade_scale = float(grade_scale)
        self.speed_window = int(speed_window)
        self.lat_gain = float(lat_gain)
        self.max_accel = float(os.environ.get(
            "S10_AUTO_MAX_ACCEL", str(max_accel)))
        self.yaw_damp = float(yaw_damp)   # 实测 yaw rate 阻尼，防航向过冲
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
        amp = float(os.environ.get("S10_STAIR_CORRIDOR_X", "0.6"))
        if amp <= 0.0:
            return out
        y = out[:, 1]
        y_in0, y_in1 = 33.0, 39.5      # 进入斜坡：0→amp
        y_out0, y_out1 = 39.5, 41.2    # 退出斜坡：amp→0（收敛回 wp7）
        t_in = np.clip((y - y_in0) / (y_in1 - y_in0), 0.0, 1.0)
        t_out = np.clip((y - y_out1) / (y_out0 - y_out1), 0.0, 1.0)
        ramp = np.where(y < y_in1,
                        np.sin(0.5 * np.pi * t_in) ** 2,
                        np.sin(0.5 * np.pi * t_out) ** 2)
        out[:, 0] += amp * ramp
        return out

    def _stair_diag_bump(self, xy):
        """v133: shared diagonal bump for dense path points (y in [37.8,40.6]).
        Amp from S10_STAIR_DIAG_AMP (0=off). Returns copy."""
        out = np.asarray(xy, dtype=np.float64).copy()
        a = float(os.environ.get("S10_STAIR_DIAG_AMP", "0.0"))
        if a <= 0.0:
            return out
        y0, y1 = 37.8, 40.6
        t = np.clip((out[:, 1] - y0) / (y1 - y0), 0.0, 1.0)
        out[:, 0] = out[:, 0] + a * np.sin(np.pi * t)
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
            # 坡度：前向航段
            if i < n - 1:
                dz = wp[i + 1, 2] - wp[i, 2]
                grade = dz / max(self.seg_len[i], 1e-3)
                # 平地段用满速；坡度越大减速越多，下限 climb_max_speed
                v_grade = self.max_speed / (1.0 + self.grade_scale * abs(grade))
                v_grade = max(v_grade, self.climb_max_speed)
            else:
                v_grade = self.max_speed
            if (i < n - 1
                    and wp[i + 1, 2] - wp[i, 2] > 0.08):
                self.step_zone[i] = True     # 本航段终点是台阶/陡升
            if (i < n - 1
                    and wp[i + 1, 2] - wp[i, 2] > 0.25):
                self.stair_zone[i] = True    # 连续楼梯（多级台阶）
            self.speed_limit[i] = min(self.max_speed, v_curve, v_grade)

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

    def _racing_line_arc(self, raw, xy, n_per, cut_r):
        """把转角航点两侧的样条替换为相切圆弧（赛车线切弯，完整版）。

        对每个转角>阈值的航点 i：沿路径弧长取切点 p_in/p_out
        （d=R·tan(θ/2)），删除两点之间样条，插入半径 R 的圆弧。
        多弯按弧长合并；重叠弯（连续反向弯段短）跳过（保守）。
        """
        n = len(xy)
        seg = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        cuts = []
        for i in range(1, n - 1):
            a = xy[i - 1]
            b = xy[i]
            c = xy[i + 1]
            v1 = b - a
            v2 = c - b
            L1 = float(np.linalg.norm(v1))
            L2 = float(np.linalg.norm(v2))
            if L1 < 1e-3 or L2 < 1e-3:
                continue
            v1 = v1 / L1
            v2 = v2 / L2
            cos_t = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
            theta = np.arccos(cos_t)
            if theta < float(os.environ.get("S10_RACING_CUT_MIN_TURN", "0.45")):
                continue
            d = float(cut_r * np.tan(theta / 2.0))
            d = min(d, L1 * 0.45, L2 * 0.45)
            if d < 0.15:
                continue
            k = int(np.argmin(np.sum((raw - b[None, :]) ** 2, axis=1)))
            s_wp = float(cum[k])
            s_in = max(0.0, s_wp - d)
            s_out = min(cum[-1], s_wp + d)
            p_in = self._raw_point_at(raw, cum, s_in)
            p_out = self._raw_point_at(raw, cum, s_out)
            side = float(v1[0] * v2[1] - v1[1] * v2[0])
            ni = np.array([-v1[1], v1[0]])
            no = np.array([-v2[1], v2[0]])
            c_dir = -1.0 if side > 0 else 1.0
            center = (p_in + c_dir * ni * cut_r
                      + p_out + c_dir * no * cut_r) / 2.0
            ang_in = np.arctan2(p_in[1] - center[1], p_in[0] - center[0])
            ang_out = np.arctan2(p_out[1] - center[1], p_out[0] - center[0])
            n_arc = max(int(2.0 * d / max(float(seg.mean()), 1e-4)), 8)
            if side > 0:
                if ang_out < ang_in:
                    ang_out += 2.0 * np.pi
            else:
                if ang_out > ang_in:
                    ang_out -= 2.0 * np.pi
            angs = np.linspace(ang_in, ang_out, n_arc)
            arc = np.column_stack([
                center[0] + cut_r * np.cos(angs),
                center[1] + cut_r * np.sin(angs)])
            cuts.append((s_in, s_out, arc))
        if not cuts:
            return raw
        cuts.sort(key=lambda c: c[0])
        merged = []
        for s_in, s_out, arc in cuts:
            if merged and s_in < merged[-1][1] - 1e-3:
                continue
            merged.append((s_in, s_out, arc))
        pts = []
        cur = 0.0
        for s_in, s_out, arc in merged:
            mask = (cum >= cur - 1e-6) & (cum <= s_in + 1e-6)
            pts.extend(raw[mask])
            pts.extend(arc[1:-1])
            cur = s_out
        mask = cum >= cur - 1e-6
        pts.extend(raw[mask])
        return np.asarray(pts, dtype=np.float64)

    def _raw_point_at(self, raw, cum, s):
        if s <= 0:
            return raw[0].copy()
        if s >= cum[-1]:
            return raw[-1].copy()
        k = int(np.searchsorted(cum, s, side="right") - 1)
        k = max(0, min(k, len(raw) - 2))
        t = (s - cum[k]) / max(cum[k + 1] - cum[k], 1e-6)
        return raw[k] + t * (raw[k + 1] - raw[k])

    def _build_smooth_path(self):
        """Catmull-Rom 样条过航点 → 均匀弧长采样 + 数值曲率/速度剖面。

        Catmull-Rom：三次 Hermite 插值，**严格经过每个航点**（判点 0.2m
        半径保证），C1 连续平滑转弯；实现简单无几何 bug（替代圆弧/切角，
        2026-08-06 圆弧两个几何 bug 导致绕圈/变长/外偏 0.5m 已弃用）。
        速度剖面：数值曲率 κ=|dθ/ds| 平滑后 v=min(v_max, √(a_lat/κ))，
        曲率 clamp（R_min=0.8m → v_min≈2.2m/s）防过冲造成过慢。
        """
        wp = self.wp
        n = len(wp)
        res = self.path_res
        n_per = int(os.environ.get("S10_GLOBAL_NPER_SEG", "24"))
        xy = self._stair_corridor_xy(wp[:, :2])
        raw = [xy[0].copy()]
        # 切线因子（2026-08-08）：默认 0.5 = 标准 Catmull-Rom。增大到
        # 0.7~0.8 让弯道更平缓（wp 处半径 1.36→2.1~4.0m），vlim 自动
        # 提高（2.85→3.4~4.4），路径仍严格过航点（判点安全）。
        _tk = float(os.environ.get("S10_GLOBAL_TANGENT_K", "0.5"))
        for i in range(n - 1):
            p0 = xy[max(i - 1, 0)]
            p1 = xy[i]
            p2 = xy[i + 1]
            p3 = xy[min(i + 2, n - 1)]
            m1 = (p2 - p0) * _tk
            m2 = (p3 - p1) * _tk
            for k in range(1, n_per):
                t = k / n_per
                t2 = t * t
                t3 = t2 * t
                h00 = 2 * t3 - 3 * t2 + 1
                h10 = t3 - 2 * t2 + t
                h01 = -2 * t3 + 3 * t2
                h11 = t3 - t2
                raw.append(h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2)
            raw.append(xy[i + 1].copy())
        raw = np.asarray(raw, dtype=np.float64)

        # 赛车线圆弧切弯（2026-08-07，用户"参考 MPPI 赛车/摩托"）：
        # 对每个转角>阈值的航点，把前后各 cut_len 内的样条替换为与
        # 两切点相切的圆弧（半径 R≥min_r，外偏 apex 控制在航点判点内）。
        _cut_r = float(os.environ.get("S10_RACING_CUT_R", "0.0"))
        if _cut_r > 0.1:
            raw = self._racing_line_arc(raw, xy, n_per, _cut_r)
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

        # 速度剖面：v(s) = min(v_max, √(a_lat·R))，圆角处按曲率限速
        curve_accel = float(os.environ.get("S10_CURVE_ACCEL", "6.0"))
        vlim = np.full(len(pts), self.max_speed, dtype=np.float64)
        for k, c in enumerate(self.path_curv):
            if c > 1e-6:
                vlim[k] = min(vlim[k], np.sqrt(
                    curve_accel / max(c, 1e-4)))
        # 转向能力约束（2026-08-07）：弯道速度不能超过 vyaw_max * R——
        # 否则 MPC 实际 yaw 跟不上，转向滞后→过冲→振荡。v = ω·R。
        for k, c in enumerate(self.path_curv):
            if c > 1e-6:
                vlim[k] = min(vlim[k],
                              float(os.environ.get("S10_AUTO_VYAW_MAX", "2.0"))
                              / max(c, 1e-4))
        # S 弯组合限速（2026-08-07，nr15/16 wp3 侧翻复现）：wp2→3→4 是
        # 连续反向弯（右66.7°→左55°→右71.3°，段长仅4.6~4.8m），机器人在
        # 3 m/s 下转向来不及，err 爆发→饱和→侧翻。检测 6m 窗口内同时存在
        # 正负大曲率（连续反向弯），组合段整体限速更保守。
        _sw = int(float(os.environ.get("S10_CURVE_SWING_WINDOW", "6.0")) / res)
        _swing_v = float(os.environ.get("S10_CURVE_SWING_VX", "2.6"))
        for k in range(len(vlim)):
            lo = max(0, k - _sw)
            hi = min(len(vlim), k + _sw + 1)
            if (float(np.max(self.path_curv_signed[lo:hi])) > 0.25
                    and float(np.min(self.path_curv_signed[lo:hi])) < -0.25):
                vlim[k] = min(vlim[k], _swing_v)
        # 弯道减速前向传播（2026-08-06 用户 1.1）：曲率大的点往前 5m
        # 线性压低 vlim——4m/s 冲进 71° 弯转向不及（wp3→4 北偏复现），
        # 弯道前必须提前减速（距离前瞻而非瞬时曲率）。
        _decel_ahead = int(float(os.environ.get(
            "S10_CURVE_DECEL_AHEAD", "5.0")) / res)
        for k in range(len(vlim)):
            if vlim[k] < self.max_speed - 0.5:
                for j in range(max(0, k - _decel_ahead), k):
                    vlim[j] = min(vlim[j], vlim[k])
        # 台阶/楼梯段限速映射到路径弧长（按航点区间）
        for i in range(n - 1):
            if not (self.step_zone[i] or self.stair_zone[i]):
                continue
            if i == 0:
                continue   # wp0→1 起步缓坡（z 升 0.475）不是台阶
            s0 = self.cum_len[i]
            s1 = self.cum_len[i + 1]
            v_zone = (self.stair_vx if self.stair_zone[i] else self.step_vx)
            _zm = float(os.environ.get("S10_ZONE_MARGIN", "1.0"))
            mask = (cum >= s0 - _zm) & (cum <= s1 + _zm)
            vlim[mask] = np.minimum(vlim[mask], v_zone)
        self.path_vlim = vlim
        self.path_total = float(cum[-1])
        # 航点在平滑路径上的弧长（统一 pursuit 的 passed 判断标尺：
        # 平滑弧长 ≠ 折线弧长，直接用折线 cum_len 会错位 10m+）
        self.path_wp_s = np.array([
            float(self.path_cum[np.argmin(np.sum(
                (self.path_pts[:, :2] - self.wp[i, :2]) ** 2, axis=1))])
            for i in range(len(self.wp))])

    def compute_cmd(self, robot_xy, yaw, next_idx, robot_z=None, yaw_rate=0.0):
        """返回 (vx, vyaw)。robot_xy: (2,) 全局位置；next_idx: 下一个未到达航点。"""
        if next_idx >= len(self.wp):
            return 0.0, 0.0
        wp_next = self.wp[next_idx]
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
        self._s_cur = max(
            getattr(self, "_s_cur", 0.0) + _proj * 0.5, s_proj)
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
        if d_wp < float(os.environ.get("S10_AUTO_WP_AIM", "0.8")) or passed:
            target = wp_next
        else:
            s_target = min(self._s_cur + self.lookahead, self.path_total)
            target = self._path_point_at(s_target)
        err = np.arctan2(target[1] - robot_xy[1],
                         target[0] - robot_xy[0]) - yaw
        err = float(np.arctan2(np.sin(err), np.cos(err)))
        self._last_err = err
        self._last_dwp = d_wp
        self._last_tgt = target
        # 高架/坡顶段限制速度与转向：离地越高，侧翻风险越大（窄轮距）
        z_ahead = float(robot_z if robot_z is not None else 0.0)
        for j in range(self.speed_window):
            if next_idx + j < len(self.wp):
                z_ahead = max(z_ahead, float(self.wp[next_idx + j, 2]))
        # 高架限速系数（S10_AUTO_ELEV_K，默认 0.6）：z 越高限速越狠。
        # 实测 z=3.75 平直高架段 elev_f=0.33 → 巡航 ~1.3 m/s，对竞速明显过慢
        # （用户反馈"限速不合理"）；0.6 为保守默认，全圈验证后可调小。
        # 原 1.5 把坡上速度压到 2.1 m/s，动量不足过 0.13m 台阶——
        # 复现证明 4.5 m/s 可连续过台阶（台阶区由 step_zone 单独限速）。
        elev_k = float(os.environ.get("S10_AUTO_ELEV_K", "0.6"))
        elev_factor = 1.0 / (1.0 + elev_k * max(0.0, z_ahead - 0.4))
        vyaw_max_eff = self.vyaw_max * elev_factor
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
            cte_corr = -self.cte_gain * float(np.clip(cte / 2.0, -1.0, 1.0))
            if (abs(cte) < 6.0 and abs(err) < float(os.environ.get(
                    "S10_AUTO_CTE_ERR_GATE", "0.6"))
                    and cte_corr * err >= -0.5):
                vyaw = float(self.yaw_gain * err + yaw_ff
                             - self.yaw_damp * yaw_rate + cte_corr)
            else:
                vyaw = float(self.yaw_gain * err + yaw_ff
                             - self.yaw_damp * yaw_rate)
        else:
            vyaw = float(self.yaw_gain * err + yaw_ff
                         - self.yaw_damp * yaw_rate)
        vyaw = float(np.clip(vyaw, -vyaw_max_eff, vyaw_max_eff))
        # 变化率限制（防反馈振荡；每次调用 = 0.05s）
        vyaw = float(np.clip(
            vyaw, self._last_vyaw_out - self.vyaw_slew,
            self._last_vyaw_out + self.vyaw_slew))
        self._last_vyaw_out = vyaw

        # 限速：平滑路径速度剖面（全局导航层，曲率/坡度/台阶已编码；
        # 8m 前瞻 = 4m/s 下提前 2s 减速，弯道转向跟得上）
        _k_far = min(self._k_near + int(float(os.environ.get(
            "S10_AUTO_VLIM_LOOKAHEAD", "5.0")) / self.path_res),
                     len(self.path_vlim) - 1)
        v_lim = float(np.min(self.path_vlim[self._k_near:_k_far + 1]))
        # 转向速度分级：|err|>0.3 时——
        #   近点（d_wp<3m，如起步/航点大转角）：0.4 m/s 原地转向，避免冲过航点；
        #   远点（如爬坡段）：1.5 m/s 慢速转弯，保持推力爬台阶。
        if abs(err) > float(os.environ.get("S10_AUTO_ERR_GATE", "0.30")):
            # 弯道/起步限速（2026-08-07）：err>1.0（起步 90° 或急弯）必须
            # 慢速转向防侧翻；gate~1.0 用 turn_vx（0.45 可让直线段小偏差
            # 不触发减速）；进弯前 3m 线性减速。
            big_err_vx = float(os.environ.get("S10_AUTO_BIGERR_VX", "1.5"))
            turn_vx = float(os.environ.get("S10_AUTO_TURN_VX", "2.5"))
            near_vx = float(os.environ.get("S10_AUTO_NEAR_VX", "1.8"))
            if abs(err) > 1.0:
                v_lim = min(v_lim, big_err_vx * elev_factor)
            elif d_wp < 0.5:
                v_lim = min(v_lim, near_vx * elev_factor)
            elif d_wp < float(os.environ.get("S10_AUTO_NEAR_DIST", "2.0")):
                v_lim = min(v_lim, max(2.0, self.max_speed * d_wp
                                       / float(os.environ.get(
                                           "S10_AUTO_NEAR_RAMP", "2.0")))
                            * elev_factor)
            else:
                v_lim = min(v_lim, turn_vx * elev_factor)
        else:
            v_lim = min(v_lim, self.max_speed * elev_factor)
        # 接近当前航点时减速，保证进入 0.2m 到达半径判定。
        # 高架（z>0.9）时额外减半——坡顶最后一级台阶高速接近会前翻（实测）。
        if d_wp < float(os.environ.get("S10_AUTO_NEAR_DIST", "2.0")):
            # 高架（z>0.9）接近速度压到 1.0 m/s——坡顶侧翻实测（roll -0.5→翻）
            if robot_z is not None and robot_z > 0.9:
                v_lim = min(v_lim, 1.0)
            else:
                # 航点是 0.5m 半径检查点而非停车点（2026-08-07）：窗口 3m→2m、
                # 下限 0.2×vmax→1.2 m/s，避免短航段（4.6m）整段被拖慢。
                v_lim = min(v_lim, max(
                    float(os.environ.get("S10_AUTO_NEAR_MIN", "1.2")),
                    self.max_speed * d_wp
                    / float(os.environ.get("S10_AUTO_NEAR_RAMP", "2.0"))))
        # 台阶区限速（航点 z 兜底，已知地图，无感知滞后）：目标航段是陡升
        # 且机器人已越过前一航点（或接近该航点）→ 限速 step_vx。
        # 解决 §3.7 翻车机制：3.1 m/s 撞 0.125m riser → 前轮爬升翘头后仰翻。
        # 当前航段 = (next_idx-1 → next_idx)；step_zone 在该段终点是陡升时置位
        # 链 52：连续楼梯段（stair_zone，z 升 >0.25）用 stair_vx（可快），
        # 单级横脊仍用 step_vx（保守，防高速撞脊侧翻）。
        if (next_idx >= 2 and next_idx - 1 < len(self.step_zone)
                and self.step_zone[next_idx - 1] and d_wp < self.step_dist):
            # wp0→1 起步缓坡（z 升 0.475）不是台阶：next_idx>=2 才判台阶限速
            if (next_idx - 1 < len(self.stair_zone)
                    and self.stair_zone[next_idx - 1]):
                v_lim = min(v_lim, self.stair_vx)
            else:
                v_lim = min(v_lim, self.step_vx)
        # 速度限幅：避免转向后瞬间 0→4 m/s 的侧向冲击（侧翻风险）
        dv = self.max_accel * 0.05   # 每 10 步(0.05s)更新的速度增量
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
                      f"ff={yaw_ff:.2f} elev={elev_factor:.2f}", flush=True)
        return vx, vyaw

    def speed_limit_at(self, idx):
        if idx >= len(self.wp):
            return 0.0
        return float(self.speed_limit[idx])

    def update_mode(self, robot_xy, next_idx, yaw=None, local_map=None):
        """双模式判定：已知航段 z 大升（>0.25）**且感知确认离散台阶** →
        STAIR_SEQUENCE。2026-08-06 修复：仅按航点 z 升会把大坡度（wp0→1
        缓坡 +0.47m）误判为楼梯（STAIR σ=2.0 在坡上乱伸腿）；感知 step_flag
        只认离散台阶，陡坡/缓坡无 flag → 保持 CRUISE 轮子爬坡。
        滞回（2026-08-06 凌晨）：进入 STAIR 后**保持**直到离开楼梯区
        （y>40.5 或 next 推进）——狗在台阶底部来回时感知确认瞬时失败会
        反复切 CRUISE（权重突变 → MPC 行为突变 → 侧翻，batch v31 复现）。
        """
        if self.mode == "STAIR":
            if next_idx > 7 or robot_xy[1] > 40.5:
                self.mode = "CRUISE"
            return
        _dbg = os.environ.get("S10_MODE_DEBUG")
        _sz = None
        _d = None
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
            _use_global = d_wp < self.stair_mode_dist
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
        i0 = max(next_idx - 1, 0)
        i1 = min(i0 + n_wp, len(self.wp))
        if i1 - i0 < 3:
            return None
        win = self._stair_corridor_xy(self.wp[i0:i1][:, :2])
        seg = np.linalg.norm(np.diff(win, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total < 0.5:
            return None
        # 按 speed 滚动：从机器人投影起，前方总长按 spacing 采样（弧长均匀）
        s_start = 0.0
        s_targets = np.arange(s_start, total, spacing)
        pts = np.zeros((len(s_targets), 3))
        for k, s in enumerate(s_targets):
            i = int(np.searchsorted(cum, s, side="right") - 1)
            i = max(0, min(i, len(seg) - 1))
            t = (s - cum[i]) / max(seg[i], 1e-6)
            pts[k, :2] = win[i] + t * (win[i + 1] - win[i])
        # v129/v133: shared diagonal bump so MPC ref and navigation pursuit
        # use the identical path.
        pts[:, :2] = self._stair_diag_bump(pts[:, :2])
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
