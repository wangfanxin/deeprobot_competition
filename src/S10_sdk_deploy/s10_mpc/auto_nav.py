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
                 yaw_gain=2.5, lookahead=2.5, lat_accel_max=6.0,
                 climb_max_speed=1.5, grade_scale=5.0, speed_window=3,
                 lat_gain=1.5, max_accel=5.0, yaw_damp=0.6, cte_gain=1.2):
        """
        waypoints: (N,3) 全局航点 [x, y, z]
        """
        self.wp = np.asarray(waypoints, dtype=np.float64)
        self.max_speed = float(max_speed)
        self.vyaw_max = float(vyaw_max)
        self.yaw_gain = float(yaw_gain)
        self.lookahead = float(lookahead)
        self.lat_accel_max = float(lat_accel_max)
        self.climb_max_speed = float(climb_max_speed)
        self.grade_scale = float(grade_scale)
        self.speed_window = int(speed_window)
        self.lat_gain = float(lat_gain)
        self.max_accel = float(max_accel)
        self.yaw_damp = float(yaw_damp)   # 实测 yaw rate 阻尼，防航向过冲
        self.cte_gain = float(cte_gain)   # 横向偏差修正增益（与航向一致时生效）
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
        # vyaw 变化率限制（rad/s 每 0.05s 更新，S10_AUTO_VYAW_SLEW 默认 0.8）：
        # 反馈 ±1.28 瞬跳 + 轮 FF 放大 → yaw 打转（全航点 #7 wp2 处 ±3rad 振荡），
        # 限制指令变化率可消振荡（2026-08-05）。
        self.vyaw_slew = float(os.environ.get("S10_AUTO_VYAW_SLEW", "0.8"))
        self._last_vx = 0.0
        self._last_vyaw_out = 0.0
        self._ref_dbg_cnt = 0
        self._precompute()

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
        """沿路径取距起点 dist 处的点（线性插值）。"""
        if dist <= 0:
            return self.wp[0].copy()
        if dist >= self.cum_len[-1]:
            return self.wp[-1].copy()
        k = int(np.searchsorted(self.cum_len, dist, side="right") - 1)
        k = max(0, min(k, len(self.seg_len) - 1))
        t = (dist - self.cum_len[k]) / max(self.seg_len[k], 1e-6)
        return self.wp[k] + t * (self.wp[k + 1] - self.wp[k])

    def compute_cmd(self, robot_xy, yaw, next_idx, robot_z=None, yaw_rate=0.0):
        """返回 (vx, vyaw)。robot_xy: (2,) 全局位置；next_idx: 下一个未到达航点。"""
        if next_idx >= len(self.wp):
            return 0.0, 0.0
        wp_next = self.wp[next_idx]
        d_wp = float(np.linalg.norm(robot_xy - wp_next[:2]))

        # 纯 pursuit：机器人在当前航段上的投影进度（弧长基准）
        seg_a = self.wp[max(next_idx - 1, 0)]
        seg_b = self.wp[next_idx]
        d = seg_b[:2] - seg_a[:2]
        L2 = max(float(np.dot(d, d)), 1e-6)
        t = float(np.clip(np.dot(robot_xy - seg_a[:2], d) / L2, 0.0, 1.0))
        s_proj = self.cum_len[max(next_idx - 1, 0)] + t * np.sqrt(L2)

        # 目标点：视距内或已越过航点平面 → 瞄准航点本身（保证 0.2m 判点）；
        # 否则 → 路径前视点（平滑跟线，不切弦离轨）
        passed = s_proj > self.cum_len[next_idx] - 0.05
        if d_wp < self.lookahead or passed:
            target = wp_next
        else:
            s_target = min(s_proj + self.lookahead, self.cum_len[-1])
            target = self._path_point_at(s_target)
        err = np.arctan2(target[1] - robot_xy[1],
                         target[0] - robot_xy[0]) - yaw
        err = float(np.arctan2(np.sin(err), np.cos(err)))
        self._last_err = err
        self._last_dwp = d_wp
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
        # 横向偏差修正（防漂移）：相对当前航段计算 cte，修正方向与 pursuit
        # 航向一致时才叠加（点积>0），避免与航点航向打架形成极限环。
        seg_a = self.wp[max(next_idx - 1, 0)]
        seg_b = self.wp[next_idx]
        d = seg_b[:2] - seg_a[:2]
        L = float(np.linalg.norm(d))
        cte = 0.0
        if L > 1e-6:
            n = d / L
            rel = robot_xy - seg_a[:2]
            cte = float(n[0] * rel[1] - n[1] * rel[0])   # 左为正
            self._last_cte = cte
            # 修正方向：左偏→右转（vyaw 负）
            cte_corr = -self.cte_gain * float(np.clip(cte / 2.0, -1.0, 1.0))
            # 与 pursuit 航向的一致性：航向朝目标=err 符号；修正不反向则叠加
            if abs(cte) < 6.0 and cte_corr * err >= -0.5:
                vyaw = float(self.yaw_gain * err - self.yaw_damp * yaw_rate
                             + cte_corr)
            else:
                vyaw = float(self.yaw_gain * err - self.yaw_damp * yaw_rate)
        else:
            vyaw = float(self.yaw_gain * err - self.yaw_damp * yaw_rate)
        vyaw = float(np.clip(vyaw, -vyaw_max_eff, vyaw_max_eff))
        # 变化率限制（防反馈振荡；每次调用 = 0.05s）
        vyaw = float(np.clip(
            vyaw, self._last_vyaw_out - self.vyaw_slew,
            self._last_vyaw_out + self.vyaw_slew))
        self._last_vyaw_out = vyaw

        # 限速：前方 speed_window 个航点的最小限速
        v_lim = self.max_speed
        for j in range(self.speed_window):
            if next_idx + j < len(self.wp):
                v_lim = min(v_lim, self.speed_limit[next_idx + j])
        # 转向速度分级：|err|>0.3 时——
        #   近点（d_wp<3m，如起步/航点大转角）：0.4 m/s 原地转向，避免冲过航点；
        #   远点（如爬坡段）：1.5 m/s 慢速转弯，保持推力爬台阶。
        if abs(err) > 0.30:
            turn_vx = float(os.environ.get("S10_AUTO_TURN_VX", "1.5"))
            v_lim = min(v_lim, (0.4 if d_wp < 3.0 else turn_vx) * elev_factor)
        else:
            v_lim = min(v_lim, self.max_speed * elev_factor)
        # 接近当前航点时减速，保证进入 0.2m 到达半径判定。
        # 高架（z>0.9）时额外减半——坡顶最后一级台阶高速接近会前翻（实测）。
        if d_wp < 3.0:
            # 高架（z>0.9）接近速度压到 1.0 m/s——坡顶侧翻实测（roll -0.5→翻）
            if robot_z is not None and robot_z > 0.9:
                v_lim = min(v_lim, 1.0)
            else:
                v_lim = min(v_lim, self.max_speed * max(0.2, d_wp / 2.5))
        # 台阶区限速（航点 z 兜底，已知地图，无感知滞后）：目标航段是陡升
        # 且机器人已越过前一航点（或接近该航点）→ 限速 step_vx。
        # 解决 §3.7 翻车机制：3.1 m/s 撞 0.125m riser → 前轮爬升翘头后仰翻。
        # 当前航段 = (next_idx-1 → next_idx)；step_zone 在该段终点是陡升时置位
        # 链 52：连续楼梯段（stair_zone，z 升 >0.25）用 stair_vx（可快），
        # 单级横脊仍用 step_vx（保守，防高速撞脊侧翻）。
        if (next_idx >= 1 and next_idx - 1 < len(self.step_zone)
                and self.step_zone[next_idx - 1] and d_wp < self.step_dist):
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
        return vx, vyaw

    def speed_limit_at(self, idx):
        if idx >= len(self.wp):
            return 0.0
        return float(self.speed_limit[idx])

    def update_mode(self, robot_xy, next_idx):
        """按当前航段 stair_zone 判定 CRUISE/STAIR_SEQUENCE（已知地图，无感知滞后）。"""
        _dbg = os.environ.get("S10_MODE_DEBUG")
        _sz = None
        _d = None
        if (next_idx >= 1 and next_idx - 1 < len(self.stair_zone)
                and self.stair_zone[next_idx - 1]):
            _sz = bool(self.stair_zone[next_idx - 1])
            d_wp = float(np.linalg.norm(
                robot_xy - self.wp[next_idx, :2]))
            _d = d_wp
            if d_wp < self.stair_mode_dist:
                self.mode = "STAIR"
                if _dbg:
                    print(f"[MODE] STAIR next={next_idx} sz={_sz} "
                          f"d={d_wp:.1f}", flush=True)
                return
        if _dbg and int(robot_xy[1] * 2) % 10 == 0:
            print(f"[MODE] CRUISE next={next_idx} sz={_sz} d={_d} "
                  f"y={robot_xy[1]:.1f}", flush=True)
        self.mode = "CRUISE"

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
        win = self.wp[i0:i1][:, :2]
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
        # z：高程图地形高 + 站姿高（平滑 + 无效继承）
        z_ref = np.full(len(pts), 0.205)
        last_ok = None
        if local_map is not None:
            hm = local_map.get("heightmap")
            valid = local_map.get("valid")
            if hm is not None:
                ox = float(local_map["origin"][0])
                oy = float(local_map["origin"][1])
                res = float(local_map["resolution"])
                for k in range(len(pts)):
                    j = int(np.floor((pts[k, 0] - ox) / res))
                    i = int(np.floor((pts[k, 1] - oy) / res))
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
