"""local_map: 纯 numpy 世界对齐高程瓦片（感知-voxel 链路）。

数据流（"高程图-感知-voxel"）：
  [LiDAR 点云(传感器系)] --位姿--> [世界系点云]
      --3D 体素栅格(降采样/去噪/多层结构分离)--> [2.5D 高程瓦片 min-z 落地面]
      --世界对齐累积(重锚定+重叠带继承+最近观测优先)--> [固定形状 (60,60) + 瓦片原点]
      --> 注入 DIAL-MPC rollout（查图见 perception/elevation_lookup.py）

冻结契约（doc/0806.md §2.1）：
- 锚定瓦片 8×8 m @0.1m（含 1m 重叠带），有效输出区 6×6 m
  -> height/valid 固定形状 (60,60) float32 / bool；
- 每帧注入 3600 float（~14KB），JAX 按值传递、形状固定、零 retrace；
- 重锚定：机器人距瓦片中心 >2m -> 重锚定到新中心；重叠带数据继承；
- 融合：同格多观测 = min-z 去噪；跨帧 = 最近观测优先（新观测覆盖旧观测）；
- 空洞/未知 = valid=False（下游按"未知不惩罚"）；
- 纯 numpy，禁止 import mujoco/ros/jax/taichi；真机可移植（上游只换点云来源）。
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

import numpy as np


def _snap_floor(value: float, res: float) -> float:
    """把坐标向下对齐到 res 网格（保证跨瓦片单元格边界一致，重叠可逐格继承）。"""
    return float(np.floor(float(value) / res) * res)


@dataclass
class LocalMapConfig:
    resolution: float = 0.1          # 栅格边长 (m)
    tile_size: float = 8.0           # 锚定瓦片边长 (m)，含 1m 重叠带
    effective_size: float = 6.0      # 有效输出区边长 (m) -> (60,60)
    voxel_size: float = 0.05         # 3D 体素边长 (m)，点云降采样/去噪
    reanchor_dist: float = 2.0       # 机器人距瓦片中心超过该值 -> 重锚定 (m)
                                     # （链 9 实测 1.5 反而不稳：重锚定更频繁、
                                     #  新区域逐帧补洞，riser 前地图可靠性下降；
                                     #  保持 2.0。后轮落点由 6×6 有效区天然覆盖，
                                     #  ±3m 内始终包含 0.456m 轴距的后轮）
    max_hang: float = 1.5            # 相对 base z 的上界：过滤悬空/顶部结构 (m)
    max_drop: float = 1.5            # 相对 base z 的下界：过滤远低于机身的第二层地面 (m)
    min_points_per_cell: int = 1     # 每格至少体素数才有效
    max_age: float = 0.0             # 跨帧老化：>0 时超过该秒数未刷新置为空洞；0=关闭（初赛默认）
    fill_value: float = 10.0         # 空洞填充值（下游只认 valid）
    step_threshold: float = 0.08     # 台阶判定阈值 (m)，见 0806.md §2.1 公式
    inpaint_iter: int = 3            # 近场空洞填补传播格数（0=关闭；3 格=0.3m）

    def __post_init__(self):
        self.n = int(round(self.tile_size / self.resolution))
        self.n_out = int(round(self.effective_size / self.resolution))
        self.band = (self.n - self.n_out) // 2
        assert self.n_out + 2 * self.band == self.n, \
            f"tile_size/effective_size 与 resolution 不匹配: n={self.n}, n_out={self.n_out}"


@dataclass
class LocalMapTile:
    heightmap: np.ndarray                 # (60,60) float32
    valid: np.ndarray                     # (60,60) bool
    origin: Tuple[float, float]           # 输出瓦片左下角世界坐标 (x0, y0)
    resolution: float
    nx: int
    ny: int
    stamp: float
    frame: str = "map"


class LocalMap:
    """世界对齐高程瓦片：体素化点云 -> min-z 落地面 -> 跨帧累积。

    线程约定：update() 只在采集线程调用；下游通过 get_tile() 快照读取
    （快照由调用方加锁，本类不持锁）。
    """

    def __init__(self, cfg: LocalMapConfig = LocalMapConfig(),
                 robot_xy: Optional[np.ndarray] = None, stamp: float = 0.0):
        self.cfg = cfg
        n = cfg.n
        self.height = np.full((n, n), cfg.fill_value, dtype=np.float32)
        self.valid = np.zeros((n, n), dtype=np.bool_)
        self.last_seen = np.full((n, n), -np.inf, dtype=np.float32)
        if robot_xy is not None and len(robot_xy) >= 2:
            self.x0 = _snap_floor(robot_xy[0] - cfg.tile_size / 2.0, cfg.resolution)
            self.y0 = _snap_floor(robot_xy[1] - cfg.tile_size / 2.0, cfg.resolution)
        else:
            self.x0 = 0.0
            self.y0 = 0.0
        self.stamp = float(stamp)
        self.last_voxel_count = 0
        self.last_points_in_band = 0

    # ---------- 对外接口 ----------
    def reset(self, robot_xy: np.ndarray, stamp: float = 0.0) -> None:
        n = self.cfg.n
        self.height[...] = self.cfg.fill_value
        self.valid[...] = False
        self.last_seen[...] = -np.inf
        self.x0 = _snap_floor(robot_xy[0] - self.cfg.tile_size / 2.0, self.cfg.resolution)
        self.y0 = _snap_floor(robot_xy[1] - self.cfg.tile_size / 2.0, self.cfg.resolution)
        self.stamp = float(stamp)

    def update(self, points_world: Optional[np.ndarray],
               robot_xy: np.ndarray, robot_z: float, stamp: float):
        """世界系点云 -> 体素栅格 -> 融合进世界对齐瓦片。

        Args:
            points_world: (N,3) 世界系命中点（含 z，单位 m）；None/空则仅老化。
            robot_xy: (2,) base 世界 xy（用于锚定与高度带参考）。
            robot_z:  base 世界 z。
            stamp:    仿真时间（秒），用于最近观测优先与老化。
        Returns:
            tile dict（同 get_tile()），失败时返回当前快照。
        """
        self.stamp = float(stamp)
        if self._need_reanchor(robot_xy):
            self._reanchor(robot_xy)
        pts = self._filter_band(points_world, robot_z)
        if pts is not None and len(pts) > 0:
            vox = self._voxelize(pts)
            self.last_voxel_count = int(len(vox))
            self._fuse(vox, stamp)
        else:
            self.last_voxel_count = 0
        if self.cfg.inpaint_iter > 0:
            self._inpaint(self.cfg.inpaint_iter)
        if self.cfg.max_age > 0.0:
            self._age(stamp)
        return self.get_tile()

    def _inpaint(self, max_iter: int):
        """近场空洞填补（min 邻域传播，最多 max_iter 格 = 0.3m）。

        修复 LiDAR riser 立面阴影：轮下/台阶立面地图恒空洞，导致 rollout
        的 r_ground（轮-地形贴合）与 ref-z（路径高程决策）失效 → wp7 台阶区
        卡死（2026-08-06 链 41 诊断 valid_sample=0）。
        用**有效邻域最小值**填充：台阶立面填成下台面高，台阶顶的观测保持
        原值，高度差在顶缘重新形成 step_flag（>0.08m）；下坡/落差则保守地
        填成高侧（不把空洞当平地）。
        只传播短距离：远处从未观测区保持 valid=False（"未知不惩罚"语义保留）。
        """
        h = self.height
        v = self.valid
        n = self.cfg.n
        if v.all():
            return
        for _ in range(max_iter):
            inv = ~v
            if not inv.any():
                return
            hp = np.pad(h, 1, mode="constant", constant_values=np.nan)
            vp = np.pad(v, 1, mode="constant", constant_values=False)
            acc = np.full((n, n), np.inf, dtype=np.float64)
            has = np.zeros((n, n), dtype=bool)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ok = vp[1 + di:1 + di + n, 1 + dj:1 + dj + n]
                    val = np.where(
                        ok, hp[1 + di:1 + di + n, 1 + dj:1 + dj + n], np.inf)
                    acc = np.minimum(acc, val)
                    has |= ok
            fillable = inv & has & np.isfinite(acc)
            if not fillable.any():
                return
            h[...] = np.where(fillable, acc, h)
            v[...] = v | fillable
    def get_tile(self, features: bool = True) -> dict:
        """固定形状输出：heightmap/valid (60,60) + origin + res + 派生特征网格。

        features=True 时额外给出 slope/roughness/step_flag（rollout cost 直接 gather）。
        """
        cfg = self.cfg
        b = cfg.band
        h = self.height[b:cfg.n - b, b:cfg.n - b].copy()
        v = self.valid[b:cfg.n - b, b:cfg.n - b].copy()
        origin = (self.x0 + b * cfg.resolution, self.y0 + b * cfg.resolution)
        tile = LocalMapTile(h, v, origin, cfg.resolution, cfg.n_out, cfg.n_out,
                            self.stamp)
        out = {
            "heightmap": h,
            "valid": v,
            "origin": np.array(origin, dtype=np.float32),
            "resolution": cfg.resolution,
            "nx": cfg.n_out,
            "ny": cfg.n_out,
            "stamp": self.stamp,
            "frame": "map",
        }
        if features:
            out["features"] = compute_terrain_features(tile, cfg.step_threshold)
        return out

    def stats(self) -> dict:
        return {
            "origin": (self.x0, self.y0),
            "stamp": self.stamp,
            "valid": int(self.valid.sum()),
            "voxels": int(self.last_voxel_count),
            "points_in_band": int(self.last_points_in_band),
        }

    # ---------- 内部：锚定 ----------
    def _need_reanchor(self, robot_xy: np.ndarray) -> bool:
        cfg = self.cfg
        cx = self.x0 + cfg.tile_size / 2.0
        cy = self.y0 + cfg.tile_size / 2.0
        dx = float(robot_xy[0]) - cx
        dy = float(robot_xy[1]) - cy
        return (dx * dx + dy * dy) > (cfg.reanchor_dist * cfg.reanchor_dist)

    def _reanchor(self, robot_xy: np.ndarray) -> None:
        """重锚定到新中心（机器人位置向下对齐到网格），重叠区逐格继承。"""
        cfg = self.cfg
        n = cfg.n
        new_x0 = _snap_floor(robot_xy[0] - cfg.tile_size / 2.0, cfg.resolution)
        new_y0 = _snap_floor(robot_xy[1] - cfg.tile_size / 2.0, cfg.resolution)
        dcol = int(round((new_x0 - self.x0) / cfg.resolution))   # 新列 k <-> 旧列 k+dcol
        drow = int(round((new_y0 - self.y0) / cfg.resolution))

        new_h = np.full((n, n), cfg.fill_value, dtype=np.float32)
        new_v = np.zeros((n, n), dtype=np.bool_)
        new_ls = np.full((n, n), -np.inf, dtype=np.float32)

        dst_c0 = max(0, -dcol)
        dst_c1 = min(n, n - dcol)
        dst_r0 = max(0, -drow)
        dst_r1 = min(n, n - drow)
        if dst_c0 < dst_c1 and dst_r0 < dst_r1:
            src_c0 = dst_c0 + dcol
            src_c1 = dst_c1 + dcol
            src_r0 = dst_r0 + drow
            src_r1 = dst_r1 + drow
            new_h[dst_r0:dst_r1, dst_c0:dst_c1] = self.height[src_r0:src_r1, src_c0:src_c1]
            new_v[dst_r0:dst_r1, dst_c0:dst_c1] = self.valid[src_r0:src_r1, src_c0:src_c1]
            new_ls[dst_r0:dst_r1, dst_c0:dst_c1] = self.last_seen[src_r0:src_r1, src_c0:src_c1]

        self.height, self.valid, self.last_seen = new_h, new_v, new_ls
        self.x0, self.y0 = new_x0, new_y0

    # ---------- 内部：点云 -> 体素 -> 瓦片 ----------
    def _filter_band(self, points_world: Optional[np.ndarray],
                     robot_z: float) -> Optional[np.ndarray]:
        cfg = self.cfg
        if points_world is None:
            return None
        pts = np.asarray(points_world, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            return None
        pts = pts[:, :3]
        mask = np.isfinite(pts).all(axis=1)
        zmin = float(robot_z) - cfg.max_drop
        zmax = float(robot_z) + cfg.max_hang
        mask &= (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        margin = cfg.voxel_size * 2.0
        mask &= (pts[:, 0] >= self.x0 - margin) \
            & (pts[:, 0] <= self.x0 + cfg.tile_size + margin) \
            & (pts[:, 1] >= self.y0 - margin) \
            & (pts[:, 1] <= self.y0 + cfg.tile_size + margin)
        pts = pts[mask]
        self.last_points_in_band = int(len(pts))
        return pts

    def _voxelize(self, pts: np.ndarray) -> np.ndarray:
        """3D 体素栅格：以世界原点对齐的 voxel_size 立方体，每体素取 min-z
        （去噪），体素重心 (x,y) 作为代表点。"""
        vs = self.cfg.voxel_size
        keys = np.floor(pts / vs).astype(np.int64)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True,
                                   return_counts=True)
        n_vox = counts.shape[0]
        if n_vox == 0:
            return np.empty((0, 3), dtype=np.float64)
        zmin = np.full(n_vox, np.inf, dtype=np.float64)
        np.minimum.at(zmin, inv, pts[:, 2])
        xsum = np.zeros(n_vox, dtype=np.float64)
        ysum = np.zeros(n_vox, dtype=np.float64)
        np.add.at(xsum, inv, pts[:, 0])
        np.add.at(ysum, inv, pts[:, 1])
        vox = np.column_stack([
            xsum / counts,
            ysum / counts,
            zmin,
        ])
        return vox

    def _fuse(self, vox: np.ndarray, stamp: float) -> None:
        cfg = self.cfg
        n = cfg.n
        ci = np.floor((vox[:, 1] - self.y0) / cfg.resolution).astype(np.int64)
        cj = np.floor((vox[:, 0] - self.x0) / cfg.resolution).astype(np.int64)
        inb = (ci >= 0) & (ci < n) & (cj >= 0) & (cj < n)
        ci, cj, z = ci[inb], cj[inb], vox[inb, 2]
        if len(ci) == 0:
            return
        # 同帧去噪：每格取 min-z（体素已降采样，此步合并跨体素同格点）
        cell_min = np.full((n, n), np.inf, dtype=np.float64)
        np.minimum.at(cell_min, (ci, cj), z)
        new_valid = np.isfinite(cell_min)
        # 跨帧：最近观测优先（stamp 单调递增，新观测直接覆盖旧值）
        self.height[new_valid] = cell_min[new_valid].astype(np.float32)
        self.valid |= new_valid
        self.last_seen[new_valid] = float(stamp)

    def _age(self, stamp: float) -> None:
        cfg = self.cfg
        stale = self.valid & ((stamp - self.last_seen) > cfg.max_age)
        self.valid[stale] = False
        self.height[stale] = cfg.fill_value


def compute_terrain_features(tile: LocalMapTile,
                             step_threshold: Optional[float] = None) -> dict:
    """由高程瓦片派生 rollout 用特征网格（0806.md §2.1 公式）。

    - slope:     坡度 = |z(i,j+Δ) - z(i,j-Δ)| / (2Δ·res)，Δ=1 格；取 x/y 两方向较大值
    - roughness: 3×3 邻域 max - min
    - step:      4 邻域最大高差 (m)
    - step_flag: step > 0.08m 为 1，其余 0（强惩罚开关）

    空洞/邻域不足的格子保持 NaN；下游查图时按"未知不惩罚"处理。
    """
    res = tile.resolution
    thr = tile.step_threshold if step_threshold is None else step_threshold
    h = tile.heightmap.astype(np.float64)
    v = tile.valid
    ny, nx = h.shape
    hv = np.where(v, h, np.nan)

    slope = np.full((ny, nx), np.nan, dtype=np.float32)
    rough = np.full((ny, nx), np.nan, dtype=np.float32)
    step = np.full((ny, nx), np.nan, dtype=np.float32)
    step_flag = np.zeros((ny, nx), dtype=np.float32)

    if nx >= 3 and ny >= 3:
        # 中心差分坡度（Δ=1 格），中心位于 [1:-1, 1:-1]
        sx = np.abs(hv[:, 2:] - hv[:, :-2]) / (2.0 * res)
        sy = np.abs(hv[2:, :] - hv[:-2, :]) / (2.0 * res)
        slope[1:-1, 1:-1] = np.maximum(sx[1:-1, :], sy[:, 1:-1])

        # 3×3 邻域粗糙度（全 NaN 窗口保持 NaN，不产生 RuntimeWarning）
        stack = np.stack(
            [hv[i0:i0 + ny - 2, j0:j0 + nx - 2]
             for i0 in range(3) for j0 in range(3)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mx = np.nanmax(stack, axis=0)
            mn = np.nanmin(stack, axis=0)
        rough[1:-1, 1:-1] = mx - mn

        # 4 邻域最大高差（台阶）
        d = np.zeros((ny, nx), dtype=np.float64)
        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            i0 = max(0, di); i1 = ny + min(0, di)
            j0 = max(0, dj); j1 = nx + min(0, dj)
            i0b = max(0, -di); i1b = ny + min(0, -di)
            j0b = max(0, -dj); j1b = nx + min(0, -dj)
            d[i0:i1, j0:j1] = np.maximum(
                d[i0:i1, j0:j1],
                np.abs(hv[i0b:i1b, j0b:j1b] - hv[i0:i1, j0:j1]))
        step[...] = d
        step_flag[...] = (d > thr).astype(np.float32)

    return {
        "slope": slope,
        "roughness": rough,
        "step": step,
        "step_flag": step_flag,
    }
