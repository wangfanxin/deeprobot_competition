"""points_to_heightmap: 纯 numpy 局部高程图生成（真机可移植）。

硬约束（设计契约见 doc/0806.md §2.1；旧 .kiro 设计文档已移除）：
  - 本模块禁止 import mujoco / ros / taichi / jax，仅依赖 numpy；
  - 真机移植时直接复用，不修改。

契约（冻结）：
  - 输入点云：base 系 xy（前向 x、左向 y）+ **重力方向**相对机身高度 z（单位 m）。
    z = 世界系点高度 - 机身高度，机身 roll/pitch 倾斜时平地的 z 仍一致，不会误判为斜面。
  - 输出：局部高程图，前向 1.6 m × 侧向 1.0 m，0.1 m 栅格 → 形状 (10, 16)
  - 每格值 = 该格内点云 min-z（重力方向离地高度语义）；空洞 = fill_value
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ElevationMapConfig:
    x_min: float = 0.0          # 前向 ROI 起点（相对机身）
    x_max: float = 1.6          # 前向 ROI 终点
    y_min: float = -0.5         # 侧向左边界
    y_max: float = 0.5          # 侧向右边界
    resolution: float = 0.1     # 栅格边长 (m)
    max_hang_height: float = 0.5    # 相对机身 z 高于此值视为悬空点，滤除
    fill_value: float = 10.0    # 空洞填远值（显著高于正常地形）
    min_points_per_cell: int = 1    # 每格至少点数才有效

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))


def points_to_heightmap(
    points_base: np.ndarray,
    cfg: ElevationMapConfig = ElevationMapConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """生成局部高程图。

    Args:
        points_base: (N,3) 点云：xy 为 base 系（x 前、y 左），z 为重力方向相对机身高度。
    Returns:
        heightmap: (H, W) float32，行=y（左侧为第 0 行），列=x（近端为第 0 列），
                   值为重力方向离地高度（相对机身原点）；空洞填 cfg.fill_value。
        valid:     (H, W) bool，该格是否有有效测量。
    """
    if points_base is None or len(points_base) == 0:
        return _empty_map(cfg)

    pts = np.asarray(points_base, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return _empty_map(cfg)
    pts = pts[:, :3]

    # 1) ROI 裁剪
    mask = (
        (pts[:, 0] >= cfg.x_min) & (pts[:, 0] <= cfg.x_max)
        & (pts[:, 1] >= cfg.y_min) & (pts[:, 1] <= cfg.y_max)
    )
    pts = pts[mask]
    if len(pts) == 0:
        return _empty_map(cfg)

    # 2) 滤除悬空点（机器人腿、障碍上部、waypoint 球等）
    pts = pts[pts[:, 2] <= cfg.max_hang_height]
    if len(pts) == 0:
        return _empty_map(cfg)

    # 3) 栅格化：每格取 min-z 作为落地面
    nx, ny = cfg.nx, cfg.ny
    ci = np.floor((pts[:, 1] - cfg.y_min) / cfg.resolution).astype(np.int64)  # 行 (y)
    cj = np.floor((pts[:, 0] - cfg.x_min) / cfg.resolution).astype(np.int64)  # 列 (x)
    np.clip(ci, 0, ny - 1, out=ci)
    np.clip(cj, 0, nx - 1, out=cj)

    cell_min = np.full((ny, nx), np.inf, dtype=np.float64)
    np.minimum.at(cell_min, (ci, cj), pts[:, 2])

    heightmap = np.full((ny, nx), cfg.fill_value, dtype=np.float32)
    valid = np.isfinite(cell_min)
    heightmap[valid] = cell_min[valid].astype(np.float32)

    return heightmap, valid


def _empty_map(cfg: ElevationMapConfig) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.full((cfg.ny, cfg.nx), cfg.fill_value, dtype=np.float32),
        np.zeros((cfg.ny, cfg.nx), dtype=bool),
    )
