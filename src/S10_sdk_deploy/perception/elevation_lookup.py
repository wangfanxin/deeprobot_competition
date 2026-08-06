"""elevation_lookup: dial-mpc rollout 侧固定形状高程查图（纯 jnp，零 retrace）。

背景（doc/0806.md §2.1 / 难点）：
- rollout 查图必须固定形状 + 纯 jnp 向量化，否则触发 retrace；
- 越界/空洞按"未知不惩罚"（valid=False -> cost=0，不产生梯度惩罚）；
- 坡度/粗糙度/台阶网格已在感知侧（perception/local_map.py
  compute_terrain_features）按固定形状预计算，rollout 只做 gather。

接入示意（后续在 S10WheeledEnv._reward 中实现，本文件无需改动）：
    from perception.elevation_lookup import terrain_cost
    elev = info["elevation_map"]                  # 仿真节点 get_local_map() 注入
    feats = elev["features"]                      # valid/slope/roughness/step_flag (60,60)
    origin = elev["origin"]; res = elev["resolution"]
    # 轮落点 = 预测轮体 xpos 的 (x,y)（接触点≈轮心 xy，z 不参与索引）
    wheel_xy = d.xpos[[5, 9, 13, 17]][:, :2]     # fl/fr/hl/hr 轮体世界 xy
    cost, ok = terrain_cost(feats, origin, res, wheel_xy,
                            w_slope=..., w_rough=..., w_step=...)
    reward = reward - cost.sum()

注意：shape (60,60) 与 origin/res 构成固定契约；查询点数量固定
（Hsample+1 × 4 轮），首次 trace 后不再 retrace。
"""
from typing import Mapping, Tuple

import jax.numpy as jnp


def _cell_xy(xy: jnp.ndarray, origin: jnp.ndarray, res: float) -> jnp.ndarray:
    """世界 xy -> 栅格坐标（允许负值/越界，不裁剪）。xy: (...,2)。"""
    return (xy - origin) / res


def sample_grid(grid: jnp.ndarray, valid: jnp.ndarray,
                origin: jnp.ndarray, res: float, xy: jnp.ndarray,
                fill: float = 0.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """在固定形状栅格上按世界 xy 双线性最近邻采样。

    Args:
        grid:  (H,W) float32（height/slope/roughness/...）
        valid: (H,W) bool
        origin: (2,) 瓦片左下角世界坐标
        res:   栅格边长 (m)
        xy:    (...,2) 世界坐标查询点
        fill:  未知格回填值（默认 0，即"未知不惩罚"）
    Returns:
        (values, ok)：values 与 xy 同 shape（去尾维），ok 为有效查询掩码。
    """
    f = _cell_xy(xy, origin, res)
    i = jnp.floor(f[..., 1]).astype(jnp.int32)   # 行 = y
    j = jnp.floor(f[..., 0]).astype(jnp.int32)   # 列 = x
    h, w = grid.shape[0], grid.shape[1]
    inb = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    ii = jnp.clip(i, 0, h - 1)
    jj = jnp.clip(j, 0, w - 1)
    val = grid[ii, jj]
    ok = valid[ii, jj] & inb
    safe = jnp.where(jnp.isfinite(val), val, fill)
    return jnp.where(ok, safe, fill), ok


def terrain_cost(features: Mapping[str, jnp.ndarray],
                 origin: jnp.ndarray, res: float, xy: jnp.ndarray,
                 w_slope: float = 1.0, w_rough: float = 1.0,
                 w_step: float = 3.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """地形代价：坡度/粗糙度/台阶在轮落点处 gather 后加权求和。

    features: dict(valid, slope, roughness, step_flag)，均为 (60,60) float32。
    xy: (...,2) 世界系轮心 xy。
    Returns:
        (cost, ok)：cost 与 xy 同 shape（去尾维）；ok 为全部三项均有效掩码。
    """
    valid = jnp.asarray(features["valid"], dtype=jnp.bool_)
    slope = jnp.asarray(features["slope"], dtype=jnp.float32)
    rough = jnp.asarray(features["roughness"], dtype=jnp.float32)
    step = jnp.asarray(features["step_flag"], dtype=jnp.float32)

    s, ok_s = sample_grid(slope, valid, origin, res, xy, fill=0.0)
    r, ok_r = sample_grid(rough, valid, origin, res, xy, fill=0.0)
    t, ok_t = sample_grid(step, valid, origin, res, xy, fill=0.0)
    cost = w_slope * s + w_rough * r + w_step * t
    ok = ok_s & ok_r & ok_t
    return cost, ok
