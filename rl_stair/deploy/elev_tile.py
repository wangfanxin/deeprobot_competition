"""elev_tile.py: build the auto_nav.py `local_map` tile from LidarTerrainV2 (lidar
elevation map). Deployable: lidar rays only, no god-view.

local_map format consumed by AutoNavFollower.update_mode:
  {"heightmap": h (ny,nx surface/max-z), "valid": v (bool),
   "origin": (x0,y0), "resolution": res, "features": {"step_flag": sf}}

step_flag = |surface (hmax) gradient| > step_th, marking riser/ridge edges
(stairs AND ridges) so the elevation-map STAIR-mode entry/exit can detect them.
"""
import numpy as np


def build_local_tile(lterr, cx, cy, half=3.0, step_th=0.08):
    """Build a (2*half x 2*half) tile centered on (cx, cy) from LidarTerrainV2."""
    res = float(lterr.res)
    i0 = max(int(np.floor((cy - half) / res)), 0)
    i1 = min(int(np.floor((cy + half) / res)), lterr.h.shape[0] - 1)
    j0 = max(int(np.floor((cx - half) / res)), 0)
    j1 = min(int(np.floor((cx + half) / res)), lterr.h.shape[1] - 1)
    if i1 <= i0 or j1 <= j0:
        return None
    h = lterr.hmax[i0:i1 + 1, j0:j1 + 1].copy()
    v = lterr.valid[i0:i1 + 1, j0:j1 + 1].copy()
    # surface gradient -> step_flag (riser/ridge edges); -inf (no hit) -> 0
    hs = np.where(v, h, np.nan)
    gy, gx = np.gradient(hs)
    grad = np.nan_to_num(np.hypot(gx, gy), nan=0.0)
    sf = (grad > step_th).astype(np.float32)
    origin = np.array([j0 * res + float(lterr.ox),
                       i0 * res + float(lterr.oy)], dtype=np.float32)
    return {"heightmap": h, "valid": v, "origin": origin,
            "resolution": res, "nx": h.shape[1], "ny": h.shape[0],
            "features": {"step_flag": sf}}
