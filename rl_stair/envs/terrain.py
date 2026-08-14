import os, re
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# ---------------------------------------------------------------------------
# Terrain representation (robot drives along +y, obstacles start at y0)
# ---------------------------------------------------------------------------

@dataclass
class Terrain:
    boxes: List[dict] = field(default_factory=list)   # geom dicts
    riser_y: List[float] = field(default_factory=list)      # obstacle front edges (y)
    riser_top_z: List[float] = field(default_factory=list)  # obstacle top heights
    start_y: float = -3.0      # spawn band start (y)
    spawn_lo: float = 0.5      # spawn distance before first obstacle
    spawn_hi: float = 2.0
    goal_y: float = 10.0       # target y (behind last obstacle)
    course_end: float = 14.0   # terrain extends to here
    ground_friction: float = 1.0

def _box(half_x, half_y, half_z, x, y, z_top, friction, rgba="0.6 0.65 0.6 1"):
    return {"type": "box", "size": [half_x, half_y, half_z],
            "pos": [x, y, z_top - half_z], "friction": friction, "rgba": rgba}

def flat(course=18.0, ground_friction=1.0):
    return Terrain(ground_friction=ground_friction, goal_y=course,
                   course_end=course + 2.0)

def single_step(height, y0=1.5, width_x=3.0, friction=0.8, course=10.0):
    t = Terrain(ground_friction=1.0, goal_y=y0 + 2.0, course_end=y0 + 4.0)
    t.boxes.append(_box(width_x/2, 0.5, height/2, 0.0, y0 + 0.5, height, friction))
    t.riser_y = [y0]
    t.riser_top_z = [height]
    return t

def stairs(risers, tread=0.4, y0=1.5, width_x=3.0, friction=0.8):
    """risers: list of riser heights (competition: [0.061,0.125,0.125,0.125,0.125,0.125])."""
    t = Terrain(ground_friction=1.0)
    z, y = 0.0, y0
    for dh in risers:
        z += dh
        t.boxes.append(_box(width_x/2, tread/2, z/2, 0.0, y + tread/2, z, friction))
        t.riser_y.append(y)
        t.riser_top_z.append(z)
        y += tread
    top_len = 4.0
    t.boxes.append(_box(width_x/2, top_len/2, z/2, 0.0, y + top_len/2, z, friction))
    t.goal_y = y + 0.5
    t.course_end = y + top_len
    return t

def ridge(height, width_y=0.2, y0=1.5, width_x=3.0, friction=0.8, course=8.0):
    t = Terrain(ground_friction=1.0, goal_y=y0 + 2.0, course_end=y0 + 4.0)
    t.boxes.append(_box(width_x/2, width_y/2, height/2, 0.0, y0 + width_y/2, height, friction))
    t.riser_y = [y0]
    t.riser_top_z = [height]
    return t

def mixed(items, width_x=3.0, friction=0.8):
    """items: list of (kind, kwargs) placed sequentially along y, e.g.
       [("ridge", dict(height=0.12, y0=1.5)), ("stairs", dict(risers=[...], y0=4.5))]"""
    t = Terrain(ground_friction=1.0)
    y_cursor = 0.5
    all_riser_y, all_top = [], []
    for kind, kw in items:
        if "y0" not in kw:
            kw["y0"] = y_cursor
        if kind == "ridge":
            st = ridge(kw["height"], kw.get("width_y", 0.2), kw["y0"], width_x, friction)
        elif kind == "step":
            st = single_step(kw["height"], kw["y0"], width_x, friction)
        elif kind == "stairs":
            st = stairs(kw["risers"], kw.get("tread", 0.4), kw["y0"], width_x, friction)
        else:
            raise ValueError(kind)
        t.boxes += st.boxes
        all_riser_y += st.riser_y
        all_top += st.riser_top_z
        y_cursor = max(st.course_end, st.goal_y) + 1.0
    t.riser_y, t.riser_top_z = all_riser_y, all_top
    t.goal_y = y_cursor
    t.course_end = y_cursor + 2.0
    return t

# ---------------------------------------------------------------------------
# Training model builder: S10.xml -> MJX-compatible XML + terrain
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
ROBOT_XML = os.path.join(_REPO, "src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10.xml")
MESH_DIR = os.path.abspath(os.path.join(_REPO, "src/S10_sdk_deploy/S10_description/s10_mjcf/meshes"))
ASSET_DIR = os.path.join(_HERE, "assets")
WHEEL_COL_XML = os.path.join(ASSET_DIR, "wheel_col.xml")

def _cylinder_mesh(r, h, n=16):
    verts = []
    for k in range(n):
        a = 2*np.pi*k/n
        verts += [r*np.cos(a), r*np.sin(a), h]
    for k in range(n):
        a = 2*np.pi*k/n
        verts += [r*np.cos(a), r*np.sin(a), -h]
    verts += [0.0, 0.0, h, 0.0, 0.0, -h]
    top, bot = 2*n, 2*n+1
    faces = []
    for k in range(n):
        k2 = (k+1) % n
        faces += [k, k2, k2+n, k, k2+n, k+n, top, k, k2, bot, k2+n, k+n]
    return verts, faces

def write_wheel_mesh(radius=0.081, half=0.0185, n=16):
    v, f = _cylinder_mesh(radius, half, n)
    vs = " ".join(f"{x:.6f}" for x in v)
    fs = " ".join(str(int(x)) for x in f)
    xml = f'<mesh name="wheel_col" vertex="{vs}" face="{fs}"/>'
    os.makedirs(ASSET_DIR, exist_ok=True)
    with open(WHEEL_COL_XML, "w") as fh:
        fh.write(xml)
    return xml

def build_model_xml(terrain: Terrain, robot_xml=ROBOT_XML, mesh_dir=MESH_DIR,
                    wheel_segments=16, terrain_friction_override=None):
    xml = open(robot_xml, encoding="utf-8").read()

    # 1) convert collision cylinders: wheel -> inline mesh, legs -> capsule
    wheel_mesh_asset = write_wheel_mesh(n=wheel_segments)
    body = None
    out = []
    for ln in xml.splitlines():
        mb = re.search(r'<body name="(\w+)"', ln)
        if mb:
            body = mb.group(1)
        if 'type="cylinder"' in ln and 'class="collision"' in ln:
            msz = re.search(r'size="([\d.]+) ([\d.]+)"', ln)
            mquat = re.search(r'quat="([^"]+)"', ln)
            if msz:
                r, h = float(msz.group(1)), float(msz.group(2))
                if 'friction="1 0.8 0.001"' in ln and body and body.endswith("_wheel"):
                    # wheel -> capsule (same r,h), terrain-only contact (contype=2)
                    # avoids self-collision with wide capsule and MJX cylinder-box gap
                    quat = mquat.group(1) if mquat else "1 0 0 0"
                    ln = (f'<geom type="capsule" class="collision" size="{r:.6g} {h:.6g}" '
                          f'quat="{quat}" friction="1 0.8 0.001" contype="2" conaffinity="2"/>')
                else:
                    ln = ln.replace('type="cylinder"', 'type="capsule"')
                    ln = ln.replace(msz.group(0), f'size="{r+h:.6g} {h:.6g}"')
        out.append(ln)
    xml = "\n".join(out)

    # 2) wheel mesh asset
    xml = xml.replace("</asset>", wheel_mesh_asset + "\n</asset>")
    # 3) remove stock floor, fix meshdir
    xml = re.sub(r'<geom name=\'floor\'[^>]*/>', '', xml)
    xml = xml.replace('meshdir="../meshes/"', f'meshdir="{mesh_dir}"')

    # 4) terrain geoms
    geo = [f'<geom type="plane" size="20 60 0.02" pos="0 20 0" conaffinity="3" '
           f'rgba="0.55 0.6 0.55 1" friction="{terrain.ground_friction} 0.01 0.001"/>']
    for b in terrain.boxes:
        fric = terrain_friction_override if terrain_friction_override is not None else b["friction"]
        sz = " ".join(f"{v:.4f}" for v in b["size"])
        geo.append(f'<geom type="box" size="{sz}" pos="{b["pos"][0]:.4f} {b["pos"][1]:.4f} {b["pos"][2]:.4f}" '
                   f'rgba="{b["rgba"]}" friction="{fric} 0.8 0.001" conaffinity="3"/>')
    return xml.replace("</worldbody>", "\n".join(geo) + "\n</worldbody>")
