from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/stair_contact_planner.py')
s=p.read_text(encoding='utf-8')
needle='''            features["foothold_y"] = fy.astype(np.float32)\n            features["foothold_valid"] = fy_ok\n        return out\n'''
add='''            features["foothold_y"] = fy.astype(np.float32)\n            features["foothold_valid"] = fy_ok\n        if os.environ.get("S10_KNOWN_TERRAIN", "0") == "1" and hasattr(fol, "stair_known_tile"):\n            kt = fol.stair_known_tile(x0, y0, n, n, self.res)\n            if kt is not None:\n                mk = kt["valid"]\n                out["heightmap"] = np.where(mk, kt["heightmap"], out["heightmap"])\n                out["valid"] = out["valid"] | mk\n                for _key in ("slope", "roughness", "step", "step_flag"):\n                    features[_key] = np.where(mk, kt[_key], features[_key])\n        return out\n'''
if needle not in s:
    raise SystemExit('needle not found')
s=s.replace(needle,add)
p.write_text(s, encoding='utf-8')
print('patched')
