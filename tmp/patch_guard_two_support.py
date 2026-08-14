from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/stair_stance_guard.py')
s=p.read_text(encoding='utf-8')
old="""            stance = [j for j in range(4) if j != i and contact[j]]\n            if len(stance) < 3:\n                request_swing[i] = False\n                continue\n            pts = np.array([self.d.xpos[self.wheel_body_ids[j]][:2] for j in stance])\n            if not self._point_in_polygon(np.asarray(com_xy, dtype=np.float64), pts, self.support_margin):\n                request_swing[i] = False\n"""
new="""            stance = [j for j in range(4) if j != i and contact[j]]\n            if len(stance) < 2:\n                request_swing[i] = False\n                continue\n            pts = np.array([self.d.xpos[self.wheel_body_ids[j]][:2] for j in stance])\n            if not self._point_in_support(np.asarray(com_xy, dtype=np.float64), pts, self.support_margin):\n                request_swing[i] = False\n"""
s=s.replace(old,new)
old2="""    @staticmethod\n    def _point_in_polygon(p, pts, margin):\n"""
new2="""    @staticmethod\n    def _point_in_support(p, pts, margin):\n        pts = np.asarray(pts, dtype=np.float64)\n        if pts.shape[0] == 2:\n            a, b = pts[0], pts[1]\n            ab = b - a\n            denom = float(np.dot(ab, ab)) + 1e-9\n            t = float(np.dot(p - a, ab) / denom)\n            t = min(max(t, 0.0), 1.0)\n            closest = a + t * ab\n            dist = float(np.linalg.norm(p - closest))\n            return dist >= margin\n        if pts.shape[0] < 3:\n            return False\n        return StairStanceGuard._point_in_polygon(p, pts, margin)\n\n    @staticmethod\n    def _point_in_polygon(p, pts, margin):\n"""
s=s.replace(old2,new2)
p.write_text(s, encoding='utf-8')
print('patched guard 2-support')
