from pathlib import Path
p=Path('src/S10_sdk_deploy/s10_mpc/stair_stance_guard.py')
s=p.read_text(encoding='utf-8')
old="""    def apply(self, tau, gait_swing, com_xy):\n"""
new="""    def apply(self, tau, gait_swing, com_xy, wheel_y=None, wheel_z=None, terrain_z=None):\n"""
s=s.replace(old,new)
old2="""        fn = np.array([float(self.d.cfrc_ext[wb][2]) for wb in self.wheel_body_ids])\n        contact = fn > self.contact_min_n\n"""
new2="""        # Geometric contact is more reliable than cfrc_ext in MuJoCo wheel\n        # contacts, especially while climbing.\n        wheel_y = np.asarray(wheel_y if wheel_y is not None else [self.d.xpos[wb][1] for wb in self.wheel_body_ids], dtype=np.float64)\n        wheel_z = np.asarray(wheel_z if wheel_z is not None else [self.d.xpos[wb][2] for wb in self.wheel_body_ids], dtype=np.float64)\n        terrain_z = np.asarray(terrain_z if terrain_z is not None else (wheel_z - self.wheel_radius), dtype=np.float64)\n        contact = wheel_z < (terrain_z + self.wheel_radius + 0.02)\n        fn = np.where(contact, self.contact_min_n, 0.0)\n"""
s=s.replace(old2,new2)
p.write_text(s, encoding='utf-8')
print('patched guard geom contact')
