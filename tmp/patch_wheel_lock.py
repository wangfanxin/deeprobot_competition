from pathlib import Path
p=Path('dial-mpc/dial_mpc/envs/s10_env.py')
s=p.read_text(encoding='utf-8')
needle='''        _lock_prox = float(os.environ.get("S10_WHEEL_LOCK_PROX", "0.25"))\n        _lock_on = (in_stairs > 0) & (_prox < _lock_prox)\n        r_wheel_lock = -cfg["stair_wheel_lock_w"] * jnp.sum(\n            jnp.square(qd_wheel) * _lock_on.astype(jnp.float32))\n'''
add='''        _lock_prox = float(os.environ.get("S10_WHEEL_LOCK_PROX", "0.25"))\n        # Lock only swing-phase wheels. Locking all near-riser wheels removes\n        # propulsion from the supporting axle and causes rear-wheel free-spin.\n        if _gsw is not None:\n            _sw_lock = _gsw * _prox_ok.astype(jnp.float32)\n        else:\n            _sw_lock = ((_ms > 0.5) & _wr_ok & _prox_ok & (\n                _wr - (h_terrain + cfg["wheel_radius"]) > _sw_th)\n            ).astype(jnp.float32)\n        _lock_on = (in_stairs > 0) & (_prox < _lock_prox) & (_sw_lock > 0.5)\n        r_wheel_lock = -cfg["stair_wheel_lock_w"] * jnp.sum(\n            jnp.square(qd_wheel) * _lock_on.astype(jnp.float32))\n'''
if needle not in s:
    raise SystemExit('wheel lock needle not found')
s=s.replace(needle,add)
p.write_text(s, encoding='utf-8')
print('patched wheel lock swing-only')
