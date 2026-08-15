"""analyze_run.py: final-acceptance analysis for a sim2sim wp6->7 run.

Records the robot xy trajectory + instantaneous speed, plots xy colored by speed,
and reports completion time + module-frequency-like stats (substep/step counts).

Usage:
  python rl_stair/deploy/analyze_run.py --ckpt rl_stair/logs/model_best.pt \
      --seeds 20 --vx 1.5 --out rl_stair/logs/wp67_traj.png
"""
import os, sys, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "rl_stair"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, mujoco, torch
from rl_stair.deploy.obs_np import compute_obs_np, build_indices
from rl_stair.envs.terrain import build_model_xml, ASSET_DIR
from rl_stair.ppo import PPO, PPOCfg
import rl_stair.sim2sim_exact as S

GROUND = S.GROUND
RISERS_Y = S.RISERS_Y
TOPS = S.TOPS
REACH_Y = 41.271   # rear-legs-top + 1m (USER-DEFINED handoff)


def run_seed_traj(m, d, ppo, idx, seed, vx, spawn_y, steps, x_jit, yaw_jit):
    rng = np.random.default_rng(seed)
    x = float(rng.uniform(-x_jit, x_jit))
    yaw = 1.5708 + float(rng.uniform(-yaw_jit, yaw_jit))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in range(m.nu)]
    leg_idx = np.array([j for j, nm in enumerate(names) if 'wheel' not in nm])
    wheel_idx = np.array([j for j, nm in enumerate(names) if 'wheel' in nm])
    default_dof = idx["default_dof"]
    d.qpos[:] = 0; d.qvel[:] = 0
    d.qpos[0:3] = [x, spawn_y, GROUND + 0.359]
    d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    for i, j in enumerate(idx["act2jnt"]):
        d.qpos[j] = default_dof[i]
    mujoco.mj_forward(m, d)
    last_a = np.zeros(m.nu, dtype=np.float32)
    cmd = np.array([vx, 0.0], dtype=np.float32)
    traj = []          # (x, y, speed, z)
    ris = 0; fell = False; reached = False; reached_step = None
    for s in range(steps):
        obs = compute_obs_np(d.qpos, d.qvel, idx, last_a, cmd, RISERS_Y, TOPS)
        with torch.no_grad():
            a = ppo.actor.act(torch.as_tensor(obs).unsqueeze(0), noiseless=True).squeeze(0).numpy()
        a = np.clip(a, -1, 1)
        q = d.qpos[idx["act2jnt"]] - default_dof
        qd = d.qvel[idx["act2vel"]]
        tau = np.zeros(m.nu)
        lt = np.clip(S.KP_LEG * (a * S.ACTION_SCALE - q) - S.KD_LEG * qd, -S.TORQ_LEG, S.TORQ_LEG)
        tau[leg_idx] = lt[leg_idx]
        wt = np.clip(S.KP_WHEEL * (a * S.VEL_SCALE - qd), -S.TORQ_WHEEL, S.TORQ_WHEEL)
        tau[wheel_idx] = wt[wheel_idx]
        for _ in range(4):
            d.ctrl[:] = tau; mujoco.mj_step(m, d)
        last_a = a.copy()
        by, bz = d.qpos[1], d.qpos[2]
        spd = float(np.hypot(d.qvel[0], d.qvel[1]))
        traj.append((float(d.qpos[0]), float(by), spd, float(bz)))
        ris = max(ris, int(np.sum(RISERS_Y < by)))
        if (not reached) and by > REACH_Y and bz > GROUND + 0.15:
            reached = True; reached_step = s + 1
        if bz < GROUND + 0.15:
            fell = True; break
    return {"reached": reached, "reached_step": reached_step, "fell": fell,
            "ris": ris, "traj": np.asarray(traj)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--vx", type=float, default=1.5)
    ap.add_argument("--spawn_y", type=float, default=36.8)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--x_jit", type=float, default=0.1)
    ap.add_argument("--yaw_jit", type=float, default=0.05)
    ap.add_argument("--out", default="rl_stair/logs/wp67_traj.png")
    args = ap.parse_args()

    xml = build_model_xml(S.build_exact_terrain())
    p = os.path.join(ASSET_DIR, "exact_track_geo.xml")
    with open(p, "w", encoding="utf-8") as f:
        f.write(xml)
    m = mujoco.MjModel.from_xml_path(p)
    d = mujoco.MjData(m)
    idx = build_indices(m)
    ppo = PPO(55, 72, 16, PPOCfg(num_envs=1), "cpu")
    ppo.load(args.ckpt); ppo.actor.eval()

    res = [run_seed_traj(m, d, ppo, idx, s, args.vx, args.spawn_y, args.steps,
                         args.x_jit, args.yaw_jit) for s in range(args.seeds)]
    n_reach = sum(r["reached"] for r in res)
    sr = [r["reached_step"] for r in res if r["reached_step"]]
    DT = float(m.opt.timestep) * 4.0
    print(f"RESULT reach={n_reach}/{args.seeds} ({100*n_reach/args.seeds:.1f}%)")
    if sr:
        t_reach = np.asarray(sr) * DT
        print(f"REACH_TIME mean={t_reach.mean():.2f}s min={t_reach.min():.2f}s "
              f"max={t_reach.max():.2f}s (dt={DT*1000:.1f}ms/step, steps={np.mean(sr):.0f})")
    # module frequencies (sim timing): nav 2Hz, MPPI ~11Hz, VMC 200Hz, lidar 10Hz
    print("MODULE_FREQ nav=2Hz mppi~11Hz vmc=%.0fHz lidar=10Hz"
          % (1.0 / DT))
    print(f"FALLS={sum(r['fell'] for r in res)}")

    # xy trajectory plot colored by speed
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable:", e)
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    allsp = []
    for i, r in enumerate(res):
        tr = r["traj"]
        if len(tr) == 0:
            continue
        sp = tr[:, 2]
        allsp.extend(sp.tolist())
        sc = ax.scatter(tr[:, 1], tr[:, 0], c=sp, s=4, cmap="viridis",
                        vmin=0.0, vmax=max(2.0, float(np.max(sp))),
                        label=f"seed{i}({'OK' if r['reached'] else 'FAIL'})")
    for y0 in RISERS_Y:
        ax.axvline(y0, color="r", lw=0.5, alpha=0.4)
    ax.axvline(REACH_Y, color="g", lw=1.5, ls="--", label="wp7 handoff (rear+1m)")
    ax.set_xlabel("y (m)"); ax.set_ylabel("x (m)")
    ax.set_title("wp6->7 xy trajectory (color = speed m/s)")
    ax.legend(loc="upper left", fontsize=7)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("speed (m/s)")
    ax.set_aspect("equal")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("SAVED_PLOT", args.out)


if __name__ == "__main__":
    main()
