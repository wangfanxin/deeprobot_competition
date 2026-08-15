"""sim2sim_exact.py: C++ MuJoCo transfer eval on EXACT competition stair geometry.

Replaces the broken mesh terrain (S10_track.xml meshes do not support the robot in
mujoco 3.11 - see doc RL_stair_方案_20260814.md §3.38) with an equivalent BOX terrain
in this harness only (competition files untouched): approach platform @z=0.48 +
the real 6 risers (0.06/0.13/0.12/0.13/0.12/0.13, treads 0.475/0.40/0.45/0.40/0.40)
at the real track coordinates. CPU-only (no training-GPU use).

Usage:
  python rl_stair/sim2sim_exact.py --ckpt rl_stair/logs/model_latest.pt --seeds 10
"""
import os, sys, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco, torch
from rl_stair.deploy.obs_np import compute_obs_np, build_indices
from rl_stair.envs.terrain import build_model_xml, ASSET_DIR
from rl_stair.ppo import PPO, PPOCfg

GROUND = 0.48
RISERS_Y = np.array([37.90, 38.375, 38.775, 39.225, 39.625, 40.025])
TOPS = np.array([0.54, 0.67, 0.79, 0.92, 1.04, 1.17])
TREADS = [0.475, 0.40, 0.45, 0.40, 0.40, 0.40]
KP_LEG, KD_LEG, KP_WHEEL = 50.0, 1.0, 2.0
TORQ_LEG, TORQ_WHEEL, ACTION_SCALE, VEL_SCALE = 48.0, 13.5, 0.7, 24.0

def build_exact_terrain():
    class T: pass
    t = T(); t.boxes = []
    t.boxes.append({"type":"box","size":[1.5, (37.9-36.4)/2, GROUND/2],
                    "pos":[0.0, (36.4+37.9)/2, GROUND/2], "friction":1.0, "rgba":"0.6 0.65 0.6 1"})
    for i in range(6):
        y0 = RISERS_Y[i]; tr = TREADS[i]; top = TOPS[i]
        t.boxes.append({"type":"box","size":[1.5, tr/2, top/2], "pos":[0.0, y0+tr/2, top/2],
                        "friction":0.8, "rgba":"0.6 0.65 0.6 1"})
    # BUGFIX 2026-08-15 22:06: harness was MISSING the top platform after the last riser.
    # Training terrain (terrain.py stairs()) has top_len=4.0m at the last-riser height, and
    # the real track's stair zone extends +2m past the last riser (cruise_vmc_noros.py:558).
    # Without it the robot climbed all 6 steps then drove OFF THE EDGE -> "fell after reach"
    # artifact (was the entire clean=0/10 story). Add 4m top platform @z=TOPS[-1] to match.
    last_end = RISERS_Y[-1] + TREADS[-1]      # 40.025 + 0.40 = 40.425
    # top platform covers the whole eval window (~10.8m travel in 1200 steps from spawn
    # 36.8 -> y~47.6), so an upright robot never drives off an artificial edge.
    top_len = 10.0
    top_z = float(TOPS[-1])
    t.boxes.append({"type":"box","size":[1.5, top_len/2, top_z/2],
                    "pos":[0.0, last_end + top_len/2, top_z/2],
                    "friction":0.8, "rgba":"0.6 0.65 0.6 1"})
    t.riser_y = RISERS_Y; t.riser_top_z = TOPS
    t.start_y = 35.0; t.goal_y = 41.5; t.ground_friction = 1.0
    return t

def run_seed(m, d, ppo, idx, seed, vx, spawn_y, steps, x_jit, yaw_jit):
    rng = np.random.default_rng(seed)
    x = float(rng.uniform(-x_jit, x_jit))
    yaw = 1.5708 + float(rng.uniform(-yaw_jit, yaw_jit))
    act2jnt, act2vel = idx["act2jnt"], idx["act2vel"]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in range(m.nu)]
    leg_idx = np.array([j for j,nm in enumerate(names) if 'wheel' not in nm])
    wheel_idx = np.array([j for j,nm in enumerate(names) if 'wheel' in nm])
    default_dof = idx["default_dof"]
    d.qpos[:]=0; d.qvel[:]=0
    d.qpos[0:3]=[x, spawn_y, GROUND+0.359]
    d.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
    for i,j in enumerate(act2jnt): d.qpos[j]=default_dof[i]
    mujoco.mj_forward(m,d)
    last_a=np.zeros(m.nu,dtype=np.float32); cmd=np.array([vx,0.0],dtype=np.float32)
    ris=0; fell=False; maxy=spawn_y; reached=False; fell_after_reach=False; fall_y=None; fall_z=None
    # USER-DEFINED handoff (2026-08-15 22:10): "交接在后腿登顶后1米左右".
    # rear axle = base_y - half_wheelbase(0.246); rear legs top when rear axle crosses
    # last riser front (40.025) -> base_y = 40.271; +1.0m -> reach_y = 41.271.
    reach_y = 41.271
    for s in range(steps):
        obs=compute_obs_np(d.qpos,d.qvel,idx,last_a,cmd,RISERS_Y,TOPS)
        with torch.no_grad():
            a=ppo.actor.act(torch.as_tensor(obs).unsqueeze(0),noiseless=True).squeeze(0).numpy()
        a=np.clip(a,-1,1)
        q=d.qpos[act2jnt]-default_dof; qd=d.qvel[act2vel]
        tau=np.zeros(m.nu)
        lt=np.clip(KP_LEG*(a*ACTION_SCALE-q)-KD_LEG*qd,-TORQ_LEG,TORQ_LEG); tau[leg_idx]=lt[leg_idx]
        wt=np.clip(KP_WHEEL*(a*VEL_SCALE-qd),-TORQ_WHEEL,TORQ_WHEEL); tau[wheel_idx]=wt[wheel_idx]
        for _ in range(4):
            d.ctrl[:]=tau; mujoco.mj_step(m,d)
        last_a=a.copy()
        by,bz=d.qpos[1],d.qpos[2]; maxy=max(maxy,by)
        ris=max(ris,int(np.sum(RISERS_Y<by)))
        if (not reached) and (by > reach_y) and (bz > GROUND+0.15):
            reached = True; steps_to_reach = s+1
        if bz<GROUND+0.15:
            fell=True; fell_after_reach = reached; fall_y=float(by); fall_z=float(bz); break
    succ = reached   # RL acceptance: climbed + upright at wp7 handoff
    return {"succ": succ, "ris": ris, "fell": fell, "fell_after_reach": fell_after_reach,
            "reached": reached, "steps_to_reach": steps_to_reach if reached else None,
            "maxy": maxy, "steps": s+1, "fall_y": fall_y, "fall_z": fall_z}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--vx", type=float, default=1.5)
    ap.add_argument("--spawn_y", type=float, default=36.8)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--x_jit", type=float, default=0.1)
    ap.add_argument("--yaw_jit", type=float, default=0.05)
    args = ap.parse_args()

    xml = build_model_xml(build_exact_terrain())
    p = os.path.join(ASSET_DIR, "exact_track_geo.xml")
    with open(p, "w", encoding="utf-8") as f: f.write(xml)
    m = mujoco.MjModel.from_xml_path(p)
    d = mujoco.MjData(m)
    idx = build_indices(m)
    ppo = PPO(55, 72, 16, PPOCfg(num_envs=1), "cpu")
    ppo.load(args.ckpt); ppo.actor.eval()
    print(f"ckpt={args.ckpt} vx={args.vx} seeds={args.seeds}", flush=True)
    res = [run_seed(m, d, ppo, idx, s, args.vx, args.spawn_y, args.steps, args.x_jit, args.yaw_jit)
           for s in range(args.seeds)]
    n_succ = sum(r["succ"] for r in res)
    n_reach = sum(r["reached"] for r in res)
    n_clean = sum(1 for r in res if r["reached"] and not r["fell"])
    ris_hist = {}
    for r in res: ris_hist[r["ris"]] = ris_hist.get(r["ris"], 0) + 1
    fell = sum(r["fell"] for r in res)
    fall_after = sum(r["fell_after_reach"] for r in res)
    sr = [r["steps_to_reach"] for r in res if r["steps_to_reach"]]
    avg_to_reach = int(np.mean(sr)) if sr else None
    fy = [round(r["fall_y"],2) for r in res if r["fell"]]
    print(f"FALL_POS y={sorted(fy)}", flush=True)
    print(f"RESULT reach={n_reach}/{args.seeds} ({100*n_reach/args.seeds:.1f}%) "
          f"clean={n_clean}/{args.seeds} succ={n_succ}/{args.seeds} "
          f"riser_hist={dict(sorted(ris_hist.items()))} fell={fell}(after_reach={fall_after}) "
          f"avg_steps_to_reach={avg_to_reach}", flush=True)

if __name__ == "__main__":
    main()
