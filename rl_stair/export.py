"""Export a trained RL-stair actor to TorchScript (.pt) / optional ONNX.

IMPORTANT (JIT lesson): trace the DETERMINISTIC forward path
`forward() = tanh(MLP(obs))`. Do NOT trace `act()` -- it mutates `self.std`
and fails torch.jit.trace ("Cannot insert a Tensor that requires grad as a
constant"). Verified on torch 2.7: forward trace round-trip max diff = 0.0.

Usage:
  python rl_stair/export.py --ckpt rl_stair/logs/model_latest.pt \
      --out rl_stair/deploy/policy.pt [--onnx rl_stair/deploy/policy.onnx]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rl_stair.ppo import Actor

OBS_LAYOUT = (
    "53-dim actor obs order (deploy must reproduce EXACTLY):\n"
    "  [0:3]   base ang vel (body) *0.25\n"
    "  [3:6]   projected gravity\n"
    "  [6:8]   cmd [vx, yaw]\n"
    "  [8:20]  leg joint pos error (12) *1 (default 0/0.67/-1.3)\n"
    "  [20:32] leg joint vel (12) *0.05\n"
    "  [32:48] last action (16)\n"
    "  [48:52] terrain ctx: front/rear axle distance + height diff to next riser (4)\n"
    "  [52]    rough bool (stair OR ridge)\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="rl_stair/deploy/policy.pt")
    ap.add_argument("--onnx", type=str, default="")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit("checkpoint not found: " + args.ckpt)
    print(OBS_LAYOUT)
    ck = torch.load(args.ckpt, map_location="cpu")
    am = ck["actor"]
    obs_dim = am["mean.0.weight"].shape[1]
    act_dim = am["mean.6.weight"].shape[0]
    print("inferred obs_dim={} action_dim={} it={}".format(obs_dim, act_dim, ck.get("it", 0)))

    actor = Actor(obs_dim, act_dim).eval()
    actor.load_state_dict(am)
    x = torch.randn(args.batch, obs_dim)

    traced = torch.jit.trace(actor, x)   # forward() path
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.jit.save(traced, args.out)

    loaded = torch.jit.load(args.out)
    with torch.no_grad():
        d = float((actor(x) - loaded(x)).abs().max())
    print("saved {}  jit round-trip max diff = {}".format(args.out, d))
    assert d == 0.0, "JIT round-trip mismatch!"

    if args.onnx:
        try:
            torch.onnx.export(actor, x, args.onnx, input_names=["obs"],
                              output_names=["action"], opset_version=13)
            print("saved " + args.onnx)
        except Exception as e:
            print("ONNX skipped: {}: {}".format(type(e).__name__, e))


if __name__ == "__main__":
    main()
