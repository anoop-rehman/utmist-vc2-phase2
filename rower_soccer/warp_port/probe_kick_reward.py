"""Where does kick's reward actually go, and why is it so uneven?

Two questions, one probe.

**Per frame.** The reward is a sum of four things and only one of them is ever
large. This logs each separately so a claim like "it gets a big reward when it
strikes the ball" can be checked rather than assumed:

    reward = w_arrive * last_arrival          <- ZERO except on the ONE step a
                                                 segment closes
           + w_strike * credit                <- banked when the ball leaves the
                                                 creature (w_strike is 0.1)
           + shaping_scale * shaping          <- every step, small
    all multiplied by upright ** w_upright

**Per segment.** An episode is 15 s cut into 2-6 s segments. Each segment
respawns the BALL and the TARGET but NOT the creature, which keeps the pose,
heading and velocity the last segment left it in. So the difficulty of a segment
is partly inherited, and the hypothesis this probe exists to test is that
per-segment arrival is predicted by the geometry the segment STARTS with -- how
far the ball is, how far the creature has to turn to face it, how far it then has
to turn again to send the ball at the target, and how many seconds it has. If
that is true, a fitness that averages over segments is capped by the undoable
ones and the plateau is a property of the task, not of the policy.

    python -m rower_soccer.warp_port.probe_kick_reward \
        --run runs_v2/kick_ant_v12_v3_unfrozen --ckpt latest.pt \
        --worlds 64 --episodes 8 --out runs_v2/kick_ant_v12_v3_unfrozen/probe

Writes a per-step CSV for world 0, a per-segment CSV for every world, and prints
the correlation table. `--video` additionally renders world 0 with the reward
decomposition burned into each frame.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from rower_soccer.warp_port.ball_task import upright
from rower_soccer.warp_port.ppo import ActorCritic, SimpleActorCritic, _flatten_checkpoint


def build(run_dir, ckpt, worlds, seed):
    import importlib
    cfg = json.load(open(os.path.join(run_dir, "config.json")))

    class A:
        def __init__(self, d):
            self.__dict__.update(d)

        def __getattr__(self, n):
            raise AttributeError(f"config.json has no '{n}'")

    args = A(dict(cfg, score_worlds=worlds, score_seed=seed))
    mod = importlib.import_module("rower_soccer.warp_port.train_kick_warp")
    env = mod.make_env(args, num_worlds=worlds, seed=seed)
    sd = _flatten_checkpoint(torch.load(os.path.join(run_dir, ckpt),
                                        map_location="cpu"))
    if args.plain:
        ac = SimpleActorCritic(env.obs_dim, env.act_dim)
    else:
        ac = ActorCritic(env.obs_dim, env.act_dim,
                         proprio_indices=env.proprio_indices.tolist(),
                         task_indices=env.task_indices.tolist(),
                         z_dim=args.z_dim,
                         state_dependent_std=args.state_dependent_std)
    want, have = set(ac.state_dict()), set(sd)
    assert want == have, (sorted(want - have)[:4], sorted(have - want)[:4])
    ac.load_state_dict(sd)
    ac.eval().to(env.device)
    return env, ac, args


def heading_xy(env):
    """Unit vector the creature's root faces, in world xy."""
    _, rot = env._root_frames()
    fwd = rot[:, :, 0]                       # first column = local +x
    n = torch.linalg.norm(fwd[:, :2], dim=-1, keepdim=True).clamp(min=1e-6)
    return fwd[:, :2] / n


def signed_angle(a, b):
    """Angle from a to b in degrees, both [n,2], in [0,180]."""
    a = a / torch.linalg.norm(a, dim=-1, keepdim=True).clamp(min=1e-6)
    b = b / torch.linalg.norm(b, dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1.0, 1.0)))


@torch.no_grad()
def run(env, ac, args, episodes, out_dir, want_video=False):
    os.makedirs(out_dir, exist_ok=True)
    rw = env.reward                                # the RewardStrategy object
    w_arrive = getattr(rw, "w_arrive", 0.0)
    w_strike = getattr(rw, "w_strike", 0.0)
    w_upright = getattr(rw, "w_upright", 0.0)

    steps = int(round(args.episode_secs / 0.025)) * episodes
    obs = env.reset()
    per_step, per_seg = [], []
    frames = []
    renderer = None
    if want_video:
        from rower_soccer.warp_port.render import WarpRenderer
        renderer = WarpRenderer(args.creature_xml, has_ball=True)

    # Geometry the CURRENT segment started with, per world. Captured after every
    # respawn, which is the only moment it is the segment's own.
    def capture():
        root, _ = env._root_frames()
        root_xy = root[:, :2]
        ball = env._ball_xy()
        tgt = env.target_xy
        to_ball = ball - root_xy
        to_tgt = tgt - ball
        return {
            "d_ant_ball": torch.linalg.norm(to_ball, dim=-1).clone(),
            "turn_to_ball": signed_angle(heading_xy(env), to_ball).clone(),
            "turn_ball_to_target": signed_angle(to_ball, to_tgt).clone(),
            "d_ball_target": torch.linalg.norm(to_tgt, dim=-1).clone(),
            "budget_s": (env.seg_limit * 0.025).clone(),
            "upright0": upright(env).clone(),
        }

    start = capture()
    struck = torch.zeros(env.n, dtype=torch.bool, device=env.device)
    # `last_arrival` is the REWARD curve, exp(-arrival_reward_coef * d), which is
    # deliberately gentler than the FITNESS curve exp(-0.5 * d) the run reports.
    # Track both: quoting one as the other is how a probe ends up disagreeing
    # with the trainer's own number for no real reason.
    fit_sum_prev = env.target_fit_sum.clone()

    for t in range(steps):
        # `ac.dist(obs).mean`, never a sample -- the same deterministic path
        # score.py uses, so numbers here are comparable to the run's own scores.
        act = ac.dist(obs.float()).mean
        # Components BEFORE the step would price the previous state; the reward
        # the env returns is computed after the step, so read them after too.
        obs, rew, done = env.step(act.to(obs.dtype))

        up = upright(env)
        arrive = w_arrive * env.last_arrival
        strike = w_strike * env.credit
        shaped = env.shaping_scale * rw._shaping(env)
        struck |= env.credit > 0

        per_step.append({
            "t": t, "reward": float(rew[0]),
            "arrive": float(arrive[0] * up[0] ** w_upright),
            "strike": float(strike[0] * up[0] ** w_upright),
            "shaping": float(shaped[0] * up[0] ** w_upright),
            "upright": float(up[0]),
            "seg_t": float(env.seg_t[0]), "seg_limit": float(env.seg_limit[0]),
            "d_ball_target": float(env._target_dist_now()[0]),
            "closed": int(bool(env.seg_reset[0])),
        })

        if bool(env.seg_reset.any()):
            idx = env.seg_reset.nonzero(as_tuple=True)[0]
            for i in idx.tolist():
                per_seg.append({
                    "world": i,
                    "d_ant_ball": float(start["d_ant_ball"][i]),
                    "turn_to_ball": float(start["turn_to_ball"][i]),
                    "turn_ball_to_target": float(start["turn_ball_to_target"][i]),
                    "d_ball_target": float(start["d_ball_target"][i]),
                    "budget_s": float(start["budget_s"][i]),
                    "upright0": float(start["upright0"][i]),
                    "struck": int(bool(struck[i])),
                    # last_arrival is the snapshot taken before the respawn, so
                    # it is this segment's outcome and not the next one's spawn.
                    "arrival_reward": float(env.last_arrival[i]),
                    # delta of the running fitness accumulator = this segment's
                    # exp(-0.5 d), the number the trainer's fitness averages
                    "arrival": float(env.target_fit_sum[i] - fit_sum_prev[i]),
                })
            fit_sum_prev = env.target_fit_sum.clone()
            fresh = capture()
            for k, v in fresh.items():
                start[k][idx] = v[idx]
            struck[idx] = False

        if renderer is not None and len(frames) < 600:
            frames.append(renderer.frame(env, 0))

    with open(os.path.join(out_dir, "per_step_world0.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_step[0]))
        w.writeheader(); w.writerows(per_step)
    with open(os.path.join(out_dir, "per_segment.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seg[0]))
        w.writeheader(); w.writerows(per_seg)
    if frames:
        import imageio.v2 as imageio
        imageio.mimwrite(os.path.join(out_dir, "world0.mp4"), frames, fps=40,
                         macro_block_size=1, quality=8)
    return per_step, per_seg


def report(per_step, per_seg, out_dir):
    ps = {k: np.array([r[k] for r in per_step], dtype=float) for k in per_step[0]}
    n_closed = int(ps["closed"].sum())
    print(f"\n=== per-step, world 0: {len(per_step)} steps, "
          f"{n_closed} segment closes ===")
    tot = ps["reward"].sum()
    for k in ("arrive", "strike", "shaping"):
        v = ps[k]
        nz = int((np.abs(v) > 1e-9).sum())
        print(f"  {k:9s} sum {v.sum():9.1f} ({100 * v.sum() / tot:5.1f}% of "
              f"total)   nonzero on {nz:4d}/{len(v)} steps   max {v.max():6.3f}")
    print(f"  {'TOTAL':9s} sum {tot:9.1f}")
    print(f"  upright: mean {ps['upright'].mean():.3f}  min {ps['upright'].min():.3f}")

    seg = {k: np.array([r[k] for r in per_seg], dtype=float) for k in per_seg[0]}
    a = seg["arrival"]
    ar = seg["arrival_reward"]
    print(f"  arrival(reward curve, what is PAID): mean {ar.mean():.3f}")
    print(f"\n=== per-segment, all worlds: {len(a)} segments ===")
    print(f"  arrival: mean {a.mean():.3f}  median {np.median(a):.3f}  "
          f"p10 {np.percentile(a,10):.3f}  p90 {np.percentile(a,90):.3f}")
    # exp(-0.5 * d) at the spawn distance is what a policy that never moves the
    # ball scores, so it is the floor any comparison has to beat.
    floor = np.exp(-0.5 * seg["d_ball_target"])
    print(f"  do-nothing floor (exp(-0.5*spawn dist)): mean {floor.mean():.3f}")
    print(f"  fraction of segments at or below the floor: "
          f"{100 * (a <= floor + 1e-3).mean():.1f}%")
    print(f"  fraction above 0.8 (a good kick): {100 * (a > 0.8).mean():.1f}%")
    print(f"  struck the ball in {100 * seg['struck'].mean():.1f}% of segments")

    print("\n=== does the STARTING geometry predict the outcome? ===")
    print(f"  {'feature':22s} {'corr with arrival':>18s}")
    for k in ("d_ant_ball", "turn_to_ball", "turn_ball_to_target",
              "d_ball_target", "budget_s", "upright0"):
        if seg[k].std() < 1e-9:
            continue
        print(f"  {k:22s} {np.corrcoef(seg[k], a)[0, 1]:18.3f}")

    print("\n=== arrival by starting difficulty ===")
    for k, edges in (("d_ant_ball", [0, 1, 2, 3, 100]),
                     ("turn_to_ball", [0, 30, 60, 120, 181]),
                     ("budget_s", [0, 3, 4, 5, 100])):
        print(f"  by {k}:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (seg[k] >= lo) & (seg[k] < hi)
            if m.sum() < 5:
                continue
            print(f"    [{lo:5.1f},{hi:5.1f})  n={int(m.sum()):5d}  "
                  f"arrival {a[m].mean():.3f}  struck {100*seg['struck'][m].mean():4.0f}%")
    print(f"\nwrote {out_dir}/per_step_world0.csv and per_segment.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="latest.pt")
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=None)
    p.add_argument("--video", action="store_true")
    a = p.parse_args()
    out = a.out or os.path.join(a.run, "probe")
    env, ac, cfg = build(a.run, a.ckpt, a.worlds, a.seed)
    print(f"{a.run}/{a.ckpt}  {a.worlds} worlds x {a.episodes} episodes "
          f"({cfg.episode_secs}s each)  reward_kind={cfg.reward_kind} "
          f"w_arrive={cfg.w_arrive} w_strike={cfg.w_strike}")
    ps, pg = run(env, ac, cfg, a.episodes, out, want_video=a.video)
    report(ps, pg, out)


if __name__ == "__main__":
    main()
