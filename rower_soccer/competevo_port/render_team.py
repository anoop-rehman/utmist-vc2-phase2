"""Watch a trained 2v2 team actually play.

`probe_2v2.py render` drives the scene with a TRANSPLANTED 1v1 pair, which is
what existed when the design doc was written. This drives it with a native
`TeamActorCritic` -- the thing 2f actually trains -- and is the only way to
look at the result.

Looking is not optional here. This project has twice shipped an environment
that was numerically fine and visually wrong, and the 2v2 scene adds two bodies,
a new contact bitmask and a new spawn layout, none of which a gate can fully
speak for. Every number 2f produces should be read next to a clip.

    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.competevo_port.render_team \
        --policies runs/competevo_port/t2v2_cold/policies.pt \
        --out /tmp/t2v2.mp4 --episodes 3

Same read-only contract as the other renderers: physics stays in the batched
backend and a separate `MjModel` is posed from world 0's qpos, with that
world's genome written through the SAME `DesignWriter` the env uses -- so the
body on screen is the body that was simulated, not an idealised one.
"""

import argparse
import collections
import os

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies",
                   default="runs/competevo_port/t2v2_cold/policies.pt")
    p.add_argument("--out", default="/tmp/t2v2.mp4")
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--episodes", type=int, default=3,
                   help="episodes recorded from world 0. The ending histogram "
                        "uses every world, so it is far better resolved")
    p.add_argument("--fps", type=int, default=40)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--back-x", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--untrained", action="store_true")
    args = p.parse_args()
    # EGL is broken on this pod (eglQueryString on NoneType); osmesa renders
    # the same pixels on the CPU, slower. Physics is unaffected either way.
    os.environ.setdefault("MUJOCO_GL", "osmesa")

    import imageio.v2 as imageio
    import mujoco

    from rower_soccer.competevo_port.probe_2v2 import _team_render_model
    from rower_soccer.competevo_port.render_designs import apply_design
    from rower_soccer.competevo_port.team_env import TeamRunToGoalDevEnv
    from rower_soccer.competevo_port.team_policy import TeamActorCritic
    from rower_soccer.competevo_port.train_team_smoke import TeamPolicyObsEnv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = TeamRunToGoalDevEnv(num_worlds=args.worlds,
                              use_gpu=(device == "cuda"), seed=args.seed,
                              down_rule="team_down", win_rule="team_first",
                              goal_credit="team",
                              scene_kwargs={"back_x": args.back_x})
    torch.manual_seed(args.seed)
    role = False
    blob = None
    if not args.untrained:
        blob = torch.load(args.policies, map_location="cpu")
        # Infer role_in_design from the checkpoint, as score_policies does:
        # building the wrong variant fails on load, and a renderer that must be
        # told which variant it holds will be pointed at the wrong one.
        from rower_soccer.competevo_port.team_policy import ROLE_DIM
        probe = TeamActorCritic(n_agents=env.n_agents)
        role = (blob["ac_0"]["scale_norm.mean"].numel()
                == probe.scale_norm.mean.numel() + ROLE_DIM)
    acs = [TeamActorCritic(n_agents=env.n_agents, role_in_design=role)
           for _ in range(2)]
    if blob is not None:
        for ac, key in zip(acs, ("ac_0", "ac_1")):
            ac.load_state_dict(blob[key])
        print(f"role_in_design={role} (inferred)")
    acs = [ac.to(device).eval() for ac in acs]
    driver = TeamPolicyObsEnv(env, acs[0])
    lanes = [torch.tensor(l, device=env.device) for l in ([0, 2], [1, 3])]

    rmodel, rmeta, rwriter, rarrays = _team_render_model(env.n_agents,
                                                         args.back_x)
    renderer = mujoco.Renderer(rmodel, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 14.0, -25.0, 90.0
    torsos = [a.torso_body for a in rmeta.agents]

    frames, shown, live = [], 0, None
    endings = collections.Counter()
    env.reset_win_stats()
    obs = driver.reset()
    budget = args.episodes * (env.max_episode_steps + 2) + 8
    with torch.no_grad():
        for _ in range(budget):
            o = obs.float()
            act = torch.zeros(env.n, env.n_agents, env.act_dim,
                              device=env.device, dtype=o.dtype)
            for e, ln in enumerate(lanes):
                act[:, ln] = acs[e].mean_action(o[:, ln])
            obs, _, done, info = driver.step(act.to(env.dtype))
            if shown < args.episodes and not bool(info["was_design"][0]):
                d0 = env.scale[0].detach().cpu().numpy()
                if live is None or not np.array_equal(d0, live):
                    apply_design(rmodel, rwriter, rarrays, d0)
                    live = d0.copy()
                rdata = mujoco.MjData(rmodel)
                rdata.qpos[:] = env.qpos[0].detach().double().cpu().numpy()
                mujoco.mj_forward(rmodel, rdata)
                cam.lookat[:] = rdata.xpos[torsos].mean(0)
                renderer.update_scene(rdata, camera=cam)
                frames.append(renderer.render())
            if bool(done.any()):
                idx = done.nonzero(as_tuple=True)[0]
                for e in env.last_end[idx].tolist():
                    endings[{0: "running", 1: "goal", 2: "wipeout",
                             3: "fall", 4: "timeout"}[e]] += 1
                if bool(done[0]):
                    shown += 1

    if frames:
        imageio.mimwrite(args.out, frames, fps=args.fps, macro_block_size=1,
                         quality=8)
    total = max(sum(endings.values()), 1)
    label = "UNTRAINED" if args.untrained else args.policies
    print(f"\npolicy: {label}")
    print(f"  {total} episodes over {env.n} worlds")
    for k in ("goal", "wipeout", "fall", "timeout"):
        print(f"    {k:8s} {endings[k]:5d}   {100 * endings[k] / total:5.1f}%")
    print(f"  team win rate {[round(float(x), 3) for x in np.atleast_1d(env.team_win_rate())]}")
    if frames:
        print(f"  video: {args.out}  ({len(frames)} frames, "
              f"{len(frames) / args.fps:.1f}s at {args.fps} fps)")
    else:
        print("  NO FRAMES -- world 0 never left the design stage")


if __name__ == "__main__":
    main()
