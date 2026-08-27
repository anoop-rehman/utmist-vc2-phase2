"""Watch any run from the 2g/2h creature sweep, reconstructed from its own log.

`render_team.py` renders the 2f/2g ANT scene: it builds `TeamRunToGoalDevEnv`
with no `creatures` and loads a `TeamActorCritic`. Point it at a sweep run and
one of two things happens -- a spider checkpoint fails to load into an
ant-shaped net (design 40 vs 20, 16 motors vs 8), or, for the runs where the
widths happen to line up, it quietly draws the WRONG BODIES. Neither is a clip
you can read a result off.

So this renderer takes no scene arguments at all. It reads `log.json`'s recorded
`args` and rebuilds exactly what trained:

    creatures -> scene composition      per_slot -> SlotTeamActorCritic
    down_rule / win_rule / goal_credit -> the same episode semantics

which means a clip cannot silently disagree with the run it claims to show. The
one thing still asserted rather than inferred is `back_x`, because the sweep did
not vary it.

    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \
        -m rower_soccer.competevo_port.render_sweep \
        --run runs/competevo_port/t2h_spsp_s42 --out /tmp/spsp.mp4

EGL, not osmesa. `render_team.py` carried a comment claiming EGL was broken on
this pod; it was not -- the EGLError came from `GLContext.__del__` at teardown,
AFTER a successful render. EGL is 2.2 ms/frame against osmesa's 46 ms.

Read the printed ending histogram next to the clip. It is resolved over every
world, while the video is world 0 only, and for these runs the histogram is the
result -- a 'win rate' in a cell that never scores is a wipeout artifact.
"""

import argparse
import collections
import json
import os

import numpy as np
import torch


def load_args(run):
    """The run's own recorded arguments -- the only source for the scene."""
    with open(os.path.join(run, "log.json")) as fh:
        return json.load(fh)["args"]


def build_env(a, worlds, seed):
    from rower_soccer.competevo_port.team_env import TeamRunToGoalDevEnv
    kw = dict(down_rule=a["down_rule"], win_rule=a["win_rule"],
              goal_credit=a["goal_credit"])
    creatures = None
    if a.get("creatures"):
        creatures = [c.strip() for c in a["creatures"].split(",")]
        kw["scene_kwargs"] = {"creatures": creatures}
    env = TeamRunToGoalDevEnv(num_worlds=worlds,
                              use_gpu=torch.cuda.is_available(),
                              seed=seed, **kw)
    return env, creatures


def build_policies(env, a, run, device):
    """Rebuild the architecture the run trained, then load its weights.

    `per_slot` is read from the log rather than inferred from tensor shapes:
    inference is what `render_team` does for `role_in_design`, and it only works
    because that one flag changes a width. Reading the recorded flag cannot be
    fooled by two variants that happen to agree.
    """
    from rower_soccer.competevo_port.slot_policy import from_env
    from rower_soccer.competevo_port.team_policy import TeamActorCritic
    team = env.meta.team
    sides = [[i for i in range(env.n_agents) if team[i] == t] for t in (0, 1)]
    if a.get("per_slot"):
        acs = [from_env(env, s) for s in sides]
    else:
        acs = [TeamActorCritic(n_agents=env.n_agents,
                               role_in_design=a.get("role_in_design", False))
               for _ in range(2)]
    blob = torch.load(os.path.join(run, "policies.pt"), map_location="cpu")
    for ac, key in zip(acs, ("ac_0", "ac_1")):
        ac.load_state_dict(blob[key])          # strict: a mismatch must raise
    return [ac.to(device).eval() for ac in acs], sides


def render_model_for(n_agents, back_x, creatures):
    """`probe_2v2._team_render_model`, but honouring the composition.

    Same read-only contract as the other renderers: physics stays in the
    batched backend and this separate `MjModel` is posed from world 0's qpos,
    with that world's genome written through the SAME `DesignWriter` the env
    uses -- so the body on screen is the body that was simulated.
    """
    from rower_soccer.competevo_port.design import (CONST_FIELDS,
                                                    WRITTEN_FIELDS,
                                                    DesignWriter,
                                                    build_design_spec)
    from rower_soccer.competevo_port.team_scene import build_dev_team_scene
    model, meta = build_dev_team_scene(n_agents=n_agents, back_x=back_x,
                                       creatures=creatures)
    spec = build_design_spec(model, meta, device="cpu", dtype=torch.float64)
    arrays = {name: torch.from_numpy(
        np.asarray(getattr(model, name))).unsqueeze(0)
        for name in tuple(WRITTEN_FIELDS) + tuple(CONST_FIELDS)}
    writer = DesignWriter(spec, arrays, model=model, exact_constants=True)
    return model, meta, writer, arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True,
                   help="a sweep run directory containing log.json + policies.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--episodes", type=int, default=3,
                   help="episodes recorded from world 0; the ending histogram "
                        "uses every world and is far better resolved")
    p.add_argument("--fps", type=int, default=40)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--back-x", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import imageio.v2 as imageio
    import mujoco

    from rower_soccer.competevo_port.render_designs import apply_design
    from rower_soccer.competevo_port.slot_policy import wrap_env

    a = load_args(args.run)
    out = args.out or os.path.join(args.run, "clip.mp4")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    env, creatures = build_env(a, args.worlds, args.seed)
    acs, sides = build_policies(env, a, args.run, device)
    print(f"[{os.path.basename(args.run)}] creatures={env.meta.creatures} "
          f"per_slot={bool(a.get('per_slot'))} lanes={sides}")
    print(f"  design_dims={env.meta.design_dims} motors={env.meta.n_motors} "
          f"obs_dim={env.obs_dim}")

    driver = wrap_env(env, acs[0])
    lanes = [torch.tensor(s, device=env.device) for s in sides]
    rmodel, rmeta, rwriter, rarrays = render_model_for(
        env.n_agents, args.back_x, creatures)
    renderer = mujoco.Renderer(rmodel, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 14.0, -25.0, 90.0
    torsos = [ag.torso_body for ag in rmeta.agents]

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
                # `env.last_end` is the env's own reason code, the same source
                # the trainer's eval uses. An `info["end_*"]` lookup silently
                # counted nothing here -- those keys do not exist -- and a
                # histogram of zeros reads as "no episodes ended" rather than
                # as a bug, so it is read off the env instead.
                idx = done.nonzero(as_tuple=True)[0]
                for e in env.last_end[idx].tolist():
                    endings[{0: "running", 1: "goal", 2: "wipeout",
                             3: "fall", 4: "timeout"}[e]] += 1
                if bool(done[0]):
                    shown += 1

    if not frames:
        print("  NO FRAMES -- world 0 never left the design stage")
        return
    imageio.mimwrite(out, frames, fps=args.fps, macro_block_size=1, quality=8)
    tot = max(1, sum(endings.values()))
    hist = "  ".join(f"{k} {endings[k]}/{tot} ({100 * endings[k] / tot:.1f}%)"
                     for k in ("goal", "wipeout", "fall", "timeout"))
    print(f"  endings over {tot} episodes, {env.n} worlds: {hist}")
    print(f"  team win rate "
          f"{[round(float(x), 3) for x in np.atleast_1d(env.team_win_rate())]}")
    print(f"  wrote {out}  ({len(frames)} frames, "
          f"{len(frames) / args.fps:.1f}s at {args.fps} fps)")


if __name__ == "__main__":
    main()
