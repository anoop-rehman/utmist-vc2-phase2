"""Watch a trained dev pair actually play, and count how their games end.

Two things at once, because they answer the same question. 2e showed our port
reproduces their LEARNING curve but scores ~12x fewer wins, and the leading
suspect is that our episodes end differently -- ours fall over. A video says
what the failure looks like; the ending histogram says how often it happens.
Rendering is the same read-only contract as `render.py`: physics stays in the
batched backend, and a separate MjModel is posed from one world's qpos to take
the picture, with that world's genome written through the SAME DesignWriter the
env uses, so the body on screen is the body that was simulated.

    python -m rower_soccer.competevo_port.render_dev_rollout \
        --policies runs/competevo_port/m2e_validation/policies.pt \
        --out /tmp/dev_pair.mp4 --worlds 64 --episodes 6
"""

import argparse
import collections
import os

import numpy as np
import torch


def _load_pair(path, device):
    from rower_soccer.competevo_port.dev_ppo import DevActorCritic
    blob = torch.load(path, map_location="cpu")
    acs = []
    for key in ("ac_0", "ac_1"):
        ac = DevActorCritic().to(device)
        ac.load_state_dict(blob[key])
        ac.eval()
        acs.append(ac)
    return acs, blob.get("args", {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", default="runs/competevo_port/m2e_validation/policies.pt")
    p.add_argument("--out", default="/tmp/dev_pair.mp4")
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--episodes", type=int, default=6,
                   help="episodes to record from world 0 (the histogram uses "
                        "every world, so it is far better resolved than this)")
    p.add_argument("--fps", type=int, default=40)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import imageio.v2 as imageio
    import mujoco

    from rower_soccer.competevo_port.dev_env import RunToGoalDevEnv
    from rower_soccer.competevo_port.render_designs import (_render_model,
                                                            apply_design)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    acs, targs = _load_pair(args.policies, device)
    env = RunToGoalDevEnv(num_worlds=args.worlds, use_gpu=(device == "cuda"),
                          seed=args.seed)

    rmodel, rmeta, rwriter, rarrays = _render_model(n_agents=env.n_agents)
    renderer = mujoco.Renderer(rmodel, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 9.0, -22.0, 90.0
    torsos = [a.torso_body for a in rmeta.agents]

    frames = []
    endings = collections.Counter()
    both_fell_and_scored = 0
    ep_lens, ep_done = [], 0
    shown = 0
    live_design = None

    obs = env.reset()
    env.reset_win_stats()
    # Long enough for `--episodes` world-0 episodes even if every one of them
    # runs to the 500-step truncation limit.
    budget = args.episodes * (env.max_episode_steps + 2) + 8
    with torch.no_grad():
        for _ in range(budget):
            o = obs.float()
            a = torch.stack([acs[i].mean_action(o[:, i])
                             for i in range(env.n_agents)], dim=1)
            obs, rew, done, info = env.step(a.to(env.dtype))

            if shown < args.episodes and not bool(info["was_design"][0]):
                d0 = env.scale[0].detach().cpu().numpy()
                if live_design is None or not np.array_equal(d0, live_design):
                    apply_design(rmodel, rwriter, rarrays, d0)
                    live_design = d0.copy()
                rdata = mujoco.MjData(rmodel)
                rdata.qpos[:] = env.qpos[0].detach().double().cpu().numpy()
                mujoco.mj_forward(rmodel, rdata)
                cam.lookat[:] = rdata.xpos[torsos].mean(0)
                renderer.update_scene(rdata, camera=cam)
                frames.append(renderer.render())

            if bool(done.any()):
                idx = done.nonzero(as_tuple=True)[0]
                won = info["winner"][idx].any(-1)
                fell = info["fell"][idx].any(-1)
                trunc = info["truncated"][idx]
                both_fell_and_scored += int((won & fell).sum())
                # Precedence matches the env's own: a crossed goal line ends
                # the game whatever the torso height is doing.
                for w, f, t in zip(won.tolist(), fell.tolist(), trunc.tolist()):
                    endings["goal" if w else "fell" if f
                            else "timeout" if t else "other"] += 1
                ep_lens.extend(env.last_len[idx].float().cpu().tolist())
                ep_done += len(idx)
                if bool(done[0]):
                    shown += 1

    if frames:
        imageio.mimwrite(args.out, frames, fps=args.fps,
                         macro_block_size=1, quality=8)

    total = sum(endings.values()) or 1
    print(f"policies: {args.policies}  (trained args epochs≈{targs.get('iters')})")
    print(f"worlds={env.n}  episodes finished={ep_done}  "
          f"mean length={np.mean(ep_lens) if ep_lens else 0:.1f} "
          f"of {env.max_episode_steps}")
    for k in ("goal", "fell", "timeout", "other"):
        print(f"  {k:8s} {endings[k]:6d}  {100.0 * endings[k] / total:5.1f}%")
    print(f"  (episodes where a goal and a fall coincided: {both_fell_and_scored})")
    print(f"win rate per agent: {env.win_rate()}")
    if frames:
        print(f"video: {args.out}  ({len(frames)} frames, "
              f"{len(frames) / args.fps:.1f}s at {args.fps} fps)")


if __name__ == "__main__":
    main()
