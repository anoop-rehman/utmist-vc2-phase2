"""Handoff gate for the ball-strike drill envs (`kick`, `shoot`). Re-runnable.

Two checks per env, both of which have caught real bugs in this stack before:

  1. 256 worlds x 600 steps of RANDOM TORQUE. Random torque is the adversarial
     case for the contact model -- it flails the creature into the ball and the
     floor at full gear with no policy smoothing it -- and it is exactly the
     probe scene.py's solref values were tuned against. Asserts: zero diverged
     worlds, every observation finite and inside ppo.OBS_SANITY_LIMIT, every
     reward finite and inside the env's own rew_clip.
  2. A single-world eval render, written to a real .mp4. This is the check that
     the env is *watchable* -- render.py builds a SEPARATE model from the same
     scene, so a mismatch between the physics and render scenes (which shoot
     could easily have: it is the one drill on the full pitch) shows up here and
     nowhere else.

Run (from the repo root, PYTHONPATH=. so the package resolves to this tree):

  MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.warp_port.smoke_ball_envs
  ... --env kick --worlds 256 --steps 600 --out /tmp/smoke
  ... --cpu            # no GPU: CpuBackend, few worlds, slow but backend-parity

Exit status is 0 only if every check passes, so it drops straight into CI.
"""

import argparse
import os
import time

import numpy as np
import torch


def build(name, worlds, seed, use_gpu, creature_xml):
    if name == "kick":
        from rower_soccer.warp_port.kick_env import WarpKickEnv
        env = WarpKickEnv(num_worlds=worlds, seed=seed, use_gpu=use_gpu,
                          use_graph=use_gpu, creature_xml=creature_xml)
    elif name == "shoot":
        from rower_soccer.warp_port.shoot_env import WarpShootEnv
        env = WarpShootEnv(num_worlds=worlds, seed=seed, use_gpu=use_gpu,
                           use_graph=use_gpu, creature_xml=creature_xml)
    else:
        raise ValueError(name)
    return env


def renderer_for(name, creature_xml, floor_half):
    from rower_soccer.warp_port.render import WarpRenderer
    if name == "shoot":
        # shoot's physics scene IS the pitch (shoot_env._base_xml returns None),
        # so the render scene must be too, or the picture is of a different world.
        return WarpRenderer(creature_xml, has_ball=True, base_xml=None,
                            distance=12.0)
    from rower_soccer.warp_port.worm_env_base import _arena_xml
    return WarpRenderer(creature_xml, has_ball=True,
                        base_xml=_arena_xml(floor_half), distance=8.0)


def random_torque_smoke(env, steps, name):
    from rower_soccer.warp_port.ppo import OBS_SANITY_LIMIT
    torch.manual_seed(0)
    obs = env.reset()
    lo, hi = env.rew_clip
    obs_max, rew_lo, rew_hi, rew_sum = 0.0, float("inf"), float("-inf"), 0.0
    n_nonfinite_obs = n_nonfinite_rew = n_over = 0
    # The segment/strike accumulators are PER EPISODE and env.reset() clears
    # them, so harvest before every reset instead of reading them at the end.
    tally = {"segments": 0.0, "strikes": 0.0, "credit": 0.0, "goals": 0.0}

    def harvest():
        tally["segments"] += float(env.n_segments.sum())
        tally["strikes"] += float(env.credit_count.sum())
        tally["credit"] += float(env.credit_sum.sum())
        if hasattr(env, "goals"):
            tally["goals"] += float(env.goals.sum())

    t0 = time.perf_counter()
    for t in range(steps):
        a = torch.rand(env.n, env.act_dim, device=env.device) * 2.0 - 1.0
        obs, rew, done = env.step(a)
        if done:
            harvest()
            obs = env.reset()
        n_nonfinite_obs += int((~torch.isfinite(obs).all(-1)).sum())
        n_nonfinite_rew += int((~torch.isfinite(rew)).sum())
        m = float(obs.abs().amax())
        obs_max = max(obs_max, m)
        n_over += int((obs.abs().amax(-1) > OBS_SANITY_LIMIT).sum())
        rew_lo = min(rew_lo, float(rew.min()))
        rew_hi = max(rew_hi, float(rew.max()))
        rew_sum += float(rew.mean())
    dt = time.perf_counter() - t0
    harvest()
    out = {
        "env": name,
        "worlds": env.n,
        "steps": steps,
        "obs_dim": env.obs_dim,
        "proprio": len(env.proprio_indices),
        "task": len(env.task_indices),
        "act_dim": env.act_dim,
        "contact_dist_m": round(env.contact_dist, 3),
        "reach_m": round(env.reach, 3),
        "diverged_worlds": env.n_diverged,
        "nonfinite_obs_rows": n_nonfinite_obs,
        "nonfinite_rewards": n_nonfinite_rew,
        "obs_abs_max": round(obs_max, 2),
        "obs_over_sanity_limit": n_over,
        "reward_min": round(rew_lo, 4),
        "reward_max": round(rew_hi, 4),
        "reward_mean_per_step": round(rew_sum / steps, 4),
        "segments_completed": int(tally["segments"]),
        "strikes_banked": int(tally["strikes"]),
        "mean_strike_speed": round(
            tally["credit"] / max(1.0, tally["strikes"]), 3),
        "fitness_mean": round(float(env.fitness().mean()), 4),
        "env_steps_per_s": int(env.n * steps / dt),
    }
    if hasattr(env, "goals"):
        out["goals"] = int(tally["goals"])
    fails = []
    # Positive control: if not one segment restarted in 600 steps the segment
    # machinery is dead and every other number above is measuring nothing.
    if tally["segments"] < 1:
        fails.append("no strike segment completed -- segment logic is inert")
    if env.n_diverged:
        fails.append(f"{env.n_diverged} diverged worlds")
    if n_nonfinite_obs:
        fails.append(f"{n_nonfinite_obs} non-finite obs rows")
    if n_nonfinite_rew:
        fails.append(f"{n_nonfinite_rew} non-finite rewards")
    if n_over:
        fails.append(f"{n_over} obs rows over OBS_SANITY_LIMIT")
    if not (lo - 1e-4 <= rew_lo and rew_hi <= hi + 1e-4):
        fails.append(f"reward outside rew_clip {env.rew_clip}")
    return out, fails


def render_smoke(name, path, creature_xml, use_gpu):
    """One-world episode driven by an untrained ActorCritic, rendered to mp4.
    Untrained rather than random-uniform so it exercises the exact code path the
    trainer's periodic eval video uses (render.eval_video + ac.dist)."""
    from rower_soccer.warp_port.ppo import ActorCritic
    from rower_soccer.warp_port.render import eval_video
    env = build(name, 1, 7, use_gpu, creature_xml)
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(), z_dim=16)
    ac = ac.to(env.device)
    ren = renderer_for(name, creature_xml, getattr(env, "_floor_half", 10.0))
    ep_rew, fit = eval_video(env, ac, path, ren)
    size = os.path.getsize(path)
    fails = []
    if size < 10_000:
        fails.append(f"{path} is only {size} bytes")
    if not np.isfinite(ep_rew) or not np.isfinite(fit):
        fails.append(f"non-finite eval ep_rew={ep_rew} fitness={fit}")
    return {"video": path, "bytes": size, "ep_rew": round(ep_rew, 3),
            "fitness": round(fit, 4)}, fails


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="both", choices=["kick", "shoot", "both"])
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--creature-xml", default="creature_configs/ant.xml")
    p.add_argument("--out", default="runs_v2/smoke_ball_envs")
    p.add_argument("--cpu", action="store_true",
                   help="CpuBackend instead of Warp (no CUDA needed; slow)")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    use_gpu = not args.cpu

    names = ["kick", "shoot"] if args.env == "both" else [args.env]
    all_fails = []
    for name in names:
        env = build(name, args.worlds, args.seed, use_gpu, args.creature_xml)
        stats, fails = random_torque_smoke(env, args.steps, name)
        print(f"[{name}] random-torque smoke:", flush=True)
        for k, v in stats.items():
            print(f"    {k:26s} {v}", flush=True)
        all_fails += [f"{name}: {f}" for f in fails]
        del env
        if not args.no_video:
            path = os.path.join(args.out, f"{name}_smoke.mp4")
            vstats, vfails = render_smoke(name, path, args.creature_xml, use_gpu)
            print(f"[{name}] eval render:", flush=True)
            for k, v in vstats.items():
                print(f"    {k:26s} {v}", flush=True)
            all_fails += [f"{name}: {f}" for f in vfails]

    if all_fails:
        print("\nGATE FAILED:", flush=True)
        for f in all_fails:
            print(f"  - {f}", flush=True)
        raise SystemExit(1)
    print("\nGATE PASSED", flush=True)


if __name__ == "__main__":
    main()
