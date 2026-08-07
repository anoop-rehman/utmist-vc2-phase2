"""PPO on the Warp port of dm_control quadruped-fetch.

Two architectures, matching the reproduction question being asked:
  * --plain: SimpleActorCritic MLP (the standard-baseline arm)
  * default: our latent-bottleneck ActorCritic (expert -> z -> shared decoder),
    proprio = the 78 body-internal dims, task = ball_state + target_position.

    MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_fetch_warp \
        --run-name fetch_warp_plain --plain --worlds 1024
"""
import argparse
import json
import os
import subprocess
import time

import imageio
import mujoco
import numpy as np
import torch


def eval_episode(env, ac, video_path=None, renderer=None, cam=0, fps=50):
    """One deterministic episode on `env` (1 world). Returns (return, frames)."""
    obs = env.reset()
    total, done, frames = 0.0, False, []
    while not done:
        with torch.no_grad():
            a = ac.dist(obs.float()).mean.clamp(-1, 1)
        obs, r, done = env.step(a)
        total += float(r[0])
        if video_path is not None:
            renderer_data = renderer[1]
            renderer_data.qpos[:] = env.qpos[0].detach().cpu().numpy()
            renderer_data.qvel[:] = 0.0
            mujoco.mj_forward(renderer[2], renderer_data)
            renderer[0].update_scene(renderer_data, camera=cam)
            frames.append(renderer[0].render())
    if video_path is not None:
        with imageio.get_writer(video_path, fps=fps, quality=7) as w:
            for f in frames:
                w.append_data(f)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="fetch_warp")
    p.add_argument("--steps", type=int, default=20_000_000_000)
    p.add_argument("--worlds", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--ent-floor", type=float, default=-1.2)
    p.add_argument("--ent-ceil", type=float, default=0.0)
    p.add_argument("--ent-anneal-steps", type=int, default=400_000_000)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--plain", action="store_true",
                   help="SimpleActorCritic MLP baseline instead of the latent bottleneck")
    # Scaled-arena variant (worm comparison): shrink the arena and the ball's
    # spawn energy together. None/defaults = the faithful suite task.
    p.add_argument("--floor-size", type=float, default=None)
    p.add_argument("--ball-drop-z", type=float, default=2.0)
    p.add_argument("--ball-kick-std", type=float, default=5.0)
    # Contact buffer sizes PER WORLD. mjw's convex narrowphase allocates an
    # EPA buffer proportional to worlds x nconmax; at 2048 worlds the 256
    # default spikes >1.1 GB in one grab and OOMs a shared 16 GB card. The
    # quadruped realistically makes a few dozen contacts per world.
    p.add_argument("--nconmax", type=int, default=256)
    p.add_argument("--njmax", type=int, default=1024)
    p.add_argument("--max-hours", type=float, default=10.0)
    p.add_argument("--video-secs", type=float, default=1200.0)
    p.add_argument("--first-video-secs", type=float, default=120.0)
    p.add_argument("--ckpt-secs", type=float, default=1800.0)
    p.add_argument("--gcs-bucket", default="vc2-2026-checkpoints")
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    from rower_soccer.warp_port.fetch_env import WarpFetchEnv
    from rower_soccer.warp_port.ppo import (ActorCritic, SimpleActorCritic,
                                            PPOTrainer, save_checkpoint,
                                            load_checkpoint,
                                            export_sb3_compatible)

    run_dir = os.path.join("runs_v2", args.run_name)
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    cfg = {**vars(args), "backend": "mujoco_warp", "task": "quadruped_fetch"}
    cfg["git_sha"] = git_sha
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name,
                   id=args.run_name, resume="allow", config=cfg)
        wandb.define_metric("env_step")
        wandb.define_metric("*", step_metric="env_step")

    env_kw = dict(floor_size=args.floor_size, ball_drop_z=args.ball_drop_z,
                  ball_kick_std=args.ball_kick_std,
                  nconmax=args.nconmax, njmax=args.njmax)
    env = WarpFetchEnv(num_worlds=args.worlds, seed=args.seed, **env_kw)
    if args.plain:
        ac = SimpleActorCritic(env.obs_dim, env.act_dim)
    else:
        ac = ActorCritic(env.obs_dim, env.act_dim,
                         proprio_indices=env.proprio_indices.tolist(),
                         task_indices=env.task_indices.tolist(),
                         z_dim=args.z_dim)
    trainer = PPOTrainer(env, ac, lr=args.lr, rollout_len=args.rollout,
                         ent_coef=args.ent_coef, ent_floor=args.ent_floor,
                         ent_ceil=args.ent_ceil,
                         ent_anneal_steps=args.ent_anneal_steps)

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    latest_path = os.path.join(run_dir, "latest.pt")
    best_path = os.path.join(run_dir, "best.pt")
    start_steps, best_score = 0, -np.inf
    if args.resume and os.path.exists(ckpt_path):
        start_steps = load_checkpoint(trainer, ckpt_path)
        print(f"[setup] resumed at step {start_steps:,}", flush=True)

    # Eval: 1-world Warp env + a render-only MjData on the same model.
    eval_env = WarpFetchEnv(num_worlds=1, seed=7, use_graph=False, **env_kw)
    rmodel = eval_env.model
    rdata = mujoco.MjData(rmodel)
    renderer = (mujoco.Renderer(rmodel, height=240, width=320), rdata, rmodel)

    print(f"[setup] worlds={env.n} obs={env.obs_dim} act={env.act_dim} "
          f"arch={'plain' if args.plain else 'latent'} "
          f"steps/iter={trainer.T * trainer.N:,}", flush=True)

    t0 = time.perf_counter()
    deadline = t0 + args.max_hours * 3600.0
    last_ckpt = t0
    last_video = t0 - max(0.0, args.video_secs - args.first_video_secs)
    it = 0
    while trainer.total_steps < args.steps and time.perf_counter() < deadline:
        stats = trainer.train_iter()
        it += 1
        now = time.perf_counter()
        fps = (trainer.total_steps - start_steps) / (now - t0)
        if it % 5 == 0:
            fit = float(env.fitness().mean())
            print(f"[monitor] step={trainer.total_steps:,} fps={fps:,.0f} "
                  f"eta={(deadline-now)/60:.1f}min "
                  f"ep_rew={stats['ep_rew_env_mean']:.1f} (max 1000) "
                  f"reward_now={fit:.3f} std={stats['std']:.3f} "
                  f"diverged={trainer.n_diverged:,}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"env_step": trainer.total_steps, "monitor/fps": fps,
                           "train/ep_rew": stats["ep_rew_env_mean"],
                           "train/reward_now": fit,
                           "train/entropy": stats["ent"], "train/std": stats["std"],
                           "train/pg_loss": stats["pg"], "train/vf_loss": stats["vf"]})

        if args.video_secs > 0 and now - last_video >= args.video_secs:
            last_video = now
            vpath = os.path.join(run_dir, "videos",
                                 f"eval_step_{trainer.total_steps:010d}.mp4")
            score = eval_episode(eval_env, ac, vpath, renderer)
            print(f"[monitor] video: {vpath} (WARP eval return={score:.1f}/1000)",
                  flush=True)
            if score > best_score:
                best_score = score
                export_sb3_compatible(ac, best_path)
                print(f"[monitor] new BEST return {best_score:.1f} -> {best_path}",
                      flush=True)
                if args.gcs_bucket:
                    from rower_soccer.warp_port.gcs import sync_async
                    sync_async(best_path, args.gcs_bucket, args.run_name)
            if use_wandb:
                import wandb
                wandb.log({"env_step": trainer.total_steps,
                           "eval/video": wandb.Video(vpath, format="mp4"),
                           "eval/return_warp": score})

        if now - last_ckpt >= args.ckpt_secs:
            last_ckpt = now
            save_checkpoint(trainer, ckpt_path)
            export_sb3_compatible(ac, latest_path)
            print(f"[monitor] checkpoint saved at step {trainer.total_steps:,}",
                  flush=True)
            if args.gcs_bucket:
                from rower_soccer.warp_port.gcs import sync_async
                for f_ in (ckpt_path, latest_path,
                           os.path.join(run_dir, "config.json")):
                    sync_async(f_, args.gcs_bucket, args.run_name)

    save_checkpoint(trainer, ckpt_path)
    export_sb3_compatible(ac, os.path.join(run_dir, "final.pt"))
    if args.gcs_bucket:
        from rower_soccer.warp_port.gcs import sync_async
        sync_async(ckpt_path, args.gcs_bucket, args.run_name)
        sync_async(os.path.join(run_dir, "final.pt"), args.gcs_bucket, args.run_name)
    print(f"[setup] done in {(time.perf_counter()-t0)/60:.1f}min; saved final.pt",
          flush=True)


if __name__ == "__main__":
    main()
