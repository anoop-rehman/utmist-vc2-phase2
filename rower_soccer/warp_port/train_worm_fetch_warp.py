"""PPO on fetch for OUR worm (see worm_fetch_env.py).

    MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_worm_fetch_warp \
        --run-name fetch_worm_arena --scene arena \
        --init-from runs_v2/_init_follow_base.pt
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="fetch_worm")
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
    p.add_argument("--plain", action="store_true")
    p.add_argument("--init-from", default=None,
                   help="follow/dribble checkpoint; the decoder transfers "
                        "(identical 29-dim proprio contract)")
    p.add_argument("--scene", choices=["arena", "pitch"], default="arena")
    p.add_argument("--floor-half", type=float, default=5.0)
    p.add_argument("--spawn-frac", type=float, default=0.9)
    p.add_argument("--ball-drop-z", type=float, default=1.0)
    p.add_argument("--ball-kick-std", type=float, default=1.5)
    p.add_argument("--creature-xml", default="creature_configs/three_seg_worm.xml")
    p.add_argument("--up-axis-json",
                   default="creature_configs/three_seg_worm_up_axis.json")
    p.add_argument("--max-hours", type=float, default=10.0)
    p.add_argument("--video-secs", type=float, default=1200.0)
    p.add_argument("--first-video-secs", type=float, default=120.0)
    p.add_argument("--ckpt-secs", type=float, default=1800.0)
    p.add_argument("--gcs-bucket", default="vc2-2026-checkpoints")
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    from rower_soccer.warp_port.worm_fetch_env import (WarpWormFetchEnv,
                                                       fetch_ball, _arena_xml)
    from rower_soccer.warp_port.render import WarpRenderer, eval_video
    from rower_soccer.warp_port.ppo import (ActorCritic, SimpleActorCritic,
                                            PPOTrainer, save_checkpoint,
                                            load_checkpoint, load_pretrained,
                                            export_sb3_compatible)

    run_dir = os.path.join("runs_v2", args.run_name)
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    cfg = {**vars(args), "backend": "mujoco_warp", "task": "worm_fetch",
           "git_sha": git_sha}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name,
                   id=args.run_name, resume="allow", config=cfg)
        wandb.define_metric("env_step")
        wandb.define_metric("*", step_metric="env_step")

    env_kw = dict(creature_xml=args.creature_xml, up_axis_json=args.up_axis_json,
                  scene=args.scene, floor_half=args.floor_half,
                  spawn_frac=args.spawn_frac, ball_drop_z=args.ball_drop_z,
                  ball_kick_std=args.ball_kick_std)
    env = WarpWormFetchEnv(num_worlds=args.worlds, seed=args.seed, **env_kw)
    if args.plain:
        ac = SimpleActorCritic(env.obs_dim, env.act_dim)
    else:
        ac = ActorCritic(env.obs_dim, env.act_dim,
                         proprio_indices=env.proprio_indices.tolist(),
                         task_indices=env.task_indices.tolist(), z_dim=args.z_dim)
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
    elif args.init_from:
        load_pretrained(ac, args.init_from, device=trainer.device)

    eval_env = WarpWormFetchEnv(num_worlds=1, seed=7, use_graph=False, **env_kw)
    base = _arena_xml(args.floor_half) if args.scene == "arena" else None
    eval_ren = WarpRenderer(args.creature_xml, has_ball=True,
                            base_xml=base, ball=fetch_ball())

    print(f"[setup] worlds={env.n} obs={env.obs_dim} act={env.act_dim} "
          f"scene={args.scene} arch={'plain' if args.plain else 'latent'} "
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
                  f"ep_rew={stats['ep_rew_env_mean']:.1f} (max {env.episode_steps}) "
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
            ep_rew, _ = eval_video(eval_env, ac, vpath, eval_ren)
            print(f"[monitor] video: {vpath} (WARP eval return={ep_rew:.1f}"
                  f"/{env.episode_steps})", flush=True)
            if ep_rew > best_score:
                best_score = ep_rew
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
                           "eval/return_warp": ep_rew})

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
    print(f"[setup] done in {(time.perf_counter()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
