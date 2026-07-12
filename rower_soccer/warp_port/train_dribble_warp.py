"""Warp-accelerated dribble-drill training, warm-started from a follow policy.

Single-task and unconstrained: the decoder is NOT frozen and follow is NOT
replayed, so the policy is free to forget follow entirely. That is deliberate --
this run answers one question, "can dribble be trained at all on this body and
this ball", before any of the shared-decoder multitask machinery
(docs/STAGE2_MULTITASK.md) is worth building.

The follow checkpoint transfers everything except the task encoder and the
critic's first layer (dribble's task obs is 8 wide, follow's is 4). The decoder
-- the actual motor skill -- transfers unchanged, since it only ever sees
proprio + z.

Usage:
  MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_dribble_warp \
      --run-name dribble_v1 --init-from runs_v2/warp_C_velshape_slowtgt/latest.pt \
      --steps 500000000 --gcs-bucket vc2-2026-checkpoints
"""

import argparse
import json
import os
import shutil
import subprocess
import time

import imageio
import torch


def cpu_eval_video(ac, path, seed=7, deterministic=True, **task_kw):
    """Run current weights in the dm_control dribble env; render, return
    (ep_reward, final unshaped fitness). Doubles as the Warp->dm_control
    physics transfer check."""
    from rower_soccer.drills.dribble import make_dribble_env
    from rower_soccer.drills.gym_wrap import DrillGymEnv
    env = getattr(cpu_eval_video, "_env", None)
    if env is None:
        env = cpu_eval_video._env = DrillGymEnv(
            lambda random_state=None: make_dribble_env(random_state=random_state,
                                                       **task_kw), seed=seed)
    obs, _ = env.reset()
    done, ep_rew, frames = False, 0.0, []
    while not done:
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device="cuda").unsqueeze(0)
            # d.mean, never d.sample: at inference the action head's log_std is
            # exploration noise we do not want.
            a = ac.dist(o).mean if deterministic else ac.dist(o).sample()
        obs, r, term, trunc, _ = env.step(a.squeeze(0).clamp(-1, 1).cpu().numpy())
        done = term or trunc
        ep_rew += r
        frames.append(env.render())
    fitness = env._env.task.get_fitness(env._env.physics)
    with imageio.get_writer(path, fps=40, quality=7) as w:
        for f in frames:
            w.append_data(f)
    return ep_rew, fitness


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000_000)
    p.add_argument("--worlds", type=int, default=2048)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--ent-floor", type=float, default=None)
    p.add_argument("--ent-ceil", type=float, default=0.0,
                   help="max log_std (default 0.0 => std<=1.0, matching the "
                        "[-1,1] action clamp); pass a large value to disable")
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--init-from", default=None,
                   help="follow checkpoint to warm-start from (checkpoint.pt or "
                        "latest.pt). Task encoder + critic input layer re-init; "
                        "the decoder (the low-level controller) carries over.")
    p.add_argument("--target-speed", type=float, nargs=2, default=[0.1, 1.0],
                   help="dribbling is harder than following, so the target is "
                        "slower than follow's [0.25, 2.0]")
    p.add_argument("--bounds", type=float, default=27.0)
    p.add_argument("--ball-spawn", type=float, nargs=2, default=[5.0, 8.0],
                   help="ball spawn distance from the worm (m); must clear its "
                        "4.65 m footprint radius")
    p.add_argument("--target-dist", type=float, nargs=2, default=[2.0, 6.0],
                   help="target spawn distance from the BALL (m), not from the "
                        "worm: anchoring it to the worm leaves ball and target "
                        "~13 m apart, where exp(-c*d) is flat zero and the drill "
                        "has no gradient at all")
    p.add_argument("--reward-coef", type=float, default=0.5)
    p.add_argument("--reward-mode", default="paper", choices=["paper", "progress"])
    p.add_argument("--progress-scale", type=float, default=2.0)
    p.add_argument("--approach-scale", type=float, default=0.5,
                   help="progress mode: weight on the player->ball potential. "
                        "Without it nothing rewards walking to the ball and the "
                        "ball->target term stays identically zero forever")
    p.add_argument("--w-player-to-ball", type=float, default=0.1)
    p.add_argument("--w-ball-to-target", type=float, default=0.3)
    p.add_argument("--episode-secs", type=float, default=15.0)
    p.add_argument("--run-name", required=True)
    p.add_argument("--video-secs", type=float, default=300.0)
    p.add_argument("--ckpt-secs", type=float, default=1800.0)
    p.add_argument("--mid-ckpt-frac", type=float, default=0.5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gcs-bucket", default=None)
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    run_dir = os.path.join("runs_v2", args.run_name)
    if os.path.isdir(run_dir) and os.listdir(run_dir) and not args.resume:
        p.error(f"{run_dir} exists and is non-empty. Pass --resume to continue "
                f"that run, or pick a different --run-name.")
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    config = {**vars(args), "git_sha": git_sha, "backend": "mujoco_warp",
              "task": "dribble"}
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        n = sum(f.startswith("config_resume_") for f in os.listdir(run_dir))
        cfg_path = os.path.join(run_dir, f"config_resume_{n + 1}.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=1)

    final_path = os.path.join(run_dir, "final.pt")
    if os.path.exists(final_path):
        os.remove(final_path)
        print(f"[setup] removed stale {final_path} (from a previous run)", flush=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=config,
                   dir=run_dir, id=args.run_name.replace("/", "-"), resume="allow")
        wandb.define_metric("env_step")
        wandb.define_metric("*", step_metric="env_step")

    from rower_soccer.warp_port.dribble_env import WarpDribbleEnv
    from rower_soccer.warp_port.ppo import (ActorCritic, PPOTrainer,
                                            export_sb3_compatible,
                                            load_checkpoint, load_pretrained,
                                            save_checkpoint)

    env = WarpDribbleEnv(num_worlds=args.worlds,
                         target_speed_range=tuple(args.target_speed),
                         reward_coef=args.reward_coef,
                         episode_seconds=args.episode_secs,
                         reward_mode=args.reward_mode,
                         progress_scale=args.progress_scale,
                         approach_scale=args.approach_scale,
                         w_player_to_ball=args.w_player_to_ball,
                         w_ball_to_target=args.w_ball_to_target,
                         bounds=args.bounds,
                         target_dist_range=tuple(args.target_dist),
                         ball_spawn_range=tuple(args.ball_spawn))
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(), z_dim=args.z_dim)
    trainer = PPOTrainer(env, ac, lr=args.lr, rollout_len=args.rollout,
                         ent_coef=args.ent_coef, ent_floor=args.ent_floor,
                         ent_ceil=args.ent_ceil)

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    latest_path = os.path.join(run_dir, "latest.pt")
    mid_path = os.path.join(run_dir, "checkpoint_mid.pt")
    mid_target = int(args.steps * args.mid_ckpt_frac) if args.mid_ckpt_frac else 0
    start_steps = 0
    if args.resume and os.path.exists(ckpt_path):
        start_steps = load_checkpoint(trainer, ckpt_path)
        print(f"[setup] resumed from {ckpt_path} at step {start_steps:,}", flush=True)
    elif args.init_from:
        # Warm start only on a fresh run: on --resume the checkpoint already
        # contains these weights (further trained), and re-seeding from follow
        # would throw away the dribble progress it holds.
        load_pretrained(ac, args.init_from, device=trainer.device)

    print(f"[setup] worlds={env.n} obs={env.obs_dim} act={env.act_dim} "
          f"proprio={len(env.proprio_indices)} task={len(env.task_indices)} "
          f"steps/iter={trainer.T * trainer.N:,}", flush=True)
    t0 = time.perf_counter()
    last_video = last_ckpt = t0
    it = 0
    while trainer.total_steps < args.steps:
        stats = trainer.train_iter()
        it += 1
        now = time.perf_counter()
        fps = (trainer.total_steps - start_steps) / (now - t0)
        eta_min = (args.steps - trainer.total_steps) / fps / 60
        if it % 5 == 0:
            fit = float(env.fitness().mean())
            print(f"[monitor] step={trainer.total_steps:,}/{args.steps:,} "
                  f"({100*trainer.total_steps/args.steps:.1f}%) fps={fps:,.0f} "
                  f"eta={eta_min:.1f}min ep_rew={stats['ep_rew_env_mean']:.1f} "
                  f"fitness={fit:.3f} std={stats['std']:.3f}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"env_step": trainer.total_steps,
                           "monitor/fps": fps, "monitor/eta_min": eta_min,
                           "train/ep_rew": stats["ep_rew_env_mean"],
                           # Unshaped Table-S3 fitness: the gate metric, and the
                           # one number the velocity shaping terms cannot inflate.
                           "train/fitness": fit,
                           "train/entropy": stats["ent"], "train/std": stats["std"],
                           "train/pg_loss": stats["pg"], "train/vf_loss": stats["vf"]})
        if args.video_secs > 0 and now - last_video >= args.video_secs:
            last_video = now
            vpath = os.path.join(run_dir, "videos",
                                 f"eval_step_{trainer.total_steps:010d}.mp4")
            ep_rew, fit = cpu_eval_video(
                ac, vpath, target_speed_range=tuple(args.target_speed),
                ball_spawn_range=tuple(args.ball_spawn),
                target_dist_range=tuple(args.target_dist))
            print(f"[monitor] video: {vpath} (dm_control transfer eval "
                  f"ep_rew={ep_rew:.1f} fitness={fit:.3f})", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"env_step": trainer.total_steps,
                           "eval/video": wandb.Video(vpath, format="mp4"),
                           "eval/ep_rew_dm_control": ep_rew,
                           "eval/fitness_dm_control": fit})
        if now - last_ckpt >= args.ckpt_secs:
            last_ckpt = now
            save_checkpoint(trainer, ckpt_path)
            export_sb3_compatible(ac, latest_path)
            print(f"[monitor] checkpoint saved at step {trainer.total_steps:,} "
                  f"({os.path.getsize(ckpt_path)/1e6:.1f} MB, overwrite)", flush=True)
            wrote_mid = False
            if mid_target and not os.path.exists(mid_path) \
                    and trainer.total_steps >= mid_target:
                shutil.copy2(ckpt_path, mid_path)
                wrote_mid = True
                print(f"[monitor] rollback copy -> {mid_path}", flush=True)
            if args.gcs_bucket:
                from rower_soccer.warp_port.gcs import sync_async
                for path in (ckpt_path, cfg_path, latest_path):
                    sync_async(path, args.gcs_bucket, args.run_name)
                if wrote_mid:
                    sync_async(mid_path, args.gcs_bucket, args.run_name)

    save_checkpoint(trainer, ckpt_path)
    export_sb3_compatible(ac, latest_path)
    export_sb3_compatible(ac, final_path)
    if args.gcs_bucket:
        from rower_soccer.warp_port.gcs import sync_blocking, wait_all
        wait_all()
        for path in (cfg_path, ckpt_path, latest_path, final_path):
            sync_blocking(path, args.gcs_bucket, args.run_name)
    print(f"[setup] done in {(time.perf_counter()-t0)/60:.1f}min; saved final.pt",
          flush=True)


if __name__ == "__main__":
    main()
