"""Warp-accelerated follow-drill training with transfer-eval videos.

Every --video-secs of wallclock, current weights are loaded into the CPU
dm_control env and an eval episode is rendered — monitoring learning AND
Warp->dm_control physics transfer in one artifact.

Usage:
  MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_follow_warp \
      --steps 20000000 --worlds 2048 --run-name follow_warp_v1
"""

import argparse
import json
import os
import subprocess
import time

import imageio
import numpy as np
import torch


def cpu_eval_video(ac, path, seed=7, deterministic=True):
    """Runs current weights in the dm_control env; returns ep reward."""
    from rower_soccer.drills.follow import make_follow_env
    from rower_soccer.drills.gym_wrap import DrillGymEnv
    env = getattr(cpu_eval_video, "_env", None)
    if env is None:
        env = cpu_eval_video._env = DrillGymEnv(make_follow_env, seed=seed)
    obs, _ = env.reset()
    done, ep_rew, frames = False, 0.0, []
    while not done:
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device="cuda").unsqueeze(0)
            d = ac.dist(o)
            a = d.mean if deterministic else d.sample()
        obs, r, term, trunc, _ = env.step(a.squeeze(0).clamp(-1, 1).cpu().numpy())
        done = term or trunc
        ep_rew += r
        frames.append(env.render())
    with imageio.get_writer(path, fps=40, quality=7) as w:
        for f in frames:
            w.append_data(f)
    return ep_rew


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--worlds", type=int, default=2048)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--ent-floor", type=float, default=None)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--target-speed", type=float, nargs=2, default=[0.25, 2.0])
    p.add_argument("--reward-coef", type=float, default=0.5)
    p.add_argument("--vel-shaping", type=float, default=0.0)
    p.add_argument("--episode-secs", type=float, default=15.0)
    p.add_argument("--run-name", required=True)
    p.add_argument("--video-secs", type=float, default=300.0)
    p.add_argument("--ckpt-secs", type=float, default=1800.0,
                   help="wallclock seconds between full checkpoints (overwrite)")
    p.add_argument("--resume", action="store_true",
                   help="resume from <run_dir>/checkpoint.pt if present")
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    run_dir = os.path.join("runs_v2", args.run_name)
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    config = {**vars(args), "git_sha": git_sha, "backend": "mujoco_warp"}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=config,
                   dir=run_dir, id=args.run_name.replace("/", "-"), resume="allow")

    from rower_soccer.warp_port.follow_env import WarpFollowEnv
    from rower_soccer.warp_port.ppo import (ActorCritic, PPOTrainer,
                                            export_sb3_compatible,
                                            load_checkpoint, save_checkpoint)

    env = WarpFollowEnv(num_worlds=args.worlds,
                        target_speed_range=tuple(args.target_speed),
                        reward_coef=args.reward_coef,
                        episode_seconds=args.episode_secs,
                        w_vel_shaping=args.vel_shaping)
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(), z_dim=args.z_dim)
    trainer = PPOTrainer(env, ac, lr=args.lr, rollout_len=args.rollout,
                         ent_coef=args.ent_coef, ent_floor=args.ent_floor)

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    start_steps = 0
    if args.resume and os.path.exists(ckpt_path):
        start_steps = load_checkpoint(trainer, ckpt_path)
        print(f"[setup] resumed from {ckpt_path} at step {start_steps:,}", flush=True)

    print(f"[setup] worlds={env.n} obs={env.obs_dim} act={env.act_dim} "
          f"steps/iter={trainer.T * trainer.N:,}", flush=True)
    t0 = time.perf_counter()
    last_video = t0
    last_ckpt = t0
    it = 0
    while trainer.total_steps < args.steps:
        stats = trainer.train_iter()
        it += 1
        now = time.perf_counter()
        fps = (trainer.total_steps - start_steps) / (now - t0)
        eta_min = (args.steps - trainer.total_steps) / fps / 60
        if it % 5 == 0:
            print(f"[monitor] step={trainer.total_steps:,}/{args.steps:,} "
                  f"({100*trainer.total_steps/args.steps:.1f}%) fps={fps:,.0f} "
                  f"eta={eta_min:.1f}min ep_rew={stats['ep_rew_env_mean']:.1f} "
                  f"std={stats['std']:.3f}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"monitor/fps": fps, "monitor/eta_min": eta_min,
                           "train/ep_rew": stats["ep_rew_env_mean"],
                           "train/entropy": stats["ent"], "train/std": stats["std"],
                           "train/pg_loss": stats["pg"], "train/vf_loss": stats["vf"]},
                          step=trainer.total_steps)
        if args.video_secs > 0 and now - last_video >= args.video_secs:
            last_video = now
            vpath = os.path.join(run_dir, "videos",
                                 f"eval_step_{trainer.total_steps:010d}.mp4")
            ep_rew = cpu_eval_video(ac, vpath)
            print(f"[monitor] video: {vpath} (dm_control transfer eval "
                  f"ep_rew={ep_rew:.1f})", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"eval/video": wandb.Video(vpath, format="mp4"),
                           "eval/ep_rew_dm_control": ep_rew},
                          step=trainer.total_steps)
            export_sb3_compatible(ac, os.path.join(run_dir, "latest.pt"))
        if now - last_ckpt >= args.ckpt_secs:
            last_ckpt = now
            save_checkpoint(trainer, ckpt_path)
            print(f"[monitor] checkpoint saved at step {trainer.total_steps:,} "
                  f"({os.path.getsize(ckpt_path)/1e6:.1f} MB, overwrite)", flush=True)

    save_checkpoint(trainer, ckpt_path)
    export_sb3_compatible(ac, os.path.join(run_dir, "final.pt"))
    print(f"[setup] done in {(time.perf_counter()-t0)/60:.1f}min; saved final.pt",
          flush=True)


if __name__ == "__main__":
    main()
