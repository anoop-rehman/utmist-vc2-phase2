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
import shutil
import subprocess
import time

import imageio
import numpy as np
import torch


def make_eval(args, has_ball=False):
    """One-world Warp env + renderer, built once and reused.

    Warp is ground truth: the eval runs in the SAME simulator the policy trains in,
    and the video is drawn from that simulator's state. The dm_control CPU drill is
    no longer in the loop -- see warp_port/render.py for why.
    """
    from rower_soccer.warp_port.follow_env import WarpFollowEnv
    from rower_soccer.warp_port.render import WarpRenderer
    env = WarpFollowEnv(
        num_worlds=1, use_graph=False, seed=7,
        target_speed_range=tuple(args.target_speed),
        spawn_dist_range=tuple(args.spawn_dist),
        bounds=args.bounds, reward_coef=args.reward_coef,
        w_vel_shaping=args.vel_shaping, reward_mode=args.reward_mode,
        progress_scale=args.progress_scale, episode_seconds=args.episode_secs,
        energy_coef=args.energy_coef, smooth_coef=args.smooth_coef)
    return env, WarpRenderer(args.creature_xml, has_ball=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--worlds", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0,
                   help="env + torch init seed; vary for replica runs")
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--ent-floor", type=float, default=None)
    p.add_argument("--ent-ceil", type=float, default=0.0,
                   help="max log_std (default 0.0 => std<=1.0, matching the "
                        "[-1,1] action clamp); pass a large value to disable")
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--init-from", default=None,
                   help="checkpoint to warm-start weights from (checkpoint.pt or "
                        "latest.pt), e.g. a follow policy trained on the same "
                        "body at a different mass scale. Weights only, fresh "
                        "optimizer. Ignored when --resume finds a checkpoint.")
    # Froude-scale of C's [0.1, 0.8] (sqrt(0.1768) = 0.4205), NOT of the old
    # [0.25, 2.0] default -- that is the abandoned FAST target earlier runs failed
    # on, and C's "slowtgt" name is precisely that finding. Keeps target speed at
    # ~0.3x the worm's achievable speed, matching C's ratio.
    # Calibrated against the ONE follow run that ever worked (warp_C_velshape_slowtgt:
    # 445-495 reward, follows to within 0.5-1.3 m). What matters is not the absolute
    # speed but target_max / achievable_speed -- the margin the worm has to catch AND
    # hold the target while turning and correcting:
    #
    #   C            0.80 / 2.830 = 0.28   <- worked
    #   follow_s176  0.85 / 1.040 = 0.82   <- failed, plateaued at 182/600
    #   follow_v4    0.34 / 0.759 = 0.45   <- stuck in the do-nothing optimum
    #
    # probe_speed is NONDETERMINISTIC run to run -- the worm spawns as an unstable
    # vertical stack and topples chaotically, which amplifies float nondeterminism, so
    # even a fixed seed varies. Measured spread: 0.76 / 0.87 / 0.87 / 0.88 / 1.06 /
    # 1.32 / 1.50 m/s. Never calibrate off one sample.
    #
    # Target speed uses the MINIMUM (0.76), deliberately: a target the worm cannot
    # physically catch makes the drill unlearnable, with nothing in the training loop
    # to say so, while a slightly-too-slow target is merely easy. 0.283 * 0.76 = 0.215.
    #
    # Do NOT take probe_speed's old "80% of achievable" suggestion. That is exactly the
    # number that produced follow_s176.
    p.add_argument("--target-speed", type=float, nargs=2, default=[0.03, 0.21])
    p.add_argument("--bounds", type=float, default=10.0,
                   help="target roaming half-extent (m)")
    p.add_argument("--spawn-dist", type=float, nargs=2, default=[1.76, 5.28],
                   help="target spawn distance (m): 1-3 body lengths")
    p.add_argument("--reward-coef", type=float, default=0.5)
    # Realism regularizers (default OFF -> baseline is unchanged). Energy penalises
    # brute thrust; smooth is CAPS temporal smoothness (penalises jerk, not speed).
    p.add_argument("--energy-coef", type=float, default=0.0)
    p.add_argument("--smooth-coef", type=float, default=0.0)
    # Anneal the entropy bonus to 0 over this many env-steps (0 = constant). Fixes
    # the late-training entropy runaway that collapsed follow_v5.
    p.add_argument("--ent-anneal-steps", type=int, default=0)
    # NOT 0.0. The bare `paper` reward is exp(-c*dist), which pays a worm for standing
    # still and gives it almost no gradient to discover locomotion -- follow_v4 sat in
    # that do-nothing optimum for 800M steps (ep_rew 134 vs ~130 for doing nothing).
    # Every follow run that ever learned had a dense per-step locomotion signal:
    # C used paper + vel_shaping, follow_v2 used reward_mode=progress. A run with
    # NEITHER cannot learn, and that is what follow_v4 was.
    #
    # Scaled to preserve C's shaping magnitude, since our worm is slower. This one uses
    # the MEDIAN achievable speed (~0.90 m/s), not the minimum: it sets how strong the
    # shaping is in typical motion, and calibrating it off the pessimistic tail would
    # over-shape by ~2x at the top of the speed range.
    #   C:    0.05 * 2.830 m/s = 0.1415 reward/step at full speed
    #   ours: 0.15 * 0.900 m/s = 0.1350  ->  w = 0.15
    p.add_argument("--vel-shaping", type=float, default=0.15)
    p.add_argument("--reward-mode", default="paper",
                   choices=["paper", "velshape", "progress"])
    p.add_argument("--progress-scale", type=float, default=2.0)
    p.add_argument("--episode-secs", type=float, default=15.0)
    # Wall-clock budget. Runs are sized in HOURS, not steps: throughput swings with
    # how many runs share the GPU (69k steps/s alone, ~50k with two), so a step target
    # is really an unpredictable time target. --steps stays as a backstop.
    p.add_argument("--max-hours", type=float, default=48.0,
                   help="stop after this much wallclock, whatever step count that is")
    p.add_argument("--creature-xml",
                   default="creature_configs/three_seg_worm.xml")
    p.add_argument("--run-name", required=True)
    p.add_argument("--video-secs", type=float, default=300.0)
    # Fire the FIRST transfer-eval video early, so a broken run (bad obs layout,
    # bad reward, creature glitching) is visible in minutes instead of after the
    # first full --video-secs interval. Subsequent videos keep the normal cadence.
    p.add_argument("--first-video-secs", type=float, default=60.0)
    p.add_argument("--ckpt-secs", type=float, default=1800.0,
                   help="wallclock seconds between full checkpoints (overwrite)")
    p.add_argument("--mid-ckpt-frac", type=float, default=0.5,
                   help="write a one-shot rollback copy (checkpoint_mid.pt) at "
                        "the first checkpoint past this fraction of --steps; "
                        "0 disables")
    p.add_argument("--resume", action="store_true",
                   help="resume from <run_dir>/checkpoint.pt if present")
    p.add_argument("--gcs-bucket", default=None,
                   help="upload each checkpoint to gs://<bucket>/<run_name>/ "
                        "(e.g. vc2-2026-checkpoints); best-effort, non-blocking")
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    run_dir = os.path.join("runs_v2", args.run_name)
    # Reusing a run name without --resume silently mixes artifacts from two
    # different runs into one directory (and one GCS prefix): config.json gets
    # clobbered, and a final.pt left by the earlier run outlives the later one.
    if os.path.isdir(run_dir) and os.listdir(run_dir) and not args.resume:
        p.error(f"{run_dir} exists and is non-empty. Pass --resume to continue "
                f"that run, or pick a different --run-name.")
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    config = {**vars(args), "git_sha": git_sha, "backend": "mujoco_warp"}
    # Never overwrite the originating run's config: each resume leg records its
    # own args/git_sha alongside it, so provenance survives.
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        n = sum(f.startswith("config_resume_") for f in os.listdir(run_dir))
        cfg_path = os.path.join(run_dir, f"config_resume_{n + 1}.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=1)

    # final.pt is only written on a clean exit, so any copy present now belongs
    # to an earlier run under this name. Drop it rather than let its
    # authoritative-sounding name outrank the checkpoint we are about to train.
    final_path = os.path.join(run_dir, "final.pt")
    if os.path.exists(final_path):
        os.remove(final_path)
        print(f"[setup] removed stale {final_path} (from a previous run)",
              flush=True)
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=config,
                   dir=run_dir, id=args.run_name.replace("/", "-"), resume="allow")
        # Use env_step as an explicit x-axis metric and let wandb's own internal
        # step auto-increment. On resume the checkpoint's env_step may be behind
        # wandb's internal counter (e.g. a prior run logged further before being
        # killed without checkpointing); logging against wandb's auto step avoids
        # the monotonic-step drop that silently discards replayed points.
        wandb.define_metric("env_step")
        wandb.define_metric("*", step_metric="env_step")

    from rower_soccer.warp_port.follow_env import WarpFollowEnv
    from rower_soccer.warp_port.ppo import (ActorCritic, PPOTrainer,
                                            export_sb3_compatible,
                                            load_checkpoint, load_pretrained,
                                            save_checkpoint)

    env = WarpFollowEnv(num_worlds=args.worlds, seed=args.seed,
                        target_speed_range=tuple(args.target_speed),
                        reward_coef=args.reward_coef,
                        episode_seconds=args.episode_secs,
                        w_vel_shaping=args.vel_shaping,
                        reward_mode=args.reward_mode,
                        progress_scale=args.progress_scale,
                        bounds=args.bounds,
                        spawn_dist_range=tuple(args.spawn_dist),
                        energy_coef=args.energy_coef, smooth_coef=args.smooth_coef)
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(), z_dim=args.z_dim)
    trainer = PPOTrainer(env, ac, lr=args.lr, rollout_len=args.rollout,
                         ent_coef=args.ent_coef, ent_floor=args.ent_floor,
                         ent_ceil=args.ent_ceil,
                         ent_anneal_steps=args.ent_anneal_steps)

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    latest_path = os.path.join(run_dir, "latest.pt")
    mid_path = os.path.join(run_dir, "checkpoint_mid.pt")
    best_path = os.path.join(run_dir, "best.pt")
    best_score = float("-inf")
    mid_target = int(args.steps * args.mid_ckpt_frac) if args.mid_ckpt_frac else 0
    start_steps = 0
    if args.resume and os.path.exists(ckpt_path):
        start_steps = load_checkpoint(trainer, ckpt_path)
        print(f"[setup] resumed from {ckpt_path} at step {start_steps:,}", flush=True)
    elif args.init_from:
        # Fresh run only: on a real --resume the checkpoint already holds these
        # weights, further trained, and re-seeding would throw that away.
        load_pretrained(ac, args.init_from, device=trainer.device)

    print(f"[setup] worlds={env.n} obs={env.obs_dim} act={env.act_dim} "
          f"steps/iter={trainer.T * trainer.N:,}", flush=True)
    eval_env, eval_ren = make_eval(args)
    t0 = time.perf_counter()
    # Back-date the video timer so the first one lands at --first-video-secs.
    last_video = t0 - max(0.0, args.video_secs - args.first_video_secs)
    last_ckpt = t0
    it = 0
    deadline = t0 + args.max_hours * 3600.0
    while trainer.total_steps < args.steps and time.perf_counter() < deadline:
        stats = trainer.train_iter()
        it += 1
        now = time.perf_counter()
        fps = (trainer.total_steps - start_steps) / (now - t0)
        # ETA is now the wall-clock deadline, not the step target.
        eta_min = max(0.0, (deadline - now) / 60)
        if it % 5 == 0:
            print(f"[monitor] step={trainer.total_steps:,}/{args.steps:,} "
                  f"({100*trainer.total_steps/args.steps:.1f}%) fps={fps:,.0f} "
                  f"eta={eta_min:.1f}min ep_rew={stats['ep_rew_env_mean']:.1f} "
                  f"std={stats['std']:.3f}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"env_step": trainer.total_steps,
                           "monitor/fps": fps, "monitor/eta_min": eta_min,
                           "train/ep_rew": stats["ep_rew_env_mean"],
                           "train/entropy": stats["ent"], "train/std": stats["std"],
                           "train/pg_loss": stats["pg"], "train/vf_loss": stats["vf"]})
        if args.video_secs > 0 and now - last_video >= args.video_secs:
            last_video = now
            vpath = os.path.join(run_dir, "videos",
                                 f"eval_step_{trainer.total_steps:010d}.mp4")
            from rower_soccer.warp_port.render import eval_video
            ep_rew, fit = eval_video(eval_env, ac, vpath, eval_ren)
            print(f"[monitor] video: {vpath} (WARP eval "
                  f"ep_rew={ep_rew:.1f} fitness={fit:.3f})", flush=True)
            # Keep the BEST policy, not just the latest.
            #
            # follow_v5_velshape's transfer eval went 262 -> 351 -> 465 -> 476.5 and
            # then COLLAPSED to 166.6 in its final stretch (log_std pinned at the
            # entropy ceiling). final.pt and latest.pt both hold the collapsed 166
            # policy. The 476 weights -- comfortably in C's 445-495 band -- existed,
            # were never saved, and are gone.
            #
            # Late collapse is not exotic in long PPO runs, and 48-hour runs give it
            # far more room. Scored on the DETERMINISTIC dm_control transfer eval,
            # which is the number we actually care about.
            if ep_rew > best_score:
                best_score = ep_rew
                export_sb3_compatible(ac, best_path)
                print(f"[monitor] new BEST transfer eval {best_score:.1f} "
                      f"-> {best_path}", flush=True)
                if args.gcs_bucket:
                    from rower_soccer.warp_port.gcs import sync_async
                    sync_async(best_path, args.gcs_bucket, args.run_name)
            if use_wandb:
                import wandb
                wandb.log({"env_step": trainer.total_steps,
                           "eval/video": wandb.Video(vpath, format="mp4"),
                           "eval/ep_rew_warp": ep_rew, "eval/fitness_warp": fit})
        if now - last_ckpt >= args.ckpt_secs:
            last_ckpt = now
            save_checkpoint(trainer, ckpt_path)
            # latest.pt is the weights-only view of checkpoint.pt, written in
            # the same breath so the two never disagree. It used to be exported
            # from the video block, which meant no videos => no latest.pt.
            export_sb3_compatible(ac, latest_path)
            print(f"[monitor] checkpoint saved at step {trainer.total_steps:,} "
                  f"({os.path.getsize(ckpt_path)/1e6:.1f} MB, overwrite)", flush=True)
            # One extra restore point, written once at the first checkpoint past
            # --mid-ckpt-frac. checkpoint.pt is overwritten in place, so without
            # this a policy collapse leaves nothing to roll back to.
            wrote_mid = False
            if mid_target and not os.path.exists(mid_path) \
                    and trainer.total_steps >= mid_target:
                shutil.copy2(ckpt_path, mid_path)
                wrote_mid = True
                print(f"[monitor] rollback copy -> {mid_path} at step "
                      f"{trainer.total_steps:,}", flush=True)
            if args.gcs_bucket:
                from rower_soccer.warp_port.gcs import sync_async
                sync_async(ckpt_path, args.gcs_bucket, args.run_name)
                sync_async(cfg_path, args.gcs_bucket, args.run_name)
                # latest.pt is the export used for inference; it was previously
                # written but never uploaded.
                sync_async(latest_path, args.gcs_bucket, args.run_name)
                if wrote_mid:
                    sync_async(mid_path, args.gcs_bucket, args.run_name)

    save_checkpoint(trainer, ckpt_path)
    export_sb3_compatible(ac, latest_path)
    export_sb3_compatible(ac, final_path)
    if args.gcs_bucket:
        from rower_soccer.warp_port.gcs import sync_blocking, wait_all
        # Drain any mid-run uploads first so their (older) bytes cannot land on
        # top of the final ones, then upload synchronously: returning from
        # main() kills the daemon upload threads mid-transfer.
        wait_all()
        for path in (cfg_path, ckpt_path, latest_path, final_path):
            sync_blocking(path, args.gcs_bucket, args.run_name)
    print(f"[setup] done in {(time.perf_counter()-t0)/60:.1f}min; saved final.pt",
          flush=True)


if __name__ == "__main__":
    main()
