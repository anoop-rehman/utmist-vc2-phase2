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


def make_eval(args):
    """One-world Warp env + renderer, built once and reused. Warp is ground truth."""
    from rower_soccer.warp_port.dribble_env import WarpDribbleEnv
    from rower_soccer.warp_port.render import WarpRenderer
    env = WarpDribbleEnv(
        num_worlds=1, use_graph=False, seed=7,
        target_speed_range=tuple(args.target_speed),
        ball_spawn_range=tuple(args.ball_spawn),
        target_dist_range=tuple(args.target_dist),
        bounds=args.bounds, reward_coef=args.reward_coef,
        w_player_to_ball=args.w_player_to_ball,
        w_ball_to_target=args.w_ball_to_target,
        reward_mode=args.reward_mode, progress_scale=args.progress_scale,
        approach_scale=args.approach_scale, episode_seconds=args.episode_secs,
        energy_coef=args.energy_coef, smooth_coef=args.smooth_coef)
    return env, WarpRenderer(args.creature_xml, has_ball=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000_000)
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
                   help="follow checkpoint to warm-start from (checkpoint.pt or "
                        "latest.pt). Task encoder + critic input layer re-init; "
                        "the decoder (the low-level controller) carries over.")
    # Rescaled off probe_speed's 0.759 m/s (post contact-stiffening), keeping dribble's
    # target slower than follow's 0.21 cap because dribbling is harder: the worm must
    # shepherd a ball, not just chase a point. 0.15 / 0.759 = 0.20 margin, vs follow's
    # 0.28 and C's 0.28. See train_follow_warp's --target-speed for the full reasoning.
    p.add_argument("--target-speed", type=float, nargs=2, default=[0.03, 0.15],
                   help="dribbling is harder than following, so the target is "
                        "slower still than follow's [0.03, 0.21] cap")
    p.add_argument("--bounds", type=float, default=10.0)
    p.add_argument("--ball-spawn", type=float, nargs=2, default=[1.5, 3.0],
                   help="ball spawn distance from the worm (m); dm_control's own "
                        "1-3 m, which the 1.76 m worm's 0.82 m footprint allows")
    p.add_argument("--target-dist", type=float, nargs=2, default=[2.0, 5.0],
                   help="target spawn distance from the BALL (m), not from the "
                        "worm: anchoring it to the worm leaves ball and target "
                        "~13 m apart, where exp(-c*d) is flat zero and the drill "
                        "has no gradient at all")
    p.add_argument("--reward-coef", type=float, default=0.5)
    # Realism regularizers (default OFF -> baseline is unchanged). Energy penalises
    # brute thrust; smooth is CAPS temporal smoothness (penalises jerk, not speed).
    p.add_argument("--energy-coef", type=float, default=0.0)
    p.add_argument("--smooth-coef", type=float, default=0.0)
    # Anneal the entropy bonus to 0 over this many env-steps (0 = constant). Fixes
    # the late-training entropy runaway that collapsed follow_v5.
    p.add_argument("--ent-anneal-steps", type=int, default=0)
    # Fine control: make the exploration std a learned function of state (gSDE-flavored)
    # so the policy can go quiet near the ball. Its bounds are wide/low by design, so
    # --ent-floor does NOT apply to it.
    p.add_argument("--state-dependent-std", action="store_true")
    # Reward-parking: anneal the velocity-shaping terms to 0 over N steps, so late
    # training optimizes pure fitness (ball AT target) instead of ball velocity.
    p.add_argument("--shaping-anneal-steps", type=int, default=0)
    p.add_argument("--reward-mode", default="paper", choices=["paper", "progress"])
    p.add_argument("--progress-scale", type=float, default=2.0)
    p.add_argument("--approach-scale", type=float, default=0.5,
                   help="progress mode: weight on the player->ball potential. "
                        "Without it nothing rewards walking to the ball and the "
                        "ball->target term stays identically zero forever")
    # w_player_to_ball is dribble's locomotion signal -- the exact analogue of follow's
    # --vel-shaping, and it was mis-scaled the same way. The `paper` reward is
    #     fitness + w_p2b * v(player->ball) + w_b2t * v(ball->target)
    # and a worm that never moves collects fitness = exp(-c*d_bt) ~ 0.19 EVERY step for
    # free: 600 * 0.19 = 114. dribble_paper_v4 scored 116-137. It sat exactly on the
    # do-nothing value for 800M steps, because at w=0.1 and 0.759 m/s the most it could
    # earn for walking to the ball was 0.076/step -- less than the free fitness.
    #
    # Scaled to C's proven shaping magnitude (~0.14/step at full speed), as follow, off
    # the MEDIAN achievable speed (~0.90 m/s; probe_speed is nondeterministic, spread
    # 0.76-1.50 -- never calibrate off one sample):
    #   0.15 * 0.900 m/s = 0.135
    p.add_argument("--w-player-to-ball", type=float, default=0.15)
    # NOT rescaled: this multiplies the BALL's velocity, which is not limited by the
    # creature's speed (a struck ball reaches ~7 m/s). Only creature-velocity terms
    # need the achievable-speed correction.
    p.add_argument("--w-ball-to-target", type=float, default=0.3)
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
    # First transfer-eval video fires early so a broken run is visible in minutes.
    p.add_argument("--first-video-secs", type=float, default=60.0)
    p.add_argument("--ckpt-secs", type=float, default=1800.0)
    p.add_argument("--mid-ckpt-frac", type=float, default=0.5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gcs-bucket", default=None)
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed)

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

    env = WarpDribbleEnv(num_worlds=args.worlds, seed=args.seed,
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
                         ball_spawn_range=tuple(args.ball_spawn),
                         energy_coef=args.energy_coef, smooth_coef=args.smooth_coef)
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(), z_dim=args.z_dim,
                     state_dependent_std=args.state_dependent_std)
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
        # Warm start only on a fresh run: on --resume the checkpoint already
        # contains these weights (further trained), and re-seeding from follow
        # would throw away the dribble progress it holds.
        load_pretrained(ac, args.init_from, device=trainer.device)

    print(f"[setup] worlds={env.n} obs={env.obs_dim} act={env.act_dim} "
          f"proprio={len(env.proprio_indices)} task={len(env.task_indices)} "
          f"steps/iter={trainer.T * trainer.N:,}", flush=True)
    eval_env, eval_ren = make_eval(args)
    t0 = time.perf_counter()
    last_ckpt = t0
    # Back-date the video timer so the first one lands at --first-video-secs.
    last_video = t0 - max(0.0, args.video_secs - args.first_video_secs)
    it = 0
    deadline = t0 + args.max_hours * 3600.0
    while trainer.total_steps < args.steps and time.perf_counter() < deadline:
        if args.shaping_anneal_steps > 0:
            env.shaping_scale = max(0.0, 1.0 - trainer.total_steps / args.shaping_anneal_steps)
        stats = trainer.train_iter()
        it += 1
        now = time.perf_counter()
        fps = (trainer.total_steps - start_steps) / (now - t0)
        # ETA is now the wall-clock deadline, not the step target.
        eta_min = max(0.0, (deadline - now) / 60)
        if it % 5 == 0:
            fit = float(env.fitness().mean())
            # diverged: world-steps whose physics went non-finite (see ppo.collect).
            # Expected to be 0 or a trickle. If it climbs, the contact model is wrong
            # and the run is training on garbage -- do not ignore it.
            print(f"[monitor] step={trainer.total_steps:,}/{args.steps:,} "
                  f"({100*trainer.total_steps/args.steps:.1f}%) fps={fps:,.0f} "
                  f"eta={eta_min:.1f}min ep_rew={stats['ep_rew_env_mean']:.1f} "
                  f"fitness={fit:.3f} std={stats['std']:.3f} "
                  f"diverged={trainer.n_diverged:,}", flush=True)
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
            # The shaping weights must be passed too: the CPU task defaults to its own
            # w_p2b/w_b2t, so without these the transfer eval would report ep_rew under
            # a DIFFERENT reward than the one being trained on.
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
            # Scored on FITNESS, not ep_rew: fitness is the unshaped Table-S3
            # metric (ball close to target) and is the one number the velocity
            # shaping terms cannot inflate. It is the milestone-1 gate.
            if fit > best_score:
                best_score = fit
                export_sb3_compatible(ac, best_path)
                print(f"[monitor] new BEST fitness {best_score:.3f} "
                      f"-> {best_path}", flush=True)
                if args.gcs_bucket:
                    from rower_soccer.warp_port.gcs import sync_async
                    sync_async(best_path, args.gcs_bucket, args.run_name)
            if use_wandb:
                import wandb
                # _warp, not _dm_control: this eval runs in the Warp env now (see
                # render.py). The value was always the Warp number; only the label
                # was stale, left over from when the eval was dm_control.
                wandb.log({"env_step": trainer.total_steps,
                           "eval/video": wandb.Video(vpath, format="mp4"),
                           "eval/ep_rew_warp": ep_rew,
                           "eval/fitness_warp": fit})
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
