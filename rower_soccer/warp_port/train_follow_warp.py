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

from rower_soccer.warp_port import curriculum
from rower_soccer.warp_port import score


def make_eval_env(args, num_worlds, seed):
    """The env the trainer EVALUATES in: same task, same knobs, N worlds.

    Factored out of make_eval so the one-world render env and the N-world
    scoring env are built from one list of kwargs. They must agree exactly -- if
    they drift, the number that selects best.pt and the video meant to show it
    stop describing the same task, and nothing would say so. (Note that this
    list already differs from the TRAINING env's, which passes no arena /
    pitch_scale; that is pre-existing and left alone here.)
    """
    from rower_soccer.warp_port.follow_env import WarpFollowEnv
    return WarpFollowEnv(
        # use_graph=True: capturing a CUDA graph for the eval env is
        # ~16x faster per step (measured on the dribble env, the same scene
        # plus a ball: 1462.1 -> 92.4 ms/step, i.e. 877 -> 55 s per 600-step
        # video at the default --video-secs 300). Warp is still ground truth
        # either way; the graph changes only how the same kernels are launched.
        num_worlds=num_worlds, use_graph=True, seed=seed,
        creature_xml=args.creature_xml, arena=args.arena,
        pitch_scale=args.pitch_scale,
        target_speed_range=tuple(args.target_speed),
        spawn_dist_range=tuple(args.spawn_dist),
        bounds=args.bounds, reward_coef=args.reward_coef,
        w_vel_shaping=args.vel_shaping, reward_mode=args.reward_mode,
        progress_scale=args.progress_scale, episode_seconds=args.episode_secs,
        energy_coef=args.energy_coef, smooth_coef=args.smooth_coef)


def make_eval(args, has_ball=False):
    """One-world Warp env + renderer, built once and reused.

    Warp is ground truth: the eval runs in the SAME simulator the policy trains in,
    and the video is drawn from that simulator's state. The dm_control CPU drill is
    no longer in the loop -- see warp_port/render.py for why.

    One world because the RENDERER can only draw one. That constraint belongs to
    the video and to nothing else, which is why scoring no longer runs here --
    see make_score_env and docs/DRILL_V4_NOTES.md section 10.
    """
    from rower_soccer.warp_port.render import WarpRenderer
    env = make_eval_env(args, num_worlds=1, seed=7)
    # Render the arena (the physics scene), not the default pitch background.
    return env, WarpRenderer(args.creature_xml, has_ball=False,
                             base_xml=env._base_xml())


def make_score_env(args):
    """The N-world scoring env that selects best.pt. No renderer: the score
    never needed one, and the one-world env was only ever the render env."""
    return make_eval_env(args, num_worlds=args.score_worlds,
                         seed=args.score_seed)


_STYLE_REF_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "runs_v2", "rower_ref_gait.npz")


def _style_of(q, ref, settle_s=2.0, control_dt=0.025):
    """Style of one eval episode, or None if no reference applies to this body."""
    if ref is None or q is None:
        return None
    try:
        from rower_soccer.tools.style import style_score
        r = style_score(q[int(settle_s / control_dt):], control_dt, ref)
        return {k: float(r[k]) for k in ("style", "amp", "freq", "shape", "gait_hz")}
    except Exception as e:                                  # noqa: BLE001
        # Never let a diagnostic kill a training run.
        print(f"[monitor] style failed: {e}", flush=True)
        return None


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
    p.add_argument("--freeze-decoder", action="store_true",
                   help="Freeze the low-level controller (decoder + action head) "
                        "and train only the task expert that emits z. This is the "
                        "NPMP/Liu-et-al. arrangement: the motor skill is learned "
                        "once from the reference motion and then reused, so every "
                        "task shares one motor style and cannot degrade it. Pair "
                        "with --init-from a tracking run's decoder.")
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
    p.add_argument("--pitch-scale", type=float, default=0.3125,
                   help="uniform scale on dm_soccer's pitch (ground, walls and "
                        "both goals together). 1.0 = its 96x72 m 2v2 pitch, "
                        "sized for BoxHead; 0.3125 = 30x22.5 m with a 7.4 m "
                        "goal, which our ant can actually cross in a match.")
    p.add_argument("--arena", default="fenced", choices=["fenced", "pitch"],
                   help="'fenced' is the small walled arena (wall at "
                        "--floor-half); 'pitch' is the real 2v2 soccer pitch. "
                        "Geometry only -- timestep, cone, floor friction and "
                        "solref are identical, verified on the compiled models. "
                        "Use 'pitch' so the fence stops being part of the task: "
                        "23.5%% of mid-episode kick targets land outside a 10 m "
                        "wall, asking the ant to arc the ball over it.")
    p.add_argument("--creature-xml",
                   default="creature_configs/three_seg_worm.xml")
    p.add_argument("--run-name", required=True)
    p.add_argument("--video-secs", type=float, default=300.0)
    # Fire the FIRST transfer-eval video early, so a broken run (bad obs layout,
    # bad reward, creature glitching) is visible in minutes instead of after the
    # first full --video-secs interval. Subsequent videos keep the normal cadence.
    p.add_argument("--first-video-secs", type=float, default=60.0)
    score.add_args(p)
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
    # The AR(1) latent prior, ||z_t - alpha*z_{t-1}||^2. train_track_warp has
    # carried this since PIPELINE_V2 and defaults it ON at 0.01 -- so the decoder
    # is trained having only ever seen SMOOTH latents. This trainer never passed
    # it, which left the task expert free to drive that decoder at any rate it
    # liked, and it does: measured z on follow_rower_npmp_v2 runs at 12.67 Hz
    # against tracking's 1.15 Hz, at 3.4x the amplitude. A frozen decoder driven
    # an order of magnitude outside its training distribution is not a preserved
    # motor skill, which is the likeliest reason that run reached fitness 0.972
    # while scoring style 0.379. Default stays 0.0 so existing runs reproduce.
    p.add_argument("--z-ar-coef", type=float, default=0.0,
                   help="AR(1) latent smoothness ||z_t - alpha*z_{t-1}||^2; "
                        "match the tracking run (0.01) when using --init-from")
    p.add_argument("--z-ar-alpha", type=float, default=0.95)
    p.add_argument("--z-smooth-coef", type=float, default=0.0,
                   help="static ||z||^2 prior; NOT the same object as --z-ar-coef")
    curriculum.add_args(p)
    p.add_argument("--freeze-log-std", action="store_true",
                   help="with --freeze-decoder, also hold the inherited "
                        "per-joint exploration noise. The prior learns std "
                        "0.07-0.14 on the gait-carrying arms; left trainable it "
                        "floats up under ent_coef because fitness does not care "
                        "about gait quality.")
    p.add_argument("--style-ref", default=_STYLE_REF_DEFAULT,
                   help="gait reference used to score HOW the policy moves; "
                        "fitness cannot see this axis")
    p.add_argument("--no-style", action="store_true",
                   help="skip the style diagnostic entirely")
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
                        creature_xml=args.creature_xml,
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
                         ent_anneal_steps=args.ent_anneal_steps,
                         z_smooth_coef=args.z_smooth_coef,
                         z_ar_coef=args.z_ar_coef, z_ar_alpha=args.z_ar_alpha)

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    latest_path = os.path.join(run_dir, "latest.pt")
    mid_path = os.path.join(run_dir, "checkpoint_mid.pt")
    best_path = os.path.join(run_dir, "best.pt")
    best_score = float("-inf")
    mid_target = int(args.steps * args.mid_ckpt_frac) if args.mid_ckpt_frac else 0
    start_steps = 0
    if args.freeze_decoder:
        # log_std stays trainable by default. The original reasoning was that it
        # is exploration noise, not motor skill, and that pinning it would deny
        # the task policy any way to quiet down for fine control.
        #
        # Measurement disagrees with the premise. The noise is PER JOINT, and the
        # tracking prior learns a very specific structure: std 0.07-0.14 on the
        # four arm joints that carry the gait, ~1.0 on the rest. That is fine
        # motor control, learned. In follow it does not quiet down further -- it
        # floats UP (0.07 -> 0.11, 0.14 -> 0.58), because ent_coef rewards
        # entropy and fitness is indifferent to gait quality. --freeze-log-std
        # tests whether holding that structure preserves the gait.
        frozen = 0
        mods = [ac.mlp_extractor.decoder, ac.action_net]
        for mod in mods:
            for prm in mod.parameters():
                prm.requires_grad_(False)
                frozen += prm.numel()
        if args.freeze_log_std:
            if ac.state_dependent_std:
                for prm in ac.log_std_net.parameters():
                    prm.requires_grad_(False)
                    frozen += prm.numel()
            else:
                ac.log_std.requires_grad_(False)
                frozen += ac.log_std.numel()
        trainable = sum(p.numel() for p in ac.parameters() if p.requires_grad)
        print(f"[setup] decoder FROZEN: {frozen:,} params held, "
              f"{trainable:,} trainable (expert + critic"
              f"{'' if args.freeze_log_std else ' + log_std'})", flush=True)
        if args.freeze_log_std:
            import numpy as _np
            _s = ac.log_std.detach().exp().cpu().numpy() if not ac.state_dependent_std else None
            print(f"[setup] log_std FROZEN at inherited per-joint std "
                  f"{_np.round(_s, 3) if _s is not None else '(state-dependent)'}",
                  flush=True)
        # Adam was built over every parameter; rebuild it over the live ones so
        # frozen weights cannot drift via weight decay or stale moments.
        trainer.opt = torch.optim.Adam(
            [p for p in ac.parameters() if p.requires_grad], lr=args.lr)

    # The --freeze-decoder block sits ABOVE this deliberately: it rebuilds
    # trainer.opt over only the trainable parameters, and load_checkpoint
    # restores a saved optimizer state into it. Freeze after loading and a
    # frozen-decoder run can be checkpointed but never resumed -- the saved
    # state's single reduced parameter group does not match a full-parameter
    # Adam ("loaded state dict contains a parameter group that doesn't match
    # the size of optimizer's group").
    if args.resume and os.path.exists(ckpt_path):
        start_steps = load_checkpoint(trainer, ckpt_path)
        print(f"[setup] resumed from {ckpt_path} at step {start_steps:,}", flush=True)
    elif args.init_from:
        # Fresh run only: on a real --resume the checkpoint already holds these
        # weights, further trained, and re-seeding would throw that away.
        before = ac.mlp_extractor.decoder[0].weight.detach().clone()
        load_pretrained(ac, args.init_from, device=trainer.device)
        if torch.equal(before, ac.mlp_extractor.decoder[0].weight.detach()):
            raise SystemExit(
                f"\n--init-from {args.init_from} transferred NOTHING to the "
                f"decoder.\nload_pretrained copies only shape-matching tensors, "
                f"so a decoder\nbuilt for a different body is skipped in silence. "
                f"This env's proprio\nis {len(env.proprio_indices)} wide; the "
                f"checkpoint's decoder expects something else.\nCheck "
                f"--creature-xml matches the body the prior was trained on.")

    # Style reference. Optional and body-specific: it grades this creature's gait
    # against the evolved gait the NPMP tracker was built from, so a worm run (2
    # joints) must not be scored against the rower's 8-joint reference. Mismatch
    # disables it loudly rather than silently reporting nonsense.
    style_ref = None
    if not args.no_style:
        try:
            from rower_soccer.tools.style import load_reference
            style_ref = load_reference(args.style_ref)
            if len(style_ref["names"]) != env.act_dim:
                print(f"[setup] style DISABLED: reference has "
                      f"{len(style_ref['names'])} joints, this body has "
                      f"{env.act_dim}", flush=True)
                style_ref = None
            else:
                print(f"[setup] style reference {args.style_ref} "
                      f"({len(style_ref['names'])} joints)", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"[setup] style DISABLED: {e}", flush=True)

    print(f"[setup] worlds={env.n} obs={env.obs_dim} act={env.act_dim} "
          f"steps/iter={trainer.T * trainer.N:,}", flush=True)
    eval_env, eval_ren = make_eval(args)
    # The BATCHED scoring env: N worlds, no renderer, built ONCE (a Warp env
    # costs a scene compile plus a graph capture) and reused every evaluation.
    # best.pt is selected on this, not on the one-world render eval -- a running
    # max over single-episode draws selects the luckiest draw rather than the
    # best policy, measured across all four drills in
    # docs/DRILL_V4_NOTES.md section 10.
    score_env = make_score_env(args) if args.score_worlds > 0 else None
    if score_env is not None:
        print(f"[setup] scoring env: {args.score_worlds} worlds, seed "
              f"{args.score_seed} re-applied every rollout (paired draws)",
              flush=True)
    t0 = time.perf_counter()
    last_steps, last_t = start_steps, t0
    # Back-date the video timer so the first one lands at --first-video-secs.
    last_video = t0 - max(0.0, args.video_secs - args.first_video_secs)
    # Default (--score-secs 0) puts scoring on the video cadence, so the two
    # numbers in the log describe the SAME weights and can be compared directly.
    score_cadence = args.score_secs if args.score_secs > 0 else args.video_secs
    last_score = last_video
    last_ckpt = t0
    speed_curr = curriculum.from_args(args)
    it = 0
    deadline = t0 + args.max_hours * 3600.0
    while trainer.total_steps < args.steps and time.perf_counter() < deadline:
        stats = trainer.train_iter()
        it += 1
        now = time.perf_counter()
        # Lifetime average since the run started. It lags hard after any slow
        # period -- a run that spent four hours CPU-starved and then recovered
        # reports the four hours forever -- so `fps_now` is the number to read
        # when asking "how fast is it going", and `fps` when asking "how long
        # until it finishes".
        fps = (trainer.total_steps - start_steps) / (now - t0)
        fps_now = ((trainer.total_steps - last_steps) / (now - last_t)
                   if now > last_t else fps)
        last_steps, last_t = trainer.total_steps, now
        # ETA is now the wall-clock deadline, not the step target.
        eta_min = max(0.0, (deadline - now) / 60)
        if it % 5 == 0:
            line = speed_curr.update(env, eval_env)
            if line:
                print(line, flush=True)
            # The scoring env has to track the curriculum too, or best.pt is
            # selected at a difficulty the run left behind long ago. Synced
            # here rather than threaded through SpeedCurriculum.update, which
            # takes a single eval env.
            if score_env is not None:
                score_env.speed_range = env.speed_range
            print(f"[monitor] step={trainer.total_steps:,}/{args.steps:,} "
                  f"({100*trainer.total_steps/args.steps:.1f}%) fps={fps:,.0f} fps_now={fps_now:,.0f} "
                  f"eta={eta_min:.1f}min ep_rew={stats['ep_rew_env_mean']:.1f} "
                  f"std={stats['std']:.3f} tgt_spd={env.speed_range[1]:.2f}",
                  flush=True)
            if use_wandb:
                import wandb
                wandb.log({"env_step": trainer.total_steps,
                           "monitor/fps": fps, "monitor/fps_now": fps_now, "monitor/eta_min": eta_min,
                           "train/ep_rew": stats["ep_rew_env_mean"],
                           "train/entropy": stats["ent"], "train/std": stats["std"],
                           "train/pg_loss": stats["pg"], "train/vf_loss": stats["vf"]})
        do_video = args.video_secs > 0 and now - last_video >= args.video_secs
        do_score = (score_env is not None and score_cadence > 0
                    and now - last_score >= score_cadence)
        if do_video:
            last_video = now
            vpath = os.path.join(run_dir, "videos",
                                 f"eval_step_{trainer.total_steps:010d}.mp4")
            from rower_soccer.warp_port.render import eval_video
            ep_rew, fit, ev_q = eval_video(eval_env, ac, vpath, eval_ren,
                                           record_joints=True)
            # Fitness is exp(-c*||player - target||): it grades ARRIVING and is
            # structurally blind to HOW. follow_rower_baseline reached 0.950 by
            # vibrating at 0.46 Hz with its arms folded, and nothing in this loop
            # could tell it apart from the NPMP-primed policy at 0.960 that
            # actually rows. `style` is that missing axis. Logged from the eval
            # episode itself, so n=1 and noisy (+/-0.1); the authoritative number
            # is `python -m rower_soccer.tools.style score --checkpoint ...`,
            # which averages worlds.
            st = _style_of(ev_q, style_ref)
            print(f"[monitor] video: {vpath} (WARP eval "
                  f"ep_rew={ep_rew:.1f} fitness={fit:.3f}"
                  + (f" style={st['style']:.3f}"
                     f" [amp {st['amp']:.2f} freq {st['freq']:.2f}"
                     f" shape {st['shape']:.2f}]" if st else "")
                  + ")", flush=True)
            if use_wandb:
                import wandb
                # eval/ep_rew_warp and eval/fitness_warp keep their names and
                # their meaning: the OLD one-world single-episode numbers,
                # retained so they can be compared against the batched pair.
                # They no longer select anything.
                logs = {"env_step": trainer.total_steps,
                        "eval/video": wandb.Video(vpath, format="mp4"),
                        "eval/ep_rew_warp": ep_rew, "eval/fitness_warp": fit}
                if st:
                    logs.update({f"eval/style_{k}" if k != "style" else "eval/style": v
                                 for k, v in st.items()})
                wandb.log(logs)
        sel = None
        if do_score:
            last_score = now
            sc = score.score_policy(score_env, ac, seed=args.score_seed)
            print(f"[monitor] score: ep_rew_batched={sc.ep_rew:.1f} "
                  f"fitness_batched={sc.fitness:.3f} +/-{sc.fitness_sem:.3f} "
                  f"(sem over {sc.worlds} worlds, world spread "
                  f"{sc.fitness_std:.3f})", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"env_step": trainer.total_steps,
                           "eval/ep_rew_batched": sc.ep_rew,
                           "eval/fitness_batched": sc.fitness,
                           "eval/fitness_batched_sem": sc.fitness_sem,
                           "eval/fitness_batched_std": sc.fitness_std})
            # follow selects on ep_rew, not fitness -- unlike the three ball
            # drills. Left as it is: follow's fitness is exp(-c*dist) read at
            # the final step only, so it grades where the creature happened to
            # be standing when the clock ran out, while ep_rew integrates the
            # whole episode. Changing WHICH statistic selects is a separate
            # decision from fixing how noisily it is measured.
            sel = sc.ep_rew
        elif do_video and score_env is None:
            # --score-worlds 0: explicit opt-out, back to the old behaviour.
            sel = ep_rew
        # Keep the BEST policy, not just the latest.
        #
        # follow_v5_velshape's transfer eval went 262 -> 351 -> 465 -> 476.5 and
        # then COLLAPSED to 166.6 in its final stretch (log_std pinned at the
        # entropy ceiling). final.pt and latest.pt both hold the collapsed 166
        # policy. The 476 weights -- comfortably in C's 445-495 band -- existed,
        # were never saved, and are gone.
        #
        # Late collapse is not exotic in long PPO runs, and 48-hour runs give it
        # far more room. Scored on the DETERMINISTIC eval -- and, since
        # 2026-08-11, on the BATCHED one: a running max over a ONE-episode
        # estimator selects the luckiest draw, not the best policy
        # (docs/DRILL_V4_NOTES.md 10).
        if sel is not None and sel > best_score:
            best_score = sel
            export_sb3_compatible(ac, best_path)
            print(f"[monitor] new BEST transfer eval {best_score:.1f} "
                  f"-> {best_path}", flush=True)
            if args.gcs_bucket:
                from rower_soccer.warp_port.gcs import sync_async
                sync_async(best_path, args.gcs_bucket, args.run_name)
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
