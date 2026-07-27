"""Train the NPMP tracking policy on the evolved rower gait (Warp, PPO).

    python -m rower_soccer.warp_port.train_track_warp --run-name npmp_rower_v2

Stage 1 of NPMP (Merel et al. 2019; Liu et al. 2022 §"motor control"): learn to
reproduce a reference motion under our own physics and observation contract.
The artifact that matters is not this policy but the DECODER it leaves behind --
`pi(a | proprio, z)` -- which carries evolution's rowing style into every later
rower task via `--init-from`.

The information split is enforced by the env's obs layout, not by this script:
the expert/encoder sees the reference lookahead, the decoder sees only proprio
and z. See track_env.py.

BEFORE TRAINING, RUN THE GATE:

    python -m rower_soccer.tools.rower_ref check

If the worst torque margin is below 1, the body cannot drive the reference and
no amount of RL will fix it. That is not hypothetical -- `npmp_rower_track`
spent 400M steps failing exactly this way, against a reference that was 2x too
fast on a rower whose gears were 20x too weak. This script refuses to start in
that case rather than let it happen twice.
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--steps", type=int, default=150_000_000)
    p.add_argument("--worlds", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--ent-floor", type=float, default=None)
    p.add_argument("--ent-ceil", type=float, default=0.0)
    p.add_argument("--ent-anneal-steps", type=int, default=0)
    p.add_argument("--z-dim", type=int, default=16,
                   help="latent width; PIPELINE_V2 default 16 (paper used 60 "
                        "for a 56-DOF humanoid, this rower has 8)")
    # NPMP's autoregressive latent prior. alpha 0.95 is the paper's value.
    p.add_argument("--z-ar-coef", type=float, default=0.01)
    p.add_argument("--z-ar-alpha", type=float, default=0.95)
    p.add_argument("--z-smooth-coef", type=float, default=0.0)
    p.add_argument("--init-from", default=None)
    p.add_argument("--resume", action="store_true")
    # env
    p.add_argument("--creature-xml",
                   default="creature_configs/two_arm_rower_scaled.xml")
    p.add_argument("--ref", default=os.path.join(REPO, "runs_v2", "rower_ref_gait.npz"))
    p.add_argument("--episode-secs", type=float, default=10.0)
    p.add_argument("--track-coef", type=float, default=2.0,
                   help="exp(-c * weighted joint error); 2.0 puts a 0.5 rad "
                        "error at reward 0.37")
    p.add_argument("--upright-coef", type=float, default=1.0)
    p.add_argument("--energy-coef", type=float, default=0.0)
    p.add_argument("--smooth-coef", type=float, default=0.0)
    p.add_argument("--no-rsi", action="store_true",
                   help="disable reference state initialisation (ablation)")
    # bookkeeping
    p.add_argument("--max-hours", type=float, default=6.0)
    p.add_argument("--video-secs", type=float, default=900.0)
    p.add_argument("--first-video-secs", type=float, default=120.0)
    p.add_argument("--ckpt-secs", type=float, default=1800.0)
    p.add_argument("--mid-ckpt-frac", type=float, default=0.5)
    p.add_argument("--gcs-bucket", default=None)
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--skip-gate", action="store_true",
                   help="train even if the body cannot drive the reference "
                        "(you almost certainly do not want this)")
    return p.parse_args()


def gate(args):
    """Refuse to burn GPU-hours on a physically unreachable reference."""
    from rower_soccer.tools.rower_ref import check
    xml = args.creature_xml if os.path.isabs(args.creature_xml) \
        else os.path.join(REPO, args.creature_xml)
    worst = check(ref=args.ref, xml=xml)
    if worst < 1.0 and not args.skip_gate:
        raise SystemExit(
            f"\nGATE FAILED: worst torque margin {worst:.2f}x < 1.0.\n"
            "The body cannot drive this reference; RL cannot fix that.\n"
            "Regenerate the body with a larger --gear-scale, or rebuild the\n"
            "reference slower (rower_ref.py build --no-froude). Override with\n"
            "--skip-gate only if you know why you are doing it.")
    return worst


def main():
    args = parse_args()
    run_dir = os.path.join(REPO, "runs_v2", args.run_name)
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)

    print("[gate] torque margin of the body against the reference:", flush=True)
    margin = gate(args)
    print(f"[gate] passed, worst margin {margin:.2f}x\n", flush=True)

    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump({**vars(args), "gate_margin": margin}, f, indent=2)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.run_name,
                       id=args.run_name, resume="allow", config=vars(args))
            wandb.define_metric("env_step")
            wandb.define_metric("*", step_metric="env_step")
        except Exception as e:                      # noqa: BLE001
            print(f"[setup] wandb disabled: {e}", flush=True)
            use_wandb = False

    from rower_soccer.warp_port.track_env import WarpTrackEnv
    from rower_soccer.warp_port.ppo import (ActorCritic, PPOTrainer,
                                            export_sb3_compatible,
                                            load_checkpoint, load_pretrained,
                                            save_checkpoint)
    from rower_soccer.warp_port.render import WarpRenderer, eval_video

    xml = args.creature_xml if os.path.isabs(args.creature_xml) \
        else os.path.join(REPO, args.creature_xml)

    env = WarpTrackEnv(num_worlds=args.worlds, creature_xml=xml, ref_path=args.ref,
                       episode_seconds=args.episode_secs, seed=args.seed,
                       track_coef=args.track_coef, upright_coef=args.upright_coef,
                       energy_coef=args.energy_coef, smooth_coef=args.smooth_coef,
                       rsi=not args.no_rsi)
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
    final_path = os.path.join(run_dir, "final.pt")
    # The actual deliverable: the reusable low-level controller.
    decoder_path = os.path.join(REPO, "runs_v2", "_init_rower_npmp.pt")

    # Persist best_score. Without it a --resume restarts the comparison at
    # -inf, so the first post-resume eval overwrites best.pt even when it is
    # worse -- which is how this run lost its 262.34 policy to a later 251.13.
    # ppo.py already carries a comment about follow_v5 losing its best weights
    # exactly this way; same bug, different door.
    best_meta = os.path.join(run_dir, "best_score.json")
    best_score = float("-inf")
    if os.path.exists(best_meta):
        try:
            with open(best_meta) as f:
                best_score = float(json.load(f)["best_score"])
            print(f"[setup] best score so far: {best_score:.2f}", flush=True)
        except Exception:                                   # noqa: BLE001
            pass
    mid_target = int(args.steps * args.mid_ckpt_frac) if args.mid_ckpt_frac else 0
    start_steps = 0
    if args.resume and os.path.exists(ckpt_path):
        start_steps = load_checkpoint(trainer, ckpt_path)
        print(f"[setup] resumed at step {start_steps:,}", flush=True)
    elif args.init_from:
        load_pretrained(ac, args.init_from, device=trainer.device)

    print(f"[setup] worlds={env.n} obs={env.obs_dim} "
          f"(proprio {len(env.proprio_indices)} + task {len(env.task_indices)}) "
          f"act={env.act_dim} z={args.z_dim}", flush=True)
    print(f"[setup] reference {env.K} frames @ {env.freq_tgt:.3f} Hz, "
          f"steps/iter={trainer.T * trainer.N:,}", flush=True)

    eval_env = WarpTrackEnv(num_worlds=1, creature_xml=xml, ref_path=args.ref,
                            episode_seconds=args.episode_secs, seed=args.seed + 1,
                            use_graph=False, rsi=False)
    eval_ren = WarpRenderer(xml, has_ball=False)

    t0 = time.perf_counter()
    last_video = t0 - max(0.0, args.video_secs - args.first_video_secs)
    last_ckpt = t0
    it = 0
    deadline = t0 + args.max_hours * 3600.0
    while trainer.total_steps < args.steps and time.perf_counter() < deadline:
        stats = trainer.train_iter()
        it += 1
        now = time.perf_counter()
        fps = (trainer.total_steps - start_steps) / (now - t0)
        eta_min = max(0.0, (deadline - now) / 60)
        if it % 5 == 0:
            jerr = env.mean_joint_err()
            print(f"[monitor] step={trainer.total_steps:,}/{args.steps:,} "
                  f"({100*trainer.total_steps/args.steps:.1f}%) fps={fps:,.0f} "
                  f"eta={eta_min:.1f}min rew={stats['ep_rew_env_mean']:.3f} "
                  f"jerr={jerr:.3f}rad std={stats['std']:.3f}", flush=True)
            if use_wandb:
                import wandb
                log = {"env_step": trainer.total_steps, "monitor/fps": fps,
                       "train/ep_rew": stats["ep_rew_env_mean"],
                       "train/joint_err_rad": jerr,
                       "train/joint_err_deg": float(np.rad2deg(jerr)),
                       "train/entropy": stats["ent"], "train/std": stats["std"],
                       "train/pg_loss": stats["pg"], "train/vf_loss": stats["vf"],
                       "train/diverged": trainer.n_diverged}
                if "z_ar" in stats:
                    log["train/z_ar"] = stats["z_ar"]
                wandb.log(log)
        if args.video_secs > 0 and now - last_video >= args.video_secs:
            last_video = now
            vpath = os.path.join(run_dir, "videos",
                                 f"eval_step_{trainer.total_steps:010d}.mp4")
            ep_rew, fit = eval_video(eval_env, ac, vpath, eval_ren)
            ev_jerr = eval_env.mean_joint_err()
            print(f"[monitor] video {vpath} (ep_rew={ep_rew:.2f} fitness={fit:.3f} "
                  f"jerr={np.rad2deg(ev_jerr):.1f}deg)", flush=True)
            if ep_rew > best_score:
                best_score = ep_rew
                export_sb3_compatible(ac, best_path)
                # Same export format as best.pt, NOT a raw state_dict: this file
                # is consumed by load_pretrained, whose _flatten_checkpoint wants
                # {"mlp_extractor", "action_net", ...}. A raw dump loads nowhere.
                export_sb3_compatible(ac, decoder_path)
                with open(best_meta, "w") as f:
                    json.dump({"best_score": best_score,
                               "step": trainer.total_steps}, f)
                print(f"[monitor] new BEST {best_score:.2f} -> {best_path} "
                      f"(+ decoder {decoder_path})", flush=True)
                if args.gcs_bucket:
                    from rower_soccer.warp_port.gcs import sync_async
                    sync_async(best_path, args.gcs_bucket, args.run_name)
            if use_wandb:
                import wandb
                wandb.log({"env_step": trainer.total_steps,
                           "eval/video": wandb.Video(vpath, format="mp4"),
                           "eval/ep_rew": ep_rew, "eval/fitness": fit,
                           "eval/joint_err_deg": float(np.rad2deg(ev_jerr))})
        if now - last_ckpt >= args.ckpt_secs:
            last_ckpt = now
            save_checkpoint(trainer, ckpt_path)
            export_sb3_compatible(ac, latest_path)
            print(f"[monitor] checkpoint at step {trainer.total_steps:,}", flush=True)
            if mid_target and not os.path.exists(mid_path) \
                    and trainer.total_steps >= mid_target:
                shutil.copy2(ckpt_path, mid_path)
            if args.gcs_bucket:
                from rower_soccer.warp_port.gcs import sync_async
                for pth in (ckpt_path, cfg_path, latest_path):
                    sync_async(pth, args.gcs_bucket, args.run_name)

    save_checkpoint(trainer, ckpt_path)
    export_sb3_compatible(ac, latest_path)
    export_sb3_compatible(ac, final_path)
    export_sb3_compatible(ac, decoder_path)
    if args.gcs_bucket:
        from rower_soccer.warp_port.gcs import sync_blocking, wait_all
        wait_all()
        for path in (cfg_path, ckpt_path, latest_path, final_path):
            sync_blocking(path, args.gcs_bucket, args.run_name)
    print(f"[setup] done in {(time.perf_counter()-t0)/60:.1f}min; "
          f"decoder -> {decoder_path}", flush=True)


if __name__ == "__main__":
    main()
