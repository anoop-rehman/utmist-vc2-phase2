"""Warp-accelerated kick-drill training on a frozen follow/dribble decoder.

The intended arrangement (ANT_SPRINT_WORKSTREAMS, WS1's queue) is NPMP's: the
motor skill is learned once by `follow`, and every drill after it trains only the
task expert that emits z, so all four skills share one gait and none can degrade
it. That is `--init-from <follow best.pt> --freeze-decoder`.

The checkpoint transfers everything except the task encoder and the critic's
input layer. Kick's task obs is 12 wide -- the same width as dribble's -- so a
DRIBBLE checkpoint transfers those two layers as well; a FOLLOW checkpoint
(task 4) re-inits them. Either way the decoder, which is the motor skill, carries
over unchanged: it only ever sees proprio + z.

Usage (ant, on the frozen follow decoder):
  MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.warp_port.train_kick_warp \
      --run-name kick_ant_v1 --creature-xml creature_configs/ant.xml \
      --init-from runs_v2/follow_ant_v1/best.pt --freeze-decoder \
      --gcs-bucket vc2-2026-checkpoints
"""

import argparse
import json
import os
import shutil
import subprocess
import time

import numpy as np
import torch


from rower_soccer.warp_port.scene import BallSpec


def make_env(args, num_worlds, seed, use_graph=True):
    from rower_soccer.warp_port.kick_env import WarpKickEnv
    return WarpKickEnv(
        num_worlds=num_worlds, seed=seed, use_graph=use_graph,
        creature_xml=args.creature_xml,
        ball=BallSpec(radius=args.ball_radius, mass=args.ball_mass), arena=args.arena, pitch_scale=args.pitch_scale,
        episode_seconds=args.episode_secs, segment_seconds=args.segment_secs,
        ball_spawn_range=tuple(args.ball_spawn), target_dist=args.target_dist,
        speed_clip=args.speed_clip, w_strike=args.w_strike,
        w_player_to_ball=args.w_player_to_ball, w_ball_to_cmd=args.w_ball_to_cmd,
        approach_scale=args.approach_scale, reward_mode=args.reward_mode,
        reward_kind=args.reward_kind, w_upright=args.w_upright, w_arrive=args.w_arrive,
        segment_seconds_range=tuple(args.segment_secs_range),
        target_dist_range=tuple(args.target_dist_range),
        pace_range=tuple(args.pace_range),
        deadline_range=tuple(args.deadline_range),
        arrival_reward_coef=args.arrival_reward_coef,
        w_anchor=args.w_anchor, anchor_free_radius=args.anchor_free_radius,
        time_coef=args.time_coef,
        energy_coef=args.energy_coef, smooth_coef=args.smooth_coef,
        floor_half=args.floor_half, fixed_start=args.fixed_start,
        target_cone=args.target_cone)


def make_eval(args):
    """One-world Warp env + renderer, built once and reused. Warp is ground truth.

    use_graph=True, where the follow/dribble trainers pass False. Measured on the
    shoot env (the heavier of the two -- it runs on the full pitch), one control
    step of the ONE-world eval env:

        graph=True    80 ms/step        graph=False   1280 ms/step

    A 15 s eval episode is 600 steps, so that is the difference between a ~50 s
    eval and a ~13 min one, taken every --video-secs (default 300 s) with
    training BLOCKED throughout. Verified in the same process that the training
    env's 256-world graph and the eval env's 1-world graph coexist: both keep
    stepping after the second capture, zero divergence in either.
    """
    from rower_soccer.warp_port.render import WarpRenderer
    from rower_soccer.warp_port.worm_env_base import _arena_xml
    env = make_env(args, num_worlds=1, seed=7, use_graph=True)
    # Render the arena the physics actually runs in, not the default pitch.
    return env, WarpRenderer(args.creature_xml, has_ball=True,
                             # 12 m, not 8: at 8 the TARGET (3-6 m out, and up
                             # to 6 m the other side of the ball) sits off-frame
                             # for much of a segment, so a human cannot see
                             # where the ball was actually meant to go.
                             base_xml=env._base_xml(), distance=12.0,
                             ball=env._ball_spec())


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
    p.add_argument("--ent-anneal-steps", type=int, default=0)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--init-from", default=None,
                   help="checkpoint to warm-start from (best.pt / latest.pt). "
                        "The decoder -- the low-level controller -- carries "
                        "over; task encoder + critic input layer re-init unless "
                        "the source drill's task width also happens to be 12.")
    p.add_argument("--freeze-decoder", action="store_true",
                   help="Freeze the low-level controller (decoder + action head) "
                        "and train only the task expert that emits z. The "
                        "NPMP/Liu-et-al. arrangement: one motor skill, shared by "
                        "every drill and degradable by none. Pair with "
                        "--init-from.")
    p.add_argument("--freeze-log-std", action="store_true",
                   help="with --freeze-decoder, also hold the inherited "
                        "per-joint exploration noise (it floats UP otherwise, "
                        "because ent_coef pays for entropy and fitness is "
                        "indifferent to gait quality)")
    p.add_argument("--state-dependent-std", action="store_true")
    p.add_argument("--plain", action="store_true",
                   help="plain-MLP baseline: no latent bottleneck, no decoder")

    # -- task ---------------------------------------------------------------
    p.add_argument("--episode-secs", type=float, default=15.0)
    p.add_argument("--segment-secs", type=float, default=5.0,
                   help="max seconds per STRIKE SEGMENT: the ball is re-placed "
                        "and a fresh direction commanded either when the "
                        "creature strikes and the ball leaves, or when this "
                        "runs out. 3 attempts per 15 s episode at the default.")
    p.add_argument("--ball-spawn", type=float, nargs=2, default=[1.5, 3.0],
                   help="ball spawn distance from the creature (m); dm_control's "
                        "own 1-3 m band")
    p.add_argument("--target-dist", type=float, default=4.0,
                   help="how far along the commanded direction the aim POINT is "
                        "drawn. It is scored only through the direction, so this "
                        "is a display/obs convention, not a reward parameter -- "
                        "keep it in the range a human would click.")
    p.add_argument("--floor-half", type=float, default=10.0,
                   help="arena half-size (m). Larger than dribble's 5 because a "
                        "struck ball travels; the walls should not be what stops "
                        "it inside a segment.")

    # -- reward -------------------------------------------------------------
    p.add_argument("--reward-mode", default="paper", choices=["paper", "progress"])
    p.add_argument("--w-strike", type=float, default=None,
                   help="weight on the banked strike speed. Under "
                        "--reward-kind direction/point it defaults to 0.5 and "
                        "is THE task reward, paid on exactly one step per "
                        "segment, so a good 6 m/s strike is worth 3.0 against "
                        "a shaping trickle of ~0.1/step. Under 'timed' it "
                        "defaults to 0: power is now a CONSEQUENCE of the "
                        "deadline (a far target must be struck hard, a near "
                        "one gently or it overshoots), so paying for it "
                        "separately would price the same thing twice and in "
                        "one direction only.")
    p.add_argument("--speed-clip", type=float, default=8.0,
                   help="cap (m/s) on the credited strike speed. The warp ball "
                        "occasionally leaves a bad contact at 20-30 m/s (see "
                        "scene.py); without a cap that solver artefact would be "
                        "the single most rewarded event in training.")
    p.add_argument("--w-player-to-ball", type=float, default=0.15,
                   help="`paper` mode: velocity toward the ball. Dribble's "
                        "value, calibrated to ~0.135/step at full speed.")
    p.add_argument("--w-ball-to-cmd", type=float, default=0.1,
                   help="dense ball-velocity-along-command term. Deliberately "
                        "small: it is the hackable version of the strike credit "
                        "(nudge repeatedly instead of striking once), and "
                        "--shaping-anneal-steps exists to park it.")
    p.add_argument("--approach-scale", type=float, default=0.5,
                   help="`progress` mode: weight on the player->ball potential")
    p.add_argument("--shaping-anneal-steps", type=int, default=0,
                   help="anneal ALL shaping to 0 over N env-steps, so late "
                        "training optimizes the strike alone")
    p.add_argument("--energy-coef", type=float, default=0.0)
    p.add_argument("--smooth-coef", type=float, default=0.0)

    # -- curriculum ---------------------------------------------------------
    p.add_argument("--fixed-start", action="store_true",
                   help="stage 1: ball dead ahead and the command colinear with "
                        "it, so walking forward strikes it at the target")
    p.add_argument("--reward-kind", default="direction",
                   choices=["direction", "point", "timed"],
                   help="'timed' (drill v4) is a PASS: a deadline "
                        "T = target_dist / v_pace is drawn per attempt, the "
                        "segment ends at exactly T, and the reward is "
                        "exp(-c * ||ball(T) - target||). Early and late are "
                        "punished alike and dribbling is excluded by "
                        "arithmetic rather than by a penalty term -- but the "
                        "task obs grows to 14 (see kick_env). "
                        "'direction' scores max(ball_vel . command) -- a "
                        "projection, so it cannot distinguish a hard wild kick "
                        "from a gentle accurate one, and RL climbs the easier "
                        "'hit harder' gradient (kick_ant_v1: median aim error "
                        "35 deg, 16%% of strikes backwards). 'point' scores "
                        "exp(-c*d) to the commanded point at closest approach, "
                        "which is what shoot already does.")
    p.add_argument("--w-arrive", type=float, default=3.0,
                   help="weight of the arrival term under --reward-kind point")
    p.add_argument("--segment-secs-range", type=float, nargs=2, default=[2.0, 6.0],
                   help="Table S2's randomized kick window. This, not a contact "
                        "budget or a body penalty, is what separates kick from "
                        "dribble: the ant tops out near 0.6 m/s, so it cannot "
                        "carry the ball to a 4-8 m target inside 2-6 s.")
    p.add_argument("--target-dist-range", type=float, nargs=2, default=[4.0, 8.0],
                   help="Table S2 calls the kick target DISTANT; randomized per "
                        "attempt so the policy cannot memorise one range")
    p.add_argument("--arrival-reward-coef", type=float, default=None,
                   help="decay constant for the arrival term IN THE REWARD "
                        "only; fitness always keeps --reward-coef so arms stay "
                        "comparable. Default None = share it (v4 behaviour). "
                        "Set ~0.2 to escape the flat-reward desert measured on "
                        "v4: the ant overshoots 3-6 m targets by 2-3x and "
                        "exp(-0.5*d) is numerically flat out there (d=10 -> "
                        "0.007), so nothing gradients it toward striking "
                        "softer. At 0.2, d=10 -> 0.135.")
    p.add_argument("--w-anchor", type=float, default=0.0,
                   help="--reward-kind timed (v7): weight on the SPAWN ANCHOR, "
                        "per step. Two effects, both aimed at 'walk to the "
                        "ball and strike it FROM THERE' rather than 'shove it "
                        "downfield': the me->ball approach shaping is re-aimed "
                        "at the ball's spawn point (before contact that is the "
                        "ball, so approach is unchanged; after contact the old "
                        "term was literally paying for the chase), and the "
                        "creature pays w_anchor per step per metre it strays "
                        "past --anchor-free-radius from that point. A segment "
                        "is 50-200 steps, so 0.01 makes a full-segment 2.5 m "
                        "dribble cost ~3, the same order as a perfect pass "
                        "(w_arrive * 1.0 = 3). 0 = off.")
    p.add_argument("--anchor-free-radius", type=float, default=1.0,
                   help="metres around the ball's spawn point that cost "
                        "nothing (see --w-anchor). The creature has to stand "
                        "beside the ball to swing at it, so this must exceed "
                        "its standing reach or the anchor fights the strike.")
    p.add_argument("--pace-range", type=float, nargs=2, default=[1.5, 3.0],
                   help="--reward-kind timed: band the pass pace v_pace is "
                        "drawn from (m/s), where the deadline is "
                        "T = target_dist / v_pace. MEASURED, not chosen: "
                        "probe_strike_speed on kick_ant_v3/best.pt gives a "
                        "realised pace (approach + flight, which is what T "
                        "covers) of median 1.6 m/s over 3 m and 2.9 m/s over "
                        "6 m, p90 2.7 and 4.4. The spec's U(2,6) would put "
                        "most of the band beyond the body -- at 6 m/s a 3 m "
                        "pass is due in 0.5 s and the ant needs 1.4 s just to "
                        "REACH the ball -- and an unreachable band is a flat "
                        "gradient, not a hard task.")
    p.add_argument("--deadline-range", type=float, nargs=2, default=[0.5, 4.0],
                   help="--reward-kind timed: clamp on T (s). With the default "
                        "pace band and a 3-6 m target this spans 1.0-4.0 s, so "
                        "the clamp is a guard rail rather than the thing "
                        "setting the difficulty.")
    p.add_argument("--time-coef", type=float, default=0.0,
                   help="decay arrival by exp(-k*t) at closest approach, so a "
                        "fast pass beats a slow trickle. 0 = paper-faithful")
    p.add_argument("--target-cone", type=float, default=0.0,
                   help="stage 2+: command may sit up to +/- this many RADIANS "
                        "off the colinear line (with --fixed-start)")
    p.add_argument("--cone-anneal-steps", type=int, default=0)
    p.add_argument("--cone-start", type=float, default=0.0)
    p.add_argument("--cone-max", type=float, default=np.pi)

    # -- run plumbing -------------------------------------------------------
    p.add_argument("--max-hours", type=float, default=48.0,
                   help="stop after this much wallclock, whatever step count "
                        "that is; --steps stays as a backstop")
    p.add_argument("--ball-radius", type=float, default=0.15,
                   help="dm_control's SoccerBall takes radius/mass as ARGUMENTS "
                        "(0.35/0.045 are defaults, not a spec) -- what makes it "
                        "a soccer ball is condim 6 + rolling friction + "
                        "priority 1, which are size-independent. 0.35 m put the "
                        "ball at the ant's torso height (ratio 1.43) so it could "
                        "only be shoved with the body; 0.15 puts it at leg "
                        "height, matching dm_control fetch's 0.53.")
    p.add_argument("--ball-mass", type=float, default=0.045,
                   help="rolling deceleration is mass-independent (measured: "
                        "1.81 m from 4 m/s at every mass tried), so this only "
                        "affects how much the ball squirts on contact")
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
    p.add_argument("--w-upright", type=float, default=1.0,
                   help="exponent on the uprightness factor the whole reward is "
                        "MULTIPLIED by, as dm_control's fetch reward does. 0 "
                        "disables it, which is what runs before 2026-08-09 "
                        "effectively used -- and with posture unpriced the ant "
                        "learned to splay flat and shove the ball with its "
                        "torso instead of its legs.")
    p.add_argument("--creature-xml", default="creature_configs/ant.xml")
    p.add_argument("--run-name", required=True)
    p.add_argument("--video-secs", type=float, default=300.0)
    p.add_argument("--first-video-secs", type=float, default=60.0)
    p.add_argument("--ckpt-secs", type=float, default=1800.0)
    p.add_argument("--mid-ckpt-frac", type=float, default=0.5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gcs-bucket", default=None)
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    if args.w_strike is None:
        # Resolved here, not in the parser, so the value that lands in
        # config.json is the one the run actually used.
        args.w_strike = 0.0 if args.reward_kind == "timed" else 0.5
    run(args, task="kick", make_env_fn=make_env, make_eval_fn=make_eval)


def run(args, task, make_env_fn, make_eval_fn):
    """Shared train loop for kick and shoot. Kept in this module (rather than
    edited into an existing trainer) so the two new drills add files and change
    none -- WS1 merges these without touching dribble's."""
    torch.manual_seed(args.seed)

    run_dir = os.path.join("runs_v2", args.run_name)
    if os.path.isdir(run_dir) and os.listdir(run_dir) and not args.resume:
        raise SystemExit(f"{run_dir} exists and is non-empty. Pass --resume to "
                         f"continue that run, or pick a different --run-name.")
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    config = {**vars(args), "git_sha": git_sha, "backend": "mujoco_warp",
              "task": task}
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

    from rower_soccer.warp_port.ppo import (ActorCritic, SimpleActorCritic,
                                            PPOTrainer, export_sb3_compatible,
                                            load_checkpoint, load_pretrained,
                                            save_checkpoint)

    env = make_env_fn(args, num_worlds=args.worlds, seed=args.seed)
    if args.plain:
        ac = SimpleActorCritic(env.obs_dim, env.act_dim)
    else:
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
    # Freeze BEFORE any checkpoint load: it rebuilds the optimizer over
    # only the trainable parameters, and load_checkpoint restores a saved
    # optimizer state into it. Freeze afterwards and a frozen-decoder run
    # can be checkpointed but never resumed -- the saved single reduced
    # parameter group does not match a full-parameter Adam.
    if args.freeze_decoder:
        if args.plain:
            raise SystemExit("--freeze-decoder is meaningless with --plain: the "
                             "plain baseline has no decoder.")
        frozen = 0
        for mod in (ac.mlp_extractor.decoder, ac.action_net):
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
        # Adam was built over every parameter; rebuild it over the live ones so
        # frozen weights cannot drift via stale moments.
        trainer.opt = torch.optim.Adam(
            [p for p in ac.parameters() if p.requires_grad], lr=args.lr)

    if args.resume and os.path.exists(ckpt_path):
        start_steps = load_checkpoint(trainer, ckpt_path)
        print(f"[setup] resumed from {ckpt_path} at step {start_steps:,}", flush=True)
    elif args.init_from:
        # Warm start only on a fresh run: on --resume the checkpoint already
        # holds these weights, further trained.
        load_pretrained(ac, args.init_from, device=trainer.device)


    print(f"[setup] task={task} worlds={env.n} obs={env.obs_dim} "
          f"act={env.act_dim} proprio={len(env.proprio_indices)} "
          f"task_obs={len(env.task_indices)} contact_dist={env.contact_dist:.2f}m "
          f"segment_steps={env.segment_steps} "
          f"steps/iter={trainer.T * trainer.N:,}", flush=True)
    eval_env, eval_ren = make_eval_fn(args)
    t0 = time.perf_counter()
    last_ckpt = t0
    last_video = t0 - max(0.0, args.video_secs - args.first_video_secs)
    it = 0
    deadline = t0 + args.max_hours * 3600.0
    cone = getattr(args, "target_cone", 0.0)
    while trainer.total_steps < args.steps and time.perf_counter() < deadline:
        if args.shaping_anneal_steps > 0:
            env.shaping_scale = max(
                0.0, 1.0 - trainer.total_steps / args.shaping_anneal_steps)
        if getattr(args, "cone_anneal_steps", 0) > 0:
            frac = min(1.0, trainer.total_steps / args.cone_anneal_steps)
            cone = args.cone_start + frac * (args.cone_max - args.cone_start)
            env.target_cone = cone
            eval_env.target_cone = cone
        stats = trainer.train_iter()
        it += 1
        now = time.perf_counter()
        fps = (trainer.total_steps - start_steps) / (now - t0)
        eta_min = max(0.0, (deadline - now) / 60)
        if it % 5 == 0:
            fit = float(env.fitness().mean())
            strikes = float(env.credit_count.mean())
            extra = ""
            if hasattr(env, "goals"):
                extra = f" goals/ep={float(env.goals.mean()):.2f}"
            # v7: the whole point of the anchor is visible here. A creature
            # that strikes and stays sits near 0; one that walks the ball
            # downfield climbs. Instantaneous, not an episode mean, so it is a
            # snapshot of where the batch happens to be standing.
            anchor = None
            if getattr(args, "w_anchor", 0.0) > 0:
                anchor = float(env.anchor_excess(args.anchor_free_radius).mean())
                extra += f" anchor={anchor:.2f}m"
            # diverged: world-steps whose physics went non-finite (ppo.collect).
            # Expected 0 or a trickle; if it climbs the contact model is wrong.
            print(f"[monitor] step={trainer.total_steps:,}/{args.steps:,} "
                  f"({100*trainer.total_steps/args.steps:.1f}%) fps={fps:,.0f} "
                  f"eta={eta_min:.1f}min ep_rew={stats['ep_rew_env_mean']:.1f} "
                  f"fitness={fit:.3f} strikes/ep={strikes:.2f}{extra} "
                  f"std={stats['std']:.3f} diverged={trainer.n_diverged:,}",
                  flush=True)
            if use_wandb:
                import wandb
                log = {"env_step": trainer.total_steps,
                       "monitor/fps": fps, "monitor/eta_min": eta_min,
                       "train/ep_rew": stats["ep_rew_env_mean"],
                       # Unshaped gate metric; the shaping terms cannot inflate it.
                       "train/fitness": fit,
                       "train/strikes_per_ep": strikes,
                       "train/entropy": stats["ent"], "train/std": stats["std"],
                       "train/pg_loss": stats["pg"], "train/vf_loss": stats["vf"]}
                if hasattr(env, "goals"):
                    log["train/goals_per_ep"] = float(env.goals.mean())
                if anchor is not None:
                    log["train/anchor_excess_m"] = anchor
                if getattr(args, "cone_anneal_steps", 0) > 0:
                    log["train/cone_deg"] = float(np.rad2deg(cone))
                wandb.log(log)
        if args.video_secs > 0 and now - last_video >= args.video_secs:
            last_video = now
            vpath = os.path.join(run_dir, "videos",
                                 f"eval_step_{trainer.total_steps:010d}.mp4")
            from rower_soccer.warp_port.render import eval_video
            ep_rew, fit = eval_video(eval_env, ac, vpath, eval_ren)
            print(f"[monitor] video: {vpath} (WARP eval "
                  f"ep_rew={ep_rew:.1f} fitness={fit:.3f})", flush=True)
            # Keep the BEST policy, not just the latest: late collapse is not
            # exotic in long PPO runs and has cost this project a good policy
            # before (follow_v5 went 476 -> 166 in its final stretch, and only
            # the 166 was saved). Scored on FITNESS, which no shaping term can
            # inflate.
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
