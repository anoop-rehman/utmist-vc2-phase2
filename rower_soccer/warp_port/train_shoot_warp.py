"""Warp-accelerated shoot-drill training on a frozen follow/dribble/kick decoder.

Shoot is kick with the command pinned at dm_soccer's own goal (see shoot_env),
so this trainer is train_kick_warp's loop with shoot's flags -- the loop itself
is imported, not copied.

The natural queue order is follow -> dribble -> kick -> shoot, each
`--init-from` the last and `--freeze-decoder` throughout, so one motor skill
serves all four. Kick's task obs is 12 wide and shoot's is 13, so a kick
checkpoint re-inits the task encoder and the critic's input layer and transfers
everything else, decoder included.

Usage (ant, on the frozen follow decoder):
  MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.warp_port.train_shoot_warp \
      --run-name shoot_ant_v1 --creature-xml creature_configs/ant.xml \
      --init-from runs_v2/follow_ant_v1/best.pt --freeze-decoder \
      --gcs-bucket vc2-2026-checkpoints
"""

import argparse

import numpy as np
import torch

from rower_soccer.warp_port.train_kick_warp import run


def make_env(args, num_worlds, seed, use_graph=True):
    from rower_soccer.warp_port.shoot_env import WarpShootEnv
    return WarpShootEnv(
        num_worlds=num_worlds, seed=seed, use_graph=use_graph,
        creature_xml=args.creature_xml,
        episode_seconds=args.episode_secs, segment_seconds=args.segment_secs,
        shoot_dist_range=tuple(args.shoot_dist),
        ball_spawn_range=tuple(args.ball_spawn),
        shoot_y_frac=args.shoot_y_frac, spawn_cone=args.spawn_cone,
        out_of_play_dist=args.out_of_play,
        speed_clip=args.speed_clip, w_strike=args.w_strike,
        goal_bonus=args.goal_bonus, reward_coef=args.reward_coef,
        w_player_to_ball=args.w_player_to_ball, w_ball_to_cmd=args.w_ball_to_cmd,
        approach_scale=args.approach_scale, reward_mode=args.reward_mode,
        energy_coef=args.energy_coef, smooth_coef=args.smooth_coef,
        fixed_start=args.fixed_start)


def make_eval(args):
    """One-world Warp env + renderer. base_xml=None so the render scene is the
    PITCH -- the same scene shoot's physics runs in. Passing the arena here (as
    the other drills do) would draw a goal-less world and hide the whole task."""
    from rower_soccer.warp_port.render import WarpRenderer
    env = make_env(args, num_worlds=1, seed=7, use_graph=False)
    return env, WarpRenderer(args.creature_xml, has_ball=True, base_xml=None,
                             distance=14.0, elevation=-25.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000_000)
    p.add_argument("--worlds", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--ent-floor", type=float, default=None)
    p.add_argument("--ent-ceil", type=float, default=0.0)
    p.add_argument("--ent-anneal-steps", type=int, default=0)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--init-from", default=None,
                   help="checkpoint to warm-start from; the decoder (the motor "
                        "skill) carries over, task encoder + critic input layer "
                        "re-init")
    p.add_argument("--freeze-decoder", action="store_true",
                   help="freeze decoder + action head and train only the task "
                        "expert that emits z (the NPMP arrangement)")
    p.add_argument("--freeze-log-std", action="store_true")
    p.add_argument("--state-dependent-std", action="store_true")
    p.add_argument("--plain", action="store_true")

    # -- task ---------------------------------------------------------------
    p.add_argument("--episode-secs", type=float, default=15.0)
    p.add_argument("--segment-secs", type=float, default=5.0,
                   help="max seconds per SHOT ATTEMPT. The attempt also ends "
                        "early on a goal, on the ball crossing the line wide, "
                        "or on the ball leaving --out-of-play.")
    p.add_argument("--shoot-dist", type=float, nargs=2, default=[2.0, 5.0],
                   help="ball spawn distance from the goal mouth (m). Small on "
                        "purpose: the pitch is 96 m long and a 1 m/s ant would "
                        "spend the whole episode walking. Widen it only once "
                        "the drill is learned at short range.")
    p.add_argument("--ball-spawn", type=float, nargs=2, default=[1.5, 3.0],
                   help="creature spawn distance BEHIND the ball (m)")
    p.add_argument("--shoot-y-frac", type=float, default=0.4,
                   help="lateral ball spawn as a fraction of the goal's "
                        "half-width (11.88 m), so 0.4 => +/-4.75 m")
    p.add_argument("--spawn-cone", type=float, default=np.pi / 3,
                   help="radians the creature's spawn may sit off the "
                        "ball->goal line")
    p.add_argument("--out-of-play", type=float, default=20.0,
                   help="attempt is abandoned once the ball is this far from "
                        "the goal mouth (m)")

    # -- reward -------------------------------------------------------------
    p.add_argument("--reward-mode", default="paper", choices=["paper", "progress"])
    p.add_argument("--w-strike", type=float, default=0.5,
                   help="weight on the banked ball speed toward the goal, paid "
                        "at contact-break")
    p.add_argument("--goal-bonus", type=float, default=5.0,
                   help="paid once, on the step the ball crosses the line "
                        "between the posts and under the bar. Sized to dominate "
                        "a good strike (0.5 * 8 = 4) without saturating "
                        "rew_clip, so scoring beats striking hard at nothing.")
    p.add_argument("--speed-clip", type=float, default=8.0)
    p.add_argument("--reward-coef", type=float, default=0.5,
                   help="c in the fitness exp(-c * d), d = closest the ball got "
                        "to the goal MOUTH (0 anywhere inside it)")
    p.add_argument("--w-player-to-ball", type=float, default=0.15)
    p.add_argument("--w-ball-to-cmd", type=float, default=0.1)
    p.add_argument("--approach-scale", type=float, default=0.5)
    p.add_argument("--shaping-anneal-steps", type=int, default=0)
    p.add_argument("--energy-coef", type=float, default=0.0)
    p.add_argument("--smooth-coef", type=float, default=0.0)

    # -- curriculum ---------------------------------------------------------
    p.add_argument("--fixed-start", action="store_true",
                   help="stage 1: spawn the creature already FACING the ball "
                        "(and so, roughly, the goal) instead of at a random yaw")

    # -- run plumbing -------------------------------------------------------
    p.add_argument("--max-hours", type=float, default=48.0)
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
    # shoot has no steering cone (the goal does not move); `run` reads these
    # generically, so declare them off.
    args.target_cone = 0.0
    args.cone_anneal_steps = 0
    torch.manual_seed(args.seed)
    run(args, task="shoot", make_env_fn=make_env, make_eval_fn=make_eval)


if __name__ == "__main__":
    main()
