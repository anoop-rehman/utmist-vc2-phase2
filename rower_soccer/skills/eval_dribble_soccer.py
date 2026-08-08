"""Score a `dribble` checkpoint in the CPU soccer env, before pinning it.

The registry's `follow` entry records why this script exists: `best.pt` is
whichever checkpoint won the deterministic WARP eval, and for `follow_ant_v1`
that one carried an outright pathology (a symmetric-state fixed point) while
`final.pt` did not. "Best" means best on the metric the trainer logged, in the
simulator it ran. So measure in the env the GAME uses before pinning.

Dribble's metric is not follow's. Follow asks how close the CREATURE got; dribble
asks how close the BALL got, which a policy can fail in two extra ways: never
touching the ball at all, and shoving it past the target. Both are reported.

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m \
      rower_soccer.skills.eval_dribble_soccer --model runs_v2/dribble_ant_v1/best.pt
"""

import argparse
import numpy as np

# Ball-relative legs: the target is placed `offset` from the BALL, so each leg
# asks for the same push regardless of where the previous leg left things. An
# ant-relative or absolute schedule degenerates into "how fast can it cross the
# pitch" once the ball drifts.
DEFAULT_PLAN = [
    ((3.0, 0.0), 20.0),
    ((-1.5, 3.0), 20.0),
    ((0.0, -3.0), 20.0),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--creature", default="ant")
    p.add_argument("--model", default=None,
                   help="dribble checkpoint to score (default: the registry's)")
    p.add_argument("--action-mode", default="mean", choices=["mean", "noise"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--video", default=None)
    p.add_argument("--fps", type=int, default=40)
    args = p.parse_args()

    import torch
    torch.set_num_threads(1)

    from rower_soccer.skills import SkillController
    from rower_soccer.skills.soccer import SoccerFrameSource, make_skill_soccer_env

    legs, touched_any = [], 0
    for seed in range(args.seeds):
        env = make_skill_soccer_env(home=(args.creature,), time_limit=1e6,
                                    random_state=seed, match_dt=True)
        src = SoccerFrameSource(env)
        ctrl = SkillController(args.creature, action_mode=args.action_mode,
                               seed=seed, name="home_0",
                               checkpoints=({"dribble": args.model} if args.model
                                            else None))
        if seed == 0:
            print(f"[eval] {ctrl.contract.describe()}  "
                  f"skills={ctrl.available_skills()}", flush=True)

        ts = env.reset()
        hz = int(round(1.0 / env.task.control_timestep))
        # Put the ball within reach so the leg measures dribbling, not searching:
        # the drill spawns it close, and "walk 8 m to the ball first" is the
        # follow skill's job, not this metric's.
        _place_ball_near_ant(env, src, dist=1.0)
        ts = env.step([np.zeros(ctrl.act_dim)])

        for offset, secs in DEFAULT_PLAN:
            ball0 = src.ball_xy().copy()
            target = tuple(ball0 + np.asarray(offset))
            ctrl.set_command("dribble", target_xy=target)
            d0 = float(np.linalg.norm(ball0 - np.asarray(target)))
            moved = 0.0
            for _ in range(int(secs * hz)):
                a = ctrl.action(src.frame(ts, 0))
                ts = env.step([a])
                moved = max(moved, float(np.linalg.norm(src.ball_xy() - ball0)))
            d1 = float(np.linalg.norm(src.ball_xy() - np.asarray(target)))
            legs.append(dict(seed=seed, d0=d0, d1=d1, moved=moved,
                             fitness=float(np.exp(-0.5 * d1))))
            touched_any += int(moved > 0.15)
            print(f"  seed {seed} leg -> target {d0:.2f}m away: ball ended "
                  f"{d1:.2f}m out (moved {moved:.2f}m) fitness {np.exp(-0.5*d1):.3f}",
                  flush=True)

    d1 = np.array([l["d1"] for l in legs])
    fit = np.array([l["fitness"] for l in legs])
    print(f"\n[eval] {args.model or 'registry default'}")
    print(f"  legs                 {len(legs)}")
    print(f"  ball moved >0.15 m   {touched_any}/{len(legs)}   "
          "(a policy that never touches the ball scores 0 here)")
    print(f"  final ball distance  median {np.median(d1):.2f} m  "
          f"mean {d1.mean():.2f} m  worst {d1.max():.2f} m")
    print(f"  fitness exp(-d/2)    median {np.median(fit):.3f}  mean {fit.mean():.3f}")


def _place_ball_near_ant(env, src, dist=1.0):
    """Drop the ball `dist` metres in front of the ant, at rest."""
    physics = env.physics
    ball = env.task.ball
    joint = physics.bind(ball.root_body.freejoint)
    ant_xy = src.root_xy(0)
    qpos = np.array(joint.qpos)
    qpos[0:2] = ant_xy + np.array([dist, 0.0])
    qpos[2] = float(physics.bind(ball.root_body).xpos[2])
    joint.qpos = qpos
    joint.qvel = np.zeros(6)


if __name__ == "__main__":
    main()
