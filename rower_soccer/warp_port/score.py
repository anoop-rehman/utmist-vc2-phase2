"""Batched deterministic scoring of a policy, for `best.pt` selection.

Why this module exists
----------------------
`best.pt` used to be saved whenever the RENDER eval's fitness beat a running
max. That eval is `render.eval_video`: ONE world, ONE 15 s episode, with the
target / spawn band drawn exactly once. So `best.pt` was
`max` over ~136 single-episode draws taken across a run -- the LUCKIEST DRAW,
not the best policy. Measured across all four live drills (2026-08-11,
docs/DRILL_V4_NOTES.md section 10):

    run                | best.pt fitness | typical | ratio
    dribble_ant_v3     | 0.980           | ~0.60   | 1.6x
    kick_ant_v4_timed  | 0.312           | ~0.12   | 2.6x
    kick_ant_v6_timed  | 0.300           | ~0.10   | 3.0x
    shoot_ant_v4       | 0.689           | ~0.40   | 1.7x

Re-scoring dribble's OWN best.pt ten times, one episode each, then gave mean
0.895 with a max of 0.973 and a single-episode sd of 0.176 (pooled over 640
episodes): the 0.980 that got pinned is the top of that checkpoint's own draw
distribution, not evidence of a better policy. Two consequences, both live:
every registry pin is a luck-selected checkpoint, and once `best_score` ratchets
to an outlier a genuinely better policy stops being saved at all.

The fix is not a different metric. It is the SAME quantity estimated with less
noise: run the same deterministic policy in N worlds instead of 1 and average.
The one-world env was never chosen for scoring reasons -- it is the RENDER env,
and the renderer can only draw one world. Fitness needs no renderer.

Measured at 64 worlds on dribble: single-episode sd 0.1758, batched sd 0.0231,
against the 0.1758/sqrt(64) = 0.0220 that averaging predicts. The two estimators
agree on the mean (0.8948 vs 0.8801, inside a combined 3-sigma of 0.168), which
is what makes this a variance fix rather than a new metric.

Design notes
------------
* `score_policy` takes an already-built env. Building a Warp env costs a scene
  compile plus a CUDA-graph capture, so the trainers build the scoring env ONCE
  next to `make_eval`'s render env and reuse it every evaluation.

* The scoring env must be built with `use_graph=True`. Measured on the dribble
  eval env: graph=False is 1462.1 ms/step against 92.4 ms/step captured, i.e. a
  600-step episode costs 15 minutes uncaptured versus 55 s captured. A scoring
  env that takes 15 minutes per call would eat the run.

* `seed` re-seeds the env's RNG before the rollout, so every call scores the
  policy on the SAME N task draws: identical spawn positions, target distances
  and target velocities, verified bitwise on `qpos` and `target_xy` in
  tests/test_batched_score.py. Successive evaluations in a run are therefore a
  PAIRED comparison over one fixed task set, which removes the between-task
  component of the noise on top of the N-fold averaging.

  What it does NOT buy is a bit-reproducible number, and it cannot: mujoco_warp
  is not bitwise deterministic run-to-run (its solver accumulates with atomics,
  so reduction order varies). Measured on the dribble eval env, two rollouts
  from the same seed start from a bit-identical qpos and then drift apart over
  600 chaotic steps. The residual is measured in the same test file -- see
  docs/DRILL_V4_NOTES.md section 12 for the numbers. It is far below the
  single-episode spread this module exists to remove, which is the claim that
  actually matters.

* Deterministic actions: `ac.dist(obs).mean`, never a sample. At our entropy
  floor log_std is 0.30 on a +/-1 action, which would swamp exactly the fine
  control the eval is meant to measure. Same rule as `render.eval_video`.
"""

from collections import namedtuple

import torch

# fitness / ep_rew: means over worlds, the quantities `eval_video` returns for
# its single world. fitness_std / fitness_sem: the spread across worlds and the
# standard error of the mean -- the number that says how much of a policy
# difference this eval can actually resolve. worlds: N, so a log line can say
# what the mean was taken over.
ScoreResult = namedtuple(
    "ScoreResult", "fitness ep_rew fitness_std fitness_sem worlds")


@torch.no_grad()
def score_policy(env, ac, seed=None, deterministic=True):
    """Run ONE full deterministic episode in every world of `env` and average.

    Returns a `ScoreResult`. `fitness` is the mean over worlds of the env's own
    `fitness()` read at the end of the episode -- byte-for-byte the quantity
    `render.eval_video` returns, only averaged over N worlds instead of read off
    one. `ep_rew` is the mean episode return, which is what the follow trainer
    selects on.

    `seed` (int or None): re-seed `env.gen` before the reset so the N task draws
    are identical on every call. Pass None to let the env's generator run on.
    This fixes the TASK, not the trajectory -- mujoco_warp is not bitwise
    reproducible, so the same seed does not give the same number to the last
    digit. See the module docstring.

    No renderer is involved and no frames are kept, so the cost is pure physics:
    `env.episode_steps` control steps of an N-world captured graph.
    """
    if seed is not None:
        # env.gen drives every task draw: spawn positions, target bands, and --
        # in kick/shoot -- the mid-episode segment respawns too. Re-seeding here
        # fixes the whole episode's randomness, which is what makes successive
        # calls a paired comparison rather than two independent samples.
        env.gen.manual_seed(int(seed))

    obs = env.reset()
    ep_rew = torch.zeros(env.n, device=env.device)
    done = False
    while not done:
        d = ac.dist(obs.float())
        a = (d.mean if deterministic else d.sample()).clamp(-1, 1)
        obs, r, done = env.step(a)
        ep_rew += r

    fit = env.fitness().float()
    # ddof=1: this is a sample spread over N drawn tasks, used to size the
    # standard error of their mean. N=1 would make it NaN, so guard it -- a
    # one-world scoring env is a degenerate but legal configuration (it is
    # exactly the old behaviour, which is useful for the variance tests).
    std = float(fit.std(unbiased=True)) if env.n > 1 else 0.0
    return ScoreResult(fitness=float(fit.mean()),
                       ep_rew=float(ep_rew.mean()),
                       fitness_std=std,
                       fitness_sem=std / (env.n ** 0.5),
                       worlds=int(env.n))


def add_args(p, default_worlds=64):
    """Attach the scoring flags to a trainer's ArgumentParser.

    Shared by all four drill trainers so the flag names, defaults and help text
    cannot drift apart between them.
    """
    p.add_argument("--score-worlds", type=int, default=default_worlds,
                   help="worlds in the deterministic SCORING env that selects "
                        "best.pt. 64 cuts the standard error ~8x versus the "
                        "one-world render eval, which is what made best.pt a "
                        "max over lucky draws (docs/DRILL_V4_NOTES.md 10). "
                        "0 disables batched scoring and falls back to the old "
                        "single-episode selection.")
    p.add_argument("--score-secs", type=float, default=0.0,
                   help="wallclock cadence for the batched score. 0 (default) "
                        "reuses --video-secs, so scoring happens exactly when a "
                        "video is rendered and the two numbers in the log line "
                        "describe the same weights.")
    p.add_argument("--score-seed", type=int, default=12345,
                   help="RNG seed re-applied to the scoring env before EVERY "
                        "rollout, so all evaluations in a run face the same N "
                        "task draws. That makes score-vs-score a paired "
                        "comparison: the difference is the policy, not the draw.")
