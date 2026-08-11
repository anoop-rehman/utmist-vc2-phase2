"""best.pt selection: the batched deterministic score (warp_port/score.py).

Plain python, no pytest in this venv:

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m tests.test_batched_score
    ... --no-physics    # stub-env algebra only, no GPU, ~1 s
    ... --k 10          # samples per estimator in the variance measurement

The physics group is EXPENSIVE: 2*K + repeats full 15 s episodes, ~50 s of GPU
each on a shared card, so ~20 min at the defaults. It is a measurement, not a
smoke test; run it when score.py changes, not on every edit.

What this exists to defend
--------------------------
`best.pt` was saved on a running max over the RENDER eval: one world, one 15 s
episode, task drawn once. Max-of-N over a noisy estimator grows with N, so
across ~136 evals `best.pt` became the luckiest draw rather than the best
policy -- measured at 1.6-3.0x the typical fitness in all four live drills
(docs/DRILL_V4_NOTES.md section 10).

The claim of the fix is precise, and each part is a test below:

 1. the seed REPRODUCES THE TASK DRAWS bitwise, which is what makes successive
    evals a paired comparison over one fixed task set. Note what is NOT claimed:
    the score is not bit-reproducible, because mujoco_warp is not bitwise
    deterministic run to run (atomics in the solver). Measured: two resets from
    the same seed give identical qpos while the trajectories drift apart over
    the episode;
 2. that residual same-seed noise is MEASURED, so the floor on what the score
    can resolve is a number and not an adjective;
 3. the score has MATERIALLY LOWER VARIANCE than the single-episode number.
    Measured, not asserted from theory: K single-episode evals at K different
    seeds against K batched evals at K different seeds, both standard
    deviations reported;
 4. it estimates the SAME QUANTITY -- the two means agree inside their combined
    standard error. Without this the "fix" could just be a different metric
    that happens to be quieter.

The physics group runs on the dribble drill, whose config is read from the live
run's own config.json when present, so the numbers describe the task the fix
was motivated by. The checkpoint is COPIED to a temp file before loading: the
live trainer rewrites best.pt in place and reading it mid-write would fail for
reasons that have nothing to do with this code.
"""

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from rower_soccer.warp_port import score  # noqa: E402

_results = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        ok, err = True, ""
    except Exception:                                               # noqa: BLE001
        import traceback
        ok, detail, err = False, "", traceback.format_exc()
    dt = time.perf_counter() - t0
    _results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt:.1f}s) {detail}", flush=True)
    if err:
        print(err, flush=True)


# ---------------------------------------------------------------------------
# algebra: score_policy against a stub env, no GPU
# ---------------------------------------------------------------------------
class SpyGen:
    """Stands in for the env's torch.Generator and records the seed applied.

    A wrapper, not a patched Generator: torch._C.Generator.manual_seed is
    read-only, so the spy has to sit beside it.
    """

    def __init__(self):
        self.real = torch.Generator(device="cpu")
        self.seeded_with = None

    def manual_seed(self, x):
        self.seeded_with = x
        return self.real.manual_seed(x)


class StubEnv:
    """The handful of attributes score_policy touches.

    Per-world fitness is fixed and KNOWN (`fit_values`), so the mean/std/sem it
    reports can be checked exactly rather than approximately. Reward is 1.0 per
    world per step, so ep_rew must come out as the step count.
    """

    def __init__(self, fit_values, episode_steps=5, obs_dim=3, act_dim=2):
        self.n = len(fit_values)
        self.device = "cpu"
        self.episode_steps = episode_steps
        self.obs_dim, self.act_dim = obs_dim, act_dim
        # Records what manual_seed was called with -- the paired-draw guarantee
        # is exactly "the env RNG is re-seeded before the rollout".
        self.gen = SpyGen()
        self._fit = torch.tensor(fit_values, dtype=torch.float32)
        self.t = 0
        self.n_resets = 0
        self.seen_actions = []

    @property
    def seeded_with(self):
        return self.gen.seeded_with

    def reset(self):
        self.t = 0
        self.n_resets += 1
        return torch.zeros(self.n, self.obs_dim)

    def step(self, a):
        self.seen_actions.append(a.clone())
        self.t += 1
        return (torch.zeros(self.n, self.obs_dim),
                torch.ones(self.n),
                self.t >= self.episode_steps)

    def fitness(self):
        return self._fit


class StubAC:
    """dist() returns a Normal whose mean is a constant well OUTSIDE [-1, 1] and
    whose std is huge. Deterministic scoring must therefore emit exactly the
    clamp of the mean, every step, in every world -- if it ever sampled, the
    actions would not be constant."""

    def __init__(self, mean=2.5, std=5.0, act_dim=2):
        self.mean, self.std, self.act_dim = mean, std, act_dim

    def dist(self, obs):
        n = obs.shape[0]
        return torch.distributions.Normal(
            torch.full((n, self.act_dim), self.mean),
            torch.full((n, self.act_dim), self.std))


def t_score_is_the_mean_over_worlds():
    vals = [0.1, 0.2, 0.9, 0.4]
    env = StubEnv(vals, episode_steps=7)
    r = score.score_policy(env, StubAC())
    assert r.worlds == 4, r.worlds
    assert abs(r.fitness - statistics.mean(vals)) < 1e-6, r.fitness
    # 1.0 reward per step per world, so the mean episode return is the step count.
    assert abs(r.ep_rew - 7.0) < 1e-6, r.ep_rew
    exp_std = statistics.stdev(vals)
    assert abs(r.fitness_std - exp_std) < 1e-5, (r.fitness_std, exp_std)
    assert abs(r.fitness_sem - exp_std / 2.0) < 1e-5, r.fitness_sem
    return f"mean {r.fitness:.3f} sem {r.fitness_sem:.3f} over {r.worlds} worlds"


def t_score_runs_exactly_one_full_episode():
    env = StubEnv([0.5] * 3, episode_steps=11)
    score.score_policy(env, StubAC())
    assert env.n_resets == 1, env.n_resets
    assert len(env.seen_actions) == 11, len(env.seen_actions)
    return "1 reset, 11 steps"


def t_score_uses_the_distribution_mean_never_a_sample():
    env = StubEnv([0.5] * 3, episode_steps=4)
    score.score_policy(env, StubAC(mean=2.5, std=5.0))
    a = torch.stack(env.seen_actions)
    # mean 2.5 clamped to 1.0, identically, everywhere. A sample from
    # Normal(2.5, 5) would clamp to a mix of -1 and 1.
    assert torch.allclose(a, torch.ones_like(a)), a
    # And the sampling path, for contrast, must NOT be constant -- otherwise
    # this test would pass even if `deterministic` were ignored.
    env2 = StubEnv([0.5] * 3, episode_steps=4)
    score.score_policy(env2, StubAC(mean=0.0, std=5.0), deterministic=False)
    b = torch.stack(env2.seen_actions)
    assert not torch.allclose(b, b[0].expand_as(b)), "sampling produced constants"
    return "deterministic actions are clamp(mean); sampling differs"


def t_score_reseeds_the_env_rng():
    env = StubEnv([0.5] * 3, episode_steps=2)
    score.score_policy(env, StubAC(), seed=4242)
    assert env.seeded_with == 4242, env.seeded_with
    env2 = StubEnv([0.5] * 3, episode_steps=2)
    score.score_policy(env2, StubAC(), seed=None)
    assert env2.seeded_with is None, env2.seeded_with
    return "seed=4242 re-applied; seed=None leaves the generator alone"


def t_one_world_score_has_no_spread():
    """A one-world scoring env is legal (it is the OLD behaviour) and must not
    produce a NaN std from an unbiased estimator over a single sample."""
    env = StubEnv([0.37], episode_steps=3)
    r = score.score_policy(env, StubAC())
    assert r.fitness_std == 0.0 and r.fitness_sem == 0.0, r
    assert abs(r.fitness - 0.37) < 1e-6, r.fitness
    return "n=1 -> std 0.0, not NaN"


# ---------------------------------------------------------------------------
# physics: the real dribble env, the real checkpoint
# ---------------------------------------------------------------------------
_DRIBBLE_RUN = os.path.join(_ROOT, "runs_v2", "dribble_ant_v3")


def _dribble_args():
    """An args Namespace for train_dribble_warp.make_eval_env.

    Prefers the LIVE run's config.json -- its keys are argparse dests, so it
    deserialises straight into a Namespace and the test then grades the task the
    drill is actually running. Falls back to the trainer's own defaults.
    """
    cfg_path = os.path.join(_DRIBBLE_RUN, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        return argparse.Namespace(**cfg), cfg_path
    cfg = dict(creature_xml="creature_configs/ant.xml", ball_radius=0.15,
               ball_mass=0.045, arena="pitch", pitch_scale=0.3125,
               target_speed=[0.2, 1.0], ball_spawn=[1.5, 3.0],
               target_dist=[2.0, 5.0], bounds=10.0, reward_coef=0.5,
               w_player_to_ball=0.15, w_ball_to_target=0.3,
               reward_mode="paper", progress_scale=2.0, approach_scale=0.5,
               episode_secs=15.0, energy_coef=0.0, smooth_coef=0.0,
               fixed_start=False, target_cone=0.0)
    return argparse.Namespace(**cfg), "(trainer defaults)"


def _load_policy(env, args, checkpoint):
    from rower_soccer.warp_port.ppo import ActorCritic, load_pretrained
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(),
                     z_dim=getattr(args, "z_dim", 16),
                     state_dependent_std=getattr(args, "state_dependent_std",
                                                 False)).to(env.device)
    if checkpoint:
        # Copy first: the live trainer rewrites best.pt in place.
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            tmp = tf.name
        shutil.copy2(checkpoint, tmp)
        try:
            load_pretrained(ac, tmp, device=env.device, verbose=False)
        finally:
            os.unlink(tmp)
    ac.eval()
    return ac


class _Fixture:
    """Built once (a Warp env costs a scene compile + a graph capture) and shared
    by the three physics tests, exactly as the trainers reuse their scoring env."""

    def __init__(self, worlds, checkpoint):
        from rower_soccer.warp_port.train_dribble_warp import make_eval_env
        self.args, src = _dribble_args()
        print(f"[fixture] dribble config from {src}", flush=True)
        # seed here is irrelevant: score_policy re-seeds before every rollout.
        self.env1 = make_eval_env(self.args, num_worlds=1, seed=0)
        self.envN = make_eval_env(self.args, num_worlds=worlds, seed=0)
        self.ac = _load_policy(self.env1, self.args, checkpoint)
        self.worlds = worlds
        print(f"[fixture] worlds={worlds} episode_steps={self.env1.episode_steps} "
              f"checkpoint={checkpoint or '(random init)'}", flush=True)
        # Filled by the measurement tests and read by the later ones, so the
        # rollouts are paid for once.
        self.singles = None
        self.batched = None
        self.repeats = None
        self.sigma1 = None


_FIX = None


def t_task_draws_are_reproducible():
    """The seed fixes the TASK, bitwise. That is the part score.py controls.

    It does NOT fix the trajectory, and no amount of seeding can: mujoco_warp
    accumulates constraint work with atomics, so the reduction order -- and
    therefore the last bits of qacc -- varies run to run. That divergence is
    measured by t_repeat_noise_is_small, not here. The reset observation is
    reported below rather than asserted on precisely because it is sometimes
    bit-equal and sometimes not; qpos and target_xy, which are OURS, are
    asserted.
    """
    env = _FIX.envN
    env.gen.manual_seed(777)
    o1, q1, t1 = env.reset().clone(), env.qpos.clone(), env.target_xy.clone()
    env.gen.manual_seed(777)
    o2, q2, t2 = env.reset().clone(), env.qpos.clone(), env.target_xy.clone()
    assert torch.equal(q1, q2), "same seed drew a different spawn"
    assert torch.equal(t1, t2), "same seed drew a different target"
    # A different seed must move the draw, or the check above is vacuous.
    env.gen.manual_seed(778)
    env.reset()
    assert not torch.equal(t1, env.target_xy), "the seed is being ignored"
    d_obs = float((o1 - o2).abs().max())
    return (f"qpos + target bitwise equal; reset obs max delta {d_obs:.2e} "
            f"(the simulator's, not the draw's)")


def t_repeat_noise_is_small(k):
    """MEASURED: how much the batched score moves when NOTHING changes.

    Same weights, same seed, K rollouts. This is the floor on what the score
    can resolve, and it exists only because the simulator is not bitwise
    reproducible. Recorded here so section 12's claim is a number rather than
    an adjective; the assertion it has to satisfy lives in the variance test,
    which compares it against the single-episode spread.
    """
    reps = [score.score_policy(_FIX.envN, _FIX.ac, seed=777).fitness
            for _ in range(k)]
    _FIX.repeats = reps
    sd = statistics.stdev(reps)
    print(f"    repeat (same seed, n={_FIX.worlds}): mean "
          f"{statistics.mean(reps):.4f} sd {sd:.4f}  "
          f"min {min(reps):.3f} max {max(reps):.3f}", flush=True)
    assert all(0.0 <= x <= 1.0 for x in reps), reps
    return f"sd {sd:.4f} over {k} identical calls"


def t_batched_score_has_lower_variance(k):
    """THE measurement: K single-episode evals vs K batched evals, both at K
    different seeds, both standard deviations reported.

    Two estimates of the single-episode spread are reported, and the difference
    between them matters:

      sd(direct)  -- the sample sd of the K one-world evals. This is the
                     head-to-head number, and it is a BAD estimator at K=8: the
                     single-episode fitness distribution is bounded above and
                     has a long lower tail, so whether one bad draw lands in the
                     sample swings the sd by 3x. Two runs of this file at the
                     same K seeds measured 0.1132 and 0.0315.
      sigma1      -- pooled from the per-world spread the batched rollouts
                     already report (K * worlds episodes, 512 at the defaults).
                     Every world of a batched rollout IS an independent
                     single-episode draw, so this is the same quantity measured
                     with 64x the data and no extra GPU time.

    The strong assertion is therefore made against sigma1; the direct pair is
    kept because it is the comparison the change was asked to demonstrate.
    """
    singles = [score.score_policy(_FIX.env1, _FIX.ac, seed=1000 + i).fitness
               for i in range(k)]
    runs = [score.score_policy(_FIX.envN, _FIX.ac, seed=2000 + i)
            for i in range(k)]
    batched = [r.fitness for r in runs]
    _FIX.singles, _FIX.batched = singles, batched
    sd1 = statistics.stdev(singles)
    sdN = statistics.stdev(batched)
    # Pool the per-run world variances: sqrt(mean of variances), the usual
    # pooled-sd of groups with equal n.
    sigma1 = (sum(r.fitness_std ** 2 for r in runs) / len(runs)) ** 0.5
    _FIX.sigma1 = sigma1
    n_ep = k * _FIX.worlds
    print(f"    single-episode (n=1):  mean {statistics.mean(singles):.4f} "
          f"sd {sd1:.4f}  min {min(singles):.3f} max {max(singles):.3f}  "
          f"[K={k} direct]", flush=True)
    print(f"    single-episode (pooled over {n_ep} worlds): sd {sigma1:.4f}",
          flush=True)
    print(f"    batched (n={_FIX.worlds}):      mean {statistics.mean(batched):.4f} "
          f"sd {sdN:.4f}  min {min(batched):.3f} max {max(batched):.3f}  "
          f"[K={k}]", flush=True)
    print(f"    predicted batched sd = sigma1/sqrt({_FIX.worlds}) = "
          f"{sigma1 / _FIX.worlds ** 0.5:.4f}", flush=True)
    assert sd1 > 0, "single-episode eval had zero spread; nothing to fix"
    # The head-to-head, as asked. Only strict inequality: at K=8 the direct sd
    # is too unstable to carry a tighter bound (see the docstring).
    assert sdN < sd1, (sdN, sd1)
    # The well-powered claim. 1/sqrt(64) predicts an 8x reduction; require 4x,
    # which leaves room for the ~25% CV on sdN at K=8 while still failing loudly
    # if the batching were not averaging independent draws.
    assert sigma1 > 4.0 * sdN, (sigma1, sdN)
    detail = (f"sd {sd1:.4f} (direct) / {sigma1:.4f} (pooled, {n_ep} eps) "
              f"-> {sdN:.4f}; {sigma1 / max(sdN, 1e-12):.1f}x lower against "
              f"sqrt({_FIX.worlds}) = {_FIX.worlds ** 0.5:.1f}x expected")
    if _FIX.repeats:
        # The simulator's own nondeterminism (same seed, same weights) is the
        # floor under sdN, so it must also sit below the single-episode spread.
        # If it did not, the batched score would be quiet about the task draw
        # and loud about nothing.
        sdr = statistics.stdev(_FIX.repeats)
        assert sdr < sigma1 / 2.0, (sdr, sigma1)
        detail += f"; same-seed repeat sd {sdr:.4f}"
    return detail


def t_batched_mean_matches_single_episode_mean():
    """Variance fix, not a different metric: the two estimators must agree on
    the mean inside their combined standard error.

    The single-episode arm's standard error is built from sigma1 (pooled over
    K*worlds episodes), not from the K-sample direct sd -- the direct sd is the
    unstable estimator described in the variance test, and a tolerance built on
    it would be unstable in exactly the same way.
    """
    assert _FIX.singles and _FIX.batched, "run the variance test first"
    k = len(_FIX.singles)
    m1, mN = statistics.mean(_FIX.singles), statistics.mean(_FIX.batched)
    sdN = statistics.stdev(_FIX.batched)
    se = (_FIX.sigma1 ** 2 / k + sdN ** 2 / k) ** 0.5
    # 3 sigma, plus a small absolute floor so a near-degenerate sd cannot make
    # the test hair-trigger.
    tol = max(3.0 * se, 0.02)
    diff = abs(m1 - mN)
    print(f"    single mean {m1:.4f}  batched mean {mN:.4f}  "
          f"diff {diff:.4f}  3*se {3 * se:.4f}", flush=True)
    assert diff <= tol, (diff, tol)
    return f"|{m1:.4f} - {mN:.4f}| = {diff:.4f} <= {tol:.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-physics", action="store_true",
                   help="stub-env algebra only: no GPU, no Warp compile")
    p.add_argument("--worlds", type=int, default=64,
                   help="worlds in the batched estimator (the trainers' default)")
    p.add_argument("--k", type=int, default=10,
                   help="samples per estimator in the variance measurement. Each "
                        "sample is a full 15 s episode (~24 s of GPU on a shared "
                        "card), so 2*K rollouts is the cost of this file.")
    p.add_argument("--repeats", type=int, default=5,
                   help="identical (same-seed) batched rollouts used to measure "
                        "the simulator's own run-to-run noise")
    p.add_argument("--checkpoint",
                   default=os.path.join(_DRIBBLE_RUN, "best.pt"),
                   help="policy to score. A trained one makes the variance "
                        "measurement representative; pass '' for a random init.")
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    check("score: fitness is the mean over worlds", t_score_is_the_mean_over_worlds)
    check("score: exactly one full episode", t_score_runs_exactly_one_full_episode)
    check("score: deterministic actions are the distribution mean",
          t_score_uses_the_distribution_mean_never_a_sample)
    check("score: the env RNG is re-seeded per rollout", t_score_reseeds_the_env_rng)
    check("score: one world gives 0 spread, not NaN", t_one_world_score_has_no_spread)

    if not args.no_physics:
        ckpt = args.checkpoint if args.checkpoint and os.path.exists(
            args.checkpoint) else None
        if args.checkpoint and ckpt is None:
            print(f"[fixture] {args.checkpoint} missing -> random-init policy",
                  flush=True)
        global _FIX
        _FIX = _Fixture(args.worlds, ckpt)
        check("score: the seed reproduces the task draws exactly",
              t_task_draws_are_reproducible)
        check("score: same-seed repeat noise (MEASURED)",
              lambda: t_repeat_noise_is_small(args.repeats))
        check("score: batched variance is materially lower (MEASURED)",
              lambda: t_batched_score_has_lower_variance(args.k))
        check("score: batched mean matches the single-episode mean",
              t_batched_mean_matches_single_episode_mean)

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
