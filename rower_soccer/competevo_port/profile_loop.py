"""Where an iteration of the dev loop actually spends its wall clock.

    PYTHONPATH=. python -m rower_soccer.competevo_port.profile_loop \
        --worlds 1024 --rollout 64 --iters 5

Motivation: stage 2 established that the PHYSICS port is fast (a 1024-world
`RunToGoalDevEnv.step()` is ~10^2 ms, i.e. ~10^4 world-steps/s) while the
end-to-end training loop was running orders of magnitude below that. This
answers "so where does it go?" with measurements instead of a guess, and it
answers it for BOTH trainers, so the cost of stage 3's second learner is a
measured A/B rather than an assertion.

Method, and why it is built this way:

  * Nothing in the production classes is edited. The script monkeypatches
    timing wrappers around `env.step`, `DesignWriter.write` and the policy
    `act`/`value` calls for the duration of the run, so the thing being timed is
    the code that ships.
  * Every wrapper calls `torch.cuda.synchronize()` on both sides. Without that,
    an async launch queue moves the cost to whichever line happens to sync
    first and the split is fiction. The sync itself is charged to the section
    it belongs to, which is the honest attribution: a host-bound `.item()` in
    the middle of a rollout really does cost the pipeline that stall.
  * `sync_probe` times the ONE thing that is unavoidably host-bound per step in
    the current rollout -- the `float(info["forward"].mean())` logging line --
    by running it 200 times on a live info dict.
  * Iterations are reported as a median over `--iters` (after one warm-up),
    because this card is shared with other trainers and single-shot timings on
    it swing by 5x. The one-learner / two-learner comparison is INTERLEAVED for
    the same reason.
"""

import argparse
import statistics
import time

import torch

from rower_soccer.competevo_port import design as design_mod
from rower_soccer.competevo_port.dev_env import RunToGoalDevEnv
from rower_soccer.competevo_port.dev_ppo import DevActorCritic, DevSelfPlayPPO
from rower_soccer.competevo_port.selfplay import CoEvoPPO


class Acc:
    """Named accumulators with a CUDA sync on both edges."""

    def __init__(self, cuda):
        self.t, self.n, self.cuda = {}, {}, cuda

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def wrap(self, name, fn):
        def inner(*a, **kw):
            self.sync()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            self.sync()
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0
            self.n[name] = self.n.get(name, 0) + 1
            return out
        return inner

    def reset(self):
        self.t, self.n = {}, {}


def _instrument(env, acs, acc, extra=()):
    """Wrap the three things a rollout does: step the world, write designs,
    run a network. `extra` is a list of (module, method) pairs charged to
    `policy.act` as well -- the stacked opponent forward is one call that stands
    in for `blocks` of them, and it has to be counted in the same bucket for the
    before/after to mean anything."""
    undo = []
    env.step = acc.wrap("env.step", env.step)
    undo.append(lambda: None)
    env.writer.write = acc.wrap("design.write", env.writer.write)
    for ac in acs:
        ac.act = acc.wrap("policy.act", ac.act)
        ac.value = acc.wrap("policy.value", ac.value)
    for mod, name in extra:
        setattr(mod, name, acc.wrap("policy.act", getattr(mod, name)))
    return undo


def _timed_iter(trainer, acc):
    acc.reset()
    acc.sync()
    t0 = time.perf_counter()
    trainer.train_iter()
    acc.sync()
    d = dict(acc.t)
    d["total"] = time.perf_counter() - t0
    d["_n"] = dict(acc.n)
    return d


def _report(rows, label, iters):
    keys = sorted({k for r in rows for k in r if k != "_n"})
    med = {k: statistics.median([r.get(k, 0.0) for r in rows]) for k in keys}
    n = rows[-1]["_n"]
    step_ms = 1e3 * med.get("env.step", 0.0) / max(n.get("env.step", 1), 1)
    print(f"\n--- {label} (median of {iters}) ---")
    print(f"  total iteration        {med['total']:8.2f} s")
    print(f"  env.step               {med.get('env.step', 0):8.2f} s   "
          f"({n.get('env.step', 0)} calls, {step_ms:.1f} ms each)")
    print(f"    of which design.write{med.get('design.write', 0):8.2f} s   "
          f"({n.get('design.write', 0)} calls)")
    print(f"  policy.act (sampling)  {med.get('policy.act', 0):8.2f} s   "
          f"({n.get('policy.act', 0)} calls)")
    print(f"  policy.value           {med.get('policy.value', 0):8.2f} s   "
          f"({n.get('policy.value', 0)} calls)")
    rest = med["total"] - med.get("env.step", 0) - med.get("policy.act", 0) \
        - med.get("policy.value", 0)
    print(f"  everything else        {rest:8.2f} s   "
          f"(PPO update + per-step host work in the rollout)")
    return med


def _ab_opponents(args, acc, dev, cuda):
    """Interleaved A/B of the two opponent forward paths on ONE trainer.

    Same env, same nets, same ring, same iteration count -- the only thing that
    moves between the arms is `batched_opponents`, flipped between iterations.
    That is the strongest form of the interleave this card needs: a separate
    trainer per arm would also differ in its world states, and running all of A
    then all of B would compare two different machines (the stage-3 profile read
    10.06 s and 7.53 s for identical work minutes apart)."""
    env = RunToGoalDevEnv(num_worlds=args.worlds, use_gpu=cuda, seed=3)
    acs = [DevActorCritic(design_dim=env.design_dim,
                          sim_obs_dim=env.sim_obs_dim,
                          n_motor=env.n_motor).to(dev) for _ in range(2)]
    tr = CoEvoPPO(env, acs, rollout_len=args.rollout, epochs=args.epochs,
                  minibatch_size=args.minibatch, blocks=args.blocks,
                  device=dev)

    # -- phase 0: the changed section, isolated -----------------------------
    # A whole iteration on this card swings 4x minute to minute (env.step and
    # the PPO update dominate it and both are at the mercy of six other
    # trainers), so the iteration-level A/B below needs a lot of reps to say
    # anything. This times ONLY the thing that changed -- one step's worth of
    # opponent forwards at the production batch -- alternating A/B/A/B so the
    # drift is shared, and it is the number with the small error bar.
    tr.train_iter()                               # populate obs / warm kernels
    obs = tr._obs.float()
    tr.opp_stack.sync_from(tr.opp_nets)
    for _ in range(5):
        [tr._opponent_actions(e, obs[tr.ego_worlds[e], 1 - e]) for e in range(2)]
        tr._opponent_actions_batched(obs)
    micro = {"per-slot": [], "batched": []}
    for _ in range(args.micro_reps):
        for name, fn in (("per-slot",
                          lambda: [tr._opponent_actions(
                              e, obs[tr.ego_worlds[e], 1 - e])
                              for e in range(2)]),
                         ("batched",
                          lambda: tr._opponent_actions_batched(obs))):
            acc.sync()
            t0 = time.perf_counter()
            fn()
            acc.sync()
            micro[name].append((time.perf_counter() - t0) * 1e3)
    ma, mb = (statistics.median(micro["per-slot"]),
              statistics.median(micro["batched"]))
    print(f"\n--- ONE STEP of opponent forwards, {args.worlds // 2} rows per "
          f"side, blocks={args.blocks} (median of {args.micro_reps}, "
          f"interleaved) ---")
    for name, med in (("per-slot", ma), ("batched", mb)):
        v = sorted(micro[name])
        print(f"  {name:9s} {med:7.2f} ms   "
              f"(min {v[0]:.2f}, p90 {v[int(0.9 * len(v))]:.2f}, "
              f"max {v[-1]:.2f})")
    print(f"  speedup   {ma / mb:7.2f}x   "
          f"= {(ma - mb) * args.rollout / 1e3:.2f} s per {args.rollout}-step "
          f"rollout")

    # -- phase 1: wall clock with NO wrappers -------------------------------
    # The instrumented split below over-attributes, because it synchronises on
    # both edges of every wrapped call and so serialises work that would
    # otherwise pipeline. The headline speedup has to be measured without it.
    for b in (False, True):                       # warm up BOTH paths
        tr.batched_opponents = b
        tr.train_iter()
    clean = {False: [], True: []}
    for i in range(args.iters):
        for b in (False, True):
            tr.batched_opponents = b
            acc.sync()
            t0 = time.perf_counter()
            tr.train_iter()
            acc.sync()
            clean[b].append(time.perf_counter() - t0)
        print(f"  clean [{i + 1}/{args.iters}] per-slot {clean[False][-1]:.1f}s"
              f"  batched {clean[True][-1]:.1f}s", flush=True)
    ca, cb = (statistics.median(clean[False]), statistics.median(clean[True]))
    print(f"\n--- UNINSTRUMENTED iteration wall (median of {args.iters}, "
          f"interleaved) ---")
    print(f"  per-slot opponents  {ca:8.2f} s   {sorted(clean[False])}")
    print(f"  batched opponents   {cb:8.2f} s   {sorted(clean[True])}")
    print(f"  speedup             {ca / cb:8.2f}x")

    # -- phase 2: the same A/B with the section timers on -------------------
    _instrument(env, acs + [n for s in tr.opp_nets for n in s], acc,
                extra=[(tr.opp_stack, "act")])
    for b in (False, True):
        tr.batched_opponents = b
        tr.train_iter()
    rows = {False: [], True: []}
    for i in range(args.iters):
        for b in (False, True):
            tr.batched_opponents = b
            rows[b].append(_timed_iter(tr, acc))
        print(f"  [{i + 1}/{args.iters}] per-slot {rows[False][-1]['total']:.1f}s"
              f"  batched {rows[True][-1]['total']:.1f}s", flush=True)
    a = _report(rows[False], f"PER-SLOT opponents (blocks={args.blocks})",
                args.iters)
    b = _report(rows[True], f"BATCHED opponents (blocks={args.blocks})",
                args.iters)
    print(f"\nbatched / per-slot iteration wall: {b['total'] / a['total']:.2f}x"
          f"  ({a['total'] / b['total']:.2f}x speedup)")
    print(f"policy.act: {a.get('policy.act', 0):.2f} s -> "
          f"{b.get('policy.act', 0):.2f} s "
          f"({a.get('policy.act', 1) / max(b.get('policy.act', 1e-9), 1e-9):.2f}x)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=1024)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--micro-reps", type=int, default=100)
    p.add_argument("--mode", choices=("stages", "opponents"), default="stages",
                   help="'stages' = one-learner vs two-learner (the stage-3 "
                        "profile); 'opponents' = per-slot vs batched opponent "
                        "forward, interleaved on one trainer")
    args = p.parse_args()

    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    acc = Acc(cuda)

    if args.mode == "opponents":
        return _ab_opponents(args, acc, dev, cuda)

    # -- 1. the env on its own, no policy, no learning ----------------------
    env = RunToGoalDevEnv(num_worlds=args.worlds, use_gpu=cuda, seed=0)
    env.reset()
    a = torch.zeros(env.n, env.n_agents, env.act_dim, device=env.device,
                    dtype=env.dtype)
    for _ in range(5):
        env.step(a)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        env.step(a)
    if cuda:
        torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t0) / 30 * 1e3
    print(f"env.step alone, ZERO action, {args.worlds} worlds: "
          f"{step_ms:.1f} ms/step = {1e3 / step_ms * args.worlds:,.0f} "
          f"world-steps/s")

    # The same measurement under RANDOM actions, because a zero-action ant
    # stands still, never falls, never terminates and never resets -- it
    # exercises the cheapest contact set the scene can produce and pays no
    # design writes, so quoting it as "the physics cost" could understate it.
    # Measured, it does not: the two read 100.0 ms/step and 100.0 ms/step
    # (1.0x). The solver cost here is not contact-count-driven at these designs,
    # which is worth knowing before anyone tries to explain the gap between the
    # isolated step and the in-loop step with "the ants are moving".
    for _ in range(5):
        env.step(torch.rand_like(a) * 2 - 1)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        env.step(torch.rand_like(a) * 2 - 1)
    if cuda:
        torch.cuda.synchronize()
    rnd_ms = (time.perf_counter() - t0) / 30 * 1e3
    print(f"env.step alone, RANDOM action, {args.worlds} worlds: "
          f"{rnd_ms:.1f} ms/step = {1e3 / rnd_ms * args.worlds:,.0f} "
          f"world-steps/s ({rnd_ms / step_ms:.1f}x the zero-action cost)")

    # -- 2. the one host sync the rollout does every step -------------------
    obs, rew, done, info = env.step(a)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        float(info["forward"].mean())
    if cuda:
        torch.cuda.synchronize()
    print(f"float(info['forward'].mean()) host sync: "
          f"{(time.perf_counter() - t0) / 200 * 1e3:.2f} ms/call, i.e. "
          f"{(time.perf_counter() - t0) / 200 * args.rollout:.2f} s per rollout")

    # Three 1024-world envs do not need to be resident at once on a card that
    # is shared with six other trainers.
    del env, obs, rew, done, info, a
    if cuda:
        torch.cuda.empty_cache()

    # -- 3. one-learner (stage 2) vs two-learner (stage 3), interleaved -----
    env1 = RunToGoalDevEnv(num_worlds=args.worlds, use_gpu=cuda, seed=1)
    ac1 = DevActorCritic(design_dim=env1.design_dim,
                         sim_obs_dim=env1.sim_obs_dim,
                         n_motor=env1.n_motor).to(dev)
    one = DevSelfPlayPPO(env1, ac1, rollout_len=args.rollout,
                         epochs=args.epochs, minibatch_size=args.minibatch,
                         device=dev)
    _instrument(env1, [ac1], acc)

    env2 = RunToGoalDevEnv(num_worlds=args.worlds, use_gpu=cuda, seed=2)
    acs2 = [DevActorCritic(design_dim=env2.design_dim,
                           sim_obs_dim=env2.sim_obs_dim,
                           n_motor=env2.n_motor).to(dev) for _ in range(2)]
    two = CoEvoPPO(env2, acs2, rollout_len=args.rollout, epochs=args.epochs,
                   minibatch_size=args.minibatch, blocks=args.blocks,
                   device=dev)
    _instrument(env2, acs2 + [n for s in two.opp_nets for n in s], acc)

    # Interleaved A/B: this card is shared, so load drifts over minutes and
    # running all of A then all of B compares two different machines.
    one.train_iter(); two.train_iter()             # warm up both
    rows1, rows2 = [], []
    for i in range(args.iters):
        rows1.append(_timed_iter(one, acc))
        rows2.append(_timed_iter(two, acc))
        print(f"  [{i + 1}/{args.iters}] one {rows1[-1]['total']:.1f}s  "
              f"two {rows2[-1]['total']:.1f}s", flush=True)
    m1 = _report(rows1, "ONE learner (stage 2)", args.iters)
    m2 = _report(rows2, "TWO learners + ring (stage 3)", args.iters)
    print(f"\nstage 3 / stage 2 iteration wall: "
          f"{m2['total'] / m1['total']:.2f}x")


if __name__ == "__main__":
    main()
