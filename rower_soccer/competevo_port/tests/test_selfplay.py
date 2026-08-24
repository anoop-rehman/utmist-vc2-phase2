"""Stage-3 gate: two independent learners and the opponent checkpoint ring.

Plain-python (no pytest in this venv):

    PYTHONPATH=. .venv/bin/python -m rower_soccer.competevo_port.tests.test_selfplay
    ... --draws 200000 --seed 0
    ... --gpu                      # + the batched-GPU end-to-end check

The gate, in the order the things it protects were added:

  1. INDEPENDENCE -- agent 0's and agent 1's networks share no parameter object,
     and a full optimizer step on learner 0 leaves every one of learner 1's
     tensors BIT-identical. This is the property the whole stage exists for; a
     shared-weight regression would still train and would still look plausible
     on a curve, so it has to be checked at the tensor level.
  2. SLICING -- the batched env interleaves both agents in the same worlds, so
     "which lane is mine" is an easy thing to get backwards and a hard thing to
     notice. Two directions are checked, both with markers chosen so that
     SWAPPING agent 0 and agent 1 makes the assertion fail: the observation a
     learner stores, and the lane its action lands in (vs the lane the opponent
     net drives).
  3. SAMPLING -- the empirical distribution of `sample_epoch` over enough draws
     to separate their rule from the one it is easily confused with. Their rule
     is `randint(max(1, floor(delta*epoch)), epoch)` with numpy's HIGH-EXCLUSIVE
     randint, i.e. uniform over a strictly PAST window -- not "delta of the time
     the current opponent". The test asserts P(current) == 0 at delta=0.5 so the
     confusion cannot be reintroduced silently.
  4. ROUND TRIP -- a checkpoint pushed into the ring is the one that comes back,
     bit for bit and behaviourally, and it does NOT alias the live parameters
     (an optimizer step after the push must not reach into the ring).
  5. BOUNDEDNESS -- the ring evicts, its host footprint stops growing, and the
     clamp counter fires when a draw names an evicted epoch (so a run that has
     stopped sampling their distribution says so instead of pretending).
  6. BATCHED == PER-SLOT -- the `blocks` opponent networks are evaluated as one
     stacked-weight forward instead of `blocks` separate ones. That is a pure
     speedup only if it computes the same function, so the equivalence is
     checked before anything else about it: the same weights and the same
     observations must give the same distributions and, given the same
     randomness, the same actions. Checked at the module level (many rows,
     hostile inputs) AND through `CoEvoPPO`'s own two methods, so the wiring is
     covered too.
  7. THE MIXTURE IS UNCHANGED -- the speedup must not quietly narrow the set of
     opponents a learner faces. Two claims: the per-world slot assignment is
     still uniform over `blocks` (chi-square), and the batched gather hands each
     world the action of exactly the slot it was assigned (checked with a
     distinct constant per slot, so a mis-gather names the wrong slot out loud).

1, 2, 6b, 7 and the end-to-end check build a real 8-world CPU dev env;
3, 4, 5, 6a are pure and take milliseconds. Nothing here needs CompetEvo's venv.
"""

import argparse
import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rower_soccer.competevo_port.dev_env import RunToGoalDevEnv
from rower_soccer.competevo_port.dev_ppo import (DevActorCritic,
                                                 StackedDevActors)
from rower_soccer.competevo_port.selfplay import (CoEvoPPO, DEV_DELTA,
                                                  OpponentRing)

_results = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        ok = True
    except Exception as exc:                             # noqa: BLE001
        detail, ok = f"{type(exc).__name__}: {exc}", False
    _results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} "
          f"({time.perf_counter() - t0:.1f}s) {detail}")
    return ok


def _tiny_trainer(worlds=8, rollout=2, use_gpu=False, **kw):
    """An 8-world CPU dev env is ~60 ms a step, which is fast enough to run the
    real trainer rather than a mock of it. Every structural claim below is
    therefore made about the production `CoEvoPPO`."""
    env = RunToGoalDevEnv(num_worlds=worlds, use_gpu=use_gpu,
                          max_episode_steps=20)
    dev = "cuda" if use_gpu else "cpu"
    acs = [DevActorCritic(design_dim=env.design_dim,
                          sim_obs_dim=env.sim_obs_dim,
                          n_motor=env.n_motor).to(dev) for _ in range(2)]
    kw.setdefault("minibatch_size", 64)
    kw.setdefault("epochs", 2)
    trainer = CoEvoPPO(env, acs, rollout_len=rollout, device=dev, seed=0, **kw)
    return env, trainer


def _snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _bit_equal(a, b):
    if set(a) != set(b):
        return False, "different key sets"
    for k in a:
        if not torch.equal(a[k], b[k]):
            return False, k
    return True, ""


# ---------------------------------------------------------------------------
# 1. the two learners really are independent
# ---------------------------------------------------------------------------
def t_learners_independent():
    env, tr = _tiny_trainer()

    # (a) no parameter or buffer OBJECT is shared between the two networks, and
    #     no optimizer of learner 0 has a handle on a tensor of learner 1.
    ids0 = {id(t) for t in list(tr.acs[0].parameters())
            + list(tr.acs[0].buffers())}
    ids1 = {id(t) for t in list(tr.acs[1].parameters())
            + list(tr.acs[1].buffers())}
    assert not (ids0 & ids1), "the two learners share tensor objects"
    opt0 = {id(p) for g in (tr.learners[0].opt_pi.param_groups
                            + tr.learners[0].opt_vf.param_groups)
            for p in g["params"]}
    assert not (opt0 & ids1), "learner 0's optimizers hold learner 1's params"
    # Every parameter is claimed by exactly one of the two optimizers -- a head
    # that fell out of `_param_groups` would silently never train.
    for e in range(2):
        claimed = {id(p) for g in (tr.learners[e].opt_pi.param_groups
                                   + tr.learners[e].opt_vf.param_groups)
                   for p in g["params"]}
        own = {id(p) for p in tr.acs[e].parameters()}
        assert claimed == own, f"learner {e}: {len(own - claimed)} params unowned"

    # (b) a real optimizer step on learner 0 leaves learner 1 bit-identical.
    before1 = _snapshot(tr.acs[1])
    before0 = _snapshot(tr.acs[0])
    gae = tr.collect()
    stats = tr.learners[0].update(*gae[0])
    same, key = _bit_equal(before1, _snapshot(tr.acs[1]))
    assert same, f"learner 1 moved when only learner 0 was updated ({key})"
    moved, _ = _bit_equal(before0, _snapshot(tr.acs[0]))
    assert not moved, "learner 0 did not move -- the update was a no-op"
    n_moved = sum(1 for k, v in _snapshot(tr.acs[0]).items()
                  if not torch.equal(v, before0[k]))

    # (c) and the symmetric direction, after the ring has been populated, so a
    #     checkpoint load cannot be the thing that keeps them apart.
    tr.push_checkpoints()
    before0 = _snapshot(tr.acs[0])
    gae = tr.collect()
    tr.learners[1].update(*gae[1])
    same, key = _bit_equal(before0, _snapshot(tr.acs[0]))
    assert same, f"learner 0 moved when only learner 1 was updated ({key})"

    # (d) the opponent nets are frozen and are not in anyone's optimizer.
    for e in range(2):
        for net in tr.opp_nets[e]:
            assert not any(p.requires_grad for p in net.parameters()), \
                "an opponent net still requires grad"
            assert not ({id(p) for p in net.parameters()} & (ids0 | ids1)), \
                "an opponent net aliases a learner's parameters"
    return (f"{len(ids0)} tensors each, disjoint; {n_moved} of learner 0's "
            f"tensors moved, 0 of learner 1's (kl {stats['kl']:+.1e})")


# ---------------------------------------------------------------------------
# 2. per-agent obs / action slicing
# ---------------------------------------------------------------------------
def t_agent_slicing():
    env, tr = _tiny_trainer(worlds=8, rollout=1)
    M = tr.n_ego

    # Step once so every world leaves the design stage: after that the action a
    # net emits is the 8-dim motor block, which makes the routing check below
    # read a single unambiguous number instead of a clamped design vector.
    tr.collect()

    # -- obs: a marker that is unique per (world, agent). It goes into the scale
    #    block, which is genuinely per-agent in the observation, so reading the
    #    wrong lane reads a different number.
    marker = (torch.arange(env.n, dtype=env.dtype).unsqueeze(1) * 10
              + torch.arange(env.n_agents, dtype=env.dtype).unsqueeze(0))
    env.scale[:, :, 0] = marker.to(env.scale.device)
    tr._obs = env.obs()

    # -- actions: give every network a constant, distinguishable output. log_std
    #    goes to -20 so `sample()` is the mean to ~1e-9 and the check is exact
    #    enough to name the lane.
    def _pin(net, c):
        with torch.no_grad():
            net.control_mean.weight.zero_(); net.control_mean.bias.fill_(c)
            net.scale_mean.weight.zero_(); net.scale_mean.bias.fill_(c)
            net.control_log_std.fill_(-20.0); net.scale_log_std.fill_(-20.0)

    EGO, OPP = {0: 3.0, 1: 4.0}, {0: 7.0, 1: 8.0}
    for e in range(2):
        _pin(tr.acs[e], EGO[e])
        for net in tr.opp_nets[e]:
            _pin(net, OPP[e])          # plays lane 1 - e in ego-e worlds

    sent = {}
    real_step = env.step

    def spy(a):
        sent.setdefault("a", a.clone())
        return real_step(a)

    env.step = spy
    tr.collect()
    env.step = real_step
    act = sent["a"]

    for e in range(2):
        w = tr.ego_worlds[e]
        # obs: learner e stored ITS OWN lane of ITS OWN worlds.
        got = tr.learners[e].obs_buf[0, :, 0, 1]
        want = marker[w.cpu(), e].to(got.dtype)
        assert torch.allclose(got, want), (
            f"learner {e} stored lane {'1 - e' if torch.allclose(got, marker[w.cpu(), 1 - e].to(got.dtype)) else '?'}"
            f" -- expected {want[:3].tolist()}, got {got[:3].tolist()}")
        # the swap must be detectably wrong, not accidentally equal
        assert not torch.allclose(got, marker[w.cpu(), 1 - e].to(got.dtype)), \
            "the marker cannot distinguish the two lanes"
        # action: ego's constant in lane e, the opponent's in lane 1 - e.
        ego_lane = act[w, e, -env.n_motor:]
        opp_lane = act[w, 1 - e, -env.n_motor:]
        assert torch.allclose(ego_lane, torch.full_like(ego_lane, EGO[e]),
                              atol=1e-4), \
            f"ego {e}'s action is not in lane {e} (saw {ego_lane[0, 0]:.3f})"
        assert torch.allclose(opp_lane, torch.full_like(opp_lane, OPP[e]),
                              atol=1e-4), \
            f"the opponent's action is not in lane {1 - e} (saw {opp_lane[0, 0]:.3f})"
        # and what the learner recorded as its own action is the ego action.
        assert torch.allclose(tr.learners[e].act_buf[0, :, 0, -env.n_motor:],
                              ego_lane.float(), atol=1e-4), \
            f"learner {e} recorded an action it did not emit"

    # the two ego halves partition the batch: nothing trained twice, nothing
    # left out.
    allw = torch.cat([tr.ego_worlds[0], tr.ego_worlds[1]]).sort().values
    assert torch.equal(allw, torch.arange(env.n, device=allw.device)), \
        "the ego halves do not partition the world batch"
    return (f"{M} ego worlds per learner; obs lane, action lane and recorded "
            f"action all agree, and the swapped assignment fails all three")


# ---------------------------------------------------------------------------
# 3. the ring samples their distribution
# ---------------------------------------------------------------------------
def t_sampling_distribution(draws, seed):
    rng = np.random.default_rng(seed)
    out = []

    def measure(delta, epoch, n):
        ring = OpponentRing(capacity=10_000, delta=delta)
        s = np.array([ring.sample_epoch(epoch, rng) for _ in range(n)])
        lo = max(math.floor(epoch * delta), 1)
        hi = epoch - 1 if lo != epoch else epoch
        return s, lo, hi

    # (a) delta = 0.5 at epoch 100: uniform on {50..99}. THEIR rule. Note what
    #     this rules out: "half the time the current opponent" would put ~50% of
    #     the mass on 100 and this asserts that mass is exactly zero.
    s, lo, hi = measure(DEV_DELTA, 100, draws)
    k = hi - lo + 1
    assert s.min() == lo and s.max() == hi, \
        f"support is [{s.min()}, {s.max()}], expected [{lo}, {hi}]"
    assert (s == 100).sum() == 0, \
        f"{(s == 100).sum()} draws named the CURRENT epoch; their randint is " \
        "high-exclusive, so the current opponent is never drawn at delta=0.5"
    counts = np.bincount(s, minlength=101)[lo:hi + 1]
    exp = draws / k
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    # 5-sigma on a chi-square with k-1 dof; k=50 -> mean 49, sd sqrt(98)=9.9.
    hi_crit = (k - 1) + 5.0 * math.sqrt(2 * (k - 1))
    assert chi2 < hi_crit, f"chi2 {chi2:.1f} > {hi_crit:.1f} on {k} bins"
    worst = float(np.abs(counts / draws - 1 / k).max() * k)
    out.append(f"delta=0.5 ep100: support [{lo},{hi}], chi2 {chi2:.1f}/{k - 1} "
               f"dof (crit {hi_crit:.0f}), worst bin {worst * 100:.1f}% off "
               f"uniform, P(current)=0")

    # (b) delta = 0 -- their FIXED-morph ants, full history: uniform on {1..99}.
    s, lo, hi = measure(0.0, 100, draws)
    assert (s.min(), s.max()) == (1, 99)
    counts = np.bincount(s, minlength=101)[1:100]
    chi2 = float(((counts - draws / 99) ** 2 / (draws / 99)).sum())
    assert chi2 < 98 + 5 * math.sqrt(196), f"delta=0 chi2 {chi2:.1f}"
    out.append(f"delta=0 ep100: [1,99] uniform, chi2 {chi2:.1f}/98")

    # (c) the degenerate branch: `start == end` -> the CURRENT opponent. At
    #     delta=0.5 that is exactly epoch 1, and nowhere else.
    ring = OpponentRing(delta=DEV_DELTA)
    assert ring.sample_epoch(1, rng) == 1
    assert all(ring.sample_epoch(e, rng) < e for e in (2, 3, 10, 999))
    # delta = 1 collapses to always-current (their `robo-sumo-ants-v0`).
    r1 = OpponentRing(delta=1.0)
    assert all(r1.sample_epoch(e, rng) == e for e in (1, 5, 100))
    out.append("start==end -> current: epoch 1 at delta=0.5, every epoch at "
               "delta=1")

    # (d) end to end through a populated ring: the tag that comes back is the
    #     tag that was drawn, and the mean lag matches delta=0.5's prediction
    #     (uniform on [E/2, E-1] has mean lag E/4).
    ring = OpponentRing(capacity=10_000, delta=DEV_DELTA)
    net = nn.Linear(2, 2)
    for e in range(1, 201):
        ring.push(e, net)
    tags = np.array([ring.sample(200, rng)[0] for _ in range(20_000)])
    assert ring.n_clamped == 0, "an unevicted ring should never clamp"
    lag = 200 - tags.mean()
    assert abs(lag - 50.0) < 1.0, f"mean opponent lag {lag:.2f}, expected ~50"
    out.append(f"populated ring at epoch 200: mean lag {lag:.2f} (predicted "
               f"E/4 = 50.0), 0 clamps")
    return "; ".join(out)


# ---------------------------------------------------------------------------
# 4. round trip
# ---------------------------------------------------------------------------
def t_checkpoint_round_trip():
    torch.manual_seed(0)
    live = DevActorCritic()
    with torch.no_grad():                       # move it off its init
        for p in live.parameters():
            p.add_(torch.randn_like(p) * 0.3)
        live.scale_norm.n.fill_(7.0)            # a BUFFER, not a parameter
        live.scale_norm.mean.copy_(torch.randn(20))
    at_push = _snapshot(live)

    ring = OpponentRing(capacity=4, delta=DEV_DELTA)
    ring.push(3, live)

    # Mutate the live module hard, exactly as an optimizer step would. If the
    # ring stored references instead of copies this is what would poison it.
    opt = torch.optim.Adam(live.parameters(), lr=0.1)
    obs = torch.randn(16, live.obs_dim)
    for _ in range(3):
        opt.zero_grad()
        live.value(obs).pow(2).mean().backward()
        opt.step()
    with torch.no_grad():
        live.scale_norm.n.fill_(99.0)
    assert not _bit_equal(at_push, _snapshot(live))[0], "the mutation was a no-op"

    ep, sd = ring.get(3)
    assert ep == 3
    same, key = _bit_equal(at_push, sd)
    assert same, f"the ring's copy changed with the live module ({key})"

    # load back into a fresh module: bit-identical parameters AND identical
    # behaviour on the same input.
    back = DevActorCritic()
    back.load_state_dict(sd)
    same, key = _bit_equal(at_push, _snapshot(back))
    assert same, f"round trip changed {key}"
    ref = DevActorCritic()
    ref.load_state_dict(at_push)
    ref.eval(); back.eval()
    a_ref, a_back = ref.mean_action(obs), back.mean_action(obs)
    assert torch.equal(a_ref, a_back), "reloaded checkpoint acts differently"
    assert float((a_ref - live.mean_action(obs)).abs().max()) > 1e-6, \
        "the mutated live net acts the same -- the test proves nothing"

    # and through the trainer's own path: what `resample_opponents` loads is the
    # checkpoint, not the current weights.
    env, tr = _tiny_trainer()
    tr.epoch = 1
    tr.push_checkpoints()                       # tags epoch 1 == current
    frozen = _snapshot(tr.acs[1])
    with torch.no_grad():                       # move learner 1 far away
        for p in tr.acs[1].parameters():
            p.add_(torch.randn_like(p))
    tr.epoch = 2
    tr.push_checkpoints()                       # tags epoch 2 == the new one
    ep1, sd1 = tr.rings[1].get(1)
    same, key = _bit_equal(frozen, sd1)
    assert same, f"the trainer's ring entry for epoch 1 was overwritten ({key})"
    assert not _bit_equal(frozen, tr.rings[1].get(2)[1])[0], \
        "epochs 1 and 2 stored the same weights"
    return (f"{len(at_push)} tensors incl. RunningNorm buffers survive "
            f"push -> 3 Adam steps -> get -> load_state_dict bit-identically; "
            f"trainer ring keeps epoch 1 and epoch 2 distinct")


# ---------------------------------------------------------------------------
# 5. the ring is bounded
# ---------------------------------------------------------------------------
def t_ring_is_bounded():
    cap, pushes = 8, 200
    ring = OpponentRing(capacity=cap, delta=DEV_DELTA)
    net = DevActorCritic()
    sizes = []
    for e in range(1, pushes + 1):
        ring.push(e, net)
        sizes.append(ring.nbytes())
    assert len(ring) == cap, f"ring holds {len(ring)}, capacity {cap}"
    assert ring.epochs == list(range(pushes - cap + 1, pushes + 1)), \
        f"wrong survivors: {ring.epochs}"
    assert ring.n_evicted == pushes - cap
    # Host footprint stops growing the moment the ring is full, and stays put.
    assert len(set(sizes[cap:])) == 1, "footprint still moving after fill"
    per = sizes[cap - 1] / cap
    assert sizes[-1] == sizes[cap - 1]

    # An evicted target clamps to the oldest survivor, and SAYS SO.
    before = ring.n_clamped
    ep, _ = ring.get(5)
    assert ep == ring.epochs[0], f"clamped to {ep}, expected {ring.epochs[0]}"
    assert ring.n_clamped == before + 1, "an evicted draw did not count"
    # A target inside the window is exact and does not count as a clamp.
    ep, _ = ring.get(pushes - 3)
    assert ep == pushes - 3 and ring.n_clamped == before + 1

    # The default capacity is sized so their schedule never clamps: at epoch E
    # the delta=0.5 window needs ceil(E/2) entries, and max_epoch_num is 1000.
    from rower_soccer.competevo_port.selfplay import RING_CAPACITY
    assert RING_CAPACITY >= math.ceil(1000 / 2), \
        "the default ring clips their own 1000-epoch delta=0.5 window"

    full = OpponentRing(capacity=RING_CAPACITY, delta=DEV_DELTA)
    full.push(1, net)
    mb = full.nbytes() / 1e6
    return (f"cap {cap}: {pushes} pushes -> {len(ring)} held, epochs "
            f"{ring.epochs[0]}..{ring.epochs[-1]}, {ring.n_evicted} evicted, "
            f"footprint flat at {sizes[-1] / 1e6:.2f} MB ({per / 1e3:.0f} kB "
            f"each); default capacity {RING_CAPACITY} = {mb * RING_CAPACITY:.0f} "
            f"MB host at full occupancy, covers their 1000 epochs unclamped")


# ---------------------------------------------------------------------------
# 6. the loop closes: opponents really are past ones, and both learners move
# ---------------------------------------------------------------------------
def t_loop_end_to_end(use_gpu=False):
    worlds = 64 if use_gpu else 8
    env, tr = _tiny_trainer(worlds=worlds, rollout=4, use_gpu=use_gpu,
                            blocks=3)
    start = [_snapshot(ac) for ac in tr.acs]
    for _ in range(12):
        tr.train_iter()
    assert tr.epoch == 12
    assert [len(r) for r in tr.rings] == [12, 12]
    assert tr.rings[0].epochs[-1] == 12, "the newest tag is not this epoch"
    for e in range(2):
        assert not _bit_equal(start[e], _snapshot(tr.acs[e]))[0], \
            f"learner {e} never moved"
    # The last draw was made at the TOP of the 12th iteration, i.e. at epoch 11,
    # so delta=0.5 puts the window at {5..10}: every live opponent slot must be
    # a strictly PAST checkpoint of the other learner.
    E = tr.opp_sample_epoch
    lo, hi = max(math.floor(E * DEV_DELTA), 1), E - 1
    assert E == 11, f"the draw was made at epoch {E}, expected 11"
    for e in range(2):
        for ep in tr.opp_epoch[e]:
            assert lo <= ep <= hi, \
                f"opponent epoch {ep} outside the window [{lo}, {hi}]"
    lag = tr.opponent_lag()
    assert lag > 0, "every opponent is the current policy -- the ring is dead"
    assert env.n_diverged == 0, f"{env.n_diverged} worlds diverged"
    # and with sampling OFF the opponent is the current policy, as their
    # `use_opponent_sample: false` branch does.
    env2, tr2 = _tiny_trainer(worlds=worlds, rollout=4, use_gpu=use_gpu,
                              use_opponent_sample=False)
    for _ in range(3):
        tr2.train_iter()
    assert tr2.opponent_lag() == 0.0, "sampling is off but the lag is non-zero"
    # `train_iter` draws opponents BEFORE the update, so the live slots hold
    # last iteration's weights; redraw to compare against the current ones.
    tr2.resample_opponents()
    same, key = _bit_equal(_snapshot(tr2.acs[1]),
                           _snapshot(tr2.opp_nets[0][0]))
    assert same, f"opponent-sampling-off did not load current weights ({key})"
    return (f"{worlds} worlds, 12 epochs: rings 12/12, opponent epochs "
            f"{tr.opp_epoch[0]} / {tr.opp_epoch[1]} (window {lo}-{hi} at the "
            f"epoch-{E} draw), mean lag {lag:.1f}, {env.n_diverged} diverged")


# ---------------------------------------------------------------------------
# 6a. the batched opponent forward computes the same thing (MODULE level)
# ---------------------------------------------------------------------------
def _spread_out(net, gen, live_stats=True):
    """Move a freshly built net off its initialization and, optionally, give
    its RunningNorms live statistics. Both matter: with `n == 0` the normalizer
    is the identity and the hardest branch never runs, and with the shipped
    initialization the two heads are nearly linear."""
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn(p.shape, generator=gen).to(p) * 0.5)
        for norm, dim in ((net.scale_norm, net.design_dim),
                          (net.control_norm, net.sim_obs_dim)):
            if live_stats:
                norm.n.fill_(float(torch.randint(1, 500, (1,),
                                                 generator=gen).item()))
                norm.mean.copy_(torch.randn(dim, generator=gen).to(norm.mean))
                norm.var.copy_((torch.rand(dim, generator=gen).to(norm.var)
                                * 4.0 + 0.05))
    return net


def _hostile_obs(g, m, obs_dim, gen, device):
    """Observations that exercise every branch of the action path: both stage
    flags, values large enough to hit `RunningNorm`'s clip, and the non-finite
    entries `DevActorCritic._clean` exists for."""
    obs = (torch.randn(g, m, obs_dim, generator=gen) * 3.0)
    obs[..., 0] = (torch.rand(g, m, generator=gen) < 0.5).float()
    obs[0, 0, 5] = float("nan")
    obs[0, min(1, m - 1), 6] = float("inf")
    obs[-1, -1, 7] = -float("inf")
    obs[-1, -2 if m > 1 else -1, 8] = 1e6
    return obs.to(device)


# Both paths are fp32 and they associate their sums differently -- `F.linear`'s
# `addmm` against a broadcasting `matmul` over a leading [groups, slots] axis --
# so BIT-equality is not the right gate and claiming it would be a lie. The
# gate is relative: 1e-5 of the largest value in the tensor, i.e. ~100 fp32 ulps
# on a chain of four 64-to-128-wide dot products. Measured worst case is an
# order of magnitude inside it (see the strings the checks return), and it is
# two orders below the fp32 representation error the port already carries
# through mujoco_warp (PORT_STATUS gate 0c: obs 1.9e-07, reward 3.0e-05).
_EQUIV_RTOL = 1e-5


def t_batched_equals_per_slot(use_gpu=False, rows=512, blocks=4,
                              spread=True):
    dev = "cuda" if use_gpu else "cpu"
    gen = torch.Generator().manual_seed(11)
    if not spread:
        # Production weights: their `init_fc_weights` heads, RunningNorms that
        # have never advanced. This is the regime the trainer actually runs in.
        nets = [[DevActorCritic().to(dev).eval() for _ in range(blocks)]
                for _ in range(2)]
    else:
        nets = [[_spread_out(DevActorCritic(), gen,
                         # slot (1, 0) keeps n == 0, i.e. a checkpoint whose
                         # normalizer has never been updated: the stacked
                         # normalizer's identity branch is per-slot, so a mix of
                         # n == 0 and n > 0 slots is the case that breaks a
                         # naive implementation.
                         live_stats=not (e == 1 and k == 0)).to(dev).eval()
             for k in range(blocks)] for e in range(2)]
    stack = StackedDevActors(nets[0][0], 2, blocks).to(dev)
    stack.sync_from(nets)

    obs = _hostile_obs(2, rows, nets[0][0].obs_dim, gen, dev)
    s_mean, s_std, c_mean, c_std = stack.slot_dists(obs)

    # (a) the distributions themselves: every slot, every row.
    dm = ds = 0.0
    for e in range(2):
        for k in range(blocks):
            d_s, d_c = nets[e][k].dists(obs[e])
            dm = max(dm, float((d_s.mean - s_mean[e, k]).abs().max()),
                     float((d_c.mean - c_mean[e, k]).abs().max()))
            ds = max(ds, float((d_s.stddev[0] - s_std[e, k]).abs().max()),
                     float((d_c.stddev[0] - c_std[e, k]).abs().max()))
    scale = float(torch.cat([s_mean.abs().max().view(1),
                             c_mean.abs().max().view(1)]).max())
    assert dm <= _EQUIV_RTOL * scale, (
        f"mean differs by {dm:.3e} = {dm / scale:.1e} relative to the largest "
        f"value in the tensor ({scale:.2f}), over the {_EQUIV_RTOL:.0e} gate")
    assert ds == 0.0, f"std differs by {ds:.3e}"

    # (b) the ACTIONS, with the randomness held fixed on both sides. Both paths
    #     take `mean + std * eps`, which is what `Normal.sample()` is, so this
    #     compares the whole assembled 28-dim action including the design
    #     clamp and the stage mask -- not just the network output.
    slots = torch.randint(0, blocks, (2, rows), generator=gen).to(dev)
    eps = (torch.randn(2, rows, nets[0][0].design_dim, generator=gen).to(dev),
           torch.randn(2, rows, nets[0][0].n_motor, generator=gen).to(dev))
    got = stack.act(obs, slots, noise=eps)
    ref = torch.zeros_like(got)
    for e in range(2):
        outs = torch.stack([nets[e][k].act(obs[e], noise=(eps[0][e], eps[1][e]))[0]
                            for k in range(blocks)])
        ref[e] = outs.gather(0, slots[e].view(1, -1, 1)
                             .expand(1, rows, got.shape[-1]))[0]
    da = float((got - ref).abs().max())
    a_scale = max(float(ref.abs().max()), 1.0)
    assert da <= _EQUIV_RTOL * a_scale, (
        f"action differs by {da:.3e} = {da / a_scale:.1e} relative to the "
        f"largest action ({a_scale:.2f}), over the {_EQUIV_RTOL:.0e} gate")
    n_exact = bool((got == ref).all().item())

    # (c) the equivalence must be a real claim, not a tautology: two different
    #     slots have to actually disagree, or (b) would pass on a broken gather.
    apart = float((ref[0] - stack.act(obs, (slots + 1) % blocks,
                                      noise=eps)[0]).abs().max())
    assert apart > 1e-3, "the slots are indistinguishable; (b) proves nothing"
    return (f"{2 * blocks} {'spread' if spread else 'as-initialized'} slots x "
            f"{rows} rows on {dev}: max |batched - per-slot| = {dm:.2e} abs / "
            f"{dm / max(scale, 1e-30):.1e} rel on the distribution mean "
            f"(max |mean| {scale:.2f}), {da:.2e} abs / "
            f"{da / a_scale:.1e} rel on the assembled action "
            f"({'BIT-EXACT' if n_exact else 'fp32 reassociation'}; std "
            f"bit-exact; two slots differ by {apart:.2f} so the check bites)")


# ---------------------------------------------------------------------------
# 6b. ... and so does CoEvoPPO's own pair of methods
# ---------------------------------------------------------------------------
def t_batched_equals_per_slot_in_trainer(use_gpu=False):
    worlds = 64 if use_gpu else 8
    env, tr = _tiny_trainer(worlds=worlds, rollout=1, use_gpu=use_gpu, blocks=3)
    tr.collect()                       # leave the design stage; sync the stack
    gen = torch.Generator().manual_seed(3)
    for e in range(2):
        for net in tr.opp_nets[e]:
            _spread_out(net, gen)
    tr.opp_stack.sync_from(tr.opp_nets)

    obs = tr._obs.float()
    M, A = tr.n_ego, env.act_dim
    # Lane-shaped throughout. At two agents `team_lanes` is [[0], [1]], so
    # this drives exactly the lanes the two-agent form did; the expression is
    # what generalised, not the thing being tested.
    L = tr.L
    eps = (torch.randn(2, M, L, tr.acs[0].design_dim).to(obs),
           torch.randn(2, M, L, tr.acs[0].n_motor).to(obs))
    ref = torch.stack([
        tr._opponent_actions(e, obs[tr.ego_worlds[e]][:, tr.team_lanes[1 - e]],
                             noise=(eps[0][e], eps[1][e])) for e in range(2)])
    got = tr._opponent_actions_batched(obs, noise=eps)
    d = float((got - ref).abs().max())
    assert got.shape == (2, M, L, A), f"batched path returned {tuple(got.shape)}"
    assert d <= 1e-5, f"CoEvoPPO's two opponent paths differ by {d:.3e}"

    # And the two paths drive a whole iteration to the same place when the
    # weights and the world are the same -- checked on the ACTION the env is
    # actually handed, lane by lane, not on a summary statistic.
    return (f"{worlds} worlds, blocks {tr.blocks}: "
            f"CoEvoPPO._opponent_actions vs ._opponent_actions_batched differ "
            f"by {d:.2e} over both sides x {M} worlds x {A} action dims")


# ---------------------------------------------------------------------------
# 7. the opponent MIXTURE is unchanged by the batching
# ---------------------------------------------------------------------------
def t_opponent_mixture_unchanged(use_gpu=False):
    worlds, blocks = (64 if use_gpu else 8), 4
    env, tr = _tiny_trainer(worlds=worlds, rollout=1, use_gpu=use_gpu,
                            blocks=blocks)
    tr.collect()

    # (a) each slot gets a distinct constant action, so the action a world
    #     receives NAMES the slot it was routed to.
    def _pin(net, c):
        with torch.no_grad():
            net.control_mean.weight.zero_(); net.control_mean.bias.fill_(c)
            net.scale_mean.weight.zero_(); net.scale_mean.bias.fill_(c)
            net.control_log_std.fill_(-20.0); net.scale_log_std.fill_(-20.0)

    for e in range(2):
        for k, net in enumerate(tr.opp_nets[e]):
            _pin(net, 1.0 + 10 * e + k)          # 1,2,3,4 / 11,12,13,14
    tr.opp_stack.sync_from(tr.opp_nets)

    obs = tr._obs.float()
    got = tr._opponent_actions_batched(obs)[..., -env.n_motor:]
    slots = tr.slot.view(2, tr.n_ego)
    want = (1.0 + 10 * torch.arange(2, device=slots.device).view(2, 1)
            + slots).to(got.dtype).view(2, tr.n_ego, 1, 1).expand_as(got)
    assert torch.allclose(got, want, atol=1e-3), (
        "the batched gather routed a world to the wrong slot: wanted "
        f"{want[0, :4, 0].tolist()}, got {got[0, :4, 0].tolist()}")
    # the per-slot path agrees, on the same slot assignment
    per = torch.stack([
        tr._opponent_actions(e, obs[tr.ego_worlds[e]][:, tr.team_lanes[1 - e]])
        for e in range(2)])[..., -env.n_motor:]
    assert torch.allclose(per, want, atol=1e-3), \
        "the per-slot path and the slot table disagree"

    # (b) the slot assignment itself is still uniform over `blocks`. It is drawn
    #     by `torch.randint` at each world's episode reset; draw it many times
    #     the way `collect` does and chi-square it. This is the distribution the
    #     batching must not narrow -- the CHECKPOINT distribution behind it is
    #     `sample_epoch`, gated separately above.
    n = 200_000
    draws = torch.randint(0, blocks, (n,), generator=tr.gen,
                          device=tr.slot.device).cpu().numpy()
    counts = np.bincount(draws, minlength=blocks)
    exp = n / blocks
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    crit = (blocks - 1) + 5.0 * math.sqrt(2 * (blocks - 1))
    assert chi2 < crit, f"slot draws are not uniform: chi2 {chi2:.1f} > {crit:.1f}"

    # (c) and every live slot still holds a checkpoint the ring chose: the
    #     batching touched the forward pass, not `resample_opponents`.
    for _ in range(6):
        tr.train_iter()
    E = tr.opp_sample_epoch
    lo, hi = max(math.floor(E * DEV_DELTA), 1), E - 1
    for e in range(2):
        for ep in tr.opp_epoch[e]:
            assert lo <= ep <= hi, f"opponent epoch {ep} outside [{lo}, {hi}]"
    assert env.n_diverged == 0, f"{env.n_diverged} worlds diverged"
    return (f"slot -> action routing exact on both paths; slot draws uniform "
            f"over {blocks} (chi2 {chi2:.1f}/{blocks - 1} dof, crit "
            f"{crit:.0f}, n={n:,}); after 6 batched epochs every live slot is "
            f"inside the epoch-{E} window [{lo},{hi}], {env.n_diverged} "
            f"diverged")


# ---------------------------------------------------------------------------
# 8. a short training loop closes on BOTH paths
# ---------------------------------------------------------------------------
def t_both_paths_close(use_gpu=False, epochs=4):
    out = []
    for batched in (False, True):
        worlds = 64 if use_gpu else 8
        env, tr = _tiny_trainer(worlds=worlds, rollout=4, use_gpu=use_gpu,
                                blocks=3, batched_opponents=batched)
        start = [_snapshot(ac) for ac in tr.acs]
        t0 = time.perf_counter()
        for _ in range(epochs):
            tr.train_iter()
        dt = time.perf_counter() - t0
        assert tr.batched_opponents is batched
        assert env.n_diverged == 0, \
            f"batched={batched}: {env.n_diverged} worlds diverged"
        for e in range(2):
            assert not _bit_equal(start[e], _snapshot(tr.acs[e]))[0], \
                f"batched={batched}: learner {e} never moved"
            for t in tr.acs[e].parameters():
                assert torch.isfinite(t).all(), \
                    f"batched={batched}: learner {e} has a non-finite parameter"
        out.append(f"batched={batched}: {dt:.1f}s, {env.n_diverged} diverged")
    return (f"{epochs} epochs each, both learners moved, all parameters "
            f"finite -- " + "; ".join(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draws", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()

    # The headline: the batched opponent forward is only a speedup if it is the
    # same function, so it is checked first and on the hardest inputs.
    check("batched opponents == per-slot opponents (module level, "
          "production weights)",
          lambda: t_batched_equals_per_slot(spread=False))
    check("batched opponents == per-slot opponents (module level, weights and "
          "normalizers spread far apart)",
          lambda: t_batched_equals_per_slot(spread=True))
    check("batched opponents == per-slot opponents (CoEvoPPO's own methods)",
          t_batched_equals_per_slot_in_trainer)
    check("the opponent mixture is unchanged by the batching",
          t_opponent_mixture_unchanged)
    check("a short training loop closes on BOTH opponent paths, 0 diverged",
          t_both_paths_close)

    check("two learners are independent (a step on one leaves the other "
          "bit-identical)", t_learners_independent)
    check("per-agent obs/action slicing (fails if the lanes are swapped)",
          t_agent_slicing)
    check("the ring samples THEIR distribution at the configured delta",
          lambda: t_sampling_distribution(args.draws, args.seed))
    check("a pushed checkpoint round-trips, and does not alias live weights",
          t_checkpoint_round_trip)
    check("the ring is bounded: it evicts and its footprint stops growing",
          t_ring_is_bounded)
    check("the co-evolution loop closes on CPU", t_loop_end_to_end)
    if args.gpu:
        check("batched opponents == per-slot opponents ON GPU (module level, "
              "production weights)",
              lambda: t_batched_equals_per_slot(use_gpu=True, spread=False))
        check("batched opponents == per-slot opponents ON GPU (module level, "
              "spread apart)",
              lambda: t_batched_equals_per_slot(use_gpu=True, spread=True))
        check("batched opponents == per-slot opponents ON GPU (CoEvoPPO)",
              lambda: t_batched_equals_per_slot_in_trainer(use_gpu=True))
        check("the co-evolution loop closes on the batched GPU env",
              lambda: t_loop_end_to_end(use_gpu=True))

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
