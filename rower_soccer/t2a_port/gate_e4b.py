"""D3 E4B GATE. Everything gate_e4 asserts, plus the design point the whole
experiment rests on.

E4B's characteristic failure is not a crash. It is a run that trains against
its own current self, ratchets against nothing, and reports a beautiful
equilibrium. So the load-bearing assertions here are:

  A  NEVER THE CURRENT SELF   the sampler cannot return the training epoch,
                              at any epoch, over many draws, for any delta
  B  RING IS STRICTLY PAST    every member the env actually installs during a
                              rollout is older than the current epoch
  C  MIRROR IS NOT A GRADIENT the mirror match and the ladder leave the policy
                              weights, the optimiser state and the RNG exactly
                              as they found them
  D  MIRROR DOES NOT LEAK     the transient "current self" built for the mirror
                              match never becomes a ring member -- if it did,
                              the training opponent COULD become the current
                              self and the +/-1000 term would switch off
  E  SAMPLING IS UNIFORM       the empirical draw distribution matches the
                              declared support (measured, as D2 gated theirs)
  F  NEGATIVE CONTROL          an empty ring leaves the opponent inert, and a
                              stocked ring does not -- so "the ring did
                              something" is established, not assumed
"""
import argparse, os, pickle, sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/Transform2Act")
sys.path.insert(0, "/workspace/utmist-vc2-phase2")
torch.set_default_dtype(torch.float64)

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def sha(sd):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ A, E --
def gate_sampler():
    from rower_soccer.t2a_port.e4r_ring import OpponentRing

    class Stub(OpponentRing):
        def __init__(self, delta, eps):
            self.delta = delta
            self.members = {e: None for e in eps}
            self.rs = np.random.RandomState(0)

    worst_self, checked = None, 0
    for delta in (0.0, 0.5):
        eps = list(range(0, 400, 10))
        r = Stub(delta, eps)
        for epoch in range(1, 400):
            for _ in range(40):
                e = r.sample_epoch(epoch)
                checked += 1
                if e is not None and e >= epoch:
                    worst_self = (delta, epoch, e)
    check("A NEVER THE CURRENT SELF: sampler never returns an epoch >= the "
          "training epoch", worst_self is None,
          f"{checked} draws across delta in (0, 0.5) x epochs 1-399"
          + ("" if worst_self is None else f"  VIOLATION {worst_self}"))

    # E: empirical distribution vs declared support, delta = 0 (our setting)
    r = Stub(0.0, list(range(0, 400, 10)))
    for epoch in (100, 300):
        draws = [r.sample_epoch(epoch) for _ in range(20000)]
        want = [e for e in range(0, 400, 10) if e < epoch]
        got = sorted(set(draws))
        counts = np.array([draws.count(e) for e in want], dtype=float)
        counts /= counts.sum()
        flat = abs(counts - 1.0 / len(want)).max()
        check(f"E SAMPLING UNIFORM at epoch {epoch}: support and flatness",
              got == want and flat < 0.01,
              f"support {len(got)}/{len(want)} members, max deviation from "
              f"uniform {flat:.4f}")


# ------------------------------------------------------ B, C, D, F (live) --
def gate_live(cfg_id):
    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    from design_opt.models.transform2act_policy import Transform2ActPolicy
    from khrylib.robot.xml_robot import Robot
    from khrylib.utils.torch import to_cpu, to_test
    from rower_soccer.t2a_port import e2_eval, e3_morph, rtg_scene
    from rower_soccer.t2a_port import e4_selfplay as sp
    from rower_soccer.t2a_port import e4r_ring as R

    cfg = Config(cfg_id, tmp=True)
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=0,
                               num_threads=1, training=True, checkpoint=0)
    env = agent.env
    base = os.path.join("assets", "mujoco_envs", "ant_competevo.xml")
    ring = R.OpponentRing(cfg, base, rtg_scene.build,
                          lambda: Transform2ActPolicy(cfg.policy_specs, agent),
                          Robot, delta=0.0, seed=0)
    env.set_ring(ring)

    # --- F, first half: empty ring -> opponent inert ----------------------
    env.ring_epoch = 0
    env.ring_chosen = []
    env.opp_policy = None
    st = env.reset()
    check("F NEGATIVE CONTROL (empty ring): no opponent installed, none drawn",
          env.opp_policy is None and not env.ring_chosen,
          f"opp_policy={env.opp_policy}, draws={len(env.ring_chosen)}")

    # stock the ring with three past selves
    for e in (0, 10, 20):
        with to_cpu(agent.policy_net), to_test(agent.policy_net):
            body = sp.dump_mean_action_body(env, agent.policy_net, e3_morph)
        assert body, "design stages failed while stocking the ring"
        ring.add(e, {k: v.detach().cpu()
                     for k, v in agent.policy_net.state_dict().items()}, body)
    n_before = len(ring.members)

    # --- B: every installed member is strictly past -----------------------
    env.ring_epoch = 30
    env.ring_chosen = []
    for _ in range(12):
        env.reset()
    ok = bool(env.ring_chosen) and all(e < 30 for e in env.ring_chosen)
    check("B RING IS STRICTLY PAST: every member installed over 12 resets is "
          "older than the training epoch", ok,
          f"drew {sorted(set(env.ring_chosen))} at epoch 30")

    # --- F, second half: a stocked ring is NOT inert ----------------------
    check("F NEGATIVE CONTROL (stocked ring): an opponent policy is installed",
          env.opp_policy is not None,
          f"opp_policy={'set' if env.opp_policy is not None else None}")

    # --- C: mirror + ladder leave the learner untouched -------------------
    # Populate the optimiser first. On a fresh agent its state dict is EMPTY,
    # so "unchanged" would compare empty to empty and assert nothing -- the
    # same vacuous pass that let an earlier instrument check succeed on a file
    # with no rows. One dummy step gives Adam real moment buffers to compare.
    dummy = sum(p.sum() for p in agent.policy_net.parameters())
    agent.optimizer_policy.zero_grad()
    dummy.backward()
    agent.optimizer_policy.step()
    n_opt = sum(1 for g in agent.optimizer_policy.state.values()
                for v in g.values() if torch.is_tensor(v))
    check("C0 optimiser state is non-empty, so C2 is not vacuous",
          n_opt > 0, f"{n_opt} optimiser tensors")

    before_w = sha(agent.policy_net.state_dict())
    before_opt = sha({f"o{i}": p for i, p in enumerate(
        [s for g in agent.optimizer_policy.state.values()
         for s in g.values() if torch.is_tensor(s)])}) \
        if agent.optimizer_policy.state else "empty"
    np.random.seed(1234)
    torch.manual_seed(1234)
    rng_before = (np.random.get_state()[1][:8].copy(),
                  torch.get_rng_state().clone())
    keep_epoch = env.ring_epoch

    with e3_morph.rng_guard(env), to_cpu(agent.policy_net), \
            to_test(agent.policy_net):
        mm = R.mirror_match(env, agent, ring, e2_eval, episodes=3)
        lad = R.ladder(env, agent, ring, e2_eval, episodes=2, k=2)

    after_w = sha(agent.policy_net.state_dict())
    after_opt = sha({f"o{i}": p for i, p in enumerate(
        [s for g in agent.optimizer_policy.state.values()
         for s in g.values() if torch.is_tensor(s)])}) \
        if agent.optimizer_policy.state else "empty"
    rng_after = (np.random.get_state()[1][:8].copy(),
                 torch.get_rng_state())
    check("C1 MIRROR IS NOT A GRADIENT: policy weights unchanged",
          before_w == after_w, f"{before_w} -> {after_w}")
    check("C2 MIRROR IS NOT A GRADIENT: optimiser state unchanged",
          before_opt == after_opt, f"{before_opt} -> {after_opt}")
    check("C3 MIRROR IS NOT A GRADIENT: RNG restored by rng_guard",
          np.array_equal(rng_before[0], rng_after[0])
          and torch.equal(rng_before[1], rng_after[1]),
          "numpy and torch generator states identical")
    check("C4 MIRROR IS NOT A GRADIENT: env ring_epoch restored",
          env.ring_epoch == keep_epoch,
          f"{keep_epoch} -> {env.ring_epoch}")

    # --- D: the mirror's transient current-self never joins the ring ------
    check("D MIRROR DOES NOT LEAK: ring membership unchanged, and no member "
          "equals the current epoch",
          len(ring.members) == n_before
          and all(e < keep_epoch for e in ring.members),
          f"{n_before} -> {len(ring.members)} members {sorted(ring.members)}")

    # and the mirror actually produced its three-way split
    check("D2 MIRROR REPORTS THE SPLIT: decisive/mutual/stalemate present",
          all(k in mm for k in ("decisive_rate", "mutual_rate",
                                "stalemate_rate", "fwd_mean")),
          f"stalemate {mm.get('stalemate_rate')} fwd {mm.get('fwd_mean')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="rtg_e4r_s1")
    a = ap.parse_args()
    print("=== D3 E4B GATE (ring-specific) ===", flush=True)
    gate_sampler()
    gate_live(a.cfg)
    print()
    if FAILS:
        print("GATE FAILED:", ", ".join(FAILS))
        sys.exit(1)
    print("E4B GATE PASSED")


if __name__ == "__main__":
    main()
