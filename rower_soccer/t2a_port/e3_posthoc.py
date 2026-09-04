"""D3 M3 E3 headline instrument: one function, both protocols, every arm.

`e2_posthoc.py`'s three jobs, with the first one INVERTED for the design-on
arms and a fourth added:

  1. **what the design stages did to the body, driven by the arm's OWN trained
     policy.** E2/E2.1's post-hoc asserted 134 mjModel arrays IDENTICAL; the
     E3 arms have to show the mirror -- that the arrays change, that the body
     count moves, and that the compiled model is the designed body. The
     frozen-body control arms are checked E2's way, in the same run of the
     same code, so "frozen" and "not frozen" are two answers from one
     instrument rather than two instruments. `gate_e3.py` does this before
     training with destructive RANDOM design actions; a gate that only ever
     saw random actions could miss a policy that learned some other path, so
     both are run.
  2. **both protocols** -- mean-action (the headline) and stochastic -- through
     `e2_eval.evaluate`, the same code the trainers call inline, so no number
     in the write-up comes from a training log.
  3. **the learned action std**, which is what makes the two protocols
     disagree.
  4. **the fall-dodge instrument**: `r(fall rate, return)`,
     `r(forward progress, return)` and the measured fall premium, over this
     arm's own 20 evaluation episodes. E2 measured +0.989/+0.019 across its
     seven arms and E2.1 inverted it to -0.517/+0.947; if E3's has drifted
     back toward E2's structure, the dodge reopened through morphology.

    .venv-gpu/bin/python .../t2a_port/e3_posthoc.py --cfg rtg_e3_s1 \\
        --epoch 400 --episodes 20 --out results.json
"""
import argparse
import json
import os
import sys

sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def arrays(m):
    for nm in dir(m):
        if nm.startswith("_"):
            continue
        try:
            v = getattr(m, nm)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiub":
            yield nm, np.array(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["gnn", "idle"], default="gnn")
    p.add_argument("--cfg", required=True)
    p.add_argument("--epoch", default="400")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--census-episodes", type=int, default=200)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    from rower_soccer.t2a_port import e2_eval, e3_morph
    from rower_soccer.t2a_port.e2_video import load_arm
    from rower_soccer.t2a_port.e3_video import load_gnn

    if a.arm == "idle":
        cfg, env, make, std = load_arm("idle", a.cfg, a.epoch)
        policy = None
    else:
        cfg, env, make, std = load_gnn(a.cfg, f"epoch_{int(a.epoch):04d}")
        from design_opt.agents.transform2act_agent import Transform2ActAgent
        policy = None
    act, wrap = make(True)
    maxs = cfg.done_condition.get("max_nsteps", 500) + 5
    design_on = not env.env_specs.get("force_identity_design", False)

    # ---- 1. what the design stages did, under THIS trained policy --------
    ref = dict(arrays(env.model))
    changed, counts, topos = set(), [], {}
    for i in range(a.episodes):
        np.random.seed(a.seed_base + i)
        torch.manual_seed(a.seed_base + i)
        env.seed(a.seed_base + i)
        state = wrap(env.reset())
        while env.if_use_transform_action() != 2:
            state, _, done, _ = env.step(act(state,
                                             env.if_use_transform_action()))
            state = wrap(state)
            for nm, v in arrays(env.model):
                if nm not in ref or ref[nm].shape != v.shape or \
                        not np.array_equal(ref[nm], v):
                    changed.add(nm)
            if done:
                break
        counts.append(len(env.robot.bodies))
        k, _ = e3_morph.topo_key(env)
        topos[k] = topos.get(k, 0) + 1
    frozen = not changed

    # ---- 2. the mean-action design and the sampled population ------------
    morph = {}
    if a.arm == "gnn":
        cfg2, env2, make2, _ = load_gnn(a.cfg, f"epoch_{int(a.epoch):04d}")
        from khrylib.utils.torch import to_test
        from design_opt.agents.transform2act_agent import Transform2ActAgent
        ag = Transform2ActAgent(cfg=cfg2, dtype=torch.float64,
                                device=torch.device("cpu"), seed=cfg2.seed,
                                num_threads=1, training=False,
                                checkpoint=f"epoch_{int(a.epoch):04d}")
        with to_test(ag.policy_net):
            ok = e3_morph.run_design_stages(ag.env, ag.policy_net, True,
                                            ag.running_state)
            morph["mean_action"] = e3_morph.body_summary(ag.env) if ok else {}
            morph["census"] = e3_morph.census(ag.env, ag.policy_net,
                                              a.census_episodes, False,
                                              ag.running_state)

    # ---- 3. both protocols, same episodes, same seeds --------------------
    res = {}
    for name, mean in (("mean_action", True), ("stochastic", False)):
        ai, wi = make(mean)
        r = e2_eval.evaluate(env, ai, wi, episodes=a.episodes,
                             seed_base=a.seed_base, max_steps=maxs)
        r["dodge"] = e3_morph.dodge_stats(r.get("episodes", []))
        res[name] = r

    out = dict(arm=a.arm, cfg=a.cfg, epoch=a.epoch, episodes=a.episodes,
               design_on=design_on, body_frozen=frozen, n_arrays=len(ref),
               changed=sorted(changed), body_counts=counts,
               distinct_topologies=len(topos),
               top_topology_share=(max(topos.values()) / len(counts)
                                   if counts else 0.0),
               action_std=std, opponent_speed=env.opp_speed, dt=env.dt,
               max_nsteps=env.max_nsteps, morphology=morph, results=res)

    print(f"\n{a.cfg} epoch {a.epoch}: design stages "
          f"{'LIVE' if design_on else 'IDENTITY'}, learned action std "
          f"{std:.4f}")
    if design_on:
        print(f"  DESIGN CHANGED THE BODY across {a.episodes} episodes driven "
              f"by the TRAINED policy: "
              + (f"YES -- {len(changed)} of {len(ref)} mjModel arrays, "
                 f"body counts {sorted(set(counts))}, "
                 f"{len(topos)} distinct topologies"
                 if changed else "NO -- the design stage is a NO-OP"))
    else:
        print(f"  BODY FROZEN across {a.episodes} episodes: "
              + (f"YES -- {len(ref)} arrays identical" if frozen
                 else "NO -- " + str(sorted(changed))))
    if morph.get("mean_action"):
        m = morph["mean_action"]
        print(f"  mean-action design: {m['n_bodies']} bodies "
              f"({m['model_nu_ours']} motors), mass {m['model_mass_ours']:.3f} "
              f"kg, limb length {m['limb_length']['mean']:.3f} m mean / "
              f"{m['limb_length']['sum']:.3f} m total, gear "
              f"{m['gear']['mean']:.0f}, topo {m['topo']}")
        c = morph["census"]
        print(f"  sampled ({c['sampled']} designs): "
              f"{c['distinct_topologies']} distinct topologies, most common "
              f"{100 * c['top_topology_share']:.1f}%, bodies "
              f"{c['bodies_min']}-{c['bodies_max']} (mean "
              f"{c['bodies_mean']:.1f})")
    for name in ("mean_action", "stochastic"):
        r = res[name]
        d = r["dodge"]
        print(f"  {name:12s} R {r['R_mean']:9.1f} +/- {r['R_sd']:7.1f}   "
              f"goal {r['goal_rate']:.2f}  lost {r['loss_rate']:.2f}  "
              f"fell {r['fall_rate']:.2f}  len {r['ep_len']:6.1f}  "
              f"fwd {r['max_fwd']:5.2f} m ({100 * r['frac_of_goal']:5.1f}% of "
              f"5.0 m)  speed {r['speed']:6.3f} m/s  nb {r['bodies_exec']:.1f}"
              f"  designfail {r['design_fail_rate']:.2f}")
        print(f"               r(fall,R) {d.get('r_fall_return')}  "
              f"r(fwd,R) {d.get('r_fwd_return')}  fall premium "
              f"{d.get('fall_premium')}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
