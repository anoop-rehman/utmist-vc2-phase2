"""D3 M3 E2 headline instrument: one function, both protocols, every arm.

Does three things at a saved checkpoint and nothing else:

  1. **verifies the body was frozen** for the whole trained run, driven by the
     arm's OWN trained policy through the design stages -- every mjModel array
     compared against the initial body. `gate_e2.py` phase 3 does the same with
     destructive RANDOM actions before training; a gate that only ever saw
     random actions could miss a policy that learned some other path, so both
     are run;
  2. **measures both protocols** -- mean-action (the headline) and stochastic
     (the column beside it) -- with `e2_eval.evaluate`, the same code the
     trainers call inline, so no number in the write-up comes from a training
     log;
  3. **records the learned action std**, because that is what makes the two
     protocols disagree and it differs between architectures.

    .venv-gpu/bin/python .../t2a_port/e2_posthoc.py --arm gnn \\
        --cfg rtg_gnn_s1 --epoch 100 --episodes 20 --out results.json
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
    p.add_argument("--arm", choices=["gnn", "mlp"], required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--epoch", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    from rower_soccer.t2a_port import e2_eval
    from rower_soccer.t2a_port.e2_video import load_arm

    cfg, env, make, std = load_arm(a.arm, a.cfg, a.epoch, a.tag)
    act, wrap = make(True)
    maxs = cfg.done_condition.get("max_nsteps", 500) + 5

    # ---- 1. body frozen, driven by THIS policy --------------------------
    ref = dict(arrays(env.model))
    changed = set()
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
    frozen = not changed

    # ---- 2. both protocols, same episodes, same seeds -------------------
    res = {}
    for name, mean in (("mean_action", True), ("stochastic", False)):
        ai, wi = make(mean)
        res[name] = e2_eval.evaluate(env, ai, wi, episodes=a.episodes,
                                     seed_base=a.seed_base, max_steps=maxs)

    out = dict(arm=a.arm, cfg=a.cfg, tag=a.tag, epoch=a.epoch,
               episodes=a.episodes, body_frozen=frozen,
               n_arrays=len(ref), changed=sorted(changed),
               bodies=len(env.robot.bodies), action_std=std,
               opponent_speed=env.opp_speed, dt=env.dt,
               max_nsteps=env.max_nsteps, results=res)
    tag = f"{a.cfg}{'_' + a.tag if a.tag else ''} epoch {a.epoch}"
    print(f"\n{a.arm.upper()} {tag}: {len(env.robot.bodies)} bodies, "
          f"learned action std {std:.4f}")
    print(f"  BODY FROZEN across {a.episodes} episodes driven by the TRAINED "
          f"policy: "
          + ("YES -- %d arrays identical" % len(ref) if frozen
             else "NO -- " + str(sorted(changed))))
    for name in ("mean_action", "stochastic"):
        r = res[name]
        print(f"  {name:12s} R {r['R_mean']:9.1f} +/- {r['R_sd']:7.1f}   "
              f"goal {r['goal_rate']:.2f}  lost {r['loss_rate']:.2f}  "
              f"fell {r['fall_rate']:.2f}  len {r['ep_len']:6.1f}  "
              f"dx {r['net_dx']:6.2f} m  maxx {r['max_x']:6.2f}  "
              f"speed {r['speed']:.3f} m/s  |y| {r['max_abs_y']:.2f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
