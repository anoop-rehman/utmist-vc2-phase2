"""D3 M3 E3: is the zero-motor blob actually punished by the live objective?

The argument in `D3_E3_ADVERSARIAL.md` 3b was: under `d2rep`'s alpha ~ 0.998
the objective is almost pure `dense = forward - 0.5*sum(a^2) + 1.0`, which pays
**+1.0 per step survived**, so a 0-motor body that topples at step ~21 banks
~21 where a standing actuated ant banks up to 491 -- therefore the gradient
available to the design head points away from the blob.

That was an argument, not a measurement, and an argument of exactly the shape
this project has had to retract twice. This measures it, on the live reward,
under the live alpha, with three arms that separate the two candidate causes:

  blob      the arm's OWN mean-action design (0 motors), its own control head
  ant_pol   the SAME policy's control head on the frozen 13-body ant
  ant_idle  the frozen 13-body ant at zero torque

`blob` vs `ant_pol` isolates the BODY: same weights, same control head, one
body evolved and one not. `ant_idle` is the floor -- what a standing body banks
for doing nothing at all -- and is the number the +1.0-per-step argument is
really about.

Reports the raw env return, the dense and sparse halves separately, and the
CURRICULUM objective `alpha*dense + (1-alpha)*parse` at the alpha the run is
actually at, because that -- not the env return -- is what enters the PPO
buffer and therefore what the design head is optimising.

    CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python .../t2a_port/e3_blob_probe.py \\
        --cfg rtg_e3_s1 --ckpt best --epoch-for-alpha 6 --episodes 10
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


def rollout(env, act, seed, max_steps):
    """One episode, accumulating the reward's two halves separately."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    state = env.reset()
    while env.if_use_transform_action() != 2:
        state, _, done, _ = env.step(act(state))
        if done:
            return None
    nb = len(env.robot.bodies)
    nu = sum(1 for n in env.model.actuator_names if not n.startswith("opp_"))
    R = dense = parse = 0.0
    n = 0
    info = {}
    x0 = float(env.data.subtree_com[env._our_torso_id()][0])
    for _ in range(max_steps):
        state, r, done, info = env.step(act(state))
        R += float(r)
        dense += float(info.get("dense", 0.0))
        parse += float(info.get("parse", 0.0))
        n += 1
        if done:
            break
    end = ("goal" if info.get("reached") else
           ("lost" if info.get("opp_reached") else
            ("fell" if info.get("fell") else "trunc")))
    return dict(n=n, R=R, dense=dense, parse=parse, end=end, n_bodies=nb,
                n_motors=nu,
                fwd=float(env.data.subtree_com[env._our_torso_id()][0] - x0))


def summarise(tag, eps, alpha):
    if not eps:
        return dict(arm=tag, n_eps=0)
    g = lambda k: np.array([e[k] for e in eps], dtype=float)
    obj = alpha * g("dense") + (1.0 - alpha) * g("parse")
    out = dict(arm=tag, n_eps=len(eps),
               n_bodies=float(g("n_bodies").mean()),
               n_motors=float(g("n_motors").mean()),
               steps=float(g("n").mean()), R=float(g("R").mean()),
               dense=float(g("dense").mean()), parse=float(g("parse").mean()),
               objective=float(obj.mean()), fwd=float(g("fwd").mean()),
               ends={e: [x["end"] for x in eps].count(e)
                     for e in set(x["end"] for x in eps)})
    print(f"  {tag:<10} {out['n_bodies']:4.1f} bodies {out['n_motors']:4.1f} "
          f"motors | {out['steps']:6.1f} steps | env R {out['R']:9.1f} = "
          f"dense {out['dense']:8.1f} + sparse {out['parse']:8.1f} | "
          f"OBJECTIVE at alpha {alpha:.4f}: {out['objective']:9.1f} | "
          f"fwd {out['fwd']:+.3f} m | {out['ends']}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="rtg_e3_s1")
    p.add_argument("--ckpt", default="best")
    p.add_argument("--frozen-cfg", default="rtg_e3c_s1")
    p.add_argument("--epoch-for-alpha", type=int, default=0)
    p.add_argument("--curriculum-steps", type=int, default=130208333)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--out", default=None)
    p.add_argument("--no-fall-done", action="store_true",
                   help="COUNTERFACTUAL: remove the fall from the termination "
                        "condition, by dropping stand_z to -1e9 so "
                        "`fell = s[2] < stand_z` can never fire. This is the "
                        "real form of 'remove the penalty for falling': there "
                        "is no fall PENALTY in the reward -- a fall "
                        "contributes exactly 0 and appears only in `done` "
                        "(`run_to_goal.py`) -- so what a fall actually costs "
                        "is the rest of the episode's SURVIVE_BONUS. This "
                        "probe measures what happens if it costs nothing. It "
                        "touches only this process; the live arms run the "
                        "unmodified rule.")
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    from design_opt.utils.config import Config
    from design_opt.envs import env_dict
    from rower_soccer.t2a_port import e2_eval
    from rower_soccer.t2a_port.e3_video import load_gnn
    from rower_soccer.t2a_port.train_e3_gnn import alpha_at

    cfg, env_design, make, std = load_gnn(a.cfg, a.ckpt)
    alpha = alpha_at(a.epoch_for_alpha, a.curriculum_steps, cfg.min_batch_size)
    ckpt_epoch = None
    try:
        import pickle
        ckpt_epoch = pickle.load(open(
            f"/workspace/Transform2Act/results/{a.cfg}/models/{a.ckpt}.p",
            "rb")).get("epoch")
    except Exception:
        pass
    maxs = cfg.done_condition.get("max_nsteps", 500) + 5
    act_fn, _ = make(True)

    # the same weights, on a body the design stages may not touch
    cfg_f = Config(a.frozen_cfg, tmp=True)
    env_frozen = env_dict[cfg_f.env_name](cfg_f, agent=None)
    if a.no_fall_done:
        # `fell = self.state_vector()[2] < self.stand_z` is the ONLY place the
        # fall enters, and it enters `done` alone -- never the reward. Putting
        # stand_z below any reachable height removes the fall from termination
        # exactly, with no code change and nothing else touched.
        env_design.stand_z = -1e9
        env_frozen.stand_z = -1e9
    W = env_frozen.control_action_dim + env_frozen.attr_design_dim + 1
    zero = np.zeros((len(env_frozen.robot.bodies), W))

    print(f"\n{a.cfg} checkpoint '{a.ckpt}' (saved at epoch {ckpt_epoch}), "
          f"alpha at epoch {a.epoch_for_alpha} = {alpha:.6f}, "
          f"{a.episodes} episodes, mean-action"
          + ("   [COUNTERFACTUAL: fall removed from `done`]"
             if a.no_fall_done else "") + "\n")
    print("  arm        body            | episode        | reward split"
          "                            | objective              | progress")

    arms = {}
    eps = [rollout(env_design, lambda s: act_fn(s, env_design
                                                .if_use_transform_action()),
                   a.seed_base + i, maxs) for i in range(a.episodes)]
    arms["blob"] = summarise("blob", [e for e in eps if e], alpha)

    eps = [rollout(env_frozen, lambda s: act_fn(s, env_frozen
                                                .if_use_transform_action()),
                   a.seed_base + i, maxs) for i in range(a.episodes)]
    arms["ant_pol"] = summarise("ant_pol", [e for e in eps if e], alpha)

    eps = [rollout(env_frozen, lambda s: zero, a.seed_base + i, maxs)
           for i in range(a.episodes)]
    arms["ant_idle"] = summarise("ant_idle", [e for e in eps if e], alpha)

    b, ai = arms["blob"], arms["ant_idle"]
    ap = arms["ant_pol"]
    if b.get("n_eps") and ai.get("n_eps"):
        print(f"\n  THE PREDICTION: the blob banks {b['objective']:.1f} of "
              f"objective in {b['steps']:.0f} steps; a standing 13-body ant "
              f"doing NOTHING banks {ai['objective']:.1f} in {ai['steps']:.0f} "
              f"steps.")
        print(f"  The design head's available gain from keeping the body is "
              f"{ai['objective'] - b['objective']:+.1f} per episode "
              f"({ai['objective'] / b['objective']:.1f}x) -- measured, on the "
              f"live objective, not argued.")
        print(f"  Same weights on the unevolved body: {ap['objective']:.1f} "
              f"({ap['n_motors']:.0f} motors), which separates 'no motors' "
              f"from 'bad control'.")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(cfg=a.cfg, ckpt=a.ckpt, ckpt_epoch=ckpt_epoch,
                       alpha=alpha, no_fall_done=bool(a.no_fall_done),
                       epoch_for_alpha=a.epoch_for_alpha,
                       episodes=a.episodes, arms=arms), open(a.out, "w"),
                  indent=1)
        print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
