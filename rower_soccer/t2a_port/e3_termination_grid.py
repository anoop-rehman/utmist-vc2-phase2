"""D3 M3 E3: three termination rules x two reward regimes, one instrument.

The user asked whether "removing the penalty for falling" fixes the fall-dodge.
Read from source, there IS no fall penalty: a fall contributes exactly **0** to
the reward and appears only in `done` (`design_opt/envs/run_to_goal.py`). So
what a fall actually costs is the remainder of the episode's `SURVIVE_BONUS`,
and the real form of the proposal is **removing `fell` from the termination
condition**. That is rule (ii) below.

**Why this is not a one-cell question.** Removing `fell` from `done` kills the
dodge *structurally* rather than out-weighting it, which is a different and
possibly better fix than E2.1's `d2rep` -- and if the dodge is gone
structurally, `d2rep`'s compensation may become a liability, because
down-weighting the sparse term to 0.2% also throws away the incentive to
SCORE. So termination and reward have to be crossed, not tested one at a time.

  rule (i)   current      -- a fall ends the episode, and pays nothing
  rule (ii)  no-fall-done -- a fall does not end the episode
  rule (iii) charged      -- a fall ends the episode AND is charged -1000

  regime flat   -- the env reward, `dense + parse`
  regime d2rep  -- `alpha*dense + (1-alpha)*parse`, E2.1's realised alpha

**Rule (iii) needs no separate rollout, and that is exact rather than an
approximation.** Under rule (i) an episode that ends in a fall has `parse` = 0
by construction (`goal_rewards` pays nobody), so rule (iii) is rule (i)'s
trajectory with `-1000` added on exactly the fallen episodes. Two rollout sets
per arm therefore produce all six cells.

**Arms.** The blob (each seed's own mean-action design and control head), the
COMPETENT reference (E2.1's trained `d2rep` MLP on the frozen 13-body ant --
goal 0.95/1.00, the only policy on this project that actually plays this task),
and the zero-torque floor. The competent arm is the one that matters: a blob
beats an ant that cannot play and loses to one that can, so a comparison
against the idle floor alone would compare two non-scoring options with each
other and prove nothing.

**Sample efficiency is reported beside the returns**, because it may decide
this: under rule (ii) every episode runs to truncation, so a 50,000-step batch
buys far fewer episodes, and a fallen ant cannot self-right -- those post-fall
steps are a dead state paying only the survive bonus. `dead_steps` counts them
directly.

PROBE ONLY. Nothing here touches the live E3 seeds.

    CUDA_VISIBLE_DEVICES= nice -n 19 .venv-gpu/bin/python \\
        .../t2a_port/e3_termination_grid.py --out .../posthoc/term_grid.json
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

STAND_Z = 0.28          # the real fall threshold, used to LABEL a fall even
                        # when it no longer terminates
GOAL_REWARD = 1000.0
BATCH = 50000           # min_batch_size, for episodes-per-batch


def rollout(env, act, seed, max_steps, no_fall_done):
    """One episode. Records `dense` and `parse` separately so both reward
    regimes come from the same trajectory, and records the step at which the
    agent FIRST drops below the real fall threshold even when that no longer
    ends the episode -- which is what makes `dead_steps` measurable."""
    saved = env.stand_z
    if no_fall_done:
        # the ONLY place the fall enters is `fell = state_vector()[2] <
        # stand_z`, and it enters `done` alone, never the reward. Putting
        # stand_z below any reachable height removes it from termination
        # exactly, with no code change.
        env.stand_z = -1e9
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)
        state = env.reset()
        while env.if_use_transform_action() != 2:
            state, _, done, _ = env.step(act(state, env.if_use_transform_action()))
            if done:
                return None
        nb = len(env.robot.bodies)
        nu = sum(1 for n in env.model.actuator_names if not n.startswith("opp_"))
        dense = parse = 0.0
        n, fall_step = 0, None
        info = {}
        x0 = float(env.data.subtree_com[env._our_torso_id()][0])
        for _ in range(max_steps):
            state, r, done, info = env.step(act(state, 2))
            dense += float(info.get("dense", 0.0))
            parse += float(info.get("parse", 0.0))
            n += 1
            if fall_step is None and float(env.state_vector()[2]) < STAND_Z:
                fall_step = n
            if done:
                break
        return dict(n=n, dense=dense, parse=parse, n_bodies=nb, n_motors=nu,
                    fell=fall_step is not None,
                    fall_step=fall_step,
                    dead_steps=(n - fall_step) if fall_step is not None else 0,
                    reached=bool(info.get("reached", False)),
                    opp_reached=bool(info.get("opp_reached", False)),
                    net_dx=float(env.data.subtree_com[env._our_torso_id()][0] - x0))
    finally:
        env.stand_z = saved


def score(eps, rule, alpha):
    """The six cells, from two rollout sets. `rule` is 'current'/'nofall'/
    'charged'; 'charged' is scored on the SAME episodes as 'current'."""
    if not eps:
        return {}
    g = lambda k: np.array([e[k] for e in eps], dtype=float)
    dense, parse = g("dense"), g("parse").copy()
    if rule == "charged":
        # a fall now costs the same -1000 a loss does. Under rule (i) a fallen
        # episode's parse is 0 by construction, so this is exact.
        parse = parse - GOAL_REWARD * g("fell")
    flat = dense + parse
    d2r = alpha * dense + (1.0 - alpha) * parse
    ep_len = float(g("n").mean())
    return dict(
        n_eps=len(eps), ep_len=ep_len,
        episodes_per_batch=float(BATCH / ep_len) if ep_len else 0.0,
        dead_steps=float(g("dead_steps").mean()),
        dead_frac=float(g("dead_steps").sum() / g("n").sum()),
        fall_rate=float(g("fell").mean()),
        goal_rate=float(g("reached").mean()),
        loss_rate=float(g("opp_reached").mean()),
        net_dx=float(g("net_dx").mean()),
        dense=float(dense.mean()), parse=float(parse.mean()),
        R_flat=float(flat.mean()), R_d2rep=float(d2r.mean()),
        n_bodies=float(g("n_bodies").mean()), n_motors=float(g("n_motors").mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e3-cfgs", default="rtg_e3_s1,rtg_e3_s2,rtg_e3_s3")
    p.add_argument("--e3-ckpt", default="best")
    p.add_argument("--frozen-cfg", default="rtg_e3c_s1")
    p.add_argument("--competent-cfg", default="rtg_mlp_s1")
    p.add_argument("--competent-tag", default="d2rep")
    p.add_argument("--competent-epoch", default="399")
    p.add_argument("--alpha", type=float, default=0.997696)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    from design_opt.utils.config import Config
    from design_opt.envs import env_dict
    from rower_soccer.t2a_port import e2_eval
    from rower_soccer.t2a_port.e2_video import load_arm
    from rower_soccer.t2a_port.e3_video import load_gnn

    arms = []
    for cfg_id in a.e3_cfgs.split(","):
        cfg, env, make, std = load_gnn(cfg_id, a.e3_ckpt)
        import pickle
        ep = pickle.load(open(f"/workspace/Transform2Act/results/{cfg_id}"
                              f"/models/{a.e3_ckpt}.p", "rb")).get("epoch")
        arms.append((f"blob {cfg_id} (ckpt ep{ep})", env, make(True)[0], cfg))

    cfg_c, env_c, make_c, std_c = load_arm("mlp", a.competent_cfg,
                                           a.competent_epoch, a.competent_tag)
    arms.append((f"COMPETENT ant ({a.competent_cfg}_{a.competent_tag} "
                 f"e{a.competent_epoch})", env_c, make_c(True)[0], cfg_c))

    cfg_f = Config(a.frozen_cfg, tmp=True)
    env_f = env_dict[cfg_f.env_name](cfg_f, agent=None)
    W = env_f.control_action_dim + env_f.attr_design_dim + 1
    zero = np.zeros((len(env_f.robot.bodies), W))
    arms.append(("ant_idle (zero torque)", env_f,
                 (lambda s, stage: zero), cfg_f))

    out = {"alpha": a.alpha, "episodes": a.episodes, "grid": {}}
    print(f"\n{'='*116}\nTHREE TERMINATION RULES x TWO REWARD REGIMES, "
          f"{a.episodes} mean-action episodes, alpha {a.alpha:.6f}\n{'='*116}")
    for name, env, act, cfg in arms:
        maxs = cfg.done_condition.get("max_nsteps", 500) + 5
        sets = {}
        for tag, nofall in (("term", False), ("nofall", True)):
            sets[tag] = [e for e in
                         (rollout(env, act, a.seed_base + i, maxs, nofall)
                          for i in range(a.episodes)) if e]
        cells = {
            "current": score(sets["term"], "current", a.alpha),
            "charged": score(sets["term"], "charged", a.alpha),
            "nofall": score(sets["nofall"], "nofall", a.alpha),
        }
        out["grid"][name] = cells
        print(f"\n{name}")
        print(f"  {'rule':<10}{'ep_len':>8}{'eps/50k':>9}{'dead':>7}"
              f"{'fall':>6}{'goal':>6}{'dx m':>7}{'dense':>9}{'parse':>9}"
              f"{'  R_FLAT':>11}{'  R_D2REP':>11}")
        for rule in ("current", "nofall", "charged"):
            c = cells[rule]
            if not c:
                continue
            print(f"  {rule:<10}{c['ep_len']:>8.1f}"
                  f"{c['episodes_per_batch']:>9.0f}{c['dead_steps']:>7.0f}"
                  f"{c['fall_rate']:>6.2f}{c['goal_rate']:>6.2f}"
                  f"{c['net_dx']:>7.2f}{c['dense']:>9.1f}{c['parse']:>9.1f}"
                  f"{c['R_flat']:>11.1f}{c['R_d2rep']:>11.1f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
