"""D3 M3 E3.1: how much does the body actually MOVE at each initial sigma?

`log_std_crit` = -0.8837 says which side of the basin boundary to start on. It
does not say how far below to start, and "-1.0 because it is round" is not a
justification. The competing consideration is exploration: sigma IS the action
noise, and if the initial noise does not move the body, the `forward` term
never produces a gradient and the run learns nothing for a different reason.

So this measures the thing that actually trades off -- **displacement produced
by noise alone** -- on the frozen 13-body ant with an UNTRAINED policy, at each
candidate sigma. No argument, a measurement.

Reported per sigma: mean path length travelled by the torso COM, mean |net
displacement|, fall rate, episode length, measured ctrl_cost/step, and the
resulting episode `dense`. The last column is the one 3f's table predicts, so
this also checks that arithmetic against the simulator.

    CUDA_VISIBLE_DEVICES= nice -n 19 .venv-gpu/bin/python \\
        .../t2a_port/e3_sigma_exploration.py
"""
import argparse
import json
import math
import os
import sys

sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="rtg_e3c_s1")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--log-stds", default="0.0,-0.5,-0.8837,-1.0,-1.25,-1.5,-2.0,-2.4534")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    from khrylib.utils.torch import to_test
    cfg = Config(a.cfg, tmp=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    ag = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                            device=torch.device("cpu"), seed=cfg.seed,
                            num_threads=1, training=False, checkpoint=0)
    to_test(ag.policy_net)
    env = ag.env
    maxs = cfg.done_condition.get("max_nsteps", 500) + 5

    def tf(l):
        if isinstance(l[0], list):
            return [[torch.tensor(x) for x in y] for y in l]
        return [torch.tensor(y) for y in l]

    key = "control_action_log_std"
    orig = ag.policy_net.state_dict()[key].clone()

    print(f"\n=== displacement from NOISE ALONE, untrained policy, frozen "
          f"13-body ant, {a.episodes} episodes each ===")
    print(f"  {'log_std':>9}{'sigma':>8}{'cost/step':>11}{'ep_len':>8}"
          f"{'fall':>6}{'path m':>9}{'|net dx|':>10}{'dense':>9}"
          f"{'  3f predicts':>14}")
    rows = []
    for ls in [float(x) for x in a.log_stds.split(",")]:
        with torch.no_grad():
            ag.policy_net.state_dict()[key].fill_(ls)
        paths, dxs, lens, falls, costs, denses = [], [], [], [], [], []
        for i in range(a.episodes):
            np.random.seed(5000 + i)
            torch.manual_seed(5000 + i)
            env.seed(5000 + i)
            state = env.reset()
            bad = False
            while env.if_use_transform_action() != 2:
                with torch.no_grad():
                    act = ag.policy_net.select_action(tf([state]), False).numpy()
                state, _, done, _ = env.step(act.astype(np.float64))
                if done:
                    bad = True
                    break
            if bad:
                continue
            xs = [float(env.data.subtree_com[env._our_torso_id()][0])]
            x0, n, dn, cc, fell = xs[0], 0, 0.0, 0.0, False
            for _ in range(maxs):
                with torch.no_grad():
                    act = ag.policy_net.select_action(tf([state]), False).numpy()
                state, r, done, info = env.step(act.astype(np.float64))
                n += 1
                dn += float(info.get("dense", 0.0))
                cc += float(info.get("ctrl_cost", 0.0))
                xs.append(float(info.get("com_x", xs[-1])))
                fell = bool(info.get("fell", False))
                if done:
                    break
            paths.append(float(np.abs(np.diff(xs)).sum()))
            dxs.append(abs(xs[-1] - x0))
            lens.append(n)
            falls.append(float(fell))
            costs.append(cc / max(n, 1))
            denses.append(dn)
        if not lens:
            continue
        sig = math.exp(ls)
        pred = 334.4 - 458.5 * (0.5 * 8 * sig * sig)
        row = dict(log_std=ls, sigma=sig, cost_per_step=float(np.mean(costs)),
                   ep_len=float(np.mean(lens)), fall_rate=float(np.mean(falls)),
                   path=float(np.mean(paths)), net_dx=float(np.mean(dxs)),
                   dense=float(np.mean(denses)), dense_pred_3f=pred)
        row["dense_sd"] = float(np.std(denses, ddof=1)) if len(denses) > 1 else 0.0
        rows.append(row)
        print(f"  {ls:>9.4f}{sig:>8.4f}{row['cost_per_step']:>11.4f}"
              f"{row['ep_len']:>8.1f}{row['fall_rate']:>6.2f}"
              f"{row['path']:>9.3f}{row['net_dx']:>10.3f}"
              f"{row['dense']:>9.1f}{pred:>14.1f}")
    with torch.no_grad():
        ag.policy_net.state_dict()[key].copy_(orig)

    if len(rows) >= 2:
        base = rows[0]
        print(f"\n  path length relative to log_std = 0 "
              f"({base['path']:.3f} m):")
        for r in rows:
            print(f"    log_std {r['log_std']:>8.4f}  path "
                  f"{r['path']:>7.3f} m  = {100*r['path']/base['path']:>5.1f}% "
                  f"of the noisiest setting   ep_len {r['ep_len']:>6.1f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
