"""D3 M3 E3: does the DESIGN POPULATION still contain bodies that can act?

The epoch-100 decision (`D3_E3_ADVERSARIAL.md` 3c) originally keyed on the
mean-action design's motor count alone. That was wrong, and this project's own
data says why: **E1 measured that our ant's topology distribution is NOT
concentrated at epoch 100** -- 63 and 101 distinct topologies out of 200
sampled, most-common share 5.5-7.0%, against their ant's 20-41% -- so at
exactly the horizon this experiment decides at, the greedy readout is provably
not the population it is the mode of. E3's own live census says the same thing
in real time: 18-20 distinct topologies of 20 sampled with a 0.05-0.10 top
share, while the mean-action `topo` hash has been a single value since epoch 5.

So a 0-motor mean-action design is consistent with two very different worlds,
and only the population separates them:

  * the search really has collapsed onto unactuated bodies -> stopping is right;
  * the search is still broad and merely has a degenerate MODE -> stopping
    would throw away a run that still has actuated bodies in its gradient.

This measures the second thing. It runs from a CHECKPOINT, in its own process,
so it needs no change to the live trainers.

**Step share, not just design share.** A blob dies at ~21 control steps and an
actuated ant runs to ~490, so actuated designs contribute far more of an
epoch's 50,000 training steps than their share of the sampled designs. At a 5%
design share the step share is already ~55%. The gradient sees episodes, not
designs, so both numbers are reported and the decision rule uses both.

**Run it ONE SEED AT A TIME and niced.** Three concurrent CPU-only probes of an
earlier kind pushed the live arms' `T_sample` from 61 s to 118 s under this
box's 10.2-CPU quota -- measured, see `D3_E3_ADVERSARIAL.md` 5b.

    CUDA_VISIBLE_DEVICES= nice -n 19 .venv-gpu/bin/python \\
        .../t2a_port/e3_population_probe.py --cfg rtg_e3_s1 --ckpt epoch_0100 \\
        --designs 200 --out .../census/pop_rtg_e3_s1_e0100.json
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

# Measured mean episode lengths used to convert a DESIGN share into a STEP
# share. `blob` is `e3_blob_probe.py`'s 20.9 (identical on all three seeds);
# `act` is the 491-step full episode a standing body runs to, which is the
# scripted opponent's crossing step and therefore an upper bound.
LEN_BLOB, LEN_ACT = 20.9, 491.0


def step_share(p_act, len_act=LEN_ACT, len_blob=LEN_BLOB):
    """Fraction of an epoch's TRAINING STEPS supplied by actuated designs."""
    num = p_act * len_act
    den = num + (1.0 - p_act) * len_blob
    return float(num / den) if den else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--ckpt", default="best",
                   help="checkpoint BASENAME without .p, or 'untrained'")
    p.add_argument("--designs", type=int, default=200)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    from khrylib.utils.torch import to_test
    from rower_soccer.t2a_port import e3_morph

    cfg = Config(a.cfg, tmp=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    ag = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                            device=torch.device("cpu"), seed=cfg.seed,
                            num_threads=1, training=False,
                            checkpoint=(0 if a.ckpt == "untrained"
                                        else str(a.ckpt)))
    with to_test(ag.policy_net), e3_morph.rng_guard(ag.env):
        ok = e3_morph.run_design_stages(ag.env, ag.policy_net, True,
                                        ag.running_state)
        ma = e3_morph.body_summary(ag.env) if ok else {}
        cen = e3_morph.census(ag.env, ag.policy_net, a.designs, False,
                              ag.running_state)

    out = dict(cfg=a.cfg, ckpt=a.ckpt, designs=a.designs,
               mean_action_n_motors=ma.get("model_nu_ours"),
               mean_action_n_bodies=ma.get("n_bodies"),
               mean_action_gear_mean=(ma.get("gear") or {}).get("mean"),
               mean_action_topo=ma.get("topo"),
               census=cen,
               step_share_act1=step_share(cen["p_act1"]),
               step_share_act4=step_share(cen["p_act4"]))
    print(f"\n{a.cfg} @ {a.ckpt}")
    print(f"  MEAN-ACTION (the greedy readout): "
          f"{out['mean_action_n_bodies']} bodies, "
          f"{out['mean_action_n_motors']} motors, "
          f"gear_mean {out['mean_action_gear_mean']}, "
          f"topo {out['mean_action_topo']}")
    print(f"  POPULATION ({cen['sampled']} sampled designs):")
    print(f"    distinct topologies {cen['distinct_topologies']}  "
          f"top share {cen['top_topology_share']:.3f}  "
          f"bodies {cen['bodies_min']}-{cen['bodies_max']} "
          f"(mean {cen['bodies_mean']:.2f})")
    print(f"    motors mean {cen['motors_mean']:.2f}  max {cen['motors_max']}"
          f"  histogram {cen['motors_hist']}")
    print(f"    p(>=1 motor) {cen['p_act1']:.3f} -> step share "
          f"{out['step_share_act1']:.3f}")
    print(f"    p(>=4 motors) {cen['p_act4']:.3f} -> step share "
          f"{out['step_share_act4']:.3f}")
    print(f"  (step share converts a DESIGN share into the fraction of an "
          f"epoch's 50,000 training\n   steps those designs supply, using the "
          f"measured {LEN_BLOB:.1f}-step blob and {LEN_ACT:.0f}-step actuated "
          f"episode.)")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
