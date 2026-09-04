"""D3 M3 E3.1: the control-cost break-even, DERIVED BEFORE RUNNING.

E2.1's `a_crit = 0.739` is the precedent and also the criticism: its own
write-up records that it "was derived after seeing the result, not before".
This is the same class of constant for the failure E3 actually hit, and it is
derived first.

THE QUANTITY. `run_to_goal.py`: `dense = forward - CTRL_COST_COEF*sum(a^2)
+ SURVIVE_BONUS`, with `CTRL_COST_COEF = 0.5`, `SURVIVE_BONUS = 1.0`, the sum
over the RAW unclamped action vector, and `contact_cost` a constant 0. For n
of our actuators with actions of mean mu and std sigma,

    E[cost/step] = CTRL_COST_COEF * n * (mu^2 + sigma^2)

At `control_log_std = 0`, sigma = 1 and an 8-motor ant pays 0.5*8*1 = 4.0/step
against a survive bonus of 1.0 -- the number `D3_E21_CURRICULUM.md` measured
and called "the dense reward's first gradient is quieten down".

**THE NAIVE BREAK-EVEN IS THE WRONG ONE, and that is the point of this file.**
Setting cost/step = SURVIVE_BONUS gives log_std = 0.5*ln(2/n) = -0.693 for
n = 8. But the design head is not choosing between "positive per-step reward"
and "negative per-step reward"; it is choosing between **two whole episodes**:

  * delete the actuators -> a body that topples at ~20.9 steps and banks
    ~+21.2, paying no control cost at any std (E3's measured blob);
  * keep them -> a body that stands for ~458.5 steps and banks
    334.4 - 458.5*cost (the measured idle floor's dense, minus the cost).

The blob wins whenever `334.4 - 458.5*cost < 21.2`. **Episode length is the
whole story** -- a 22x length ratio multiplies a small per-step cost into a
decisive one -- and any derivation that stops at the per-step comparison gets
the constant wrong in the permissive direction.

WHAT THIS DOES NOT MODEL, stated with it:
  * `L_ant` is held at the ZERO-TORQUE episode length. A policy emitting noise
    falls sooner, so the true threshold is STRICTER than the one printed here;
    this is an upper bound on the admissible log_std, not a target.
  * `forward` is held at the idle floor's measured value (the opponent bulldozes
    a passive ant backwards, sum(forward) ~ -124). A policy that learns to walk
    earns more and relaxes the constraint -- but it cannot learn to walk if the
    actuators are gone first, which is the whole failure.
  * mu is measured, not assumed to be 0.

    CUDA_VISIBLE_DEVICES= nice -n 19 .venv-gpu/bin/python \\
        .../t2a_port/e3_ctrl_break_even.py
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

# All measured on this project, none assumed:
CTRL_COST_COEF = 0.5      # run_to_goal.py
SURVIVE = 1.0             # run_to_goal.py
N_MOTORS = 8              # our ant
L_BLOB = 20.9             # e3_blob_probe.py, identical on all three seeds
R_BLOB = 21.2             # ditto: dense of the 0-motor body over its episode
L_ANT = 458.5             # e3_posthoc idle floor: zero-torque episode length
DENSE_ANT_FREE = 334.4    # ditto: its dense, i.e. survive + forward, no cost


def cost_per_step(log_std, n=N_MOTORS, mu2=0.0, k=CTRL_COST_COEF):
    return k * n * (mu2 + math.exp(2.0 * log_std))


def dense_ant(log_std, n=N_MOTORS, mu2=0.0, k=CTRL_COST_COEF):
    """Episode dense for an ant that STAYS UP, at this action std."""
    return DENSE_ANT_FREE - L_ANT * cost_per_step(log_std, n, mu2, k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="rtg_e3c_s1")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    # ---- 1. MEASURE mu^2 and sigma^2 at initialisation -------------------
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
    log_std0 = float(ag.policy_net.state_dict()["control_action_log_std"]
                     .mean().item())

    def tf(l):
        if isinstance(l[0], list):
            return [[torch.tensor(x) for x in y] for y in l]
        return [torch.tensor(y) for y in l]

    env.seed(0)
    state = env.reset()
    while env.if_use_transform_action() != 2:
        with torch.no_grad():
            act = ag.policy_net.select_action(tf([state]), False).numpy()
        state, _, done, _ = env.step(act.astype(np.float64))
        if done:
            break
    # THE GNN EMITS ONE SCALAR PER NODE OVER 13 BODIES AND THE ENV WRITES ONLY
    # THE 8 THAT HAVE AN ACTUATOR (`AntEnv.action_to_control`); the other 5 are
    # discarded, as E1.1 and E2 both record. `ctrl_cost` is therefore a sum over
    # 8 terms, not 13, and summing the policy's whole output column overstates
    # it by 13/8. The first version of this file did exactly that and the
    # measured-vs-predicted line caught it: E[sum a^2] read 12.64 where the
    # env's own ctrl_cost implied 7.77.
    names = list(env.model.actuator_names)
    act_rows = [i for i, b in enumerate(env.robot.bodies)
                if i > 0 and b.get_actuator_name() in names]
    mus, sq, costs = [], [], []
    for _ in range(a.steps):
        with torch.no_grad():
            mean_a = ag.policy_net.select_action(tf([state]), True).numpy()
            samp_a = ag.policy_net.select_action(tf([state]), False).numpy()
        mus.append(np.square(mean_a[act_rows, 0]).sum())
        sq.append(np.square(samp_a[act_rows, 0]).sum())
        state, r, done, info = env.step(samp_a.astype(np.float64))
        if "ctrl_cost" in info:
            costs.append(info["ctrl_cost"])
        if done:
            env.seed(1)
            state = env.reset()
            while env.if_use_transform_action() != 2:
                with torch.no_grad():
                    act = ag.policy_net.select_action(tf([state]), False).numpy()
                state, _, done, _ = env.step(act.astype(np.float64))
                if done:
                    break
    mu2 = float(np.mean(mus))
    e_sq = float(np.mean(sq))
    meas_cost = float(np.mean(costs)) if costs else float("nan")
    pred_cost = CTRL_COST_COEF * e_sq

    print(f"\n=== MEASURED at initialisation ({a.cfg}, {a.steps} steps, "
          f"frozen 13-body ant) ===")
    print(f"  control_action_log_std        {log_std0:+.4f}  "
          f"(sigma = {math.exp(log_std0):.4f})")
    print(f"  actuated nodes / GNN output   {len(act_rows)} of "
          f"{len(env.robot.bodies)}  (the other "
          f"{len(env.robot.bodies) - len(act_rows)} are discarded)")
    print(f"  E[sum mu^2] over those rows   {mu2:.4f}")
    print(f"  E[sum a^2]  over those rows   {e_sq:.4f}   "
          f"predicted n*(mu^2/n + sigma^2) = "
          f"{N_MOTORS * (mu2 / N_MOTORS + math.exp(2 * log_std0)):.4f}")
    print(f"  measured ctrl_cost/step       {meas_cost:.4f}   "
          f"(0.5 * E[sum a^2] = {pred_cost:.4f})  "
          f"residual {abs(meas_cost - pred_cost):.4f}")
    print(f"  against SURVIVE_BONUS         {SURVIVE:.4f}  ->  net "
          f"{SURVIVE - meas_cost:+.4f}/step")

    # ---- 2. THE TWO BREAK-EVENS -----------------------------------------
    naive = 0.5 * math.log(2.0 * SURVIVE / (CTRL_COST_COEF * N_MOTORS * 2))
    naive = 0.5 * math.log(SURVIVE / (CTRL_COST_COEF * N_MOTORS))
    c_crit = (DENSE_ANT_FREE - R_BLOB) / L_ANT
    ls_crit = 0.5 * math.log(c_crit / (CTRL_COST_COEF * N_MOTORS))
    k_crit = c_crit / (N_MOTORS * math.exp(2 * 0.0))     # at log_std = 0

    print(f"\n=== THE DERIVED CONSTANTS (n = {N_MOTORS} motors) ===")
    print(f"  NAIVE per-step break-even   cost/step = SURVIVE = 1.0")
    print(f"    log_std = 0.5*ln(1/(0.5*8)) = {naive:+.4f}   "
          f"sigma = {math.exp(naive):.4f}")
    print(f"    -- and this is TOO PERMISSIVE. At it the standing ant banks "
          f"{dense_ant(naive):.1f} against the blob's {R_BLOB}, so the blob "
          f"still wins.")
    print(f"\n  EPISODE break-even   dense_ant(log_std) = R_blob = {R_BLOB}")
    print(f"    critical cost/step          {c_crit:.4f}")
    print(f"    **log_std_crit              {ls_crit:+.4f}**   "
          f"sigma_crit = {math.exp(ls_crit):.4f}")
    print(f"    equivalently, at log_std = 0, **CTRL_COST_COEF_crit "
          f"= {k_crit:.4f}** (against the current {CTRL_COST_COEF}, a "
          f"{CTRL_COST_COEF / k_crit:.1f}x reduction)")
    print(f"    This is the analogue of E2.1's a_crit = 0.739, and it is an "
          f"UPPER BOUND:\n    L_ant is the zero-torque episode length, and a "
          f"noisy policy falls sooner.")

    # ---- 3. THE SWEEP ----------------------------------------------------
    print(f"\n=== dense over one episode: keep the actuators vs delete them ===")
    print(f"  {'log_std':>9}{'sigma':>8}{'cost/step':>11}{'net/step':>10}"
          f"{'ant dense':>11}{'blob':>8}{'  verdict':>12}")
    for ls in (0.0, -0.25, -0.5, naive, -0.75, ls_crit, -1.0, -1.25, -1.5,
               -2.3, math.log(0.086)):
        c = cost_per_step(ls)
        da = dense_ant(ls)
        v = "KEEP" if da > R_BLOB else "delete"
        tag = ""
        if abs(ls - naive) < 1e-9:
            tag = "  <- naive"
        if abs(ls - ls_crit) < 1e-9:
            tag = "  <- CRITICAL"
        if abs(ls + 2.3) < 1e-9:
            tag = "  <- attr_log_std default"
        if abs(ls - math.log(0.086)) < 1e-9:
            tag = "  <- E2.1 d2rep converged std"
        print(f"  {ls:>9.4f}{math.exp(ls):>8.4f}{c:>11.4f}"
              f"{SURVIVE - c:>10.4f}{da:>11.1f}{R_BLOB:>8.1f}{v:>12}{tag}")

    print(f"\n  The coordinator's proposal, log_std = -1: cost "
          f"{cost_per_step(-1.0):.4f}/step, ant banks "
          f"{dense_ant(-1.0):.1f} against the blob's {R_BLOB} -- "
          f"**KEEP wins by {dense_ant(-1.0) - R_BLOB:+.1f} "
          f"({dense_ant(-1.0) / R_BLOB:.1f}x)**, with "
          f"{ls_crit - (-1.0):+.4f} of margin below the critical value.")

    # ---- 4. THE THREE CANDIDATES, EACH ON ITS OWN -----------------------
    # With a structural floor the 0-motor option disappears, so the choice
    # becomes "stand with n motors" vs "FALL EARLY with n motors" -- the
    # fall-dodge, reached through control instead of morphology. Falling banks
    # L_blob*(SURVIVE - cost); standing banks DENSE_ANT_FREE - L_ANT*cost.
    def fall_with_motors(ls, n, k=CTRL_COST_COEF):
        return L_BLOB * (SURVIVE - cost_per_step(ls, n, 0.0, k))

    def stand_with_motors(ls, n, k=CTRL_COST_COEF):
        return dense_ant(ls, n, 0.0, k)

    print(f"\n=== THE THREE CANDIDATES, EACH EVALUATED ALONE ===")
    print(f"  Three options are available to the search at each setting. The "
          f"fix works only if\n  STAND is the best of them.")
    print(f"\n  {'candidate':<34}{'n':>3}{'log_std':>9}{'cost/st':>9}"
          f"{'STAND':>9}{'blob(0m)':>10}{'fall w/ n':>11}{'  best':>9}")
    cases = [
        ("baseline (E3 as run)", 8, 0.0, 0.5, True),
        ("(1) floor n>=4, log_std unchanged", 4, 0.0, 0.5, False),
        ("(1) floor n>=8, log_std unchanged", 8, 0.0, 0.5, False),
        ("(2) log_std = -1, no floor", 8, -1.0, 0.5, True),
        ("(2) log_std = -0.75, no floor", 8, -0.75, 0.5, True),
        ("(3) CTRL_COST_COEF 0.5 -> 0.05", 8, 0.0, 0.05, True),
        ("(1)+(2) floor n>=4 AND log_std -1", 4, -1.0, 0.5, False),
    ]
    for name, n, ls, k, blob_avail in cases:
        st = stand_with_motors(ls, n, k)
        fa = fall_with_motors(ls, n, k)
        c = cost_per_step(ls, n, 0.0, k)
        opts = {"STAND": st, "fall w/ motors": fa}
        if blob_avail:
            opts["DELETE (blob)"] = R_BLOB
        best = max(opts, key=opts.get)
        print(f"  {name:<34}{n:>3}{ls:>9.3f}{c:>9.3f}{st:>9.1f}"
              f"{(R_BLOB if blob_avail else float('nan')):>10.1f}{fa:>11.1f}"
              f"{('  ' + best):>9}")

    print(f"\n  READING: (1) ALONE DOES NOT WORK. A floor removes the 0-motor")
    print(f"  option but not the incentive -- at log_std = 0 a 4-motor ant "
          f"pays 2.0/step, so\n  FALLING EARLY still beats standing "
          f"({fall_with_motors(0.0, 4):.1f} vs "
          f"{stand_with_motors(0.0, 4):.1f}). It converts E3's morphology\n"
          f"  failure back into E2's fall-dodge, reached through control.")
    print(f"  (2) ALONE DOES WORK: at log_std = -1 the ordering is "
          f"STAND {stand_with_motors(-1.0, 8):.1f} > "
          f"DELETE {R_BLOB:.1f} > fall {fall_with_motors(-1.0, 8):.1f}.")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(cfg=a.cfg, steps=a.steps, log_std_init=log_std0,
                       mu2_sum=mu2, e_sum_a2=e_sq,
                       measured_ctrl_cost_per_step=meas_cost,
                       naive_log_std_break_even=naive,
                       critical_cost_per_step=c_crit,
                       log_std_crit=ls_crit, sigma_crit=math.exp(ls_crit),
                       ctrl_cost_coef_crit=k_crit,
                       constants=dict(CTRL_COST_COEF=CTRL_COST_COEF,
                                      SURVIVE=SURVIVE, N_MOTORS=N_MOTORS,
                                      L_BLOB=L_BLOB, R_BLOB=R_BLOB,
                                      L_ANT=L_ANT,
                                      DENSE_ANT_FREE=DENSE_ANT_FREE)),
                  open(a.out, "w"), indent=1)
        print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
