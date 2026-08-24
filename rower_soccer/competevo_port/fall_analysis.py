"""What kind of fall? The D2 residual, characterised rather than counted.

M2E section 10 left one number open: our 1v1 port ends 15.6% of episodes in a
fall against the reference's 1.0%, and that 14.6-point gap accounts for almost
exactly the 13-point goal-rate gap. "Our ants fall over more" is where the
investigation stopped. It is not a cause.

This asks four questions that have different fixes, so separating them is the
point:

1. **WHEN.** A fall in the first 30 steps is the spawn or the design -- the ant
   was never stable. A fall at step 300 mid-stride is control. These want
   opposite interventions and the ending histogram cannot tell them apart.
2. **WHICH BODY.** Each world has its own genome. If fallers and non-fallers
   are drawn from visibly different regions of the 20-dim scale vector, the
   design head is producing tippy bodies and the controller is downstream of
   that. Reported as the per-dimension standardised mean difference, so a
   dimension that matters stands out from nineteen that do not.
3. **IS IT UPRIGHTNESS OR HEIGHT?** The termination is a band on torso z
   (0.28 to 1.2). Leaving the bottom is a collapse; leaving the TOP is being
   launched, which is a contact-solver artefact and a completely different
   bug. `dev_ant.py:291` has both bounds and the fixed-morph ant has only the
   lower one, so the ceiling is dev-specific and worth checking directly.
4. **DOES THE OPPONENT DO IT?** Distance to the opponent at the moment of the
   fall, against the distance distribution over all steps. If falls happen at
   contact range they are collisions; if they happen at any range they are not.

    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.competevo_port.fall_analysis \
        --policies runs/competevo_port/m2e_fixed/policies.pt

Nothing here is causal. Every number is a difference between two populations
drawn from the same run, which is enough to say where to look next and not
enough to say why.
"""

import argparse

import numpy as np
import torch

STAND_Z_MIN, STAND_Z_MAX = 0.28, 1.2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies",
                   default="runs/competevo_port/m2e_fixed/policies.pt")
    p.add_argument("--worlds", type=int, default=384)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    from rower_soccer.competevo_port.dev_env import RunToGoalDevEnv
    from rower_soccer.competevo_port.dev_ppo import DevActorCritic

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # auto_reset=False IS THE POINT. `terms()` computes `fell` from the
    # POST-step torso z and `step()` then resets the world, so with auto-reset
    # on there is no moment at which the height that caused the termination is
    # readable -- a probe that reads z before the step is classifying on the
    # height one frame too early, which silently mislabels every collapse as a
    # launch. Worlds are never reset here; each is counted once, at its first
    # fall.
    env = RunToGoalDevEnv(num_worlds=args.worlds, use_gpu=(device == "cuda"),
                          seed=args.seed, auto_reset=False)
    blob = torch.load(args.policies, map_location="cpu")
    acs = []
    for key in ("ac_0", "ac_1"):
        ac = DevActorCritic()
        ac.load_state_dict(blob[key])
        acs.append(ac.to(device).eval())

    n, A = env.n, env.n_agents
    obs = env.reset()
    # Per (world, agent): has it fallen yet, at which step, from which side of
    # the band, how far was the opponent, and what body was it.
    fell_at = torch.full((n, A), -1, device=device, dtype=torch.long)
    fell_low = torch.zeros(n, A, device=device, dtype=torch.bool)
    fell_dist = torch.zeros(n, A, device=device)
    ended = torch.zeros(n, dtype=torch.bool, device=device)
    n_unexplained = 0
    design = env.scale.clone()                 # [n, A, 20], fixed after step 1
    all_dists = []

    with torch.no_grad():
        for t in range(env.max_episode_steps + 2):
            o = obs.float()
            a = torch.stack([acs[i].mean_action(o[:, i]) for i in range(A)],
                            dim=1)
            obs, _, done, info = env.step(a.to(env.dtype))
            # Read AFTER the step: this is the state `terms()` judged.
            z = env.qpos[:, env.root_z_idx]                  # [n, A]
            xy = env.qpos[:, env.qpos_idx[:, :2]]            # [n, A, 2]
            d = (xy[:, 0] - xy[:, 1]).norm(dim=-1)           # [n]
            live = (~ended)
            if bool(live.any()):
                all_dists.append(float(d[live].mean()))
            if t == 1:
                design = env.scale.clone()   # after the design step lands
            newly = info["fell"] & (fell_at < 0) & (~ended).unsqueeze(-1)
            if bool(newly.any()):
                fell_at = torch.where(newly, torch.full_like(fell_at, t),
                                      fell_at)
                # Which bound did it leave? Measured on the step BEFORE the
                # env applied its own reset, which is why z is read above.
                fell_low |= newly & (z < STAND_Z_MIN)
                # Anything the band should have caught but neither bound
                # explains is a bug in this probe, not a finding.
                unexplained = newly & (z >= STAND_Z_MIN) & (z <= STAND_Z_MAX)
                n_unexplained += int(unexplained.sum())
                fell_dist = torch.where(newly, d.unsqueeze(-1), fell_dist)
            ended |= done
            if bool(ended.all()):
                break

    fa = fell_at.cpu().numpy()
    fl = fell_low.cpu().numpy()
    fd = fell_dist.cpu().numpy()
    des = design.reshape(-1, design.shape[-1]).cpu().numpy()
    flat_fell = (fa >= 0).reshape(-1)
    steps = fa.reshape(-1)[flat_fell]

    print(f"\n=== fall analysis: {args.policies}")
    print(f"    {n} worlds x {A} agents = {n * A} agent-episodes, mean actions")
    print(f"    {flat_fell.sum()} of {n * A} agents fell "
          f"({100 * flat_fell.mean():.1f}%)")
    if flat_fell.sum() == 0:
        print("    nothing fell -- nothing to characterise")
        return

    print("\n-- 1. WHEN")
    for lo, hi, name in ((0, 30, "0-30      (spawn / design)"),
                         (30, 100, "30-100    (early gait)"),
                         (100, 300, "100-300   (mid)"),
                         (300, 10**9, "300+      (late)")):
        k = int(((steps >= lo) & (steps < hi)).sum())
        print(f"    step {name:26s} {k:5d}   {100 * k / len(steps):5.1f}%")
    print(f"    median fall step {np.median(steps):.0f}, "
          f"mean {steps.mean():.0f} of {env.max_episode_steps}")

    print("\n-- 2. WHICH BOUND (0.28 low = collapse, 1.2 high = launched)")
    low = int(fl.reshape(-1)[flat_fell].sum())
    if n_unexplained:
        print(f"    !! {n_unexplained} falls had z INSIDE the band when read. "
              f"This probe is not measuring what it claims; do not read on.")
    print(f"    below {STAND_Z_MIN}: {low:5d}   {100 * low / len(steps):5.1f}%")
    print(f"    above {STAND_Z_MAX}: {len(steps) - low:5d}   "
          f"{100 * (len(steps) - low) / len(steps):5.1f}%")
    print("    the upper bound is dev-specific (dev_ant.py:291); the "
          "fixed-morph ant has no ceiling, so anything here is a dev-only "
          "failure mode")

    print("\n-- 3. WHICH BODY (standardised mean difference, faller - not)")
    fell_d, ok_d = des[flat_fell], des[~flat_fell]
    if len(ok_d) > 1 and len(fell_d) > 1:
        pooled = np.sqrt((fell_d.var(0) + ok_d.var(0)) / 2) + 1e-8
        smd = (fell_d.mean(0) - ok_d.mean(0)) / pooled
        order = np.argsort(-np.abs(smd))[:5]
        for i in order:
            print(f"    scale[{i:2d}]  SMD {smd[i]:+.3f}")
        # The largest of twenty differences is large even when all twenty are
        # noise, so the number to beat is not 0 -- it is the expected maximum
        # under the null. SE(SMD) ~ sqrt(1/n1 + 1/n2); the max of d standard
        # normals sits near sqrt(2 ln d) SEs.
        n1, n2, d = len(fell_d), len(ok_d), des.shape[1]
        se = np.sqrt(1.0 / n1 + 1.0 / n2)
        null_max = np.sqrt(2 * np.log(d)) * se
        obs_max = np.abs(smd).max()
        print(f"    largest |SMD| {obs_max:.3f} over {d} dimensions "
              f"({n1} fallers vs {n2}).")
        print(f"    noise floor: SE {se:.3f}, expected max under the null "
              f"~{null_max:.3f}. Observed / null = {obs_max / null_max:.2f}x.")
        r = obs_max / null_max
        print("    " + ("below the floor -- this is what 20 noisy dimensions "
                        "look like" if r < 1.0 else
                        "SUGGESTIVE only. A ratio in [1, 2] is one marginal "
                        "dimension out of twenty and would not survive a "
                        "correction; do not build on it without a second seed"
                        if r < 2.0 else
                        "clears the floor -- worth a targeted look"))
    else:
        print("    one of the two populations is too small to compare")

    print("\n-- 4. OPPONENT DISTANCE at the moment of the fall")
    dd = fd.reshape(-1)[flat_fell]
    print(f"    at fall:      mean {dd.mean():.2f} m, median "
          f"{np.median(dd):.2f} m")
    print(f"    over all steps: mean {np.mean(all_dists):.2f} m")
    close = int((dd < 1.0).sum())
    print(f"    within 1.0 m at the fall: {close} / {len(dd)} "
          f"({100 * close / len(dd):.1f}%)")
    print("    if this is not well above the all-steps baseline, falls are "
          "not collisions")


if __name__ == "__main__":
    main()
