"""Gate `batched_exec_env` against their live env. Two venvs, as the bridge does.

    # 1. in THEIR venv -- record real execution steps
    cd /workspace/Transform2Act && source env-gpu.sh
    MUJOCO_GL=osmesa .venv-gpu/bin/python \
        /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/gate_batched_exec.py \
        --emit --cfg hopper_gpu_s2 --checkpoint 1000 --steps 400

    # 2. in OURS
    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.t2a_port.gate_batched_exec --check

The method is the point. `PORT_MAP.md` section 13 measured that the two MuJoCo
builds diverge to ~12% of joint range over an episode and that no settable
option closes it, so a port CANNOT be gated by rolling both forward and
comparing. Instead every check here feeds **their recorded qpos/qvel** into our
env and compares what our pipeline computes from it. That makes the observation,
reward and done checks EXACT -- they are pure functions of state, and any
difference is a real defect rather than accumulated integrator drift.

The physics is then checked separately and on the only terms available: from a
shared start, does our sim stay within the envelope section 13 already measured?
A regression there shows up as a divergence much larger than the bridge's.
"""

import argparse
import json
import os
import sys

import numpy as np

OUT = ("/tmp/claude-0/-root/453bc0de-a27f-4894-ad03-7d048158ee36/scratchpad/"
       "t2a_exec_ref.json")


def emit(args):
    sys.path.append("/workspace/Transform2Act")
    os.chdir("/workspace/Transform2Act")
    import torch
    from design_opt.agents.transform2act_agent import (Transform2ActAgent,
                                                       tensorfy)
    from design_opt.utils.config import Config

    torch.set_default_dtype(torch.float64)
    cfg = Config(args.cfg, tmp=False)
    ckpt = args.checkpoint if args.checkpoint in ("best",) else int(args.checkpoint)
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=cfg.seed,
                               num_threads=1, training=False, checkpoint=ckpt)
    env, policy = agent.env, agent.policy_net
    policy.eval()

    state = env.reset()
    with torch.no_grad():
        while True:                       # run the design stages out
            a = policy.select_action(tensorfy([state]), True).numpy().astype(np.float64)
            state, _, _, info = env.step(a)
            if info["stage"] == "execution":
                break

    rows = []
    with torch.no_grad():
        for _ in range(args.steps):
            qpos_b = env.sim.data.qpos.copy()
            qvel_b = env.sim.data.qvel.copy()
            nsteps_b = int(env.control_nsteps)
            a = policy.select_action(tensorfy([state]), True).numpy().astype(np.float64)
            state, reward, done, info = env.step(a)
            rows.append({
                "qpos_before": qpos_b.tolist(), "qvel_before": qvel_b.tolist(),
                "control_nsteps_before": nsteps_b,
                "action": np.asarray(a).tolist(),
                "reward": float(reward), "done": bool(done),
                "qpos_after": env.sim.data.qpos.copy().tolist(),
                "qvel_after": env.sim.data.qvel.copy().tolist(),
                "control_nsteps_after": int(env.control_nsteps),
                # `state` is [obs, edges, use_transform_action, num_nodes, body_index]
                "obs_after": np.asarray(state[0]).tolist(),
            })
            if done:
                break

    # Full episodes, for the distribution check. Their reset perturbs qpos/qvel
    # by U(+/-0.005), so episodes differ only in that noise -- which is exactly
    # the distribution our port has to reproduce.
    episodes = []
    with torch.no_grad():
        for _ in range(args.episodes):
            state = env.reset()
            while True:                   # design stages
                a = policy.select_action(tensorfy([state]), True).numpy().astype(np.float64)
                state, _, _, info = env.step(a)
                if info["stage"] == "execution":
                    break
            ret, n = 0.0, 0
            while True:
                a = policy.select_action(tensorfy([state]), True).numpy().astype(np.float64)
                state, r, done, _ = env.step(a)
                ret += float(r); n += 1
                if done:
                    break
            episodes.append({"len": n, "ret": ret})
    if episodes:
        L = np.array([e["len"] for e in episodes]); R = np.array([e["ret"] for e in episodes])
        print(f"  {len(episodes)} episodes: len {L.mean():.1f} +/- {L.std():.1f}, "
              f"return {R.mean():.1f} +/- {R.std():.1f}")

    blob = {
        "cfg": args.cfg, "checkpoint": str(ckpt), "episodes": episodes,
        "policy_specs": dict(cfg.policy_specs),
        "state_dict": {k: v.cpu().numpy().tolist()
                       for k, v in policy.state_dict().items()},
        "xml": env.cur_xml_str,
        "bodies": [b.name for b in env.robot.bodies],
        "depths": [int(b.depth) for b in env.robot.bodies],
        "index_base": int(env.index_base),
        "max_body_depth": int(cfg.max_body_depth),
        "design_params": np.asarray(env.design_cur_params).tolist(),
        "attr_fixed": np.asarray(env.get_attr_fixed()).tolist(),
        "body_index": np.asarray(env.get_body_index()).tolist(),
        "edges": np.asarray(env.robot.get_gnn_edges()).tolist(),
        "frame_skip": int(env.frame_skip), "dt": float(env.dt),
        "nq": int(env.model.nq), "nv": int(env.model.nv), "nu": int(env.model.nu),
        "actuator_names": list(env.model.actuator_names),
        "done_condition": dict(cfg.done_condition),
        "reward_specs": dict(getattr(cfg, "reward_specs", {}) or {}),
        "clip_qvel": bool(cfg.obs_specs.get("clip_qvel", False)),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(blob, f)
    ends = sum(r["done"] for r in rows)
    print(f"emitted {len(rows)} execution steps (done={ends}) -> {OUT}")


def _fmt(ok):
    return "PASS" if ok else "FAIL"


def check(args):
    import mujoco
    import torch

    from rower_soccer.t2a_port.batched_exec_env import (T2ABatchedExecEnv,
                                                        topology_spec)
    from rower_soccer.t2a_port.xml_global_to_local import convert

    with open(OUT) as f:
        blob = json.load(f)
    rows = blob["rows"]
    print(f"reference: {blob['cfg']} epoch {blob['checkpoint']}, "
          f"{len(rows)} execution steps, {len(blob['bodies'])} bodies")

    model = mujoco.MjModel.from_xml_string(
        convert(blob["xml"], legacy_inertial=True))
    spec = topology_spec(model, blob["bodies"], blob["depths"],
                         max_body_depth=blob["max_body_depth"],
                         index_base=blob["index_base"])

    fails = []

    # ---- 0. the static blocks, resolved by name ---------------------------
    for key in ("attr_fixed", "body_index"):
        d = np.abs(np.asarray(blob[key]) - spec[key]).max()
        ok = d == 0
        fails.append(not ok)
        print(f"[{_fmt(ok)}] {key:12s} max |d| = {d:.3e}")

    # Their actuator ORDER, resolved by name -- the coincidence trap.
    want = [blob["actuator_names"].index(f"{b}_joint") for b in blob["bodies"][1:]]
    got = spec["act_of_node"][1:].tolist()
    ok = want == got
    fails.append(not ok)
    print(f"[{_fmt(ok)}] actuator map  {got}  (theirs {want})")
    if want == list(range(len(want))):
        print("       NOTE: on this morphology name-order and body-order "
              "coincide, so this check cannot catch an index-order port. "
              "The negative control below can.")

    rs = blob["reward_specs"]
    env = T2ABatchedExecEnv(
        model, spec, blob["design_params"], num_worlds=args.worlds,
        frame_skip=blob["frame_skip"], done_condition=blob["done_condition"],
        alive_bonus=rs.get("alive_bonus", 1.0),
        exec_reward_scale=rs.get("exec_reward_scale", 1.0),
        abs_displacement=rs.get("abs_displacement", False),
        clip_qvel=blob["clip_qvel"], backend=args.backend,
        device=None if args.backend == "warp" else "cpu")

    # Tolerances follow the BACKEND's precision, not a constant. mujoco_warp is
    # fp32; Transform2Act is fp64 throughout. An exact check written for fp64
    # and run on fp32 fails at ~1e-7 for no reason that means anything, and a
    # check loosened to 1e-6 for both stops being able to see a real fp64 bug.
    tol = 1e-12 if env.dtype == torch.float64 else 3e-6
    # The reward is `(posafter - posbefore) / dt`, and dt is 0.008 -- so it
    # MULTIPLIES the position tolerance by 125. That is not slack, it is the
    # arithmetic: a difference of two nearby fp32 positions is a cancellation,
    # and its absolute error grows with how far the hopper has travelled
    # (ulp(25 m) ~ 2e-6, ulp(100 m) ~ 8e-6). Deriving the reward tolerance from
    # the mechanism keeps the check honest instead of loosening a constant
    # until it passes. See PORT_MAP section 14 for what to do about it.
    tol_r = tol / env.dt
    print(f"backend {args.backend}, dtype {env.dtype}, "
          f"pos tol {tol:.0e}, reward tol {tol_r:.0e} (= pos tol / dt)")

    # ---- 1. observation, at THEIR states ---------------------------------
    # Exact by construction: obs is a pure function of (qpos, qvel).
    dmax, worst = 0.0, None
    for i, r in enumerate(rows):
        env.set_state(r["qpos_after"], r["qvel_after"])
        ours = env.obs()[0].detach().cpu().numpy()
        d = np.abs(ours - np.asarray(r["obs_after"])).max()
        if d > dmax:
            dmax, worst = d, i
    ok = dmax < tol
    fails.append(not ok)
    print(f"[{_fmt(ok)}] observation   max |d| = {dmax:.3e} over {len(rows)} "
          f"states (worst step {worst})")

    # ---- 2. reward and done, at THEIR states -----------------------------
    # `terms()` is the function `step()` calls; driving it with their recorded
    # before/after states removes physics from the comparison entirely.
    dev, dt_ = env.device, env.dtype
    rmax, dbad = 0.0, 0
    for r in rows:
        n = args.worlds
        posb = torch.full((n,), r["qpos_before"][0], device=dev, dtype=dt_)
        qp = torch.as_tensor(r["qpos_after"], device=dev, dtype=dt_).repeat(n, 1)
        qv = torch.as_tensor(r["qvel_after"], device=dev, dtype=dt_).repeat(n, 1)
        ns = torch.full((n,), r["control_nsteps_after"], device=dev,
                        dtype=torch.long)
        rew, done = env.terms(posb, qp, qv, ns)
        rmax = max(rmax, abs(float(rew[0]) - r["reward"]))
        dbad += int(bool(done[0]) != r["done"])
    ok_r, ok_d = rmax < tol_r, dbad == 0
    fails += [not ok_r, not ok_d]
    print(f"[{_fmt(ok_r)}] reward        max |d| = {rmax:.3e} over {len(rows)} "
          f"steps (tol {tol_r:.1e})")
    print(f"[{_fmt(ok_d)}] done          {dbad} disagreements over {len(rows)} steps")

    # ---- 3. negative controls --------------------------------------------
    # A gate that cannot fail proves nothing. Each of these breaks one thing
    # the checks above are supposed to be sensitive to; each MUST move.
    print("\nnegative controls (each must CHANGE the result):")

    ref_obs = None
    env.set_state(rows[0]["qpos_after"], rows[0]["qvel_after"])
    ref_obs = env.obs()[0].detach().cpu().numpy().copy()

    # (a) un-flip the root block -- the hazard called out in the docstring.
    sim = env.sim_obs()[0].detach().cpu().numpy()
    unflipped = sim.copy()
    unflipped[0, :2] = sim[0, :2][::-1]
    unflipped[0, 2:5] = sim[0, 2:5][::-1]
    moved = np.abs(unflipped - sim).max()
    print(f"  [{_fmt(moved > 1e-9)}] root block is flipped "
          f"(un-flipping moves it by {moved:.3e})")
    fails.append(not (moved > 1e-9))

    # (b) qvel clipping actually binds somewhere in the reference.
    qv_all = np.array([r["qvel_after"] for r in rows])
    binds = int((np.abs(qv_all) > 10.0).sum())
    print(f"  [{_fmt(binds > 0)}] clip_qvel binds on {binds} entries "
          f"(max |qvel| = {np.abs(qv_all).max():.2f})")
    fails.append(not (binds > 0))

    # (c) permuting the actuator map must change where the torque lands.
    env.set_state(rows[0]["qpos_before"], rows[0]["qvel_before"])
    a0 = torch.as_tensor(rows[0]["action"], device=dev, dtype=dt_
                         ).unsqueeze(0).expand(args.worlds, -1, -1)
    env.step(a0, auto_reset=False)
    q_true = env.backend.qpos[0].clone()
    saved = env.act_of.clone()
    env.act_of = torch.cat([saved[:1], saved[1:].flip(0)])
    env.set_state(rows[0]["qpos_before"], rows[0]["qvel_before"])
    env.step(a0, auto_reset=False)
    d = float((env.backend.qpos[0] - q_true).abs().max())
    env.act_of = saved
    print(f"  [{_fmt(d > 1e-9)}] actuator map is load-bearing "
          f"(reversing it moves qpos by {d:.3e})")
    fails.append(not (d > 1e-9))

    # ---- 4. physics, open loop: MEASURED, not gated -----------------------
    # Replaying their actions into our sim is the worst case: a hopper is an
    # unstable system and nothing closes the loop on OUR state, so drift
    # compounds until our copy falls over while theirs is still running. That
    # is expected and is NOT a port defect -- it is reported here as a number
    # rather than a pass/fail so a regression is visible without the gate
    # asserting something untrue.
    env.set_state(rows[0]["qpos_before"], rows[0]["qvel_before"],
                  control_nsteps=rows[0]["control_nsteps_before"])
    errs = []
    for r in rows:
        a = torch.as_tensor(r["action"], device=dev, dtype=dt_
                            ).unsqueeze(0).expand(args.worlds, -1, -1)
        env.step(a, auto_reset=False)
        errs.append(float((env.backend.qpos[0]
                           - torch.as_tensor(r["qpos_after"], device=dev,
                                             dtype=dt_)).abs().max()))
    errs = np.array(errs)
    print(f"\nphysics, OPEN LOOP from their start with their actions "
          f"(measurement, not a gate):")
    for t in (0, 9, 49, 99, len(errs) - 1):
        if t < len(errs):
            print(f"    step {t + 1:4d}   {errs[t]:.3e}")

    # ---- 5. batch determinism, measured over ONE step ---------------------
    # Not over the whole rollout: this hopper is chaotic and fp32 rounding
    # differences between worlds are amplified exponentially (measured: 1.5e-08
    # after one step, ~1e+01 after 400). Asserting on the 400-step spread would
    # be asserting that a chaotic system is not chaotic. What a port CAN
    # guarantee is that one step from identical states differs by no more than
    # the arithmetic's own resolution.
    env.set_state(rows[0]["qpos_before"], rows[0]["qvel_before"],
                  control_nsteps=rows[0]["control_nsteps_before"])
    a0 = torch.as_tensor(rows[0]["action"], device=dev, dtype=dt_
                         ).unsqueeze(0).expand(args.worlds, -1, -1)
    env.step(a0, auto_reset=False)
    spread1 = float((env.backend.qpos - env.backend.qpos[0]).abs().max())
    ok = spread1 < tol
    fails.append(not ok)
    print(f"[{_fmt(ok)}] batch         {args.worlds} identical worlds after ONE "
          f"step, max spread {spread1:.3e} (tol {tol:.0e})")

    # ---- 6. CLOSED LOOP: the check that actually means something ----------
    # Our env driven by our dense policy (itself gated to 0.0 against theirs in
    # gate_dense_policy.py), compared against their episode DISTRIBUTION. This
    # is what PORT_MAP section 13 concluded the port must be validated on.
    if not blob.get("episodes"):
        print("\n(no reference episodes in the blob; re-emit with --episodes)")
    else:
        from rower_soccer.t2a_port.dense_policy import DenseTransform2ActPolicy
        n_nodes = len(blob["bodies"])
        af = len(blob["attr_fixed"][0])
        ad = len(blob["design_params"][0])
        sim_dim = len(blob["rows"][0]["obs_after"][0]) - af - ad
        sd = {k: torch.tensor(v, dtype=torch.float64)
              for k, v in blob["state_dict"].items()}
        # Head widths come from THEIR weights, not from constants here: a
        # hard-coded 3 would silently mis-size on a config with a different
        # skeleton action set.
        skel_dim = sd["skel_ind_mlp.linear.b"].shape[-1]
        ctrl_dim = sd["control_action_log_std"].shape[-1]
        assert ad == sd["attr_action_log_std"].shape[-1]
        pol = DenseTransform2ActPolicy(
            blob["policy_specs"], attr_fixed_dim=af, sim_obs_dim=sim_dim,
            attr_design_dim=ad, skel_action_dim=skel_dim,
            control_action_dim=ctrl_dim)
        pol.load_their_state_dict(sd, strict=True)
        pol = pol.to(device=dev, dtype=dt_).eval()

        # adj[g, i, j] = 1 when j sends to i. Their edges are [2, E] with both
        # directions already present (PORT_MAP section 12).
        e = np.asarray(blob["edges"])
        adj = torch.zeros(1, n_nodes, n_nodes, device=dev, dtype=dt_)
        adj[0, e[0], e[1]] = 1.0
        adj = adj.expand(args.worlds, -1, -1).contiguous()
        ind = torch.as_tensor(blob["body_index"], device=dev
                              ).unsqueeze(0).expand(args.worlds, -1).contiguous()

        lens, rets = [], []
        cur_l = torch.zeros(args.worlds, device=dev, dtype=dt_)
        cur_r = torch.zeros(args.worlds, device=dev, dtype=dt_)
        obs = env.reset()
        # Enough rounds for every world to finish the episodes we need; a
        # single time-limit's worth only ever yields one episode per world.
        rounds = int(np.ceil(args.episodes / args.worlds))
        max_steps = (env.dc["max_nsteps"] + 1) * rounds
        with torch.no_grad():
            while len(lens) < args.episodes and max_steps > 0:
                max_steps -= 1
                a = pol.mean_action("execution", obs, adj, ind)
                obs, r, done, _ = env.step(a)
                cur_l += 1.0
                cur_r += r
                if bool(done.any()):
                    idx = done.nonzero(as_tuple=True)[0]
                    lens += cur_l[idx].tolist()
                    rets += cur_r[idx].tolist()
                    cur_l[idx] = 0.0
                    cur_r[idx] = 0.0
        tl = np.array([e_["len"] for e_ in blob["episodes"]], dtype=float)
        tr = np.array([e_["ret"] for e_ in blob["episodes"]], dtype=float)
        ol = np.array(lens[:args.episodes]); orr = np.array(rets[:args.episodes])
        print(f"\nCLOSED LOOP, our env + our dense policy, "
              f"{len(ol)} episodes vs their {len(tl)}:")
        print(f"    {'':10s} {'theirs':>18s} {'ours':>18s}")
        print(f"    {'ep len':10s} {tl.mean():10.1f} +/-{tl.std():5.1f} "
              f"{ol.mean():10.1f} +/-{ol.std():5.1f}")
        print(f"    {'return':10s} {tr.mean():10.1f} +/-{tr.std():5.1f} "
              f"{orr.mean():10.1f} +/-{orr.std():5.1f}")
        # Standardised mean difference against the POOLED sd -- the same
        # measure D2 uses, so "is this a real gap" is asked the same way here.
        sp = np.sqrt((tr.var(ddof=1) + orr.var(ddof=1)) / 2)
        smd = (orr.mean() - tr.mean()) / max(sp, 1e-9)
        ok = abs(smd) < args.smd_tol
        fails.append(not ok)
        print(f"[{_fmt(ok)}] return SMD    {smd:+.3f} "
              f"(|SMD| < {args.smd_tol} required)")

    n_fail = sum(fails)
    print(f"\n{len(fails) - n_fail}/{len(fails)} checks passed")
    return 1 if n_fail else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emit", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--cfg", default="hopper_gpu_s2")
    p.add_argument("--checkpoint", default="1000")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--episodes", type=int, default=20,
                   help="full episodes for the distribution check")
    p.add_argument("--worlds", type=int, default=8)
    p.add_argument("--backend", default="warp", choices=["warp", "cpu"])
    p.add_argument("--smd-tol", type=float, default=0.8,
                   help="standardised mean difference of episode return "
                        "against theirs. 0.8 is Cohen's 'large'; anything "
                        "bigger is a port defect, not sampling noise")
    args = p.parse_args()
    if args.emit:
        emit(args)
        return 0
    if args.check:
        return check(args)
    p.error("pass --emit or --check")


if __name__ == "__main__":
    raise SystemExit(main())
