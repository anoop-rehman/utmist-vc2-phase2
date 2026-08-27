"""Gate the two-stage pipeline (D3 unit 3d step 5).

    # 1. in THEIR venv -- record real design episodes
    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python \
        /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/gate_two_stage.py \
        --emit --cfg hopper_gpu_s2 --checkpoint 1000 --episodes 40 --tag e1000
    .venv-gpu/bin/python ... --emit --checkpoint 0 --episodes 60 --tag e0

    # 2. in OURS
    cd /workspace/utmist-vc2-phase2
    PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_two_stage \
        --check --backend cpu          # the exact trajectory gate
    PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_two_stage \
        --check --backend warp         # the same, at fp32

--------------------------------------------------------------------------
What this gate is for, and what it deliberately is not
--------------------------------------------------------------------------
`gate_batched_exec.py` already checks the execution env for ONE topology
against their env. What step 5 adds is (a) the design stages off MuJoCo
entirely and (b) many morphologies sharing one compiled model. Both have a
failure mode that an observation check cannot see:

* a design stage that produces a *slightly* different body -- same graph, same
  observation shape, different capsule -- reads as a pass on every field a
  naive comparison would look at, and trains to a different number;
* a per-world field that is not written leaves that world simulating the
  GROUP REPRESENTATIVE's body while reporting its own design parameters in the
  observation. The observation is right. The physics is another robot.

So the design half is gated on the **exported XML string**, exactly, against
theirs; and the execution half is gated on the **trajectory** -- world i inside
a group of K, against world i compiled and run entirely on its own, from a
shared initial state over several hundred steps. Three negative controls make
sure that comparison can fail.

Precision, following `gate_batched_exec.py`'s method: the exact trajectory gate
runs on the fp64 CPU backend, where "the same body" means the same trajectory to
1e-12. On the fp32 warp backend the same hopper is chaotic (PORT_MAP section 14
measured identical worlds separating by ~1e1 after 400 steps), so there the gate
asserts ONE step and reports the envelope.
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

SCRATCH = ("/tmp/claude-0/-root/453bc0de-a27f-4894-ad03-7d048158ee36/"
           "scratchpad")
_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def ref_path(tag):
    return os.path.join(SCRATCH, f"t2a_design_ref_{tag}.json")


# ----------------------------------------------------------------- emit ----
def emit(args):
    sys.path.append("/workspace/Transform2Act")
    os.chdir("/workspace/Transform2Act")
    import torch
    from design_opt.agents.transform2act_agent import (Transform2ActAgent,
                                                       tensorfy)
    from design_opt.utils.config import Config

    torch.set_default_dtype(torch.float64)
    cfg = Config(args.cfg, tmp=False)
    ckpt = 0 if str(args.checkpoint) == "0" else (
        args.checkpoint if args.checkpoint == "best" else int(args.checkpoint))
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=args.seed,
                               num_threads=1, training=False, checkpoint=ckpt)
    env, pol = agent.env, agent.policy_net
    pol.eval()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eps = []
    for _ in range(args.episodes):
        s = env.reset()
        rec = {"steps": []}
        while True:
            with torch.no_grad():
                a = pol.select_action(tensorfy([s]), False
                                      ).numpy().astype(np.float64)
            stage, obs_in = env.stage, np.asarray(s[0])
            s, _, done, info = env.step(a)
            rec["steps"].append({"stage": stage, "obs_in": obs_in.tolist(),
                                 "action": a.tolist(),
                                 "obs_out": np.asarray(s[0]).tolist(),
                                 "names": [b.name for b in env.robot.bodies]})
            if info["stage"] == "execution" or done:
                break
        rec.update(xml=env.cur_xml_str,
                   design_params=np.asarray(env.design_cur_params).tolist(),
                   names=[b.name for b in env.robot.bodies],
                   depths=[int(b.depth) for b in env.robot.bodies],
                   edges=np.asarray(env.robot.get_gnn_edges()).tolist(),
                   body_index=np.asarray(env.get_body_index()).tolist())
        eps.append(rec)

    blob = {"cfg": args.cfg, "checkpoint": str(ckpt),
            "init_xml": env.init_xml_str.decode("utf-8"), "episodes": eps}
    os.makedirs(SCRATCH, exist_ok=True)
    with open(ref_path(args.tag), "w") as f:
        json.dump(blob, f)
    print(f"emitted {len(eps)} design episodes -> {ref_path(args.tag)}")


# ---------------------------------------------------------------- check ----
def replay(spec, init_xml, ep):
    """Their recorded actions, replayed through our CPU design stage."""
    from rower_soccer.t2a_port.design_stage import DesignWorld
    w = DesignWorld(spec, init_xml)
    trace = [w.obs()]
    for st in ep["steps"]:
        a = np.asarray(st["action"])
        if st["stage"] == "skeleton_transform":
            trace.append(w.skel_step(a[:, -1]))
        elif st["stage"] == "attribute_transform":
            trace.append(w.attr_step(a[:, 1:-1]))
            break
        else:
            break
    return w, trace


def check_design(spec, blobs):
    """The CPU design stage, against their env, exactly."""
    af = spec.max_body_depth
    worst_obs = worst_dp = 0.0
    bad_xml = bad_names = bad_edges = bad_ind = 0
    n_ep = n_step = 0
    for tag, blob in blobs.items():
        init = blob["init_xml"].encode()
        for ep in blob["episodes"]:
            w, trace = replay(spec, init, ep)
            n_ep += 1
            worst_obs = max(worst_obs, np.abs(
                trace[0] - np.asarray(ep["steps"][0]["obs_in"])).max())
            for k, st in enumerate(ep["steps"]):
                if st["stage"] == "skeleton_transform":
                    worst_obs = max(worst_obs, np.abs(
                        trace[k + 1] - np.asarray(st["obs_out"])).max())
                    n_step += 1
                elif st["stage"] == "attribute_transform":
                    # Their obs after the attribute step is already POST
                    # `transit_execution`, which resets the sim state WITH
                    # noise -- so the five sim columns are not comparable and
                    # the ones that are get compared instead. The sim columns
                    # at that point belong to the batched execution env, and
                    # `gate_batched_exec.py` owns them.
                    ref = np.asarray(st["obs_out"])
                    got = trace[k + 1]
                    worst_obs = max(worst_obs,
                                    np.abs(got[:, :af] - ref[:, :af]).max(),
                                    np.abs(got[:, af + 5:]
                                           - ref[:, af + 5:]).max())
                    n_step += 1
                    break
            bad_xml += w.cur_xml_str.strip() != ep["xml"].strip()
            bad_names += [b.name for b in w.robot.bodies] != ep["names"]
            bad_edges += not np.array_equal(np.asarray(w.edges()),
                                            np.asarray(ep["edges"]))
            bad_ind += not np.array_equal(w.body_index(),
                                          np.asarray(ep["body_index"]))
            worst_dp = max(worst_dp, np.abs(
                w.design_cur_params
                - np.asarray(ep["design_params"])).max())

    check("design: observation matches theirs at every design step",
          worst_obs == 0.0,
          f"{n_ep} episodes, {n_step} design steps, max abs diff "
          f"{worst_obs:.2e}")
    check("design: the exported XML is byte-identical to theirs",
          bad_xml == 0, f"{n_ep - bad_xml}/{n_ep} episodes")
    check("design: body order, edges and body_index match theirs",
          bad_names == 0 and bad_edges == 0 and bad_ind == 0,
          f"{bad_names} name, {bad_edges} edge, {bad_ind} index mismatches")
    check("design: projected design parameters match theirs",
          worst_dp == 0.0, f"max abs diff {worst_dp:.2e}")


def check_design_controls(spec, blob):
    """Negative controls for the design half."""
    from rower_soccer.t2a_port.design_stage import DesignWorld, _assert_no_child_ref
    init = blob["init_xml"].encode()

    # (a) The skeleton action has to reach the topology. Forcing every node to
    #     "add a child" must produce a different body than their action did.
    moved = tried = 0
    for ep in blob["episodes"][:20]:
        ref, _ = replay(spec, init, ep)
        w = DesignWorld(spec, init)
        for st in ep["steps"]:
            if st["stage"] != "skeleton_transform":
                break
            n = len(w.robot.bodies)
            w.skel_step(np.ones(n))
        tried += 1
        moved += tuple(w.topo_key()) != tuple(ref.topo_key())
    check("  control: the skeleton action drives the topology",
          tried and moved == tried, f"{moved}/{tried} episodes changed body")

    # (b) The attribute action has to reach the geometry. Zeroing it must
    #     change the exported XML -- if it did not, `robot_param_scale` or the
    #     action slice would be wrong and nothing above would notice.
    moved = tried = 0
    for ep in blob["episodes"][:20]:
        ref, _ = replay(spec, init, ep)
        w = DesignWorld(spec, init)
        for st in ep["steps"]:
            a = np.asarray(st["action"])
            if st["stage"] == "skeleton_transform":
                w.skel_step(a[:, -1])
            elif st["stage"] == "attribute_transform":
                w.attr_step(np.zeros_like(a[:, 1:-1]))
                break
        tried += 1
        moved += w.cur_xml_str.strip() != ref.cur_xml_str.strip()
    check("  control: the attribute action drives the geometry",
          tried and moved == tried, f"{moved}/{tried} episodes changed XML")

    # (c) The constant design-stage sim_obs rests on no generated joint
    #     carrying a `ref`. Inject one and the assertion must fire.
    xml = blob["episodes"][0]["xml"]
    hurt = xml.replace('<joint axis="0 -1 0" name="1_joint"',
                       '<joint ref="0.1" axis="0 -1 0" name="1_joint"', 1)
    fired = hurt != xml
    if fired:
        try:
            _assert_no_child_ref(hurt)
            fired = False
        except AssertionError:
            fired = True
    check("  control: a `ref` on a generated joint is caught",
          fired, "the design-stage sim_obs constant depends on it")


def build_worlds(spec, blob, limit=None):
    init = blob["init_xml"].encode()
    eps = blob["episodes"][:limit] if limit else blob["episodes"]
    return [replay(spec, init, ep)[0] for ep in eps]


def check_grouping(spec, blobs):
    from rower_soccer.t2a_port.two_stage_pipeline import (compile_design,
                                                          differing_fields,
                                                          group_designs)
    for tag, blob in blobs.items():
        worlds = build_worlds(spec, blob)
        g_ord = group_designs(worlds)
        g_set = collections.Counter(w.name_set_key() for w in worlds)
        print(f"  {tag}: {len(worlds)} designs -> {len(g_ord)} groups on the "
              f"ORDERED key, {len(g_set)} on the name-set key; sizes "
              f"{[len(v) for v in g_ord.values()]}")

    blob = blobs["e1000"]
    worlds = build_worlds(spec, blob)
    t0 = time.time()
    models = [compile_design(w.cur_xml_str) for w in worlds]
    dt = (time.time() - t0) / len(worlds)
    check("grouping: every design compiles in modern MuJoCo",
          len(models) == len(worlds),
          f"{len(models)}/{len(worlds)}, {1000 * dt:.2f} ms per design")

    key, idx = next(iter(group_designs(worlds).items()))
    diff = differing_fields([models[i] for i in idx])
    check("grouping: same-topology designs really do differ in the model",
          len(diff) > 5, f"{len(diff)} fields differ across {len(idx)} worlds")
    print(f"       {', '.join(diff)}")

    # Negative control on the group key: force two DIFFERENT topologies into
    # one group and `differing_fields` must refuse on the shape mismatch.
    sizes = collections.defaultdict(list)
    for i, w in enumerate(worlds):
        sizes[len(w.robot.bodies)].append(i)
    mixed = None
    for tag, blob2 in blobs.items():
        w2 = build_worlds(spec, blob2)
        m2 = {}
        for i, w in enumerate(w2):
            m2.setdefault(w.topo_key(), i)
        keys = list(m2)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                if len(keys[a]) != len(keys[b]):
                    mixed = (w2[m2[keys[a]]], w2[m2[keys[b]]])
                    break
            if mixed:
                break
        if mixed:
            break
    raised = False
    if mixed:
        try:
            differing_fields([compile_design(mixed[0].cur_xml_str),
                              compile_design(mixed[1].cur_xml_str)])
        except ValueError:
            raised = True
    check("  control: mixing two topologies into one group is refused",
          raised, "differing_fields() raises on the shape mismatch")


def _rollout(env, qpos0, qvel0, actions, rows):
    """Roll `env` from a given per-world state with a fixed action tape.

    `rows` selects which worlds of `actions` this env is running, so a K-world
    group and K single-world envs can be driven by the SAME tape.
    """
    import torch
    env.set_state(qpos0[rows], qvel0[rows], control_nsteps=0)
    out = []
    for t in range(actions.shape[0]):
        a = torch.as_tensor(actions[t][rows], device=env.device,
                            dtype=env.dtype)
        env.step(a, auto_reset=False)
        out.append(env.backend.qpos.detach().cpu().numpy().copy())
    return np.stack(out)                 # [T, len(rows), nq]


def check_trajectories(spec, blob, backend, steps, max_group,
                       done_condition=None):
    import torch

    from rower_soccer.t2a_port.two_stage_pipeline import (TopologyGroup,
                                                          compile_design,
                                                          group_designs)
    worlds = build_worlds(spec, blob)
    groups = group_designs(worlds)
    key, idx = next(iter(groups.items()))
    idx = idx[:max_group]
    members = [worlds[i] for i in idx]
    models = [compile_design(w.cur_xml_str) for w in members]
    K = len(members)
    nq = models[0].nq
    nv = models[0].nv
    print(f"\ntrajectory gate: {K} worlds of one topology "
          f"({len(members[0].robot.bodies)} bodies), {steps} steps, "
          f"backend {backend}")

    rng = np.random.default_rng(0)
    n_nodes = len(members[0].robot.bodies)
    actions = rng.uniform(-0.4, 0.4, size=(steps, K, n_nodes, 1))
    # Distinct per-world starts, so the comparison is not accidentally testing
    # one state K times.
    import mujoco
    md = mujoco.MjData(models[0])
    mujoco.mj_forward(models[0], md)
    qpos0 = np.repeat(md.qpos.copy()[None], K, 0) + rng.uniform(-5e-3, 5e-3,
                                                                (K, nq))
    qvel0 = np.repeat(md.qvel.copy()[None], K, 0) + rng.uniform(-5e-3, 5e-3,
                                                                (K, nv))

    common = dict(backend=backend, done_condition=done_condition,
                  reward_specs={}, clip_qvel=True, init_noise=0.0)

    def grouped(**kw):
        return TopologyGroup(key, members, models, spec, **dict(common, **kw))

    # --- the reference: each world compiled and run entirely on its own ----
    singles = []
    for j in range(K):
        g = TopologyGroup(key, [members[j]], [models[j]], spec, **common)
        singles.append(_rollout(g.env, qpos0[j:j + 1], qvel0[j:j + 1],
                                actions[:, j:j + 1], [0])[:, 0])
    ref = np.stack(singles, axis=1)       # [T, K, nq]

    g = grouped()
    got = _rollout(g.env, qpos0, qvel0, actions, list(range(K)))
    err = np.abs(got - ref)
    per_step = err.reshape(steps, -1).max(1)

    exact = g.env.dtype == torch.float64
    tol = 1e-12 if exact else 3e-6
    if exact:
        check("trajectory: a grouped world matches its own compiled model",
              per_step.max() < tol,
              f"{K} worlds x {steps} steps, max abs qpos diff "
              f"{per_step.max():.3e} (tol {tol:.0e})")
    else:
        check("trajectory: ONE step of a grouped world matches its own "
              "compiled model", per_step[0] < tol,
              f"fp32; step 1 max abs qpos diff {per_step[0]:.3e} "
              f"(tol {tol:.0e})")
        print("       fp32 envelope (chaos, not a defect): "
              + ", ".join(f"step {t + 1}: {per_step[t]:.2e}"
                          for t in (0, 9, 49, min(steps, 300) - 1)))

    # --- negative controls: each must BREAK the agreement -------------------
    print("\n  negative controls (each must break the trajectory match):")
    for label, kw in (
            ("no per-world fields written at all", dict(write=False)),
            ("per-world fields rolled by one",
             dict(field_perm=list(range(1, K)) + [0])),
            ("actuator_gear left unwritten",
             dict(drop_fields=("actuator_gear",))),
            ("body_mass/body_inertia left unwritten",
             dict(drop_fields=("body_mass", "body_inertia",
                               "body_subtreemass")))):
        gb = grouped(**kw)
        bad = _rollout(gb.env, qpos0, qvel0, actions, list(range(K)))
        d = float(np.abs(bad - ref).max())
        d1 = float(np.abs(bad[0] - ref[0]).max())
        broke = d1 > tol * 10
        check(f"    {label}", broke,
              f"step-1 diff {d1:.3e}, worst {d:.3e}")

    # The controls above would also "pass" if the grouped env were simply
    # broken. Prove it is not: with everything written, the SAME comparison is
    # the one that just passed.
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emit", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--cfg", default="hopper_gpu_s2")
    p.add_argument("--checkpoint", default="1000")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tag", default="e1000")
    p.add_argument("--backend", default="cpu", choices=["cpu", "warp"])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--max-group", type=int, default=12)
    args = p.parse_args()

    if args.emit:
        emit(args)
        return 0
    if not args.check:
        p.error("pass --emit or --check")

    import yaml
    from rower_soccer.t2a_port.design_stage import DesignSpec
    cfg_path = f"/workspace/Transform2Act/design_opt/cfg/{args.cfg}.yml"
    cfg_dict = yaml.safe_load(open(cfg_path))
    spec = DesignSpec(cfg_dict)

    blobs = {}
    for tag in ("e0", "e1000"):
        if os.path.exists(ref_path(tag)):
            blobs[tag] = json.load(open(ref_path(tag)))
    if not blobs:
        raise SystemExit(f"no reference under {SCRATCH}; run --emit first")
    print("references: " + ", ".join(
        f"{t} ({len(b['episodes'])} episodes, checkpoint {b['checkpoint']})"
        for t, b in blobs.items()))

    check_design(spec, blobs)
    check_design_controls(spec, blobs["e0"])
    check_grouping(spec, blobs)
    check_trajectories(spec, blobs["e1000"], args.backend, args.steps,
                       args.max_group,
                       done_condition=cfg_dict.get("done_condition"))

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
