"""D3 M3 E3: the morphology and fall-dodge instruments, computed AS THE RUN
PROCEEDS rather than post-hoc.

Two hazards E3 has that no earlier D3 rung had, and one measurement discipline
that both of them need.

**1. The design stage might silently no-op.** E2/E2.1 gated the MIRROR of this
-- 134 mjModel arrays IDENTICAL under destructive random design actions -- and
a design stage that quietly did nothing would reproduce E2.1's frozen-body
numbers exactly and read as a clean null. `body_summary` reads the body out of
the COMPILED mjModel, not out of the `Robot` object that asked for it, so
"the design stage wrote it" and "the simulator ran it" are two different
measurements here and `gate_e3.py` phase 2 checks they agree.

**2. Morphology is a far wider channel to the fall-dodge than control alone.**
E2 measured `r(fall rate, return) = +0.989` and `r(forward progress, return) =
+0.019` -- return measured falling, not running -- and E2.1's `d2rep` reward
regime inverted it to -0.517 / +0.947. An agent that can reshape its body could
evolve toward falling reliably, so `corr` recomputes that pair every evaluation
and the trainer logs it from epoch 0. If E3's drifts back toward E2's
structure, the dodge has reopened through morphology.

**3. An instrument must not perturb what it measures.** `rng_guard` saves and
restores numpy's, torch's and the env's own random state around every probe, so
adding a per-epoch census cannot shift the trajectory the next training epoch
samples. E2's inline evaluation did NOT do this -- `e2_eval.roll` reseeds the
global stream per episode -- and `D3_E21_CURRICULUM.md` 5c names that as its
best (untested) explanation for why its flat control did not reproduce E2
bitwise. E3 does not inherit that.
"""

import contextlib
import hashlib

import numpy as np
import torch

# The five attribute-genome columns in the order `Body.get_params` emits them
# for a non-root body, each demapped to roughly [-1, 1] by the `sin` mapping.
# Same order and meaning as `e0_analyse.GENOME_COLS`, so an E3 genome and an
# E1 genome are differenceable column for column.
GENOME_COLS = ["offset_x", "offset_y", "gear", "size", "ext_start"]
OPP_PREFIX = "opp_"


# ------------------------------------------------------------- rng guard --
@contextlib.contextmanager
def rng_guard(env=None):
    """Everything a probe draws from, saved and put back."""
    np_state = np.random.get_state()
    t_state = torch.get_rng_state()
    e_state = None
    if env is not None and hasattr(env, "np_random"):
        e_state = env.np_random.get_state()
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(t_state)
        if e_state is not None:
            env.np_random.set_state(e_state)


# ------------------------------------------------------------- topology --
def topo_key(env):
    """A topology is the SET of body names, which is a complete tree
    identifier: `xml_robot.py` names the root '0' and every child
    `str(sibling_index) + parent_name`, so a name IS its path from the root.
    Identical to `topology_census.topo_key`, restated here so E3's instrument
    has no import-time dependency on a file E0 owns."""
    names = tuple(sorted(b.name for b in env.robot.bodies))
    return hashlib.md5("|".join(names).encode()).hexdigest()[:12], names


# -------------------------------------------------------- the body itself --
def body_summary(env):
    """The current body, read TWICE and from two different places.

    `robot_*` comes from the `Robot` object -- what the design stage asked
    for. `model_*` comes from the compiled `mjModel` -- what MuJoCo is
    actually integrating. They are separate because a design stage that
    exports an XML the simulator never loads would look correct in the first
    and wrong in the second, and that failure produces a clean, boring,
    completely wrong null.

    The opponent is excluded by name from every model-side aggregate: it is a
    sibling body in the same MJCF (`D3_E2_RTG.md` 2), so `body_mass.sum()`
    over the whole model is 27 bodies of ant, not 13.
    """
    m, robot = env.model, env.robot
    key, names = topo_key(env)

    ours = [i for i, n in enumerate(m.body_names)
            if n != "world" and not n.startswith(OPP_PREFIX)]
    our_geoms = [i for i in range(m.ngeom)
                 if m.geom_bodyid[i] in ours]
    our_act = [i for i, n in enumerate(m.actuator_names)
               if not n.startswith(OPP_PREFIX)]

    # Robot-side: the genome and the physical parameters the design heads set.
    genome = {b.name: env.get_attr_design()[i].tolist()
              for i, b in enumerate(robot.bodies)}
    phys = {}
    for b in robot.bodies:
        g = b.geoms[0]
        # The root's geom is a SPHERE and carries no `fromto`; every limb is a
        # capsule, so `.start`/`.end` exist only there.
        length = (float(np.linalg.norm(np.asarray(g.end) - np.asarray(g.start)))
                  if g.type == "capsule" else 0.0)
        gear = (float(b.joints[0].actuator.gear)
                if b.joints and b.joints[0].actuator is not None else None)
        phys[b.name] = {"radius": float(np.asarray(g.size).reshape(-1)[0]),
                        "length": length, "gear": gear,
                        "depth": int(b.depth), "type": str(g.type),
                        "n_joints": len(b.joints)}

    limbs = {k: v for k, v in phys.items() if k != robot.bodies[0].name}
    ln = [v["length"] for v in limbs.values()]
    rad = [v["radius"] for v in limbs.values()]
    gr = [v["gear"] for v in limbs.values() if v["gear"] is not None]
    dep = [v["depth"] for v in limbs.values()]

    def agg(v):
        a = np.asarray(v, dtype=float)
        return dict(n=int(a.size), mean=float(a.mean()) if a.size else 0.0,
                    min=float(a.min()) if a.size else 0.0,
                    max=float(a.max()) if a.size else 0.0,
                    sum=float(a.sum()) if a.size else 0.0)

    return {
        "topo": key,
        "names": list(names),
        # -- robot side: what the design stage produced ---------------------
        "n_bodies": len(robot.bodies),
        "n_limbs": len(limbs),
        "limb_length": agg(ln),
        "limb_radius": agg(rad),
        "gear": agg(gr),
        "depth_hist": {str(d): int(dep.count(d)) for d in sorted(set(dep))},
        "genome": genome,
        # -- model side: what MuJoCo is integrating -------------------------
        "model_nbody_ours": len(ours),
        "model_nu_ours": len(our_act),
        "model_mass_ours": float(m.body_mass[ours].sum()),
        "model_geom_len_sum": float(2.0 * m.geom_size[our_geoms, 1].sum()),
        "model_gear_sum": float(m.actuator_gear[our_act, 0].sum()),
        # -- whole model, for the opponent-survives check --------------------
        "model_nbody": int(m.nbody), "model_nu": int(m.nu),
        "model_nq": int(m.nq), "model_nv": int(m.nv),
        "n_opp_bodies": int(sum(1 for n in m.body_names
                                if n.startswith(OPP_PREFIX))),
    }


def tensorfy(np_list):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x) for x in y] for y in np_list]
    return [torch.tensor(y) for y in np_list]


def run_design_stages(env, policy, mean_action, running_state=None):
    """Step ONLY the design stages -- 5 skeleton steps then 1 attribute step.
    No physics runs in either, so this is cheap enough to call every epoch."""
    state = env.reset()
    if running_state is not None:
        state = running_state(state)
    for _ in range(env.cfg.skel_transform_nsteps + 2):
        if env.if_use_transform_action() == 2:
            return True
        with torch.no_grad():
            a = policy.select_action(tensorfy([state]),
                                     mean_action).numpy().astype(np.float64)
        state, _, done, info = env.step(a)
        if running_state is not None:
            state = running_state(state)
        if done:
            return False
    return env.if_use_transform_action() == 2


def census(env, policy, episodes, mean_action=False, running_state=None):
    """`episodes` sampled designs, exactly the way a training epoch draws them.

    RUNS IN TEST MODE at the call site: the policy's three `RunningNorm` layers
    update their buffers on every forward while `training` is true, so
    sampling in train mode normalises each design against statistics the
    previous designs just moved. `D3_E0_ANT.md` 3 measured what that costs.
    """
    counts, bodies, failed = {}, [], 0
    genomes = []
    for _ in range(episodes):
        if not run_design_stages(env, policy, mean_action, running_state):
            failed += 1
            continue
        key, names = topo_key(env)
        counts[key] = counts.get(key, 0) + 1
        bodies.append(len(names))
        g = np.asarray(env.get_attr_design())
        if len(g) > 1:
            genomes.append(g[1:])          # drop the root: it pads with zeros
    n = sum(counts.values())
    top = max(counts.values()) if counts else 0
    pop = np.concatenate(genomes, axis=0) if genomes else np.zeros((0, 5))
    return {
        "sampled": n, "design_failed": failed,
        "distinct_topologies": len(counts),
        "top_topology_share": (top / n) if n else 0.0,
        "bodies_mean": float(np.mean(bodies)) if bodies else 0.0,
        "bodies_min": int(min(bodies)) if bodies else 0,
        "bodies_max": int(max(bodies)) if bodies else 0,
        "sampled_genome_std": (pop.std(axis=0, ddof=1).tolist()
                               if pop.shape[0] > 1 else [0.0] * 5),
        "sampled_genome_rows": int(pop.shape[0]),
    }


# ------------------------------------------------- the fall-dodge instrument --
def corr(x, y):
    """Pearson r, or None where a column has no variance.

    Returning None rather than 0.0 is deliberate: `D3_E21_CURRICULUM.md` 5e
    reports `d2rep`'s `r(fell, R)` as UNDEFINED because its fall rate is
    exactly 0, and a 0.0 there would read as "uncorrelated" when what it means
    is "solved". The distinction is the whole point of the statistic.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def dodge_stats(episodes):
    """E2's correlation pair, plus the premium a fall is actually worth here.

    `episodes` is `e2_eval.evaluate`'s per-episode list. `fall_premium` is
    E2 6's measurement -- mean return conditional on ending in a fall, minus
    mean return conditional on the opponent scoring -- recomputed on THIS
    arm's own episodes rather than inherited as +826 from the idle control.
    """
    if not episodes:
        return {}
    R = [e["R"] for e in episodes]
    fell = [float(e["fell"]) for e in episodes]
    fwd = [e["max_fwd"] for e in episodes]
    out = {"r_fall_return": corr(fell, R), "r_fwd_return": corr(fwd, R)}
    f = [e["R"] for e in episodes if e["fell"]]
    l = [e["R"] for e in episodes if e["opp_reached"]]
    out["n_fell"] = len(f)
    out["n_lost"] = len(l)
    out["R_given_fell"] = float(np.mean(f)) if f else None
    out["R_given_lost"] = float(np.mean(l)) if l else None
    out["fall_premium"] = (out["R_given_fell"] - out["R_given_lost"]
                           if f and l else None)
    return out


def pooled_dodge(evals):
    """The same pair over POOLED episodes from several evaluations.

    This project's own measurement rule -- aggregate before comparing rates --
    applies to correlations too: r over 10 mean-action episodes is noise, and
    a per-epoch series of it says nothing a rolling pool does not say better.
    `evals` is a list of per-episode lists, oldest first.
    """
    eps = [e for ev in evals for e in ev]
    d = dodge_stats(eps)
    d["pooled_n"] = len(eps)
    return d
