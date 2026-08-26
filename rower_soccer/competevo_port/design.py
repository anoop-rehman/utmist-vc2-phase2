"""The dev genome, applied as per-world model fields instead of a recompile.

CompetEvo's `dev_*` agents have a FIXED topology whose geometry is a flat scale
vector `s in [-1,1]^20`, emitted by the policy as the step-0 action of every
episode. Their env applies it by mutating an lxml tree
(`DevAnt.set_design_params`, dev_ant.py:53-269), re-merging the two agents'
XML strings, and calling `MjModel.from_xml_string` -- twice per episode per
worker. That cannot exist on a GPU.

This module replaces it. The topology never changes, so ONE compiled model
serves every world and a design is a write of the model fields the compiler
would have produced:

    s -> a = 1 + 0.3 s          (geometry)      b = 1 + 0.15 s   (gears)

    geom_size      capsule (radius, half-length)
    geom_pos       capsule midpoint  = base midpoint * length factor
    geom_rbound    radius + half-length          (broadphase)
    geom_aabb      (0, (r, r, h + r))            (broadphase)
    body_pos       child body offset, scaled by the PARENT capsule's length
                   factor -- this is what keeps the links attached
    actuator_gear  150 * b
    body_mass      \
    body_inertia    >  because `inertiafromgeom="true"` at density 5.0
    body_ipos      /
    body_subtreemass                              (subtree_com, i.e. the reward)

The mass block is the part that is easy to get wrong and impossible to notice:
a writer that scales only the geoms produces an ant with the right shape and the
wrong dynamics, and nothing in the observation says so. See port map risk 2.

Every body in this robot carries exactly ONE geom, so a body's mass properties
are its geom's: `body_ipos` is the capsule midpoint, `body_iquat` is the geom's
own frame (unchanged -- the genome only scales, never rotates), and
`body_inertia` is the capsule's principal inertia in that frame.

MuJoCo's capsule = a cylinder (radius r, half-length h) plus two hemispheres:

    m_cyl = rho pi r^2 2h          m_sph = rho 4/3 pi r^3
    I_axial      = m_cyl r^2/2 + m_sph 2/5 r^2
    I_transverse = m_cyl (r^2/4 + h^2/3) + m_sph (2/5 r^2 + h^2 + 3/4 h r)

(the hemisphere term is the parallel-axis shift of 2/5 m r^2 from the sphere
centre out to the capsule centre). Verified exact -- 2.5e-16 relative on mass,
3.8e-16 on inertia, against MuJoCo's own compiler over random (r, h).

The other design-dependent fields -- `body_invweight0`, `dof_invweight0`,
`actuator_acc0` -- have no closed form: MuJoCo's `mj_setConst` builds them from
the INVERSE mass matrix at qpos0. They are not decoration; `*_invweight0` sets
constraint impedance, so leaving them at the base ant's values makes contact
softness wrong for every non-base design. Measured: up to 46% off, worth 7 cm of
trajectory divergence in 0.6 s (four orders of magnitude more than the whole
PGS -> Newton solver swap costs). So they are computed EXACTLY, by calling
`mj_setConst` on a host scratch model for the handful of worlds that reset on a
given step -- 0.093 ms each, machine-epsilon agreement with a freshly compiled
model, and no recompile anywhere (`HostConstants` below). Pass
`exact_constants=False` to skip it and get the stale-constant behaviour back,
which is what the gate compares against.

2h -- MIXED CREATURES
---------------------
The paragraphs above are written for the ant, which is the only creature 2f/2g
ever ran. 2h puts an ant, a bug and a spider in the same scene, and three
things that were module constants become PER AGENT:

  * the genome table (5 params per leg, so 20 / 30 / 40 params over 4 / 6 / 8
    legs) -- read from each agent's `CreatureSpec.genome_table()`;
  * `GEOM_SCALE` / `GEAR_SCALE` -- 0.3 / 0.15 for the ant, 0.5 / 0.25 for the
    bug and the spider, so `a = 1 + geom_scale[agent] * s` is a broadcast
    against a `[1, A, 1]` tensor rather than a scalar multiply;
  * the number of scaled capsules, child offsets and gears per agent, which is
    what makes the index tables RAGGED.

Ragged is the dangerous part, and it is not the same hazard as `dev_env`'s
padded observation. There a padded column is masked to zero and reads nothing;
here every row of the table names a `geom_id` the writer SCATTERS into, and
`_gather` returns 1.0 at a `-1` parameter index -- so a padded row does not
write nothing, it writes the BASE capsule to whatever geom it names. Padding
the ant's table out to the spider's width and letting it run would silently
restore base geometry on four of the spider's capsules (or, with a padded
`geom_id` of 0, resize the FLOOR). So the padded rows are never written at all:
`DesignSpec.cap_keep` / `body_keep` / `act_keep` hold the flat indices of the
real rows and every scatter goes through them. On a homogeneous scene nothing
is padded, the keeps are `None`, and the writer is expression-for-expression
what it was.
"""

import copy
import dataclasses
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from rower_soccer.competevo_port.scene import (DESIGN_DIM, DEV_GENOME_TABLE,
                                               GEAR_SCALE, GEOM_SCALE)

DENSITY = 5.0            # `dev_ant_body.xml`'s geom default
_TWO_THIRDS_PI = 4.0 / 3.0 * np.pi


def capsule_mass_inertia(r, h, density=DENSITY):
    """(mass, I_transverse, I_axial) for a capsule of radius `r`, half-length
    `h`, about its own centre. Broadcasting over array inputs."""
    m_cyl = density * np.pi * r * r * 2.0 * h
    m_sph = density * _TWO_THIRDS_PI * r ** 3
    i_axial = m_cyl * r * r / 2.0 + m_sph * 0.4 * r * r
    i_trans = (m_cyl * (r * r / 4.0 + h * h / 3.0)
               + m_sph * (0.4 * r * r + h * h + 0.75 * h * r))
    return m_cyl + m_sph, i_trans, i_axial


@dataclass
class DesignSpec:
    """Which model entries each genome parameter owns, resolved once against
    the compiled base model. All index tensors are `[n_agents, k]`, all
    parameter tensors hold an index into that agent's OWN genome.

    This is the `set_design_params` dispatch table, as data. Building it from
    the compiled model (rather than hard-coding ids) means a scene change is
    caught by the name lookups instead of silently scaling the wrong capsule.

    On a mixed scene the per-agent tables are ragged and the short ones are
    padded to the widest. `cap_keep` / `body_keep` / `act_keep` name the REAL
    rows of the flattened `[A, k]` axis and every scatter in `design_fields`
    goes through them -- see the module docstring for why padding-and-writing
    is a silent corruption rather than a harmless no-op. They are `None` when
    nothing is padded, which is every homogeneous scene.
    """
    n_agents: int
    design_dim: int                # the WIDEST agent's genome width
    design_dims: tuple             # per agent
    geom_scale: torch.Tensor       # [1, A, 1]   a = 1 + geom_scale * s
    gear_scale: torch.Tensor       # [1, A, 1]   b = 1 + gear_scale * s
    # scaled capsules: one row per (agent, leg-link)
    geom_id: torch.Tensor          # [A, K]
    len_param: torch.Tensor        # [A, K]   index into the genome
    rad_param: torch.Tensor        # [A, K]   -1 => radius not scaled
    base_r: torch.Tensor           # [A, K]
    base_h: torch.Tensor           # [A, K]
    base_geom_pos: torch.Tensor    # [A, K, 3]
    body_of_geom: torch.Tensor     # [A, K]
    axial_slot: torch.Tensor       # [A, K]   which body_inertia slot is I_axial
    bvh_slot: torch.Tensor         # [A, K]   CPU-only: this body's BVH leaf
    cap_real: torch.Tensor         # [A, K] bool, False on a padded row
    cap_keep: torch.Tensor         # [n_real] into the flat [A*K], or None
    # scaled child-body offsets
    body_id: torch.Tensor          # [A, B]
    body_param: torch.Tensor       # [A, B]
    base_body_pos: torch.Tensor    # [A, B, 3]
    body_keep: torch.Tensor        # [n_real] into the flat [A*B], or None
    # scaled gears
    act_id: torch.Tensor           # [A, U]
    gear_param: torch.Tensor       # [A, U]
    base_gear: torch.Tensor        # [A, U, 6]
    act_keep: torch.Tensor         # [n_real] into the flat [A*U], or None
    # base copies of every field the writer touches, plus the subtree matrix
    base: dict
    descendants: torch.Tensor      # [nbody, nbody] float, D[b, c] = c in tree(b)
    nbody: int
    ngeom: int
    nu: int


def _descendant_matrix(model):
    """`D[b, c] = 1` iff body `c` is in body `b`'s subtree, so
    `body_subtreemass = D @ body_mass` -- MuJoCo's definition, as one matmul."""
    n = model.nbody
    d = np.zeros((n, n))
    for c in range(n):
        b = c
        while b >= 0:
            d[b, c] = 1.0
            b = -1 if b == 0 else int(model.body_parentid[b])
    return d


def _agent_genomes(meta):
    """Per agent: `(genome_table, design_dim, geom_scale, gear_scale)`.

    `meta.specs` is the per-agent `CreatureSpec` list a `TeamSceneMeta` carries
    (2h). Every other scene -- the 1v1 dev scene, the 2f/2g team scene built
    before `creatures.py` existed -- is the ant everywhere, and the ant's spec
    returns exactly the module constants below, so the fallback is a fallback
    and not a second definition.
    """
    specs = getattr(meta, "specs", None)
    if not specs:
        return [(DEV_GENOME_TABLE, DESIGN_DIM, GEOM_SCALE, GEAR_SCALE)
                for _ in range(meta.n_agents)]
    return [(sp.genome_table(), sp.design_dim, sp.geom_scale, sp.gear_scale)
            for sp in specs]


def _pad_rows(rows, fill):
    """Ragged per-agent lists -> a rectangular `[A, W]` list, plus the flat
    indices of the REAL entries in that rectangle (or `None` if nothing was
    padded). See `DesignSpec`: the padded rows are never written."""
    W = max(len(r) for r in rows)
    keep, out = [], []
    for a, r in enumerate(rows):
        out.append(list(r) + [fill] * (W - len(r)))
        keep += [a * W + k for k in range(len(r))]
    return out, (None if all(len(r) == W for r in rows) else keep)


def build_design_spec(model, meta, device="cpu", dtype=torch.float64):
    """Resolve each agent's genome table against a compiled dev scene.

    2h: the table, the genome width and the two scale constants come from
    `meta.specs[a]`, so an ant slot and a spider slot in the same scene get
    their own 20/40 params and their own 0.3/0.5 geometry scale.
    """
    A = meta.n_agents
    name2body = lambda s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s)
    name2geom = lambda s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, s)
    name2act = lambda s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, s)

    genomes = _agent_genomes(meta)
    geom_id, len_p, rad_p, base_r, base_h, base_gp, geom_body, axial = (
        [], [], [], [], [], [], [], [])
    bvh = []
    body_id, body_p, base_bp = [], [], []
    act_id, gear_p, base_gear = [], [], []
    for a in range(A):
        p = f"agent{a}/"
        table, a_dim = genomes[a][0], genomes[a][1]
        g_row, l_row, r_row, br_row, bh_row, bp_row, gb_row, ax_row = (
            [], [], [], [], [], [], [], [])
        bv_row = []
        b_row, bpar_row, bpos_row = [], [], []
        u_row, gp_row, bg_row = [], [], []
        for local, l_param, r_param, pos_param, gear_param in table:
            assert max(x for x in (l_param, r_param, pos_param, gear_param)
                       if x is not None) < a_dim, (
                f"agent {a}: genome table indexes past its {a_dim} parameters")
            g = name2geom(p + "geom_" + local)
            b = name2body(p + local)
            assert g >= 0 and b >= 0, f"missing {local} in the compiled scene"
            assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
            g_row.append(g)
            gb_row.append(b)
            l_row.append(l_param)
            r_row.append(-1 if r_param is None else r_param)
            br_row.append(float(model.geom_size[g, 0]))
            bh_row.append(float(model.geom_size[g, 1]))
            bp_row.append(np.asarray(model.geom_pos[g], dtype=np.float64))
            # Which body_inertia slot holds the AXIAL principal moment. A
            # capsule's transverse moments are equal, so the axial one is the
            # single entry that differs from the median; which slot MuJoCo's
            # eigen-sort puts it in is a property of the compiled model, not of
            # our formula, so read it off rather than assume.
            inert = np.asarray(model.body_inertia[b], dtype=np.float64)
            ax_row.append(int(np.argmax(np.abs(inert - np.median(inert)))))
            # CPU MuJoCo keeps a compile-time body BVH. Every leg body here has
            # exactly one geom, so its single BVH leaf IS that geom's AABB.
            assert int(model.body_bvhnum[b]) == 1, "multi-geom body BVH"
            bv_row.append(int(model.body_bvhadr[b]))
            if pos_param is not None:
                b_row.append(b)
                bpar_row.append(pos_param)
                bpos_row.append(np.asarray(model.body_pos[b], dtype=np.float64))
            if gear_param is not None:
                u = name2act(p + local + "_joint")
                assert u >= 0, f"missing motor for {local}"
                u_row.append(u)
                gp_row.append(gear_param)
                bg_row.append(np.asarray(model.actuator_gear[u],
                                         dtype=np.float64))
        for dst, src in ((geom_id, g_row), (len_p, l_row), (rad_p, r_row),
                         (base_r, br_row), (base_h, bh_row), (base_gp, bp_row),
                         (geom_body, gb_row), (axial, ax_row), (bvh, bv_row),
                         (body_id, b_row), (body_p, bpar_row),
                         (base_bp, bpos_row), (act_id, u_row),
                         (gear_p, gp_row), (base_gear, bg_row)):
            dst.append(src)

    # Ragged -> rectangular. A padded PARAMETER index is -1 (`_gather` -> 1.0)
    # and a padded model index is 0, but neither is what makes padding safe --
    # the `*_keep` tensors are, because they are what every scatter is indexed
    # by. See the module docstring.
    zero3, zero6 = [0.0, 0.0, 0.0], [0.0] * 6
    n_cap = [len(r) for r in geom_id]
    geom_id, cap_keep = _pad_rows(geom_id, 0)
    len_p, _ = _pad_rows(len_p, -1)
    rad_p, _ = _pad_rows(rad_p, -1)
    base_r, _ = _pad_rows(base_r, 0.0)
    base_h, _ = _pad_rows(base_h, 0.0)
    base_gp, _ = _pad_rows(base_gp, zero3)
    geom_body, _ = _pad_rows(geom_body, 0)
    axial, _ = _pad_rows(axial, 0)
    bvh, _ = _pad_rows(bvh, 0)
    body_id, body_keep = _pad_rows(body_id, 0)
    body_p, _ = _pad_rows(body_p, -1)
    base_bp, _ = _pad_rows(base_bp, zero3)
    act_id, act_keep = _pad_rows(act_id, 0)
    gear_p, _ = _pad_rows(gear_p, -1)
    base_gear, _ = _pad_rows(base_gear, zero6)
    K = len(geom_id[0])
    cap_real = np.zeros((A, K), dtype=bool)
    for a, k in enumerate(n_cap):
        cap_real[a, :k] = True

    T = lambda x, d=dtype: torch.as_tensor(np.asarray(x), device=device, dtype=d)
    L = lambda x: T(x, torch.long)
    base_fields = ("geom_size", "geom_pos", "geom_quat", "geom_rbound",
                   "geom_aabb", "body_pos", "body_quat", "body_mass",
                   "body_inertia", "body_ipos", "body_iquat",
                   "body_subtreemass", "actuator_gear", "qpos0", "bvh_aabb")
    base = {f: T(np.asarray(getattr(model, f), dtype=np.float64).reshape(
        _base_shape(model, f))) for f in base_fields}

    dims = tuple(int(g[1]) for g in genomes)
    scale_col = lambda i: T(np.asarray([g[i] for g in genomes],
                                       dtype=np.float64).reshape(1, A, 1))
    spec = DesignSpec(
        n_agents=A, design_dim=max(dims), design_dims=dims,
        geom_scale=scale_col(2), gear_scale=scale_col(3),
        geom_id=L(geom_id), len_param=L(len_p), rad_param=L(rad_p),
        base_r=T(base_r), base_h=T(base_h), base_geom_pos=T(base_gp),
        body_of_geom=L(geom_body), axial_slot=L(axial), bvh_slot=L(bvh),
        cap_real=torch.as_tensor(cap_real, device=device),
        cap_keep=None if cap_keep is None else L(cap_keep),
        body_id=L(body_id), body_param=L(body_p), base_body_pos=T(base_bp),
        body_keep=None if body_keep is None else L(body_keep),
        act_id=L(act_id), gear_param=L(gear_p), base_gear=T(base_gear),
        act_keep=None if act_keep is None else L(act_keep),
        base=base, descendants=T(_descendant_matrix(model)),
        nbody=model.nbody, ngeom=model.ngeom, nu=model.nu)
    _assert_inertia_ordering_is_stable(spec)
    return spec


def _spec_to(spec, device, dtype):
    """A copy of a `DesignSpec` on another device/dtype. Integer index tensors
    keep their dtype; float ones are cast."""
    kw = {}
    for f in dataclasses.fields(spec):
        v = getattr(spec, f.name)
        if torch.is_tensor(v):
            v = v.to(device=device,
                     dtype=v.dtype if not v.is_floating_point() else dtype)
        elif isinstance(v, dict):
            v = {k: t.to(device=device, dtype=dtype) for k, t in v.items()}
        kw[f.name] = v
    return DesignSpec(**kw)


def _base_shape(model, field):
    a = np.asarray(getattr(model, field))
    return (model.ngeom, 2, 3) if field == "geom_aabb" else a.shape


def _assert_inertia_ordering_is_stable(spec):
    """`body_inertia` slots follow MuJoCo's eigenvalue sort, and we keep each
    body's `body_iquat` from the base model -- which is only valid if no design
    in the box `[-1,1]^d` reorders the principal moments. Checked here at the
    corner that squashes a capsule hardest (shortest allowed, fattest allowed),
    so the writer's assumption is proved over the whole box, once, at build.

    2h: the corner is PER AGENT, because `geom_scale` is (0.3 for the ant, 0.5
    for the bug and the spider) -- a shared 0.3 would check the bug and the
    spider at a box smaller than the one they actually train in. Padded rows
    are excluded: their `base_r`/`base_h` are zeros and a degenerate capsule
    would fail an assertion about a capsule that does not exist.
    """
    gs = spec.geom_scale.reshape(-1, 1)                  # [A, 1]
    lo, hi = 1.0 - gs, 1.0 + gs
    r = spec.base_r * torch.where(spec.rad_param >= 0, hi,
                                  torch.ones_like(hi))
    h = spec.base_h * lo
    _, i_trans, i_axial = capsule_mass_inertia(r, h)
    ok = (i_axial < i_trans) | (~spec.cap_real)
    assert bool(ok.all()), (
        "a design in [-1,1]^d reorders a capsule's principal moments; the "
        "writer's fixed body_iquat/axial-slot assumption no longer holds")


def _gather(param_idx, factors):
    """`factors` is `[N, A, design_dim]`; `param_idx` is `[A, k]` with -1
    meaning "no parameter". Returns `[N, A, k]`, 1.0 where the index is -1.

    THE 1.0 IS THE TRAP. It means "leave this quantity at its base value",
    which is right for a real row whose radius the genome does not scale and
    catastrophic for a PADDED row, whose 1.0 would write a base-sized capsule
    over a correctly scaled one. Nothing here can tell the two apart -- the
    caller must not write padded rows at all, which is what `_rows`/`_real`
    below are for.
    """
    idx = param_idx.clamp(min=0).unsqueeze(0).expand(factors.shape[0], -1, -1)
    out = torch.gather(factors, 2, idx)
    return torch.where((param_idx >= 0).unsqueeze(0), out,
                       torch.ones_like(out))


def _rows(idx, keep):
    """A padded `[A, k]` INDEX tensor, flattened to the real rows only."""
    f = idx.reshape(-1)
    return f if keep is None else f.index_select(0, keep)


def _real(t, keep):
    """A padded `[N, A, k]` or `[N, A, k, C]` VALUE tensor, flattened over
    (A, k) to the same real rows `_rows` names, in the same order.

    `keep is None` -- every homogeneous scene -- returns the plain reshape the
    writer did before 2h, so that path allocates and computes exactly what it
    used to.
    """
    t = t.reshape(t.shape[0], -1, *t.shape[3:])
    return t if keep is None else t.index_select(1, keep)


def design_fields(spec, scale):
    """The model fields implied by `scale` `[N, n_agents, design_dim]`.

    Returns full-width tensors (`[N, ngeom, ...]`, `[N, nbody, ...]`,
    `[N, nu, 6]`) already filled with the base values everywhere the genome does
    not reach, so the result can be written straight into a batched Model.

    `design_dim` is the WIDEST agent's; a narrower agent's trailing genome
    columns are never indexed by its table, so they are ignored here (and
    `dev_env` keeps them masked to zero so nothing else reads them either).
    """
    scale = scale.to(spec.base_r.dtype)
    N = scale.shape[0]
    assert scale.shape[1:] == (spec.n_agents, spec.design_dim), scale.shape
    # Per agent, not per scene: the ant scales geometry by 0.3 and gears by
    # 0.15, the bug and the spider by 0.5 and 0.25.
    a = 1.0 + spec.geom_scale * scale
    b = 1.0 + spec.gear_scale * scale

    f_len = _gather(spec.len_param, a)                 # [N, A, K]
    f_rad = _gather(spec.rad_param, a)
    r = spec.base_r.unsqueeze(0) * f_rad
    h = spec.base_h.unsqueeze(0) * f_len
    mass, i_trans, i_axial = capsule_mass_inertia(r, h)

    exp = lambda t: t.unsqueeze(0).expand(N, *t.shape).clone()
    out = {k: exp(v) for k, v in spec.base.items()}

    # `keep` drops the padded rows of a mixed scene. Every destination index
    # and every source column below is taken through it, so a padded row is
    # not written anywhere rather than written harmlessly.
    keep = spec.cap_keep
    gi = _rows(spec.geom_id, keep)                      # [n_real]
    flat = lambda t: _real(t, keep)
    out["geom_size"][:, gi, 0] = flat(r)
    out["geom_size"][:, gi, 1] = flat(h)
    out["geom_pos"][:, gi] = _real(spec.base_geom_pos.unsqueeze(0)
                                   * f_len.unsqueeze(-1), keep)
    out["geom_rbound"][:, gi] = flat(r + h)
    # The tight bound, NOT padded by the contact margin. mujoco 2.3.5 (their
    # venv) pads `geom_aabb` by `geom_margin` and 3.11 (ours, and the compiler
    # mujoco_warp's model comes from) does not -- a version artifact the gate
    # separates out by compiling their emitted XML with OUR compiler.
    out["geom_aabb"][:, gi, 0] = 0.0
    out["geom_aabb"][:, gi, 1, 0] = flat(r)
    out["geom_aabb"][:, gi, 1, 1] = flat(r)
    out["geom_aabb"][:, gi, 1, 2] = flat(h + r)

    bg = _rows(spec.body_of_geom, keep)
    out["body_mass"][:, bg] = flat(mass)
    out["body_ipos"][:, bg] = out["geom_pos"][:, gi]
    # I_transverse in both non-axial slots, I_axial in the slot MuJoCo's
    # eigenvalue sort put it in for the base model (proved stable at build).
    inert = i_trans.unsqueeze(-1).expand(*i_trans.shape, 3).clone()
    slot = spec.axial_slot.unsqueeze(0).expand(N, -1, -1).unsqueeze(-1)
    inert.scatter_(3, slot, i_axial.unsqueeze(-1))
    out["body_inertia"][:, bg] = _real(inert, keep)
    out["body_subtreemass"] = (out["body_mass"]
                               @ spec.descendants.transpose(0, 1))

    bi = _rows(spec.body_id, spec.body_keep)
    out["body_pos"][:, bi] = _real(spec.base_body_pos.unsqueeze(0)
                                   * _gather(spec.body_param, a).unsqueeze(-1),
                                   spec.body_keep)

    # CPU MuJoCo only (see CPU_EXTRA_FIELDS): keep the compile-time body BVH in
    # step with the geoms it bounds, or its broadphase misses contacts on a
    # grown leg. mujoco_warp does not read this field -- it builds its own
    # broadphase from `geom_aabb`/`geom_rbound`, which are batched and written.
    bv = _rows(spec.bvh_slot, keep)
    out["bvh_aabb"][:, bv, 0:3] = 0.0
    out["bvh_aabb"][:, bv, 3] = flat(r)
    out["bvh_aabb"][:, bv, 4] = flat(r)
    out["bvh_aabb"][:, bv, 5] = flat(h + r)

    ai = _rows(spec.act_id, spec.act_keep)
    out["actuator_gear"][:, ai] = _real(spec.base_gear.unsqueeze(0)
                                        * _gather(spec.gear_param, b
                                                  ).unsqueeze(-1),
                                        spec.act_keep)
    return out


# Fields the writer owns. `geom_quat`/`body_quat`/`body_iquat`/`qpos0` are in the
# batched set but never change (the genome only scales along fixed directions);
# they are batched anyway so the Model has one consistent leading dimension and
# a future genome that rotates something has somewhere to write.
WRITTEN_FIELDS = ("geom_size", "geom_pos", "geom_rbound", "geom_aabb",
                  "body_pos", "body_mass", "body_inertia", "body_ipos",
                  "body_subtreemass", "actuator_gear")
# The three `mj_setConst` outputs, filled by `HostConstants` rather than by a
# formula. `*_invweight0` is the load-bearing one (constraint impedance);
# `actuator_acc0` comes along for free in the same call.
CONST_FIELDS = ("body_invweight0", "dof_invweight0", "actuator_acc0")
BATCHED_FIELDS = (WRITTEN_FIELDS + CONST_FIELDS
                  + ("geom_quat", "body_quat", "body_iquat"))
# Written only when the backend has them. `bvh_aabb` is MuJoCo's compile-time
# body bounding-volume hierarchy: CPU MuJoCo's broadphase descends it, so a
# stale one silently drops contacts for a design with longer legs. mujoco_warp
# has no such field (its `bvh_*` are mesh/hfield/flex only and it broadphases
# from `geom_aabb`/`geom_rbound`), so this is the CPU mirror's business alone.
CPU_EXTRA_FIELDS = ("bvh_aabb",)


class HostConstants:
    """`mj_setConst` on a host scratch model, for the worlds that just reset.

    MuJoCo derives `body_invweight0`/`dof_invweight0`/`actuator_acc0` from
    `inv(M)` at qpos0, which is not a function of the design in any form we can
    write down. But `mj_setConst` is a NUMERIC routine over an existing model --
    not a compile -- so pushing the design fields into one reusable `MjModel`
    and calling it gives the exact values a fresh compile would, at 0.23 ms per
    world (measured; machine-epsilon agreement, `tests/test_design_parity.py`).

    It is a host round-trip, which is why it is a separate object: if a future
    scene makes it expensive, this is the one thing to replace with a batched
    GPU `inv(M)`.

    The round-trip is ONE fused D2H of the five fields `mj_setConst` reads --
    236 doubles per world, one sync -- not ten separate device syncs and not a
    second host-side evaluation of `design_fields`. See `from_fields`.
    """

    # `mj_setConst` only needs inv(M) at qpos0, so only the fields that enter the
    # mass matrix and the body frames matter. Pushing the geometry and broadphase
    # fields as well would be free correctness and expensive bandwidth.
    NEEDED = ("body_mass", "body_inertia", "body_ipos", "body_pos",
              "actuator_gear")

    def __init__(self, model, spec):
        self.model = copy.deepcopy(model)
        self.data = mujoco.MjData(self.model)
        # A CPU twin of the design spec, for the standalone `compute(scale)`
        # entry point (the gate). The writer does NOT go through it -- it hands
        # over the fields it already computed, see `from_fields`.
        self.spec = _spec_to(spec, "cpu", torch.float64)
        self.shapes = {f: np.asarray(getattr(self.model, f)).shape
                       for f in CONST_FIELDS}
        # Flat [offset, size, shape] layout of the NEEDED block, so the D2H is
        # one contiguous transfer instead of one per field.
        self._slices, off = [], 0
        for f in self.NEEDED:
            shape = np.asarray(getattr(self.model, f)).shape
            n = int(np.prod(shape))
            self._slices.append((f, off, off + n, shape))
            off += n
        self._flat_width = off

    def from_fields(self, fields):
        """`fields`: the dict `design_fields` already produced, on any device.

        Only the five `NEEDED` entries are read, and they come across in ONE
        transfer. This is the production path: `design_fields` is launch-bound
        (tens of small kernels), so evaluating it a second time host-side to
        avoid the sync costs more than the sync does.
        """
        need = [fields[f] for f in self.NEEDED]
        count = need[0].shape[0]
        flat = torch.cat([t.reshape(count, -1).to(torch.float64)
                          for t in need], dim=1)
        assert flat.shape[1] == self._flat_width, (flat.shape,
                                                   self._flat_width)
        host = flat.detach().cpu().numpy()          # the one and only sync
        out = {f: np.empty((count,) + self.shapes[f]) for f in CONST_FIELDS}
        for w in range(count):
            row = host[w]
            for f, lo, hi, shape in self._slices:
                getattr(self.model, f)[:] = row[lo:hi].reshape(shape)
            mujoco.mj_setConst(self.model, self.data)
            for f in CONST_FIELDS:
                out[f][w] = getattr(self.model, f)
        return out

    def compute(self, scale):
        """`scale`: `[M, n_agents, 20]`, any device. Returns
        `{name: np.ndarray [M, ...]}`. Evaluates the design host-side in
        float64 -- the reference path the gate measures."""
        return self.from_fields(
            design_fields(self.spec, scale.detach().double().cpu()))


class DesignWriter:
    """Applies designs to a batched Model in place, for a subset of worlds.

    `model_arrays` maps field name -> a `[nworld, ...]` torch tensor aliasing
    the batched Model (mujoco_warp) or a stack of per-world MjModels (CPU). The
    writer never allocates per world, so it is safe to call every step with the
    handful of worlds that just reset.
    """

    def __init__(self, spec, model_arrays, model=None, exact_constants=True):
        self.spec = spec
        self.arrays = model_arrays
        missing = [f for f in WRITTEN_FIELDS if f not in model_arrays]
        assert not missing, f"batched Model is missing {missing}"
        self.constants = None
        if exact_constants:
            assert model is not None, "exact_constants needs the MjModel"
            missing = [f for f in CONST_FIELDS if f not in model_arrays]
            assert not missing, (f"exact_constants needs {missing} batched too")
            self.constants = HostConstants(model, spec)

    def write(self, idx, scale):
        """`idx` `[M]` world indices, `scale` `[M, n_agents, design_dim]` in
        [-1, 1] (`design_dim` = the widest agent's)."""
        if idx.numel() == 0:
            return
        fields = design_fields(self.spec, scale)
        for name in WRITTEN_FIELDS + tuple(
                f for f in CPU_EXTRA_FIELDS if f in self.arrays):
            dst = self.arrays[name]
            dst[idx] = fields[name].to(dst.dtype).reshape(idx.numel(),
                                                          *dst.shape[1:])
        if self.constants is None:
            return
        # Reuse the fields computed above rather than re-deriving them from the
        # genome on the host: `design_fields` is the expensive half of a write.
        consts = self.constants.from_fields(fields)
        for name, val in consts.items():
            dst = self.arrays[name]
            dst[idx] = torch.as_tensor(val, device=dst.device,
                                       dtype=dst.dtype).reshape(
                idx.numel(), *dst.shape[1:])
