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
    """Which model entries each of the 20 genome parameters owns, resolved once
    against the compiled base model. All index tensors are `[n_agents, k]`, all
    parameter tensors hold an index into the 20-vector.

    This is the `set_design_params` dispatch table, as data. Building it from
    the compiled model (rather than hard-coding ids) means a scene change is
    caught by the name lookups instead of silently scaling the wrong capsule.
    """
    n_agents: int
    design_dim: int
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
    # scaled child-body offsets
    body_id: torch.Tensor          # [A, B]
    body_param: torch.Tensor       # [A, B]
    base_body_pos: torch.Tensor    # [A, B, 3]
    # scaled gears
    act_id: torch.Tensor           # [A, U]
    gear_param: torch.Tensor       # [A, U]
    base_gear: torch.Tensor        # [A, U, 6]
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


def build_design_spec(model, meta, device="cpu", dtype=torch.float64):
    """Resolve `DEV_GENOME_TABLE` against a compiled dev scene."""
    A = meta.n_agents
    name2body = lambda s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s)
    name2geom = lambda s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, s)
    name2act = lambda s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, s)

    geom_id, len_p, rad_p, base_r, base_h, base_gp, geom_body, axial = (
        [], [], [], [], [], [], [], [])
    bvh = []
    body_id, body_p, base_bp = [], [], []
    act_id, gear_p, base_gear = [], [], []
    for a in range(A):
        p = f"agent{a}/"
        g_row, l_row, r_row, br_row, bh_row, bp_row, gb_row, ax_row = (
            [], [], [], [], [], [], [], [])
        bv_row = []
        b_row, bpar_row, bpos_row = [], [], []
        u_row, gp_row, bg_row = [], [], []
        for local, l_param, r_param, pos_param, gear_param in DEV_GENOME_TABLE:
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

    T = lambda x, d=dtype: torch.as_tensor(np.asarray(x), device=device, dtype=d)
    L = lambda x: T(x, torch.long)
    base_fields = ("geom_size", "geom_pos", "geom_quat", "geom_rbound",
                   "geom_aabb", "body_pos", "body_quat", "body_mass",
                   "body_inertia", "body_ipos", "body_iquat",
                   "body_subtreemass", "actuator_gear", "qpos0", "bvh_aabb")
    base = {f: T(np.asarray(getattr(model, f), dtype=np.float64).reshape(
        _base_shape(model, f))) for f in base_fields}

    spec = DesignSpec(
        n_agents=A, design_dim=DESIGN_DIM,
        geom_id=L(geom_id), len_param=L(len_p), rad_param=L(rad_p),
        base_r=T(base_r), base_h=T(base_h), base_geom_pos=T(base_gp),
        body_of_geom=L(geom_body), axial_slot=L(axial), bvh_slot=L(bvh),
        body_id=L(body_id), body_param=L(body_p), base_body_pos=T(base_bp),
        act_id=L(act_id), gear_param=L(gear_p), base_gear=T(base_gear),
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
    in the box `[-1,1]^20` reorders the principal moments. Checked here at the
    corner that squashes a capsule hardest (shortest allowed, fattest allowed),
    so the writer's assumption is proved over the whole box, once, at build."""
    lo, hi = 1.0 - GEOM_SCALE, 1.0 + GEOM_SCALE
    r = spec.base_r * torch.where(spec.rad_param >= 0, hi, 1.0)
    h = spec.base_h * lo
    _, i_trans, i_axial = capsule_mass_inertia(r, h)
    assert bool((i_axial < i_trans).all()), (
        "a design in [-1,1]^20 reorders a capsule's principal moments; the "
        "writer's fixed body_iquat/axial-slot assumption no longer holds")


def _gather(param_idx, factors):
    """`factors` is `[N, A, 20]`; `param_idx` is `[A, k]` with -1 meaning "no
    parameter". Returns `[N, A, k]`, 1.0 where the index is -1."""
    idx = param_idx.clamp(min=0).unsqueeze(0).expand(factors.shape[0], -1, -1)
    out = torch.gather(factors, 2, idx)
    return torch.where((param_idx >= 0).unsqueeze(0), out,
                       torch.ones_like(out))


def design_fields(spec, scale):
    """The model fields implied by `scale` `[N, n_agents, 20]`.

    Returns full-width tensors (`[N, ngeom, ...]`, `[N, nbody, ...]`,
    `[N, nu, 6]`) already filled with the base values everywhere the genome does
    not reach, so the result can be written straight into a batched Model.
    """
    scale = scale.to(spec.base_r.dtype)
    N = scale.shape[0]
    assert scale.shape[1:] == (spec.n_agents, spec.design_dim), scale.shape
    a = 1.0 + GEOM_SCALE * scale
    b = 1.0 + GEAR_SCALE * scale

    f_len = _gather(spec.len_param, a)                 # [N, A, K]
    f_rad = _gather(spec.rad_param, a)
    r = spec.base_r.unsqueeze(0) * f_rad
    h = spec.base_h.unsqueeze(0) * f_len
    mass, i_trans, i_axial = capsule_mass_inertia(r, h)

    exp = lambda t: t.unsqueeze(0).expand(N, *t.shape).clone()
    out = {k: exp(v) for k, v in spec.base.items()}

    gi = spec.geom_id.reshape(-1)                       # [A*K]
    flat = lambda t: t.reshape(N, -1)
    out["geom_size"][:, gi, 0] = flat(r)
    out["geom_size"][:, gi, 1] = flat(h)
    out["geom_pos"][:, gi] = (spec.base_geom_pos.unsqueeze(0)
                              * f_len.unsqueeze(-1)).reshape(N, -1, 3)
    out["geom_rbound"][:, gi] = flat(r + h)
    # The tight bound, NOT padded by the contact margin. mujoco 2.3.5 (their
    # venv) pads `geom_aabb` by `geom_margin` and 3.11 (ours, and the compiler
    # mujoco_warp's model comes from) does not -- a version artifact the gate
    # separates out by compiling their emitted XML with OUR compiler.
    out["geom_aabb"][:, gi, 0] = 0.0
    out["geom_aabb"][:, gi, 1, 0] = flat(r)
    out["geom_aabb"][:, gi, 1, 1] = flat(r)
    out["geom_aabb"][:, gi, 1, 2] = flat(h + r)

    bg = spec.body_of_geom.reshape(-1)
    out["body_mass"][:, bg] = flat(mass)
    out["body_ipos"][:, bg] = out["geom_pos"][:, gi]
    # I_transverse in both non-axial slots, I_axial in the slot MuJoCo's
    # eigenvalue sort put it in for the base model (proved stable at build).
    inert = i_trans.unsqueeze(-1).expand(*i_trans.shape, 3).clone()
    slot = spec.axial_slot.unsqueeze(0).expand(N, -1, -1).unsqueeze(-1)
    inert.scatter_(3, slot, i_axial.unsqueeze(-1))
    out["body_inertia"][:, bg] = inert.reshape(N, -1, 3)
    out["body_subtreemass"] = (out["body_mass"]
                               @ spec.descendants.transpose(0, 1))

    bi = spec.body_id.reshape(-1)
    out["body_pos"][:, bi] = (spec.base_body_pos.unsqueeze(0)
                              * _gather(spec.body_param, a).unsqueeze(-1)
                              ).reshape(N, -1, 3)

    # CPU MuJoCo only (see CPU_EXTRA_FIELDS): keep the compile-time body BVH in
    # step with the geoms it bounds, or its broadphase misses contacts on a
    # grown leg. mujoco_warp does not read this field -- it builds its own
    # broadphase from `geom_aabb`/`geom_rbound`, which are batched and written.
    bv = spec.bvh_slot.reshape(-1)
    out["bvh_aabb"][:, bv, 0:3] = 0.0
    out["bvh_aabb"][:, bv, 3] = flat(r)
    out["bvh_aabb"][:, bv, 4] = flat(r)
    out["bvh_aabb"][:, bv, 5] = flat(h + r)

    ai = spec.act_id.reshape(-1)
    out["actuator_gear"][:, ai] = (spec.base_gear.unsqueeze(0)
                                   * _gather(spec.gear_param, b).unsqueeze(-1)
                                   ).reshape(N, -1, 6)
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
        """`idx` `[M]` world indices, `scale` `[M, n_agents, 20]` in [-1, 1]."""
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
