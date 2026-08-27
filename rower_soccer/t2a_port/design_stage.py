"""D3 unit 3d step 5, part 1: Transform2Act's DESIGN stages, on the CPU, with
no MuJoCo at all.

The port map's section 7C says the design stages are free to leave on the CPU
because they never step physics. This module makes that literal: it runs their
`skeleton_transform` and `attribute_transform` stages out of their own `Robot`
(`khrylib.robot.xml_robot`, imported unmodified from the reference tree) and
never imports `mujoco`, `mujoco_py` or `mujoco_warp`.

--------------------------------------------------------------------------
Why no MuJoCo -- the one thing that had to be checked, not assumed
--------------------------------------------------------------------------
Their `HopperEnv.apply_skel_action` calls `reload_sim_model` after every
skeleton edit, and `get_sim_obs` then reads `self.data.qpos`. So the naive
reading is that a design step needs a compiled model.

Measured instead (their venv, `hopper_gpu_s2` epoch 1000, 20 sampled episodes):
the design-stage `sim_obs` is the SAME CONSTANT at every design step of every
episode --

    root row  [0, 1.25, 0, 0, 0]      every other row  [0, 0, 0, 0, 0]

-- because `reload_sim_model` leaves `data.qpos` at the freshly compiled
`qpos0`, and the only non-zero entry of `qpos0` is the root's `rootz` slide
joint, which carries `ref="1.25"` in `assets/mujoco_envs/hopper.xml` and is
never touched by any design parameter. `1.25` is NOT hard-coded here: it is
read out of the exported XML's root joint `ref` attributes on every episode,
and `_assert_no_child_ref` fails loudly if their generator ever emits a `ref`
on a joint it creates -- which would make the constant wrong without making it
look wrong.

--------------------------------------------------------------------------
What is deliberately NOT reproduced
--------------------------------------------------------------------------
Their `apply_skel_action` / `set_design_params` wrap `reload_sim_model` in a
bare `except` and end the episode when the XML fails to compile. This module
does not compile, so it cannot see that failure at the same moment they do; the
pipeline compiles once per topology after the design stages instead and reports
a failure there. `gate_two_stage.py` measures how often a design fails to
compile at all (it is 0 in every census taken so far). If a task ever produces
failing designs at a material rate, this is the difference that would show up.
"""

import sys
import xml.etree.ElementTree as ET

import numpy as np

# Their tree is the authority on the robot data structure; re-deriving it would
# be a silent behaviour change (PORT_MAP section 2 makes the same argument for
# `Body.depth`). Only `xml_robot` is imported, and it depends on numpy + lxml
# alone -- verified to import cleanly in the repo venv (python 3.11).
_T2A_ROOT = "/workspace/Transform2Act"
if _T2A_ROOT not in sys.path:
    sys.path.append(_T2A_ROOT)

from khrylib.robot.xml_robot import Robot  # noqa: E402

SKEL_NOOP, SKEL_ADD, SKEL_REMOVE = 0, 1, 2


class DesignSpec:
    """The slice of their `Config` the design stages read."""

    def __init__(self, cfg_dict):
        self.robot_cfg = cfg_dict.get("robot", {})
        self.add_body_condition = cfg_dict.get("add_body_condition", {})
        self.max_body_depth = cfg_dict.get("max_body_depth", 4)
        self.min_body_depth = cfg_dict.get("min_body_depth", 1)
        self.enable_remove = cfg_dict.get("enable_remove", True)
        self.skel_transform_nsteps = cfg_dict.get("skel_transform_nsteps", 5)
        self.robot_param_scale = cfg_dict.get("robot_param_scale", 0.1)
        obs = cfg_dict.get("obs_specs", {})
        self.attr_specs = set(obs.get("attr", []))
        self.use_projected_params = obs.get("use_projected_params", True)
        self.abs_design = obs.get("abs_design", False)
        self.use_body_ind = obs.get("use_body_ind", False)
        self.fc_graph = obs.get("fc_graph", False)
        assert not self.fc_graph, "fc_graph is not ported; hopper does not set it"
        assert self.attr_specs == {"depth"}, (
            f"attr_specs {self.attr_specs} is not ported -- only 'depth' is, "
            f"because 'jrange'/'skel' would need their own exact reproduction")
        self.index_base = self.add_body_condition.get("max_nchild", 3) + 1
        self.skel_num_action = 3 if self.enable_remove else 2


def _assert_no_child_ref(xml_str, root_body_name="0"):
    """Every joint outside the ROOT body must have no `ref`.

    This is what makes `design_sim_obs` a constant. A `ref` on a generated
    joint would move `qpos0` off zero for that joint and the design-stage
    observation would silently gain a non-zero entry.
    """
    root = ET.fromstring(xml_str)
    wb = root.find("worldbody")
    rb = None
    for el in wb.iter("body"):
        if el.get("name") == root_body_name:
            rb = el
            break
    assert rb is not None, "no root body in the exported XML"
    root_refs = {}
    for j in rb.findall("joint"):
        root_refs[j.get("name")] = float(j.get("ref", 0.0))
    bad = [j.get("name") for b in rb.iter("body") if b is not rb
           for j in b.findall("joint") if j.get("ref") is not None]
    assert not bad, f"joints outside the root carry a `ref`: {bad}"
    return root_refs


def design_sim_obs(n_nodes, root_refs, joint_order=("rootx", "rootz")):
    """`get_sim_obs()` at `qpos = qpos0, qvel = 0`, i.e. at every design step.

    Their root block is `[flip(qpos[1:3]), flip(qvel[:3])]`
    = `[ang, height, ang_vel, z_vel, x_vel]` (`batched_exec_env` docstring,
    hazard 1). At `qpos0` only `rootz` is non-zero, and it lands in the
    `height` slot.
    """
    out = np.zeros((n_nodes, 5), dtype=np.float64)
    out[0, 1] = root_refs.get("rootz", 0.0)
    return out


class DesignWorld:
    """One world's design episode: `skel_transform_nsteps` skeleton edits then
    one attribute edit. Their `HopperEnv` step logic, minus the physics.

    Node ordering is `robot.bodies` order -- append-ordered by creation, NOT
    document order -- exactly as theirs is, because that ordering is what the
    observation, the edge list and `body_index` are all indexed by.
    """

    def __init__(self, spec, init_xml_bytes):
        self.spec = spec
        self.init_xml_bytes = init_xml_bytes
        self.reset()

    # ---- their reset_model, without reload_sim_model ----------------------
    def reset(self):
        self.robot = Robot(self.spec.robot_cfg, xml=self.init_xml_bytes,
                           is_xml_str=True)
        self.cur_t = 0
        self.stage = "skel_trans"
        self.design_cur_params = self.get_attr_design()
        self.cur_xml_str = self.robot.export_xml_string().decode("utf-8")
        self.root_refs = _assert_no_child_ref(self.cur_xml_str)
        self.failed = False
        return self.obs()

    # ---- their observation blocks -----------------------------------------
    def get_attr_design(self):
        return np.stack([b.get_params([], pad_zeros=True, demap_params=True)
                         for b in self.robot.bodies])

    def get_attr_fixed(self):
        out = np.zeros((len(self.robot.bodies), self.spec.max_body_depth))
        for i, b in enumerate(self.robot.bodies):
            out[i, b.depth] = 1.0
        return out

    def body_index(self):
        return np.array([int(b.name, base=self.spec.index_base)
                         for b in self.robot.bodies], dtype=np.int64)

    def edges(self):
        return self.robot.get_gnn_edges()

    def obs(self):
        """`[n_nodes, attr_fixed + 5 + attr_design]`, their concatenation."""
        n = len(self.robot.bodies)
        return np.concatenate([self.get_attr_fixed(),
                               design_sim_obs(n, self.root_refs),
                               self.design_cur_params], axis=-1)

    # ---- their allow_* predicates -----------------------------------------
    def allow_add_body(self, body):
        c = self.spec.add_body_condition
        max_nchild = c.get("max_nchild", 3)
        min_nchild = c.get("min_nchild", 0)
        return (body.depth >= self.spec.min_body_depth
                and body.depth < self.spec.max_body_depth - 1
                and len(body.child) < max_nchild
                and len(body.child) >= min_nchild)

    def allow_remove_body(self, body):
        if body.depth >= self.spec.min_body_depth + 1 and len(body.child) == 0:
            if body.depth == 1:
                return body.parent.child.index(body) > 0
            return True
        return False

    # ---- the two design steps ---------------------------------------------
    def _export(self):
        self.cur_xml_str = self.robot.export_xml_string().decode("utf-8")

    def skel_step(self, skel_action):
        """`apply_skel_action`. The body list is SNAPSHOT before the loop but
        the predicates are evaluated live, so an add earlier in the list can
        stop a later one -- theirs does the same and the order matters."""
        assert self.stage == "skel_trans"
        bodies = list(self.robot.bodies)
        for body, a in zip(bodies, np.asarray(skel_action).reshape(-1)):
            a = int(a)
            if a == SKEL_ADD and self.allow_add_body(body):
                self.robot.add_child_to_body(body)
            if a == SKEL_REMOVE and self.allow_remove_body(body):
                self.robot.remove_body(body)
        self._export()
        self.design_cur_params = self.get_attr_design()
        self.cur_t += 1
        if self.cur_t == self.spec.skel_transform_nsteps:
            self.stage = "attr_trans"
        return self.obs()

    def attr_step(self, attr_action):
        """`set_design_params`, with their `use_projected_params` reading."""
        assert self.stage == "attr_trans"
        a = np.asarray(attr_action, dtype=np.float64)
        if self.spec.abs_design:
            params = a * self.spec.robot_param_scale
        else:
            params = self.design_cur_params + a * self.spec.robot_param_scale
        for p, body in zip(params, self.robot.bodies):
            body.set_params(p, pad_zeros=True, map_params=True)
            body.sync_node()
        self._export()
        self.design_cur_params = (self.get_attr_design()
                                  if self.spec.use_projected_params
                                  else params.copy())
        self.cur_t += 1
        self.stage = "execution"
        return self.obs()

    # ---- what the grouping keys on ----------------------------------------
    def topo_key(self):
        """The ORDERED tuple of body names.

        Names encode the path from the root (`reindex`: a body is named by its
        index in its parent's child list, prefixed by the parent's name), so
        the SET of names already determines the tree and the XML document
        order. The ORDER of `robot.bodies` is a separate thing -- it is
        creation order, so two worlds can reach the same tree by different
        skeleton actions and index their nodes differently. Keying on the
        ordered tuple makes a group share one adjacency and one `body_index`
        vector exactly, with no reordering step to get wrong.
        `gate_two_stage.py` measures the fragmentation this costs against the
        unordered key; on hopper it is zero.
        """
        return tuple(b.name for b in self.robot.bodies)

    def name_set_key(self):
        return tuple(sorted(b.name for b in self.robot.bodies))
