"""D3 M3 E4: run-to-goal with the OPPONENT driven by a frozen snapshot of
another lineage -- its evolved body and its trained controller -- instead of
by E2/E3's kinematic prescription.

Relationship to `run_to_goal.py`, which is untouched:

  * the LEARNER's code path is inherited unchanged. It is always agent 0, at
    x=-1 facing +x, with goal +4. Every observation, action, reward and design
    stage it sees is byte-identical to E3.1's. That is deliberate: the gate
    requires a non-refreshed E4 run to reproduce E3.1, and it can only do that
    if nothing on the learner's side moved.
  * everything new is confined to driving agent 1.

Why both lineages are agent 0
-----------------------------
The scene is mirror-symmetric, so "A at x=-1 racing B at x=+1" and "B at x=-1
racing A at x=+1" are the same game. Rather than train one lineage in each
slot -- which would put the two morphologies in different frames and invite a
slot confound in the divergence metric -- BOTH lineages train as agent 0 in
their own env, and each faces the other's snapshot in slot 1.

A snapshot therefore has to act in slot 1 although it was trained in slot 0.
Slot 1 is slot 0 rotated by pi about z. That is a ROTATION, not a reflection,
so it is exactly correct and introduces no chirality error -- the mistake a
naive "flip the x sign" mirror would make.

The transform, `_rotate_root_obs`
---------------------------------
Under a pi-z world rotation (X,Y,Z) -> (-X,-Y,Z):

  root z          qpos[qs+2]      unchanged
  root quat       qpos[qs+3:qs+7] q -> q_z(pi) (x) q  ==  (w,x,y,z) -> (-z,-y,x,w)
  root linear v   qvel[vs:vs+3]   world frame  -> (-vx,-vy,vz)
  root angular v  qvel[vs+3:vs+6] BODY-LOCAL   -> UNCHANGED
  joint q, qd                     intrinsic    -> unchanged

The angular-velocity row is the one that matters and it was **measured, not
derived**: a free joint's `qvel[3:6]` is expressed in the body's local frame,
so a world rotation does not touch it. (Measured by yawing a box 90 deg about
z, setting `qvel[3:6] = [1,0,0]`, and recovering the net rotation axis in the
world frame: it came out +y, i.e. the body's own x.) Deriving it instead would
have negated those three columns, which is a silent error -- the opponent
would still walk, just slightly wrong, and E4's divergence number would be
quietly corrupted. `gate_e4.py` checks the whole transform end to end.
"""
import numpy as np
import torch

from design_opt.envs.run_to_goal import RunToGoalEnv, OPP_PREFIX, GOAL_REWARD
from khrylib.robot.xml_robot import Robot
from khrylib.utils import get_single_body_qposaddr, get_graph_fc_edges


def qmul_zpi(q):
    """q_z(pi) (x) q  for q = (w, x, y, z)."""
    w, x, y, z = q
    return np.array([-z, -y, x, w])


class RunToGoalSelfPlayEnv(RunToGoalEnv):

    def __init__(self, cfg, agent):
        es = cfg.env_specs
        # The merged scene the trainer wrote for this refresh; its FIRST body
        # is ours, its opp_* sibling is the snapshot's body. `opponent_body_xml`
        # is that same snapshot body in its own (unprefixed) namespace, which
        # is what we can hand to `Robot`.
        self.opponent_body_xml = es.get('opponent_body_xml', None)
        self.opp_policy = None          # set by the trainer via set_opponent_policy
        self.opp_running_state = None
        self.opp_mode = es.get('opponent_mode', 'scripted')
        super().__init__(cfg, agent)
        if self.opp_mode == 'policy':
            assert self.opponent_body_xml, \
                "opponent_mode=policy needs env_specs.opponent_body_xml"
            self.opp_robot = Robot(cfg.robot_cfg, xml=self.opponent_body_xml)
            self._opp_name_cache = None

    # ------------------------------------------------------------- setup --
    def set_opponent_policy(self, policy, running_state=None):
        """Install the frozen snapshot. `policy` is a Transform2Act policy in
        eval mode; nothing here ever backprops through it."""
        self.opp_policy = policy
        self.opp_running_state = running_state

    def _opp_body_name(self, body):
        return OPP_PREFIX + body.name

    def _opp_joint_id(self, body):
        """The hinge id of an opponent body, or None if it has no joint.
        Resolved through the compiled model, never through index arithmetic."""
        if self._opp_name_cache is None or self._opp_name_cache[0] is not self.model:
            m = self.model
            by_body = {}
            for j in range(m.njnt):
                by_body.setdefault(int(m.jnt_bodyid[j]), []).append(j)
            self._opp_name_cache = (m, by_body)
        by_body = self._opp_name_cache[1]
        bid = self.model.body_name2id(self._opp_body_name(body))
        js = by_body.get(bid, [])
        assert len(js) <= 1, f"{self._opp_body_name(body)} has {len(js)} joints"
        return js[0] if js else None

    # -------------------------------------------------------- opponent obs --
    def _opp_sim_obs(self):
        """`AntEnv.get_sim_obs` for the OPPONENT's bodies, expressed in the
        pi-z-rotated frame so a slot-0-trained policy sees its own frame."""
        qs, vs, obid = self._opp()
        qpos, qvel = self.data.qpos, self.data.qvel
        if self.clip_qvel:
            qvel = np.clip(qvel, -10, 10)
        rows = []
        for i, body in enumerate(self.opp_robot.bodies):
            if i == 0:
                root_z = qpos[qs.start + 2]
                quat = qmul_zpi(qpos[qs.start + 3:qs.start + 7])
                # Quaternion double cover: q and -q are the SAME orientation,
                # but they are different numbers to a network. Composing the
                # pi-z rotation with a body that already starts yawed 180 deg
                # lands on w = -1 where slot 0 starts at w = +1, so a snapshot
                # would be fed a sign-flipped attitude it never saw in
                # training. Canonicalise to w >= 0, which is the convention
                # slot 0 is always in.
                if quat[0] < 0:
                    quat = -quat
                lin = qvel[vs.start:vs.start + 3] * np.array([-1.0, -1.0, 1.0])
                ang = qvel[vs.start + 3:vs.start + 6]     # body-local: as-is
                rows.append(np.concatenate([[root_z], quat, lin, ang,
                                            np.zeros(2)]))
            else:
                # `AntEnv.get_sim_obs` reads the velocity of a hinge as
                # `qvel[qpos_addr - 1]`. That -1 is the offset created by the
                # ONE free joint (7 qpos / 6 qvel) that precedes our own
                # hinges. The opponent sits behind TWO free joints, so the
                # offset is -2 for it -- and would be -3 in any future scene
                # with a third body. Rather than track that, the DOF address
                # is read straight off the model, which is exact by
                # construction. (Caught by gate 3: the last opponent hinge
                # indexed qvel[28] in a 28-DOF model and produced a
                # short row.)
                jid = self._opp_joint_id(body)
                if jid is None:
                    rows.append(np.zeros(13))
                else:
                    qa = self.model.jnt_qposadr[jid]
                    va = self.model.jnt_dofadr[jid]
                    rows.append(np.concatenate([np.zeros(11), qpos[qa:qa + 1],
                                                qvel[va:va + 1]]))
        obs = np.stack(rows)
        # the three appended task columns, in the SAME rotated frame:
        # for the opponent, "theirs" is our agent and its goal is at -goal_x,
        # which the rotation carries to +goal_x.
        com = self.data.subtree_com
        ours = com[self._our_torso_id()]
        theirs = com[obid]
        extra = np.array([theirs[0] - ours[0],      # -(ours) - -(theirs)
                          theirs[1] - ours[1],
                          self.goal_x + theirs[0]])  # goal_x - (-theirs_x)
        return np.concatenate([obs, np.tile(extra, (obs.shape[0], 1))], axis=-1)

    def _opp_obs(self):
        """The full node matrix + graph the snapshot policy expects, in the
        execution stage (its body is frozen, so no design stage ever runs)."""
        attr_fixed = []
        for body in self.opp_robot.bodies:
            row = []
            if 'depth' in self.attr_specs:
                d = np.zeros(self.cfg.max_body_depth)
                d[body.depth] = 1.0
                row.append(d)
            if 'jrange' in self.attr_specs:
                row.append(body.get_joint_range())
            if 'skel' in self.attr_specs:
                row.append(np.array([0.0, 0.0]))   # frozen: no add, no remove
            if row:
                attr_fixed.append(np.concatenate(row))
        design = np.stack([b.get_params([], pad_zeros=True, demap_params=True)
                           for b in self.opp_robot.bodies])
        sim_obs = self._opp_sim_obs()
        parts = [x for x in (np.stack(attr_fixed) if attr_fixed else None,
                             sim_obs, design) if x is not None]
        obs = np.concatenate(parts, axis=-1)
        edges = (get_graph_fc_edges(len(self.opp_robot.bodies))
                 if self.cfg.obs_specs.get('fc_graph', False)
                 else self.opp_robot.get_gnn_edges())
        # The stage flag. `if_use_transform_action()` returns 0 skeleton,
        # 1 attribute, 2 EXECUTION -- so execution is 2, not 0. With 0 here
        # the policy runs its skeleton head and every control column comes
        # back exactly 0.0: the opponent stands still and collapses, while
        # nothing errors. Measured before the fix: opponent moved 0.153 m in
        # 200 steps with max|torque| = 0.000 and root z falling 0.831 -> 0.260.
        all_obs = [obs, edges, np.array([2]), np.array([sim_obs.shape[0]])]
        if self.use_body_ind:
            all_obs.append(np.array([int(b.name, base=self.index_base)
                                     for b in self.opp_robot.bodies]))
        return all_obs

    # ------------------------------------------------------ opponent ctrl --
    def opp_action(self):
        state = self._opp_obs()
        if self.opp_running_state is not None:
            state = self.opp_running_state(state, update=False)
        # The policy consumes a BATCH of samples, so the single observation
        # must be wrapped in a one-element list -- exactly what the sampler's
        # `tensorfy([state])` does. Passing the flat list instead makes
        # `forward` read `x_i[-2]` off the observation matrix's last row and
        # raises "only one element tensors can be converted to Python
        # scalars". That failure is caught by `RunToGoalEnv.step`'s broad
        # `except Exception`, which ends the episode as a fall -- so the
        # opponent would have looked like it was "affecting the outcome"
        # while never taking a single action. Gate 6 caught exactly this.
        with torch.no_grad():
            a = self.opp_policy.select_action(
                [[torch.tensor(x) for x in state]], True
            ).numpy().astype(np.float64)
        return a[:, :self.control_action_dim]

    def opp_control(self, a):
        """Map the snapshot's per-body actions onto the opp_* motor slots.

        The pi-z rotation does not touch joint torques: every actuator is a
        hinge about a body-local axis, and body-local quantities are invariant
        under a world rotation -- the same fact measured for angular velocity.
        """
        ctrl = np.zeros_like(self.data.ctrl)
        for body, body_a in zip(self.opp_robot.bodies[1:], a[1:]):
            # jointless stub bodies have no actuator and return None here
            aname = body.get_actuator_name()
            if aname is None:
                continue
            name = OPP_PREFIX + aname
            if name in self.model.actuator_names:
                ctrl[self.model.actuator_names.index(name)] = body_a
        return ctrl

    # -------------------------------------------------------------- step --
    def set_opponent(self, k):
        """Policy mode: the opponent is a real physical body under its own
        control, so nothing overwrites its state. Scripted mode falls through
        to E2/E3's prescription unchanged."""
        if self.opp_mode == 'policy':
            return
        super().set_opponent(k)

    def do_simulation(self, ctrl, n_frames):
        """Add the opponent's torques to the physics WITHOUT adding them to
        the vector the parent charges control cost on.

        `RunToGoalEnv.step` computes

            ctrl_cost = CTRL_COST_COEF * np.square(ctrl).sum()

        over the WHOLE actuator vector, correct only because the opponent's
        slots are zero there. Injecting the opponent's control via
        `action_to_control` would therefore bill the learner 0.5*sum(a_opp^2)
        for its rival's torques -- and dense control cost is precisely the term
        that deleted every actuator in E3. So the opponent's control is added
        here, downstream of the cost, and `action_to_control` is left alone.
        """
        if (self.opp_mode == 'policy' and self.opp_policy is not None
                and self.stage == 'execution'):
            ctrl = np.asarray(ctrl).copy()
            ctrl += self.opp_control(self.opp_action())
        return super().do_simulation(ctrl, n_frames)

    # ---------------------------------------------------------- refresh --
    def swap_opponent(self, merged_xml_file, opponent_body_xml):
        """Install a new opponent body mid-training.

        Called at a refresh boundary, never inside an episode. Our own base
        body is unchanged -- only the opp_* sibling in the merged scene is --
        but `Robot` owns the whole tree, so it has to be reparsed for the new
        sibling to reach the compiled model. Every cache keyed on the old
        model is dropped, because body ids and qpos addresses all move.
        """
        self.model_xml_file = merged_xml_file
        self.robot = Robot(self.cfg.robot_cfg, xml=merged_xml_file)
        self.init_xml_str = self.robot.export_xml_string()
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        self.reload_sim_model(self.cur_xml_str)
        self.opp_robot = Robot(self.cfg.robot_cfg, xml=opponent_body_xml)
        self.opponent_body_xml = opponent_body_xml
        self._opp_cache = None
        self._opp_name_cache = None
        self._opp_frozen = None
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()
        self.design_param_names = self.robot.get_params(get_name=True)
        self._settle_opponent()
