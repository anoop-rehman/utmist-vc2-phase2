"""D3 M3 E2: CompetEvo's 1v1 run-to-goal, inside Transform2Act.

Our DeepMind ant on both sides. OUR agent is the Transform2Act `Robot` (the
first `<body>` in `assets/mujoco_envs/rtg_ant.xml`); the OPPONENT is a
sibling body the `Robot` cannot see, driven by a script.

THE SCRIPTED OPPONENT, stated exactly, because every later rung inherits it
-------------------------------------------------------------------------
At execution step `k` (k = 0 at the first control step of the episode) the
opponent's ENTIRE state is overwritten, before the physics of that step:

    root position   x = +1.0 - v * dt * k ,  y = 0 ,  z = z*
    root quaternion yaw 180 deg (facing -x), no roll, no pitch
    root velocity   (-v, 0, 0), zero angular velocity
    8 hinge angles  q*,   all hinge velocities 0
    8 motor torques 0     (its actuators are never written)

with `v = env_specs.opponent_speed` (0.68 m/s) and `dt = timestep *
frame_skip` (0.015 s). `z*` and `q*` are the stance the same ant settles into
under gravity at zero torque in this engine, measured once at construction by
`_settle_opponent` -- z* = 0.5347 m and ankles 51.87 deg, hips 0.

So the opponent is a RIGID, NON-REACTIVE, CONSTANT-SPEED obstacle in the shape
of our ant. It does not walk (no hand-written torque controller makes an ant
run, and a learned one is excluded by E2's spec), it does not steer, it does
not see our agent, and no contact can slow it, push it or knock it over. Its
trajectory is a function of `k` alone and is therefore identical in every
episode of every seed of every arm.

WHY 0.68 m/s. The task's own clock already demands a speed: 5.0 m (from
x=-1 to the goal at x=+4) inside 500 control steps x 0.015 s = 7.5 s, i.e.
0.667 m/s. The opponent is that clock made physical, advanced 2% so that
running out of time is realised as a LOSS to a visible opponent rather than as
a silent truncation: at 0.68 m/s it starts at x=+1 and crosses x=-4 at control
step 491 of 500. Beating the opponent and beating the clock are therefore the
same requirement, which is what makes E2's goal rate directly comparable to
D2's (`D2_MORPHOLOGY_COMPETENCE.md`, same body, same reward, same 5.0 m in
7.5 s).

WHAT IS TAKEN FROM CompetEvo, and what is not
---------------------------------------------
Taken verbatim from `rower_soccer/competevo_port/run_to_goal_env.py`, which is
gated against CompetEvo's own code in `competevo_port/tests/test_parity.py`:

  * dense reward `forward - ctrl_cost - contact_cost + survive`, with
    `forward = dCOM_x/dt` on the torso SUBTREE COM, `ctrl_cost = 0.5 *
    sum(a^2)` on the RAW (unclamped) action, `survive = +1.0`;
  * `contact_cost = 0` -- their `data.cfrc_ext` is never populated (no
    acceleration-stage sensor runs `mj_rnePostConstraint`), so the term is a
    constant zero in every CompetEvo run and `contact_cost_from_cfrc=False`
    reproduces that rather than inventing a cost their policies never felt;
  * sparse reward +/-1000 iff exactly one agent crosses its goal line,
    measured on the subtree COM against x = +/-4;
  * termination on a fall (root z < 0.28), on a goal, or on a non-finite
    state; truncation at 500 control steps.

Deliberately NOT taken, each because E2 has a scripted opponent:

  1. **The fall test applies to OUR agent only.** Theirs ends the episode if
     EITHER ant drops below z=0.28. Our opponent is prescribed, so its z is a
     constant 0.5347 and the test would be vacuous for it -- but if the
     opponent's stance were ever changed to a lower one, applying their rule
     would end every episode at step 0. Stated rather than inherited.
  2. **Transform2Act's `done_condition.max_ang`/`min_height`/`max_height` are
     NOT used.** CompetEvo's run-to-goal has no tilt condition; using both
     would be two termination rules for one task. E1/E1.1 keep `max_ang: 60`
     because they run Transform2Act's OWN locomotion task; E2 runs CompetEvo's.

Deliberately CHANGED, with the reason:

  3. **The observation.** CompetEvo's is a flat 31-vector `[own qpos (15) |
     own qvel (14) | opponent root x,y (2)]` in the WORLD frame. Transform2Act's
     is a per-body matrix that deliberately excludes the root's x,y so that a
     policy transfers across designs and positions. Injecting absolute world
     x,y would break that. E2 instead appends THREE columns, broadcast to every
     node row: `(opp_x - own_x, opp_y - own_y, goal_x - own_x)` -- the
     opponent's position relative to ours and the distance still to run. That
     is CompetEvo's information content in a translation-invariant frame. Both
     E2 arms are fed exactly these columns, so the comparison is unaffected.
"""

import numpy as np

from design_opt.envs.ant import AntEnv

GOAL_X = 4.0
STAND_Z = 0.28
CTRL_COST_COEF = 0.5
SURVIVE_BONUS = 1.0
GOAL_REWARD = 1000.0
OPP_PREFIX = "opp_"


class RunToGoalEnv(AntEnv):

    def __init__(self, cfg, agent):
        self._opp_cache = None
        self._opp_frozen = None
        # set BEFORE super().__init__: MujocoEnv's constructor takes a probe
        # step, which calls `_get_obs` -> `get_sim_obs`, which reads goal_x.
        es = cfg.env_specs
        self.opp_speed = float(es.get("opponent_speed", 0.68))
        self.goal_x = float(es.get("goal_x", GOAL_X))
        self.stand_z = float(es.get("stand_z", STAND_Z))
        self.max_nsteps = int(cfg.done_condition.get("max_nsteps", 500))
        super().__init__(cfg, agent)
        self._settle_opponent()

    # ---------------------------------------------------------------- opp --
    def _opp(self):
        """(qpos slice, qvel slice, body id) for the opponent, resolved by
        name against the CURRENT model. Cached per compiled model, because the
        design stages replace `self.model` even when the body is frozen."""
        if self._opp_cache is not None and self._opp_cache[0] is self.model:
            return self._opp_cache[1:]
        m = self.model
        jid = m.joint_name2id(OPP_PREFIX + "0_joint")
        qs, vs = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        nq = m.nq - qs
        nv = m.nv - vs
        bid = m.body_name2id(OPP_PREFIX + "0")
        out = (slice(qs, qs + nq), slice(vs, vs + nv), bid)
        self._opp_cache = (m,) + out
        return out

    def _our_torso_id(self):
        return self.model.body_name2id(self.robot.bodies[0].name)

    def _settle_opponent(self):
        """Measure the opponent's frozen stance: drop it at zero torque from
        the asset's pose and read the rest state. Deterministic (no noise),
        run once per env, in the engine that will run the episodes."""
        qs, vs, _ = self._opp()
        self.sim.data.qpos[:] = self.init_qpos
        self.sim.data.qvel[:] = 0.0
        self.sim.data.ctrl[:] = 0.0
        self.sim.forward()
        for _ in range(int(3.0 / self.model.opt.timestep)):
            self.sim.step()
        self._opp_frozen = self.sim.data.qpos[qs].copy()
        self.sim.data.qpos[:] = self.init_qpos
        self.sim.data.qvel[:] = 0.0
        self.sim.forward()

    def opp_x(self, k):
        """The opponent's prescribed torso x at execution step k."""
        return 1.0 - self.opp_speed * self.dt * k

    def set_opponent(self, k):
        """Overwrite the opponent's whole state with the prescription for step
        k. Nothing here reads our agent's state, which is what makes the
        opponent non-reactive by construction rather than by intention."""
        qs, vs, _ = self._opp()
        q = self._opp_frozen.copy()
        q[0] = self.opp_x(k)
        q[1] = 0.0
        self.sim.data.qpos[qs] = q
        v = np.zeros(vs.stop - vs.start)
        v[0] = -self.opp_speed
        self.sim.data.qvel[vs] = v

    # ---------------------------------------------------------------- obs --
    def get_sim_obs(self):
        obs = super().get_sim_obs()
        _, _, obid = self._opp()
        com = self.data.subtree_com
        ours = com[self._our_torso_id()]
        theirs = com[obid]
        extra = np.array([theirs[0] - ours[0], theirs[1] - ours[1],
                          self.goal_x - ours[0]])
        return np.concatenate([obs, np.tile(extra, (obs.shape[0], 1))], axis=-1)

    # --------------------------------------------------------------- step --
    def step(self, a):
        """`AntEnv.step`'s design stages verbatim; a different execution stage.

        The design branches are delegated to the parent, so E1.1's
        `force_identity_design` gate covers E2 unchanged."""
        if not self.is_inited:
            return self._get_obs(), 0, False, {"use_transform_action": False,
                                               "stage": "execution"}
        if self.stage != "execution":
            return super().step(a)

        self.control_nsteps += 1
        k = self.control_nsteps - 1

        assert np.all(a[:, self.control_action_dim:] == 0)
        control_a = a[:, :self.control_action_dim]
        ctrl = self.action_to_control(control_a)
        our_id, (_, _, opp_id) = self._our_torso_id(), self._opp()
        com_before = float(self.data.subtree_com[our_id][0])
        try:
            self.do_simulation(ctrl, self.frame_skip)
        except Exception:
            print(self.cur_xml_str)
            return (self._get_obs(), 0, True,
                    {"use_transform_action": False, "stage": "execution",
                     "reached": False, "opp_reached": False, "fell": True})
        # Snap the opponent back onto its prescription AFTER the physics, so
        # every observation, reward and rendered frame sees it at exactly
        # x(k+1). Within the five substeps our agent's contacts do push it --
        # that push is then discarded, which is what "no contact can slow it"
        # means and is why the opponent is effectively infinitely massive.
        # `sim.forward` makes subtree_com/geom_xpos consistent with the write.
        self.set_opponent(k + 1)
        self.sim.forward()
        com_after = float(self.data.subtree_com[our_id][0])

        # -- CompetEvo's dense reward -------------------------------------
        forward_r = (com_after - com_before) / self.dt
        # `ctrl` holds our 8 policy outputs (unclamped -- MuJoCo clamps to
        # ctrlrange internally) and 0 in the opponent's 8 slots, so the sum
        # over the whole vector IS `0.5 * sum(a^2)` over our raw action.
        ctrl_cost = CTRL_COST_COEF * float(np.square(ctrl).sum())
        contact_cost = 0.0
        dense = forward_r - ctrl_cost - contact_cost + SURVIVE_BONUS

        # -- CompetEvo's sparse reward ------------------------------------
        opp_com_x = float(self.data.subtree_com[opp_id][0])
        reached = com_after > self.goal_x
        opp_reached = opp_com_x < -self.goal_x
        n_reached = int(reached) + int(opp_reached)
        # `parse` is CompetEvo's name for the sparse term
        # (`competevo/evo_envs/multi_agent_env.py` -> `info['reward_parse']`);
        # `dense` is `info['reward_dense']`. They are returned SEPARATELY as
        # well as summed because their exploration curriculum optimises
        # `alpha * dense + (1 - alpha) * parse` rather than the env reward
        # (`runner/multi_agent_runner.py:150-167`, ported in
        # `rower_soccer/competevo_port/ppo.py:211-234`). D3 E2.1 needs both
        # halves to reproduce that mix. The reward this method RETURNS is
        # unchanged -- `dense + parse` is exactly the expression that was here
        # before -- so every E2 number remains reproducible from this file.
        parse = 0.0
        if n_reached == 1:
            parse = GOAL_REWARD if reached else -GOAL_REWARD
        reward = dense + parse

        s = self.state_vector()
        bad = not np.isfinite(s).all()
        fell = s[2] < self.stand_z
        done = bool(bad or fell or n_reached > 0
                    or self.control_nsteps >= self.max_nsteps)
        return (self._get_obs(), reward, done,
                {"use_transform_action": False, "stage": "execution",
                 "reached": bool(reached), "opp_reached": bool(opp_reached),
                 "fell": bool(fell), "com_x": com_after,
                 "opp_com_x": opp_com_x, "forward": forward_r,
                 "ctrl_cost": ctrl_cost, "dense": dense, "parse": parse})

    # -------------------------------------------------------------- reset --
    def transit_execution(self):
        ok = super().transit_execution()
        if ok:
            self.set_opponent(0)
            self.sim.forward()
        return ok
