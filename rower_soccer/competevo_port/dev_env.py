"""`run-to-goal-devants-v0`, batched over worlds, with per-world morphology.

Stage 2 of the port. The difference from `run_to_goal_env.RunToGoalEnv` is the
episode's first step: their `MultiDevAgentEnv.step` branches on a per-agent
stage flag, and while the stage is `attribute_transform` the incoming action is
not a torque -- it is the ant's BODY. Their env answers it by mutating two lxml
trees, re-merging them, and compiling a brand-new MjModel mid-episode
(`multi_dev_agent_env.py:274-316`). Here it is a write of that world's row of
the batched model arrays (`design.py`), and the compiled model never changes.

The episode, per world:

    reset            qpos = qpos0 + U(-0.1, 0.1), qvel = 0, stage = design,
                     scale_vector ~ U(-1,1)^20      obs = [0 | scale | sim]
    step 0  (design) action[:20] IS the genome. Model fields are written, the
                     state is replaced by a FRESH one (qpos = qpos0 exactly,
                     qvel = 0 -- their rebuild allocates a new MjData, so the
                     reset noise never reaches the simulator), reward 0,
                     no termination.                obs = [1 | design | sim]
    steps 1..500     action[-8:] is the motor command; physics, their three
                     reward layers, termination.

Deviations from `RunToGoalEnv`, all forced by their dev code (not by batching):

  * obs is 52 = [stage flag (1) | scale vector (20) | sim obs (31)], their
    `DevAnt._get_obs` list flattened in order (dev_ant.py:309-337). The sim
    block is the same 31 numbers as stage 0.
  * action is 28 = [design (20) | motor (8)] (`DevAnt.set_env`, dev_ant.py:48).
  * termination adds an UPPER bound: `0.28 <= z <= 1.2` (dev_ant.py:291). The
    fixed-morph ant only has the lower bound, so a dev ant that gets launched
    upward dies and a fixed one does not.
  * no reset noise survives into the simulated episode (above). Their run-to-goal
    dev episodes are deterministic given the design; diversity comes from the
    random `scale_vector` the policy conditions on, and from action noise.

Batching note: worlds are asynchronous, so at any wall step a few worlds are in
the design stage and the rest are executing -- exactly the mixed batch their
`DevPolicy.forward` partitions by `design_mask`. Physics is stepped for every
world regardless and the design-stage worlds' step is DISCARDED (their state is
overwritten by the post-design state). That wastes ~1/200 of the physics and
keeps a single CUDA-graph launch per step, which is worth far more.
"""

import numpy as np
import torch

from rower_soccer.competevo_port.backend import (CompeteCpuDevBackend,
                                                 CompeteWarpDevBackend)
from rower_soccer.competevo_port.design import (BATCHED_FIELDS,
                                                DesignWriter,
                                                build_design_spec)
from rower_soccer.competevo_port.run_to_goal_env import (CONTACT_COST_COEF,
                                                         CTRL_COST_COEF,
                                                         GOAL_REWARD,
                                                         MAX_EPISODE_STEPS,
                                                         MOVE_REWARD_WEIGHT,
                                                         RESET_QPOS_NOISE,
                                                         STAND_Z,
                                                         SURVIVE_BONUS)
from rower_soccer.competevo_port.scene import (CONTROL_DT, DESIGN_DIM,
                                               FRAME_SKIP, build_dev_scene)

# `DevAnt.after_step` (dev_ant.py:291): the dev ant is standing only inside a
# BAND. The fixed ant (`Ant.after_step`, ant.py:43) has no ceiling.
STAND_Z_MAX = 1.2


def _padded_list(rows, device, dtype):
    """Like `_padded_idx` but for explicit index lists of differing length."""
    W = max(len(r) for r in rows)
    idx = torch.zeros(len(rows), W, dtype=torch.long, device=device)
    mask = torch.zeros(len(rows), W, device=device, dtype=dtype)
    for i, r in enumerate(rows):
        idx[i, :len(r)] = torch.tensor(r, device=device, dtype=torch.long)
        mask[i, :len(r)] = 1.0
    return idx, mask


def _padded_idx(ranges, device, dtype):
    """`[(lo, hi), ...]` -> an `[A, max_w]` index tensor and an `[A, max_w]`
    mask that is 1 on real entries and 0 on padding.

    Padding entries index 0 rather than anything out of bounds, so the gather
    is always legal; the mask is what makes them harmless. Returning both
    (rather than a ragged list) is what keeps `sim_obs` a single gather and
    preserves the rectangular observation the batched trainer requires.
    """
    widths = [hi - lo for lo, hi in ranges]
    W = max(widths)
    idx = torch.zeros(len(ranges), W, dtype=torch.long, device=device)
    mask = torch.zeros(len(ranges), W, device=device, dtype=dtype)
    for i, ((lo, hi), w) in enumerate(zip(ranges, widths)):
        idx[i, :w] = torch.arange(lo, hi, device=device, dtype=torch.long)
        mask[i, :w] = 1.0
    return idx, mask


class RunToGoalDevEnv:
    """Batched two-dev-ant run-to-goal with per-world morphology.

    Shapes: obs `[n, A, 52]`, action `[n, A, 28]`, reward `[n, A]`, done `[n]`.
    `stage` is `[n]` bool -- True while the world's next action is its design.
    """

    # One step per episode is the design action, and their `_elapsed_steps`
    # does not count it; the eval loop has to run one step longer than the
    # episode limit or a stand-still policy never closes an episode.
    has_design_step = True

    def __init__(self, num_worlds=1024, use_gpu=True, device=None, seed=0,
                 use_graph=True, nconmax=64, njmax=512, backend_cls=None,
                 max_episode_steps=MAX_EPISODE_STEPS,
                 contact_cost_from_cfrc=False, auto_reset=True,
                 fixed_design=None, exact_constants=True, scene_kwargs=None):
        self.n = num_worlds
        self.max_episode_steps = max_episode_steps
        self.contact_cost_from_cfrc = contact_cost_from_cfrc
        self.auto_reset = auto_reset

        if backend_cls is None:
            backend_cls = CompeteWarpDevBackend if use_gpu else CompeteCpuDevBackend
        if device is None:
            device = "cuda" if use_gpu else "cpu"
        if not use_gpu:
            use_graph = False

        self.model, self.meta = self._build_scene(**(scene_kwargs or {}))
        self.backend = backend_cls(self.model, num_worlds, FRAME_SKIP,
                                   use_graph=use_graph, nconmax=nconmax,
                                   njmax=njmax, device=device,
                                   batched_fields=BATCHED_FIELDS)
        self.device = self.backend.device
        self.qpos, self.qvel, self.ctrl = (self.backend.qpos, self.backend.qvel,
                                           self.backend.ctrl)
        self.subtree_com = self.backend.subtree_com
        self.cfrc_ext = self.backend.cfrc_ext
        self.dtype = self.qpos.dtype
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        self.spec = build_design_spec(self.model, self.meta,
                                      device=self.device, dtype=self.dtype)
        self.writer = DesignWriter(self.spec, self.backend.model_arrays,
                                   model=self.model,
                                   exact_constants=exact_constants)

        m = self.meta
        self.n_agents = m.n_agents
        # 2h: on a mixed team these are per agent. The observation and action
        # tensors stay RECTANGULAR at the widest agent's width, with each
        # agent's real entries in the leading columns of each padded sub-block.
        # The action layout is [design(max_design) | motor(max_motor)], so the
        # motor block starts at the same offset for every agent regardless of
        # its genome width -- a per-agent offset would be an invitation to read
        # one creature's torques into another's actuators.
        self.n_motors = tuple(getattr(m, "n_motors", None)
                              or [m.n_motor] * m.n_agents)
        self.n_motor = max(self.n_motors)
        self.sim_obs_dim = max(getattr(m, "sim_obs_dims", None)
                               or [m.sim_obs_dim])
        # Per-agent genome widths (2h). `meta.design_dims` exists only on a
        # TeamSceneMeta; everything else is a homogeneous ant scene, where the
        # widths are all DESIGN_DIM and nothing below changes behaviour.
        self.design_dims = tuple(getattr(m, "design_dims", None)
                                 or [DESIGN_DIM] * m.n_agents)
        self.design_dim = max(self.design_dims)
        self.design_mask_row = torch.zeros(m.n_agents, self.design_dim,
                                           device=self.device, dtype=self.dtype)
        for i, w in enumerate(self.design_dims):
            self.design_mask_row[i, :w] = 1.0
        self.homogeneous_design = len(set(self.design_dims)) == 1
        self.obs_dim = 1 + self.design_dim + self.sim_obs_dim
        self.act_dim = self.design_dim + self.n_motor
        # Which flat ctrl slots each agent's motor block writes, and a mask
        # zeroing the padded motor columns so a creature can never actuate a
        # joint it does not have.
        self.motor_mask = torch.zeros(m.n_agents, self.n_motor,
                                      device=self.device, dtype=self.dtype)
        for i, w in enumerate(self.n_motors):
            self.motor_mask[i, :w] = 1.0
        self.ctrl_cols = torch.cat([
            torch.arange(a.ctrl[0], a.ctrl[1], device=self.device,
                         dtype=torch.long) for a in m.agents])
        d, L = self.device, torch.long
        # 2h: a mixed team's members differ in qpos/qvel width (ant 15/14, bug
        # 21/20, spider 27/26) and genome width (20/30/40). The old
        # `torch.stack` over per-agent ranges required them equal and would
        # raise on a mixed team -- the right failure, but not a usable one.
        # `_padded_idx` pads each agent's index row out to the widest and
        # returns a mask zeroing the padding, so `sim_obs` stays ONE
        # rectangular gather and the batched trainer keeps its rectangular
        # [n, A, obs_dim] contract. This is the same zero-pad-and-index
        # approach STAGE2_MULTITASK section 3 specifies for per-task obs.
        #
        # Homogeneous teams are unaffected: all widths are already equal, the
        # mask is all ones, and the gather is exactly the one that ran before.
        self.qpos_idx, self.qpos_mask = _padded_idx(
            [a.qpos for a in m.agents], d, self.dtype)
        self.qvel_idx, self.qvel_mask = _padded_idx(
            [a.qvel for a in m.agents], d, self.dtype)
        self.other_xy_idx = torch.tensor([a.other_qpos_xy for a in m.agents],
                                         device=d, dtype=L)
        self.torso_body = torch.tensor([a.torso_body for a in m.agents],
                                       device=d, dtype=L)
        # Padded like qpos/qvel: a mixed team's creatures have 13/19/25 bodies.
        # Only the cfrc_ext contact cost reads this, and the mask keeps a
        # padded row from contributing (it would index body 0, the world body,
        # whose cfrc is zero anyway -- but relying on that would be an
        # accident, not a design).
        self.body_ids, self.body_ids_mask = _padded_list(
            [list(a.body_ids) for a in m.agents], d, self.dtype)
        self.root_z_idx = torch.tensor([a.qpos[0] + 2 for a in m.agents],
                                       device=d, dtype=L)
        self.goal_x = torch.tensor([a.goal_x for a in m.agents], device=d,
                                   dtype=self.dtype)
        self.move_sign = torch.tensor([-1.0 if a.move_left else 1.0
                                       for a in m.agents], device=d,
                                      dtype=self.dtype)
        self.qpos0 = torch.as_tensor(np.asarray(self.model.qpos0), device=d,
                                     dtype=self.dtype)
        self.quat_idx = torch.tensor(
            [a.qpos[0] + 3 + k for a in m.agents for k in range(4)],
            device=d, dtype=L).reshape(m.n_agents, 4)

        # `fixed_design` pins every world's genome (a [A, 20] or [20] vector),
        # which is what the parity gate and any "does the ant still walk"
        # ablation need; None = their behaviour, a fresh U(-1,1) draw per reset.
        self.fixed_design = None if fixed_design is None else torch.as_tensor(
            np.broadcast_to(np.asarray(fixed_design, dtype=np.float64),
                            (m.n_agents, self.design_dim)).copy(),
            device=d, dtype=self.dtype)

        self.stage = torch.ones(self.n, device=d, dtype=torch.bool)
        self.scale = torch.zeros(self.n, m.n_agents, self.design_dim, device=d,
                                 dtype=self.dtype)
        self.ep_step = torch.zeros(self.n, device=d, dtype=L)
        self.ep_return = torch.zeros(self.n, m.n_agents, device=d,
                                     dtype=self.dtype)
        self._com_before = torch.zeros(self.n, m.n_agents, device=d,
                                       dtype=self.dtype)
        self.last_return = torch.zeros(self.n, m.n_agents, device=d,
                                       dtype=self.dtype)
        self.last_len = torch.zeros(self.n, device=d, dtype=L)
        self.last_win = torch.zeros(self.n, m.n_agents, device=d,
                                    dtype=self.dtype)
        self.games = 0
        self.wins = np.zeros(m.n_agents)
        self.n_diverged = 0

    # Subclass hooks. Both are identity/default here, so the 1v1 path is
    # bit-for-bit what it was; `team_env.TeamRunToGoalDevEnv` is the only user.
    @staticmethod
    def _build_scene(**kw):
        return build_dev_scene(**kw)

    def _mask_motors(self, motor_eff):
        """Last chance to zero an agent's torque before it is written to
        `ctrl`. 2v2 uses it to disable a downed agent."""
        return motor_eff

    # -- state helpers ------------------------------------------------------
    def _agent_com_x(self):
        return self.subtree_com[:, self.torso_body, 0]

    def _root_z(self):
        return self.qpos[:, self.root_z_idx]

    def _build_obs_cols(self):
        """`obs_cols[i]` = the columns of `obs()` that are REAL for agent i.

        obs is `[flag(1) | scale(max_design) | qpos(max_q) | qvel(max_v) |
        others(2*n_others)]`, and on a mixed team agent i occupies only the
        leading part of each padded sub-block. A per-slot policy therefore
        cannot take a contiguous slice -- it needs this gather. Homogeneous
        teams get `arange(obs_dim)`, i.e. the identity.
        """
        cols, W = [], self.design_dim
        Q = self.qpos_idx.shape[1]
        V = self.qvel_idx.shape[1]
        n_other = self.other_xy_idx.shape[1]
        for i in range(self.n_agents):
            dq = int(self.qpos_mask[i].sum().item())
            dv = int(self.qvel_mask[i].sum().item())
            c = ([0]
                 + list(range(1, 1 + self.design_dims[i]))
                 + list(range(1 + W, 1 + W + dq))
                 + list(range(1 + W + Q, 1 + W + Q + dv))
                 + list(range(1 + W + Q + V, 1 + W + Q + V + n_other)))
            cols.append(torch.tensor(c, device=self.device, dtype=torch.long))
        return cols

    @property
    def obs_cols(self):
        if getattr(self, "_obs_cols", None) is None:
            self._obs_cols = self._build_obs_cols()
        return self._obs_cols

    def sim_obs(self):
        """The 31-dim block: own qpos, own qvel, opponent root x,y.

        On a mixed team the qpos/qvel sub-blocks are padded to the widest
        agent's width and the padding masked to zero, so the block stays
        rectangular and no agent ever reads another creature's state.
        """
        q = self.qpos[:, self.qpos_idx] * self.qpos_mask
        v = self.qvel[:, self.qvel_idx] * self.qvel_mask
        o = self.qpos[:, self.other_xy_idx]
        return torch.cat([q, v, o], dim=-1)

    def obs(self):
        """[n, A, 52] = their `DevAnt._get_obs` list, flattened in its order:
        `[if_use_transform_action() | scale_vector | sim_obs]`. The flag is 0
        while the next action is the design and 1 during execution -- it is the
        index of the stage in `['attribute_transform', 'execution']`."""
        flag = (~self.stage).to(self.dtype).reshape(self.n, 1, 1).expand(
            self.n, self.n_agents, 1)
        return torch.cat([flag, self.scale, self.sim_obs()], dim=-1)

    def design_mask(self):
        """[n, A] -- their `DevPolicy.forward` partitions a mixed batch on this
        and runs the scale head on one half, the control head on the other."""
        return self.stage.reshape(self.n, 1).expand(self.n, self.n_agents)

    # -- reset --------------------------------------------------------------
    def reset(self):
        self.reset_idx(torch.arange(self.n, device=self.device))
        return self.obs()

    def reset_idx(self, idx):
        """Their `MultiDevAgentEnv.reset`: fresh agents (so designs never
        compound), a fresh random `scale_vector` per agent, and the noisy
        post-reset state their `_reset` produces. That state is only ever
        OBSERVED -- the design step throws it away (see `_apply_designs`)."""
        if idx.numel() == 0:
            return
        n = idx.numel()
        noise = (torch.rand(n, self.meta.nq, generator=self.gen,
                            device=self.device, dtype=self.dtype) * 2 - 1
                 ) * RESET_QPOS_NOISE
        self.qpos[idx] = self.qpos0.unsqueeze(0) + noise
        for a in range(self.n_agents):
            qi = self.quat_idx[a]
            q = self.qpos[idx.unsqueeze(-1), qi.unsqueeze(0)]
            self.qpos[idx.unsqueeze(-1), qi.unsqueeze(0)] = (
                q / q.norm(dim=-1, keepdim=True))
        self.qvel[idx] = 0.0
        self.ctrl[idx] = 0.0
        self.ep_step[idx] = 0
        self.ep_return[idx] = 0.0
        self.stage[idx] = True
        if self.fixed_design is None:
            # Masked: a padded genome column must stay 0, not carry a random
            # draw. Otherwise the design writer would read noise for a joint
            # the creature does not have, and the policy would see a channel
            # that changes every episode and means nothing.
            self.scale[idx] = (torch.rand(
                n, self.n_agents, self.design_dim, generator=self.gen,
                device=self.device, dtype=self.dtype) * 2 - 1
                ) * self.design_mask_row
        else:
            self.scale[idx] = self.fixed_design.unsqueeze(0)
        # No `forward()` here, unlike the fixed-morph env: the only thing read
        # from this state is the pre-design observation, which is qpos/qvel, and
        # the design step replaces the state (and re-latches the COM) before any
        # reward is computed. One less full kernel launch per step.

    def _apply_designs(self, idx, design):
        """Their `attribute_transform` step, without the recompile: write the
        design-derived model fields for these worlds, then put them in the state
        their fresh `MjData` would have -- qpos0 exactly, qvel 0."""
        # Actions arrive in the network's dtype (fp32); the CPU mirror's state
        # is float64 for the parity gate, so cast rather than assume.
        design = design.to(self.dtype)
        # 2h: a narrower creature's trailing genome columns stay 0. Nothing
        # downstream reads them (the design writer indexes only that agent's
        # own parameters) but the OBSERVATION carries them, and a channel that
        # changes every episode and means nothing is exactly the kind of input
        # a policy learns to read. On a homogeneous scene the mask is all ones.
        design = design * self.design_mask_row
        self.scale[idx] = design
        self.writer.write(idx, design)
        if hasattr(self.backend, "mark_model_dirty"):
            self.backend.mark_model_dirty()
        self.qpos[idx] = self.qpos0.unsqueeze(0)
        self.qvel[idx] = 0.0
        self.ctrl[idx] = 0.0
        self.stage[idx] = False

    def set_design(self, idx, design):
        """Apply a design out of band (the parity gate hand-sets states)."""
        self._apply_designs(idx, design)
        self.backend.forward()
        self._com_before[idx] = self._agent_com_x()[idx]

    # -- reward / termination -----------------------------------------------
    def terms(self, a, bad=None):
        """`DevAnt.after_step` + `MultiDevAgentEnv.goal_rewards` + `_get_done`.
        Identical to the fixed-morph version except for the standing BAND."""
        com_x = self._agent_com_x()
        forward_r = self.move_sign * (com_x - self._com_before) / CONTROL_DT
        ctrl_cost = CTRL_COST_COEF * (a.to(self.dtype) ** 2).sum(-1)
        if self.contact_cost_from_cfrc:
            f = (self.cfrc_ext[:, self.body_ids].clamp(-1.0, 1.0)
                 * self.body_ids_mask.unsqueeze(-1))
            contact_cost = CONTACT_COST_COEF * (f ** 2).sum((-1, -2))
        else:
            contact_cost = torch.zeros_like(ctrl_cost)
        dense = forward_r - ctrl_cost - contact_cost + SURVIVE_BONUS

        z = self._root_z()
        fell = (z < STAND_Z) | (z > STAND_Z_MAX)
        reached = torch.where(self.goal_x > 0, com_x > self.goal_x,
                              com_x < self.goal_x)
        n_reached = reached.sum(-1)
        one_winner = n_reached == 1
        parse = torch.where(one_winner.unsqueeze(-1),
                            torch.where(reached, GOAL_REWARD, -GOAL_REWARD),
                            torch.zeros_like(dense))
        game_done = n_reached > 0
        reward = parse + MOVE_REWARD_WEIGHT * dense
        if bad is None:
            bad = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        terminated = fell.any(-1) | game_done | bad
        winner = reached & one_winner.unsqueeze(-1)
        return {"reward": reward, "dense": dense, "parse": parse,
                "forward": forward_r, "ctrl_cost": ctrl_cost,
                "contact_cost": contact_cost, "terminated": terminated,
                "winner": winner, "reached": reached, "fell": fell,
                "com_x": com_x}

    # -- step ---------------------------------------------------------------
    def step(self, actions):
        """actions: `[n, A, 28]`. Worlds in the design stage read `[:20]` and
        get reward 0; the rest read `[-8:]` and are simulated."""
        a = actions.reshape(self.n, self.n_agents, self.act_dim)
        design = a[..., :self.design_dim].clamp(-1.0, 1.0)
        # THE CONTROL COST IS CHARGED ON THE RAW ACTION, NOT THE CLIPPED ONE.
        # Their `DevAnt.after_step(action)` is handed `actions[i][-8:]` straight
        # off the policy (`multi_dev_agent_env.py:311`) and computes
        # `.5 * np.square(action).sum()`; MuJoCo clamps `ctrl` to `ctrlrange`
        # inside the step, so the TORQUE is clipped but the COST is not. This
        # is not a detail at `log_std = 0`: a unit Gaussian on 8 dims costs
        # 0.5 * 8 * 1 = 4.0 per step raw against 0.5 * 8 * E[clip(a,-1,1)^2]
        # = ~2.1 clipped, and with `alpha = 1` early on the control cost IS the
        # reward. Measured before this fix, our sampled episodes ran at
        # -1.10 per step against their logged -3.0; the gates never caught it
        # because they drive `terms()` directly, which was always faithful --
        # only `step()`'s clamp was wrong.
        motor_raw = a[..., -self.n_motor:]
        motor = motor_raw.clamp(-1.0, 1.0)
        was_design = self.stage.clone()

        # Design-stage worlds are stepped and then thrown away; zero their ctrl
        # so a stale torque cannot NaN a world before it is overwritten.
        zero_design = was_design.reshape(self.n, 1, 1)
        motor_eff = torch.where(zero_design, torch.zeros_like(motor), motor)
        cost_action = torch.where(zero_design, torch.zeros_like(motor_raw),
                                  motor_raw)
        motor_eff = self._mask_motors(motor_eff) * self.motor_mask
        # Scatter each agent's REAL motor columns into the flat ctrl vector.
        # `reshape(n, -1)` was correct only while every agent had the same
        # motor count; on a mixed team it would smear one creature's torques
        # across the next creature's actuators and still run.
        flat = motor_eff[:, self.motor_mask.bool()].reshape(self.n, -1)
        self.ctrl[:, self.ctrl_cols] = flat.to(self.ctrl.dtype)
        self._com_before = self._agent_com_x().clone()
        self.backend.step()

        bad = ((~torch.isfinite(self.qpos).all(-1))
               | (~torch.isfinite(self.qvel).all(-1)))
        if bool(bad.any()):
            self.n_diverged += int(bad.sum().item())
            self.qpos[bad] = self.qpos0.unsqueeze(0)
            self.qvel[bad] = 0.0
            self.backend.forward()

        t = self.terms(cost_action, bad)
        reward, winner = t["reward"], t["winner"]
        terminated = t["terminated"]
        # The curriculum reward the trainer builds is
        # `alpha * dense + (1 - alpha) * parse`, so these two have to be zeroed
        # on the design step exactly as `reward` is -- their
        # `MultiDevAgentEnv.step` returns `reward_parse: 0, reward_dense: 0`
        # for the `attribute_transform` stage
        # (`competevo/evo_envs/multi_dev_agent_env.py:286-289`). Before this the
        # info dict leaked the executed-step terms into the design step, so a
        # curriculum-reward trainer paid ~+1 (the survive bonus) once per
        # episode that their runner never pays.
        dense_i, parse_i = t["dense"], t["parse"]

        # The design branch: reward 0, no termination, no elapsed step
        # (`_elapsed_steps` is only touched by their `_step`).
        if bool(was_design.any()):
            didx = was_design.nonzero(as_tuple=True)[0]
            self._apply_designs(didx, design[didx])
            self.backend.forward()
            self._com_before[didx] = self._agent_com_x()[didx]
            keep = (~was_design).unsqueeze(-1)
            reward = torch.where(keep, reward, torch.zeros_like(reward))
            dense_i = torch.where(keep, dense_i, torch.zeros_like(dense_i))
            parse_i = torch.where(keep, parse_i, torch.zeros_like(parse_i))
            winner = winner & keep
            terminated = terminated & (~was_design)

        self.ep_step += (~was_design).to(self.ep_step.dtype)
        truncated = self.ep_step >= self.max_episode_steps
        done = terminated | truncated
        self.ep_return += reward

        if bool(done.any()):
            self.last_return = torch.where(done.unsqueeze(-1), self.ep_return,
                                           self.last_return)
            self.last_len = torch.where(done, self.ep_step, self.last_len)
            self.last_win = torch.where(done.unsqueeze(-1),
                                        winner.to(self.dtype), self.last_win)
            self.games += int(done.sum().item())
            self.wins += winner[done].sum(0).cpu().numpy()
            if self.auto_reset:
                self.reset_idx(done.nonzero(as_tuple=True)[0])

        info = {"dense": dense_i, "parse": parse_i,
                "terminated": terminated, "truncated": truncated,
                "winner": winner, "forward": t["forward"],
                "ctrl_cost": t["ctrl_cost"], "com_x": t["com_x"],
                "fell": t["fell"], "was_design": was_design,
                "design": design}
        return self.obs(), reward, done, info

    # -- diagnostics --------------------------------------------------------
    def win_rate(self):
        if self.games == 0:
            return np.zeros(self.n_agents)
        return self.wins / self.games

    def reset_win_stats(self):
        self.games = 0
        self.wins = np.zeros(self.n_agents)
