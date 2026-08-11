"""`run-to-goal-ants-v0`, batched over worlds on mujoco_warp.

The GPU twin of CompetEvo's fixed-morph two-ant race. Their version is three
Python layers deep -- `Ant.after_step` (dense reward + fall termination),
`MultiAgentEnv.goal_rewards` / `_get_done` (the +/-1000 sparse reward and the
shared done), and `MultiAgentEnv.step` (truncation at 500) -- all looping over
agents in numpy, inside one process per environment copy. Here all three layers
are tensor ops over `[num_worlds, n_agents]`, and one compiled model serves every
world.

What is reproduced exactly (and gated in tests/test_parity.py):
  * observation ORDER and content: `[own qpos (15) | own qvel (14) | opponent
    root x,y (2)]` = 31 floats, world-frame, no ego transform (their
    `Ant._get_obs`, ant.py:59-87 -- the fixed-morph flat obs, not the dev agent's
    3-array `[stage | scale | sim_obs]` list, which is stage 2);
  * dense reward `forward - ctrl_cost - contact_cost + survive`, where `forward`
    is the torso SUBTREE-COM x displacement over the control step divided by
    dt=0.015, sign-flipped for the agent running left;
  * sparse reward: +/-1000 iff exactly one agent crosses its goal line (x=+/-4),
    measured on the subtree COM;
  * done: shared across agents -- any agent below z=0.28, or a goal crossed, or
    a non-finite state; truncation at 500 control steps.

Known deviations, each with its reason:
  1. solver PGS -> Newton and iterations 1000 -> 100 (see scene.py's docstring:
     mujoco_warp does not implement PGS). Integrator stays RK4, timestep 0.003,
     frame_skip 5.
  2. Their contact cost reads `data.cfrc_ext`, which MuJoCo leaves at zero here
     (no acceleration-stage sensors => `mj_rnePostConstraint` never runs). It is
     therefore a constant 0 in their runs, and `contact_cost_from_cfrc=False`
     (the default) reproduces that exactly instead of introducing a term their
     trained policies never felt. Set it True to compute the term from the
     backend's cfrc_ext -- which mujoco_warp DOES populate -- if a later stage
     wants the reward the code was written to express.
  3. Auto-reset is per world (theirs resets a whole env process), so worlds run
     out of phase. This is the standard batched-env change and is what makes the
     500-step episodes affordable.
  4. Their `reset_model` applies the qpos noise twice and then zeroes qvel; only
     the last draw survives, so we draw once. Their stray
     `np_random.integers(nv) * .1` qvel offset is discarded by the subsequent
     `set_xyz` (agent.py:289), so it never reaches the sim -- not reproduced.
"""

import mujoco
import numpy as np
import torch

from rower_soccer.competevo_port.backend import (CompeteCpuBackend,
                                                 CompeteWarpBackend)
from rower_soccer.competevo_port.scene import (CONTROL_DT, FRAME_SKIP,
                                               build_run_to_goal_scene)

# MultiAgentEnv default (multi_agent_env.py:77) and the yaml's episode length.
MAX_EPISODE_STEPS = 500
# Ant.after_step:43 -- the agent is "standing" while its ROOT z (not its COM) is
# at least this. Note the fixed-morph ant has no upper bound; the dev ant does.
STAND_Z = 0.28
CTRL_COST_COEF = 0.5
CONTACT_COST_COEF = 0.5e-3
SURVIVE_BONUS = 1.0
# multi_agent_env.py:70 and the `move_reward_weight=1.0` default.
GOAL_REWARD = 1000.0
MOVE_REWARD_WEIGHT = 1.0
# MultiAgentScene.reset_model:58 -- U(-0.1, 0.1) added to every qpos entry,
# including the root quaternion (which mj_forward then renormalizes).
RESET_QPOS_NOISE = 0.1


class RunToGoalEnv:
    """Batched two-ant run-to-goal.

    Shapes: obs `[n, n_agents, 31]`, action `[n, n_agents, 8]` in [-1, 1],
    reward `[n, n_agents]`, done `[n]` (shared across agents, as in theirs).
    """

    def __init__(self, num_worlds=1024, use_gpu=True, device=None, seed=0,
                 use_graph=True, nconmax=64, njmax=512, backend_cls=None,
                 max_episode_steps=MAX_EPISODE_STEPS,
                 contact_cost_from_cfrc=False, auto_reset=True,
                 scene_kwargs=None):
        self.n = num_worlds
        self.max_episode_steps = max_episode_steps
        self.contact_cost_from_cfrc = contact_cost_from_cfrc
        self.auto_reset = auto_reset

        if backend_cls is None:
            backend_cls = CompeteWarpBackend if use_gpu else CompeteCpuBackend
        if device is None:
            device = "cuda" if use_gpu else "cpu"
        if not use_gpu:
            use_graph = False

        self.model, self.meta = build_run_to_goal_scene(**(scene_kwargs or {}))
        self.backend = backend_cls(self.model, num_worlds, FRAME_SKIP,
                                   use_graph=use_graph, nconmax=nconmax,
                                   njmax=njmax, device=device)
        self.device = self.backend.device
        self.qpos, self.qvel, self.ctrl = (self.backend.qpos, self.backend.qvel,
                                           self.backend.ctrl)
        self.subtree_com = self.backend.subtree_com
        self.cfrc_ext = self.backend.cfrc_ext
        self.dtype = self.qpos.dtype
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        m = self.meta
        self.n_agents = m.n_agents
        self.obs_dim, self.act_dim = m.obs_dim, m.act_dim
        d, L = self.device, torch.long
        # Per-agent index tensors: their name-prefix filtering, done once.
        self.qpos_idx = torch.stack([torch.arange(*a.qpos, device=d, dtype=L)
                                     for a in m.agents])
        self.qvel_idx = torch.stack([torch.arange(*a.qvel, device=d, dtype=L)
                                     for a in m.agents])
        self.ctrl_idx = torch.stack([torch.arange(*a.ctrl, device=d, dtype=L)
                                     for a in m.agents])
        self.other_xy_idx = torch.tensor([a.other_qpos_xy for a in m.agents],
                                         device=d, dtype=L)
        self.torso_body = torch.tensor([a.torso_body for a in m.agents],
                                       device=d, dtype=L)
        self.body_ids = torch.stack(
            [torch.tensor(a.body_ids, device=d, dtype=L) for a in m.agents])
        # Root z lives at the third slot of each agent's own qpos slice.
        self.root_z_idx = torch.tensor([a.qpos[0] + 2 for a in m.agents],
                                       device=d, dtype=L)
        self.goal_x = torch.tensor([a.goal_x for a in m.agents], device=d,
                                   dtype=self.dtype)
        # Ant.after_step:34 -- the agent whose goal is behind it scores forward
        # progress as -dx. Precomputed as a +/-1 multiplier.
        self.move_sign = torch.tensor([-1.0 if a.move_left else 1.0
                                       for a in m.agents], device=d,
                                      dtype=self.dtype)
        self.qpos0 = torch.as_tensor(self.model.qpos0, device=d,
                                     dtype=self.dtype)
        # Free-joint quaternion slots, for renormalizing after reset noise.
        self.quat_idx = torch.tensor(
            [a.qpos[0] + 3 + k for a in m.agents for k in range(4)],
            device=d, dtype=L).reshape(m.n_agents, 4)

        self.ep_step = torch.zeros(self.n, device=d, dtype=L)
        self.ep_return = torch.zeros(self.n, m.n_agents, device=d,
                                     dtype=self.dtype)
        self._com_before = torch.zeros(self.n, m.n_agents, device=d,
                                       dtype=self.dtype)
        # Latched per-episode statistics, refreshed as episodes finish.
        self.last_return = torch.zeros(self.n, m.n_agents, device=d,
                                       dtype=self.dtype)
        self.last_len = torch.zeros(self.n, device=d, dtype=L)
        self.last_win = torch.zeros(self.n, m.n_agents, device=d,
                                    dtype=self.dtype)
        self.games = 0
        self.wins = np.zeros(m.n_agents)
        self.n_diverged = 0

    # -- state helpers ------------------------------------------------------
    def _agent_com_x(self):
        """[n, n_agents] torso subtree-COM x -- their `get_body_com('torso')[0]`,
        which is the whole ant's centre of mass, not the torso body origin."""
        return self.subtree_com[:, self.torso_body, 0]

    def _root_z(self):
        return self.qpos[:, self.root_z_idx]

    def obs(self):
        """[n, n_agents, 31] in their order: own qpos, own qvel, opponent x,y."""
        q = self.qpos[:, self.qpos_idx]                 # [n, A, 15]
        v = self.qvel[:, self.qvel_idx]                 # [n, A, 14]
        o = self.qpos[:, self.other_xy_idx]             # [n, A, 2]
        return torch.cat([q, v, o], dim=-1)

    # -- reset --------------------------------------------------------------
    def reset(self):
        self.reset_idx(torch.arange(self.n, device=self.device))
        return self.obs()

    def reset_idx(self, idx):
        if idx.numel() == 0:
            return
        noise = (torch.rand(idx.numel(), self.meta.nq, generator=self.gen,
                            device=self.device, dtype=self.dtype) * 2 - 1
                 ) * RESET_QPOS_NOISE
        self.qpos[idx] = self.qpos0.unsqueeze(0) + noise
        # mj_forward renormalizes the free-joint quats, but a batched backend
        # may read qpos before that; normalize here so obs is never garbage.
        for a in range(self.n_agents):
            qi = self.quat_idx[a]
            q = self.qpos[idx.unsqueeze(-1), qi.unsqueeze(0)]
            self.qpos[idx.unsqueeze(-1), qi.unsqueeze(0)] = (
                q / q.norm(dim=-1, keepdim=True))
        self.qvel[idx] = 0.0
        self.ctrl[idx] = 0.0
        self.ep_step[idx] = 0
        self.ep_return[idx] = 0.0
        self.backend.forward()
        self._com_before[idx] = self._agent_com_x()[idx]

    # -- reward / termination (their three layers, as tensors) --------------
    def terms(self, a, bad=None):
        """Their `Ant.after_step` + `goal_rewards` + `_get_done`, evaluated at
        the CURRENT state against the COM latched in `self._com_before`.

        Split out of `step()` so the parity harness can drive exactly this code
        on hand-set states -- their env is driven the same way (before_step, then
        set_state in place of simulate, then after_step).
        """
        com_x = self._agent_com_x()
        forward_r = self.move_sign * (com_x - self._com_before) / CONTROL_DT
        ctrl_cost = CTRL_COST_COEF * (a.to(self.dtype) ** 2).sum(-1)
        if self.contact_cost_from_cfrc:
            f = self.cfrc_ext[:, self.body_ids].clamp(-1.0, 1.0)
            contact_cost = CONTACT_COST_COEF * (f ** 2).sum((-1, -2))
        else:
            contact_cost = torch.zeros_like(ctrl_cost)
        dense = forward_r - ctrl_cost - contact_cost + SURVIVE_BONUS

        fell = self._root_z() < STAND_Z                          # [n, A]
        # reached_goal (ant.py:96-103): crossed the goal line, COM-based.
        reached = torch.where(self.goal_x > 0, com_x > self.goal_x,
                              com_x < self.goal_x)
        n_reached = reached.sum(-1)
        one_winner = n_reached == 1
        # goal_rewards: +GOAL to the crosser, -GOAL to the other, only when
        # exactly one crossed; a simultaneous double crossing pays nobody but
        # still ends the game.
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
        """actions: [n, n_agents, 8] (or [n, n_agents*8]) in [-1, 1].

        Returns (obs, reward[n, A], done[n], info). `done` is their shared done:
        termination OR truncation, identical for both agents.
        """
        # Their `Ant.after_step(action)` charges `.5 * np.square(action).sum()`
        # on the RAW policy action; MuJoCo clamps `ctrl` to `ctrlrange` inside
        # the step, so the torque is clipped but the cost is not. Keep the two
        # separate: `a` drives the actuators, `a_raw` is billed.
        a_raw = actions.reshape(self.n, self.n_agents, self.act_dim)
        a = a_raw.clamp(-1.0, 1.0)
        # Their global ctrl layout is agent0's motors then agent1's, in the
        # merged actuator order (scene._MOTOR_JOINTS).
        self.ctrl.copy_(a.reshape(self.n, -1).to(self.ctrl.dtype))

        # before_step(): latch the COM the forward reward is measured from.
        self._com_before = self._agent_com_x().clone()
        self.backend.step()

        bad = ((~torch.isfinite(self.qpos).all(-1))
               | (~torch.isfinite(self.qvel).all(-1)))
        if bool(bad.any()):
            # `_get_done` treats a non-finite state as done. Zero it first so the
            # reward tensors stay finite, then let the done path reset the world.
            self.n_diverged += int(bad.sum().item())
            self.qpos[bad] = self.qpos0.unsqueeze(0)
            self.qvel[bad] = 0.0
            self.backend.forward()

        t = self.terms(a_raw, bad)
        reward, dense, parse = t["reward"], t["dense"], t["parse"]
        forward_r, ctrl_cost = t["forward"], t["ctrl_cost"]
        com_x, fell, winner = t["com_x"], t["fell"], t["winner"]
        terminated = t["terminated"]
        self.ep_step += 1
        truncated = self.ep_step >= self.max_episode_steps
        done = terminated | truncated
        self.ep_return += reward

        if bool(done.any()):
            self.last_return = torch.where(done.unsqueeze(-1), self.ep_return,
                                           self.last_return)
            self.last_len = torch.where(done, self.ep_step, self.last_len)
            self.last_win = torch.where(done.unsqueeze(-1), winner.to(self.dtype),
                                        self.last_win)
            # Their win rate (sample_worker:284-299): every finished episode
            # counts in the denominator, including truncated draws.
            self.games += int(done.sum().item())
            self.wins += winner[done].sum(0).cpu().numpy()
            if self.auto_reset:
                self.reset_idx(done.nonzero(as_tuple=True)[0])

        info = {"dense": dense, "parse": parse, "terminated": terminated,
                "truncated": truncated, "winner": winner,
                "forward": forward_r, "ctrl_cost": ctrl_cost,
                "com_x": com_x, "fell": fell}
        return self.obs(), reward, done, info

    # -- diagnostics --------------------------------------------------------
    def win_rate(self):
        if self.games == 0:
            return np.zeros(self.n_agents)
        return self.wins / self.games

    def reset_win_stats(self):
        self.games = 0
        self.wins = np.zeros(self.n_agents)


def set_state(env, qpos, qvel):
    """Write a hand-set state into every world and refresh derived quantities.
    Mirrors gymnasium `MujocoEnv.set_state` (which runs mj_forward), so the
    parity harness drives both stacks through the same door."""
    env.qpos.copy_(torch.as_tensor(np.asarray(qpos), device=env.device,
                                   dtype=env.dtype).expand_as(env.qpos))
    env.qvel.copy_(torch.as_tensor(np.asarray(qvel), device=env.device,
                                   dtype=env.dtype).expand_as(env.qvel))
    env.backend.forward()
