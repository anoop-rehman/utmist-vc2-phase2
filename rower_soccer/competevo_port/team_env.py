"""2v2 run-to-goal: the termination, downed-player and win-condition rules.

This is the research artefact for unit 2f, not the trainer. Every rule the
design doc argues about is a flag here, so the doc's numbers come from running
the same env four ways rather than from four forks of it.

--------------------------------------------------------------------------
Why any of this is a question
--------------------------------------------------------------------------
Their `MultiDevAgentEnv._get_done` is

    done = np.any(dones)                       # dones[i] = agent i fell over
    done = game_done or not finite or done

and our 1v1 port copies it (`dev_env.terms`: `fell.any(-1) | game_done | bad`).
At two agents "any agent fell" and "the match is over" are close enough to the
same thing that the choice is invisible. At four they are not: the intended
2v2 behaviour is that a team can flip an opponent over and walk past it, and
under `np.any` flipping an opponent ENDS THE GAME instead of removing a player
from it -- so the single most interesting strategy on the list is not merely
unrewarded, it is unrepresentable.

--------------------------------------------------------------------------
`down_rule` -- what happens to an agent whose torso leaves [0.28, 1.2]
--------------------------------------------------------------------------
"any"        their rule, extended naively. Episode terminates. The control.
"ignore"     falls do nothing at all. Also a control: it upper-bounds how long
             episodes get if nothing ever ends them early.
"frozen"     the agent is OUT for the rest of the episode. Its torque is zeroed
             (`_mask_motors`), it stops collecting the survive bonus and the
             forward reward, it can no longer score -- and its body stays in
             the scene as a collidable obstacle. Episode continues.
"recover"    "frozen" for `recover_steps`, then the agent is re-posed upright
             in place (x, y kept, z = 0.75, identity yaw-only orientation,
             qvel zeroed) and resumes. An ant cannot right itself, so a
             recovery rule without a re-pose is the same as "frozen" with extra
             steps; this is the honest version of "can recover".
"team_down"  "frozen", plus: a team whose BOTH members are down loses
             immediately (the other team is paid the goal reward).

--------------------------------------------------------------------------
`win_rule` -- the 4-agent analogue of `goal_rewards`
--------------------------------------------------------------------------
Theirs pays +/-GOAL_REWARD only when EXACTLY ONE agent has reached
(`multi_dev_agent_env.py:218-234`); `num_reached_goal != 1` returns all zeros
AND `game_done = num_reached > 0`, so a simultaneous double-crossing ends the
episode paying nobody. That is a draw rule and it does not survive four agents,
because two TEAMMATES crossing on the same step is a win, not a draw.

"exactly_one"  the naive port. Kept because it is the thing to argue against.
"team_first"   a team scores when any of its members crosses. If both teams
               cross on the same step, nobody is paid (their draw rule, lifted
               to teams). RECOMMENDED.

`goal_credit` decides what the non-crossing teammate is paid under
"team_first": "team" (both members get +GOAL, both opponents -GOAL),
"scorer" (only the crosser is paid, teammate 0), "split" (+/-GOAL/2 each).
"team" is the only one of the three that makes blocking-and-walking-past
rational for the blocker, and it is also the one that doubles the sparse term
against the dense term at the TEAM level -- see the doc.
"""

import numpy as np
import torch

from rower_soccer.competevo_port.dev_env import STAND_Z_MAX, RunToGoalDevEnv
from rower_soccer.competevo_port.run_to_goal_env import (CONTACT_COST_COEF,
                                                         CTRL_COST_COEF,
                                                         GOAL_REWARD,
                                                         MOVE_REWARD_WEIGHT,
                                                         STAND_Z,
                                                         SURVIVE_BONUS)
from rower_soccer.competevo_port.scene import CONTROL_DT
from rower_soccer.competevo_port.team_scene import (SPAWN_Z,
                                                    build_dev_team_scene)

DOWN_RULES = ("any", "ignore", "frozen", "recover", "team_down")
WIN_RULES = ("exactly_one", "team_first")
GOAL_CREDITS = ("team", "scorer", "split")


class TeamRunToGoalDevEnv(RunToGoalDevEnv):
    """`RunToGoalDevEnv` with N agents in two teams and pluggable end-of-life.

    Shapes: obs `[n, A, 56]`, action `[n, A, 28]`, reward `[n, A]`, done `[n]`.
    At `n_agents=2` with `down_rule="any"`, `win_rule="exactly_one"` this is the
    1v1 env up to the observation ordering, which is identical for two agents
    (the single "other" is the opponent either way).
    """

    def __init__(self, *args, down_rule="frozen", win_rule="team_first",
                 goal_credit="team", recover_steps=50, scene_kwargs=None,
                 **kw):
        assert down_rule in DOWN_RULES, down_rule
        assert win_rule in WIN_RULES, win_rule
        assert goal_credit in GOAL_CREDITS, goal_credit
        self.down_rule, self.win_rule = down_rule, win_rule
        self.goal_credit, self.recover_steps = goal_credit, int(recover_steps)
        self._scene_kwargs = dict(scene_kwargs or {})
        super().__init__(*args, scene_kwargs=self._scene_kwargs, **kw)
        d = self.device
        A = self.n_agents
        self.team = torch.tensor(self.meta.team, device=d, dtype=torch.long)
        # [A, T] one-hot, so "did my team score" is a matmul, not a loop.
        self.n_teams = int(self.team.max().item()) + 1
        self.team_onehot = torch.zeros(A, self.n_teams, device=d,
                                       dtype=self.dtype)
        self.team_onehot[torch.arange(A, device=d), self.team] = 1.0
        self.down = torch.zeros(self.n, A, device=d, dtype=torch.bool)
        self.down_for = torch.zeros(self.n, A, device=d, dtype=torch.long)
        self.n_recoveries = 0
        # Per-episode diagnostics the probes read.
        self.last_down = torch.zeros(self.n, A, device=d, dtype=self.dtype)
        self.last_end = torch.zeros(self.n, device=d, dtype=torch.long)
        self._respawn_pending = None

    def _build_scene(self, **kw):
        return build_dev_team_scene(**kw)

    # -- downed-agent bookkeeping -------------------------------------------
    def _mask_motors(self, motor_eff):
        if self.down_rule in ("any", "ignore"):
            return motor_eff
        return torch.where(self.down.unsqueeze(-1),
                           torch.zeros_like(motor_eff), motor_eff)

    def reset_idx(self, idx):
        super().reset_idx(idx)
        if getattr(self, "down", None) is not None and idx.numel():
            self.down[idx] = False
            self.down_for[idx] = 0

    def _repose(self, world_idx, agent_idx):
        """Stand an agent back up where it fell: keep (x, y), set z = 0.75, drop
        the roll/pitch (keep the spawn yaw), zero every joint and every
        velocity. This is a teleport, and it is the only way "recover" means
        anything for an ant."""
        qi = self.qpos_idx[agent_idx]          # [k, 15]
        vi = self.qvel_idx[agent_idx]
        w = world_idx.unsqueeze(-1)
        q = self.qpos[w, qi]
        q0 = self.qpos0[qi]
        q[..., 2] = SPAWN_Z
        q[..., 3:] = q0[..., 3:]               # spawn quat + zero joint angles
        self.qpos[w, qi] = q
        self.qvel[w, vi] = 0.0

    # -- reward / termination -----------------------------------------------
    def terms(self, a, bad=None):
        com_x = self._agent_com_x()
        z = self._root_z()
        fell_now = (z < STAND_Z) | (z > STAND_Z_MAX)

        if self.down_rule == "any":
            down = fell_now
        elif self.down_rule == "ignore":
            down = torch.zeros_like(fell_now)
        else:
            down = self.down | fell_now

        alive = (~down).to(self.dtype)

        forward_r = self.move_sign * (com_x - self._com_before) / CONTROL_DT
        ctrl_cost = CTRL_COST_COEF * (a.to(self.dtype) ** 2).sum(-1)
        if self.contact_cost_from_cfrc:
            f = (self.cfrc_ext[:, self.body_ids].clamp(-1.0, 1.0)
                 * self.body_ids_mask.unsqueeze(-1))
            contact_cost = CONTACT_COST_COEF * (f ** 2).sum((-1, -2))
        else:
            contact_cost = torch.zeros_like(ctrl_cost)
        dense = forward_r - ctrl_cost - contact_cost + SURVIVE_BONUS
        # A downed agent earns nothing: no survive bonus, no forward reward, and
        # no control cost (its torque was zeroed). Without this, "frozen" pays a
        # corpse +1/step for 500 steps and lying down becomes a strategy.
        if self.down_rule not in ("any", "ignore"):
            dense = dense * alive

        reached = torch.where(self.goal_x > 0, com_x > self.goal_x,
                              com_x < self.goal_x)
        if self.down_rule not in ("any", "ignore"):
            reached = reached & (~down)

        if self.win_rule == "exactly_one":
            n_reached = reached.sum(-1)
            one = n_reached == 1
            parse = torch.where(one.unsqueeze(-1),
                                torch.where(reached, GOAL_REWARD, -GOAL_REWARD),
                                torch.zeros_like(dense))
            game_done = n_reached > 0
            winner = reached & one.unsqueeze(-1)
        else:                                            # "team_first"
            # [n, T] -- did team t have a member cross this step
            t_reached = (reached.to(self.dtype) @ self.team_onehot) > 0
            n_teams_reached = t_reached.sum(-1)
            one_team = n_teams_reached == 1
            # [n, A]: my team scored / my team conceded
            mine = t_reached.to(self.dtype) @ self.team_onehot.T > 0
            pay = torch.where(mine, GOAL_REWARD, -GOAL_REWARD)
            if self.goal_credit == "scorer":
                pay = torch.where(mine & (~reached), torch.zeros_like(pay), pay)
            elif self.goal_credit == "split":
                pay = pay * 0.5
            parse = torch.where(one_team.unsqueeze(-1), pay,
                                torch.zeros_like(dense))
            game_done = n_teams_reached > 0
            winner = mine & one_team.unsqueeze(-1)

        if bad is None:
            bad = torch.zeros(self.n, device=self.device, dtype=torch.bool)

        wiped = torch.zeros(self.n, self.n_teams, device=self.device,
                            dtype=torch.bool)
        if self.down_rule == "any":
            terminated = down.any(-1) | game_done | bad
        elif self.down_rule == "team_down":
            n_down = down.to(self.dtype) @ self.team_onehot        # [n, T]
            n_team = self.team_onehot.sum(0)                       # [T]
            wiped = n_down >= n_team
            any_wiped = wiped.any(-1)
            # Only pay the wipe-out if exactly one team is wiped, and only if
            # the goal reward did not already fire this step.
            one_wiped = wiped.sum(-1) == 1
            lose = wiped.to(self.dtype) @ self.team_onehot.T > 0
            wipe_pay = torch.where(lose, -GOAL_REWARD, GOAL_REWARD)
            fire = (one_wiped & (~game_done)).unsqueeze(-1)
            parse = torch.where(fire, wipe_pay, parse)
            winner = torch.where(fire, (~lose), winner)
            terminated = game_done | any_wiped | bad
        else:                                            # frozen / recover / ignore
            terminated = game_done | bad

        reward = parse + MOVE_REWARD_WEIGHT * dense

        # LATCH. `terms` is the only place that sees `fell_now`, and at this
        # point in `step` the pre-step stage flag is still in `self.stage`
        # (`_apply_designs` runs after), so a design step cannot latch anybody.
        # `super().step` calls `reset_idx` after this, which clears the latch
        # for worlds that just ended -- so the order is latch, then clear.
        newly = down & (~self.down) if self.down.numel() else down
        if self.down_rule not in ("any", "ignore"):
            keep = (~self.stage).unsqueeze(-1)
            self.down = down & keep
            self.down_for = torch.where(self.down, self.down_for + 1,
                                        torch.zeros_like(self.down_for))
        out = {"reward": reward, "dense": dense, "parse": parse,
               "newly_down": newly & (~self.stage).unsqueeze(-1),
                "forward": forward_r, "ctrl_cost": ctrl_cost,
                "contact_cost": contact_cost, "terminated": terminated,
               "forward": forward_r, "ctrl_cost": ctrl_cost,
               "contact_cost": contact_cost, "terminated": terminated,
               "winner": winner, "reached": reached, "fell": fell_now,
               "down": down, "wiped": wiped, "com_x": com_x, "alive": alive,
               "game_done": game_done}
        # `dev_env.step` builds its own info dict from a fixed key set, so the
        # 2v2-only fields have to travel out of band.
        self._last_terms = out
        return out

    # -- step ---------------------------------------------------------------
    def step(self, actions):
        # Recovery happens BEFORE physics, because it is a teleport: doing it
        # after would put a re-posed body into a reward that was computed from
        # its fallen pose.
        if self.down_rule == "recover":
            due = self.down & (self.down_for >= self.recover_steps)
            if bool(due.any()):
                wi, ai = due.nonzero(as_tuple=True)
                self._repose(wi, ai)
                self.backend.forward()
                self.down[due] = False
                self.down_for[due] = 0
                self.n_recoveries += int(due.sum().item())
        obs, reward, done, info = super().step(actions)
        t = self._last_terms
        for k in ("down", "wiped", "newly_down", "alive", "reached",
                  "game_done"):
            info[k] = t[k]
        down = info["down"]
        # Episode-ending classification, for the probes. 0 = still running,
        # 1 = goal, 2 = wipe-out, 3 = fall (down_rule="any"), 4 = timeout.
        if bool(done.any()):
            end = torch.zeros(self.n, device=self.device, dtype=torch.long)
            # A crossing beats a wipe-out: `terms` only fires the wipe-out
            # payout when `game_done` is false, so the classification has to
            # test the same condition rather than test `winner`, which is set
            # by both.
            end = torch.where(info["game_done"], torch.ones_like(end), end)
            if self.down_rule == "team_down":
                end = torch.where((end == 0) & info["wiped"].any(-1),
                                  torch.full_like(end, 2), end)
            if self.down_rule == "any":
                end = torch.where((end == 0) & info["down"].any(-1),
                                  torch.full_like(end, 3), end)
            end = torch.where((end == 0) & info["truncated"],
                              torch.full_like(end, 4), end)
            self.last_end = torch.where(done, end, self.last_end)
            self.last_down = torch.where(done.unsqueeze(-1),
                                         down.to(self.dtype), self.last_down)
        info["end"] = self.last_end
        return obs, reward, done, info

    def team_win_rate(self):
        if self.games == 0:
            return np.zeros(self.n_teams)
        w = self.wins.reshape(1, -1) @ self.team_onehot.cpu().numpy()
        team_sizes = self.team_onehot.sum(0).cpu().numpy()
        return (w / team_sizes).ravel() / self.games
