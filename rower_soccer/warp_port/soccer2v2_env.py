"""Batched GPU 2v2 soccer env for self-play (D1 unit 1e).

Four creatures and one ball on the scaled dm_soccer pitch, N worlds in parallel,
one shared policy driving all four players. This is the env the RL fine-tune
(unit 1f) runs in; it is deliberately NOT a drill -- there is no commanded
direction, no target marker and no per-world segment, only a match.

WHAT IT IS BATCHED OVER
-----------------------
`num_worlds` independent matches. Every observation/action/reward tensor is
FLATTENED over (world, player) with the world major:

    row = w * n_agents + k          obs [n*4, obs_dim]   act [n*4, act_dim]

so `obs.view(n, 4, -1)[w, k]` is player k of world w and a single policy can be
applied to the whole batch in one forward pass. `act_dim` is the PER-PLAYER
width (8 for the ant), not the model's 32.

SLOTS
-----
Slot order is `game/match.py`'s: (home_1, home_2, away_1, away_2). Home defends
the -x goal and attacks +x; away is the mirror. Slot k's creature carries the
MJCF prefix `p{k}-`.

OBSERVATION (proprio-first, contiguous task block -- the drills' contract)
-------------------------------------------------------------------------
    proprio(P) | ball_ego(6)
               | opp_goal_mid_ego(3) | opp_post_left_ego(2) | opp_post_right_ego(2)
               | own_goal_mid_ego(3)
               | teammate_ego(3) | teammate_vel_ego(3)
               | opp_a_ego(3) | opp_a_vel_ego(3)
               | opp_b_ego(3) | opp_b_vel_ego(3)          = P + 34   (99 for the ant)

  * proprio is BYTE-IDENTICAL to follow/dribble/kick/shoot: it is not a
    re-implementation but a call into `worm_env_base.proprio_obs`, the same
    function the drills call, with this player's index bundle. Proprio is the
    frozen decoder's entire input contract; a permuted copy here would train
    perfectly well and transfer nothing. tests/test_soccer2v2.py measures the
    identity rather than trusting this paragraph.
  * The first 13 task entries are `shoot`'s task block verbatim (ball_ego6 +
    goal_mid3 + post_left2 + post_right2, with "the goal" = the goal this team
    attacks), so a shoot checkpoint's task encoder is a meaningful warm start
    and not just its decoder.
  * ball_ego = ego position(3) + ego linear velocity(3), the same block every
    drill uses and the same shape the 2v2 game hands over
    (`ball_ego_position` + `ball_ego_linear_velocity`).
  * The other three players are ego position(3) + ego world-frame linear
    velocity(3) each, teammate first, then the two opponents in slot order.

TEAM SYMMETRY (why self-play is cheap here)
-------------------------------------------
The pitch is symmetric under a 180-degree rotation about z, M(x, y) = (-x, -y),
which swaps the two goals. That is a PROPER rotation, not a reflection: a
reflection would swap the ant's left and right legs and the mirrored state would
not be reachable by the same body.

Every task entry above is egocentric, and a global rotation rotates each root
frame with the world -- so the ego numbers are unchanged by M. Team symmetry
therefore costs nothing at all in the obs code: it is entirely carried by WHICH
goal is `opp_goal` and WHICH players are teammate/opponents. Concretely, for the
mirrored state M(s) with the home and away slots exchanged,

    obs[M(s)][mirror_slot(k)] == obs[s][k]      (exactly, up to fp)
    the action for slot k is the action for mirror_slot(k), unchanged

because actuator torques live in the creature's own frame. One policy plays both
teams, and a self-play match is symmetric by construction.

RULES (`game/match.py` / dm_soccer, transcribed)
------------------------------------------------
`match.py` is the authority and it reads dm_control's own detectors, so the
rules here are dm_control's, evaluated analytically on this scene's geometry:

  * GOAL: the ball's centre strictly inside the goal box --
    `goal_x < |x| < pitch_half_x`, `|y| < goal_half_width`, `0 < z < goal_height`
    (dm_control `PositionDetector._is_in_zone`, strict on both sides, applied to
    the ball GEOM's xpos). Counted on a RISING EDGE, per world, exactly as
    `MatchSim._detect_goal` does -- the state persists for more than one control
    step, so a level read double-counts. A goal in the -x (home) goal scores for
    AWAY, matching `Pitch.detected_goal`.
  * AFTER A GOAL: `MultiturnTask` re-runs the initializer and does NOT terminate
    (`match.py` passes `terminate_on_goal=False`). So does this env: the scoring
    world alone is re-spawned, in place, and the match clock keeps running.
  * OUT OF PLAY: THERE IS NONE, AND THERE IS NO THROW-IN. The ball BOUNCES OFF
    the pitch boundary instead. This is dm_control's own alternative --
    `Pitch(field_box=True)`, surfaced as `soccer.load(..., enable_field_box=True)`
    -- and it is what the DeepMind 2021 football paper specifies:

        "To emulate the football rules, the players can travel outside of the
         boundaries of the pitch (but cannot travel outside of the gradient-
         coloured physical hoardings), whereas the ball 'bounces off' of the
         pitch boundary. This simplification removes the need for a throw-in
         mechanism, and leaves the physics simulation to determine the range of
         strategies that players can execute (including deliberately bouncing
         the ball off the pitch boundary)."

    So there are TWO boundaries here and they are DIFFERENT SURFACES:

        field box  |x| = field_half_x, |y| = field_half_y   BALL ONLY
        wall_*     |x| = pitch_half_x, |y| = pitch_half_y   EVERYTHING

    The field box is 8 box geoms built by `scene.fieldbox_pos_size`, a
    transcription of `pitch._fieldbox_pos_size`, carrying dm_control's
    `_FIELD_BOX_CONTACT_BIT` so they collide with the ball and with nothing
    else. Players run straight through them and are stopped only by the
    hoardings 1.67 m further out, which is the paper's "players can travel
    outside of the boundaries of the pitch". Holes are left at the goal mouths
    (|y| < goal_half_width, z < goal_height) so the ball still reaches the goal
    detector: the boundary's x plane and the goal line are the SAME plane, so a
    ball crossing the goal line either scores through the mouth or bounces off
    the rest of the line, which is the football rule rather than an
    approximation of it.

    dm_control disables its throw-in the same way -- `Pitch.register_ball` only
    calls `self._field.register_entities(ball)` in the NON-field-box branch, so
    with a field box the out-of-play detector never sees the ball.

    `detected_off_court` is kept as a DIAGNOSTIC (it is what the gate reads to
    prove the ball stays in), and `throw_ins` is kept and stays 0 forever so the
    trainer's metric schema is unchanged and a regression shows up as a number
    moving rather than as a key disappearing.
  * TIME LIMIT: `match_seconds` (45 s in the game). This is the one thing that
    ends an episode, and it ends every world at once -- the drills' trainer keys
    on a single `done` bool for the whole batch and this env keeps that contract.

WHERE THIS SCENE AND THE CPU GAME DISAGREE (measured, not guessed)
------------------------------------------------------------------
The warp pitch scales dm_soccer's 96 x 72 m pitch UNIFORMLY (`pitch_scale`
0.3125 -> 30 x 22.5 m), so the goal scales with it: goal line at |x| = 13.33,
half-width 3.71, crossbar at 1.67 m. `match.py` instead builds a dm_control
`RandomizedPitch` at `pitch_half=(15, 11)` and lets dm_control size the goal --
and dm_control's goal DEPTH and HEIGHT are the absolute constant `_SIDE_WIDTH/2
= 2.667`, not a ratio, so that pitch's goal line lands at |x| = 9.67 with a
5.33 m high crossbar. Only the half-width rule (0.33 * pitch_half_y) agrees.

That is a property of the two pitches, not of this env: the drills already train
`shoot` against the scaled goal at 13.33. The rules implemented here are
dm_control's rules; the geometry is this scene's. The gate pins that claim by
running dm_control's own `Pitch` at THIS geometry (which it will accept as an
explicit `goal_size`) and comparing detector-for-detector. Reconciling the two
pitches is a separate decision and is recorded as such in
docs/PLAN_D1_ANT_PIPELINE.md.
"""

import numpy as np
import torch

from rower_soccer.warp_port.backend import CpuBackend, WarpBackend
from rower_soccer.warp_port.scene import (BallSpec, build_soccer_scene,
                                          goal_geometry)
from rower_soccer.warp_port.worm_env_base import (CONTROL_DT, SUBSTEPS,
                                                  proprio_index, proprio_obs,
                                                  to_ego3, vec_to_ego3)

# dm_control soccer constants, transcribed. See the module docstring.
DM_SPAWN_RATIO = 0.6             # initializers._SPAWN_RATIO
# The throw-in constants (_THROW_IN_BALL_Z, _throw_in's U[0.7, 0.9] shrink) are
# GONE, not merely unused: the ball bounces now. See the RULES section.
#
# How far past the field line the ball's CENTRE may be before the escape guard
# calls it out. The ball legitimately rests up to one radius past the line (the
# contact is surface-to-surface, not centre-to-plane) and legitimately goes far
# past it in x inside the goal mouth, so the guard is applied per axis and only
# outside the mouth. This is the slack it allows on top of the radius.
BALL_ESCAPE_MARGIN = 0.20


def drill_ball():
    """The ball every v3+ ant drill and `match.py` play on: dm_soccer's
    SoccerBall at radius 0.15 m. The stock 0.35 ball is a different task."""
    return BallSpec(radius=0.15, mass=0.045)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
class SoccerReward:
    """dm_soccer's per-player reward, plus OPTIONAL shaping that defaults off.

    The unshaped term is `soccer.task.Task.get_reward` verbatim: +1 to every
    player on the scoring team, -1 to every player on the conceding team, 0
    otherwise. That is what the game pays and what a policy fine-tuned here is
    ultimately judged on, so it is the default and nothing is added to it
    silently.

    The shaping knobs exist because sparse goals are not explorable from
    scratch; unit 1f anchors to a BC prior instead, and can turn these on if it
    needs to. Both are multiplied by `env.shaping_scale` so a trainer can anneal
    them to zero, exactly as the drills do:

      * w_player_to_ball: own speed toward the ball, clipped at 0.
      * w_ball_to_goal:   potential-based progress of the BALL toward the goal
        this team attacks (Ng et al. 1999), masked in any world that was
        re-spawned this step -- a kickoff teleports the ball and an unmasked
        potential reads that as several metres of free progress.
    """

    def __init__(self, w_goal=1.0, w_player_to_ball=0.0, w_ball_to_goal=0.0):
        self.w_goal = w_goal
        self.w_p2b = w_player_to_ball
        self.w_b2g = w_ball_to_goal
        self.prev_bg = None

    def bind(self, env):
        pass

    def reset(self, env):
        self.prev_bg = self._ball_to_goal(env)

    @staticmethod
    def _ball_to_goal(env):
        """[n, 2] distance from the ball to each TEAM's attacking goal MOUTH
        RECTANGLE (0 anywhere between the posts), not to its centre point --
        `shoot._goal_mouth_dist`'s reasoning applies unchanged: this goal is
        7.4 m wide and a point distance would call most of it a miss."""
        ball = env.ball_xy()
        d = []
        for team in range(2):
            s = env.attack_sign[team]
            dx = (s * (env.attack_x[team] - ball[:, 0])).clamp(min=0.0)
            dy = (ball[:, 1].abs() - env.goal_half_width).clamp(min=0.0)
            d.append(torch.sqrt(dx * dx + dy * dy + 1e-12))
        return torch.stack(d, -1)

    def __call__(self, env):
        # [n, n_agents]; flattened to [n*n_agents] on return.
        r = self.w_goal * env.goal_reward
        if self.w_b2g:
            bg = self._ball_to_goal(env)
            prog = self.prev_bg - bg
            prog = torch.where(env.world_reset.unsqueeze(-1),
                               torch.zeros_like(prog), prog)
            self.prev_bg = bg.detach()
            r = r + env.shaping_scale * self.w_b2g * prog[:, env.team]
        if self.w_p2b:
            r = r + env.shaping_scale * self.w_p2b * env.speed_to_ball()
        return r.reshape(-1)

    def fitness(self, env):
        """Per-player goal difference so far this episode. Zero-sum across the
        four players by construction, which is the honest thing for self-play:
        a single scalar cannot rank a policy against itself. Unit 1f selects on
        win rate against a FIXED opponent pool; this is the per-episode signal
        that feeds it."""
        diff = env.score[:, 0] - env.score[:, 1]
        return torch.where(env.team.unsqueeze(0) == 0,
                           diff.unsqueeze(-1), -diff.unsqueeze(-1)).reshape(-1)


# ---------------------------------------------------------------------------
# The env
# ---------------------------------------------------------------------------
class WarpSoccer2v2Env:
    """N parallel 2v2 matches. See the module docstring for the contracts."""

    def __init__(self, num_worlds=64, creature_xml="creature_configs/ant.xml",
                 n_per_team=2, match_seconds=45.0, pitch_scale=0.3125,
                 ball: BallSpec = None, use_gpu=True, device=None, seed=0,
                 # 4x the drills' 64/512: four creatures per world instead of
                 # one. Measured over a random-torque rollout the peak is ~4
                 # contacts / 33 constraints per world, so this is margin, not
                 # a fit -- but mujoco_warp silently DROPS constraints on
                 # overflow and NaNs the sim, so the margin is the point.
                 use_graph=True, nconmax=256, njmax=2048, reward=None,
                 backend_cls=None, energy_coef=0.0, smooth_coef=0.0,
                 rew_clip=(-10.0, 10.0), spawn="mirror", ball_jitter=0.0,
                 spawn_ratio=DM_SPAWN_RATIO, min_separation=2.0,
                 goal_respawn=True):
        self.n = num_worlds
        self.n_per_team = n_per_team
        self.n_agents = 2 * n_per_team
        self._pitch_scale = pitch_scale
        self._spawn = spawn
        self._spawn_ratio = spawn_ratio
        # MultiturnTask re-spawns on a goal; turning it off leaves the world
        # exactly where it was, which is what makes the rising-edge latch
        # observable (and what a "score as many as you can" variant would want).
        self._goal_respawn = goal_respawn
        self._ball_jitter = ball_jitter
        self._min_sep = min_separation
        self.episode_steps = int(round(match_seconds / CONTROL_DT))
        self.match_seconds = match_seconds
        self.energy_coef, self.smooth_coef, self.rew_clip = (energy_coef,
                                                             smooth_coef,
                                                             rew_clip)
        self.shaping_scale = 1.0
        self.n_diverged = 0

        if backend_cls is None:
            backend_cls = WarpBackend if use_gpu else CpuBackend
        if device is None:
            device = "cuda" if use_gpu else "cpu"
        if not use_gpu:
            use_graph = False

        # 1. scene: 4 creatures + ball on the scaled pitch, both goals.
        self._ball_spec = ball or drill_ball()
        self.model, self.metas, self.prefixes = build_soccer_scene(
            creature_xml, n_players=self.n_agents, ball=self._ball_spec,
            pitch_scale=pitch_scale)

        # 2. backend (owns every mjw./wp. call).
        self.backend = backend_cls(self.model, num_worlds, SUBSTEPS,
                                   use_graph=use_graph, nconmax=nconmax,
                                   njmax=njmax, device=device)
        self.device = self.backend.device
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.qpos, self.qvel, self.ctrl = (self.backend.qpos, self.backend.qvel,
                                           self.backend.ctrl)
        self.xpos, self.xmat = self.backend.xpos, self.backend.xmat
        self.sensordata = self.backend.sensordata

        # 3. per-player index plumbing. proprio_index is the drills' own.
        self.pidx = [proprio_index(m, self.device) for m in self.metas]
        m0 = self.metas[0]
        self.bq, self.bv = m0.ball_qpos, m0.ball_qvel
        self.ball_radius = m0.ball_radius
        self.spawn_z = m0.spawn_z
        self.qpos_root = [m.qpos_root for m in self.metas]
        self.qvel_root = [m.qvel_root for m in self.metas]
        self.root_body = [m.root_body for m in self.metas]
        # Creature ball-joint quaternion `w` slots (empty for hinge-only bodies
        # like the ant/worm): zeroing them and calling forward() normalises 0/0.
        self.ball_qw_idx = torch.as_tensor(
            [s for m in self.metas for s, nq in m.joint_qpos if nq == 4],
            device=self.device, dtype=torch.long)

        # 4. teams + geometry. Home (slots < n_per_team) defends -x.
        self.team = torch.tensor([0] * n_per_team + [1] * n_per_team,
                                 device=self.device, dtype=torch.long)
        self.goal_x, self.goal_half_width, self.goal_height = goal_geometry(
            pitch_scale)
        self.pitch_half = (48.0 * pitch_scale, 36.0 * pitch_scale)
        # dm_control's `field`: the pitch inset by the goal box depth on every
        # side, so its x half-extent IS the goal line. Derived from goal_x
        # rather than re-multiplying the ratio -- at 1e-5 apart the two would
        # leave a sliver that is out of play but not yet past the goal line.
        goal_depth = self.pitch_half[0] - self.goal_x
        self.field_half = (self.goal_x, self.pitch_half[1] - goal_depth)
        # attack_x[t] = the x of the goal team t shoots at; +1 home, -1 away.
        self.attack_sign = torch.tensor([1.0, -1.0], device=self.device)
        self.attack_x = self.attack_sign * self.goal_x
        # Per-player world-frame goal geometry, precomputed [n_agents, ...].
        s = self.attack_sign[self.team]                       # [A]
        z = torch.zeros_like(s)
        h = torch.full_like(s, self.goal_height / 2.0)
        gx = s * self.goal_x
        gw = s * self.goal_half_width
        self.opp_goal_mid = torch.stack([gx, z, h], -1)       # [A, 3]
        self.own_goal_mid = torch.stack([-gx, z, h], -1)
        # Posts named from the ATTACKER's own frame, so home's left post maps
        # onto away's left post under the 180-degree mirror.
        self.opp_post_left = torch.stack([gx, gw], -1)        # [A, 2]
        self.opp_post_right = torch.stack([gx, -gw], -1)
        # Teammate / opponent slot tables, in slot order within each team.
        self.mate = torch.tensor(
            [(k // n_per_team) * n_per_team + (k + 1) % n_per_team
             for k in range(self.n_agents)], device=self.device)
        self.opps = torch.tensor(
            [[(1 - k // n_per_team) * n_per_team + j
              for j in range(n_per_team)] for k in range(self.n_agents)],
            device=self.device)
        # Slot k's mirror image: home_i <-> away_i.
        self.mirror_slot = torch.tensor(
            [(k + n_per_team) % self.n_agents for k in range(self.n_agents)],
            device=self.device)

        # 5. dims.
        n_proprio = self.pidx[0].width
        self.n_proprio = n_proprio
        self.task_dim = 6 + 3 + 2 + 2 + 3 + 6 * (self.n_agents - 1)
        self.obs_dim = n_proprio + self.task_dim
        self.act_dim = self.metas[0].nu
        self.proprio_indices = np.arange(0, n_proprio)
        self.task_indices = np.arange(n_proprio, self.obs_dim)
        self.prev_ctrl = torch.zeros(self.n * self.n_agents, self.act_dim,
                                     device=self.device)

        # 6. match state.
        self.score = torch.zeros(self.n, 2, device=self.device)
        self.goal_latch = torch.zeros(self.n, dtype=torch.bool,
                                      device=self.device)
        self.scored_now = torch.zeros(self.n, dtype=torch.long,
                                      device=self.device)   # 0 none, 1 home, 2 away
        self.goal_reward = torch.zeros(self.n, self.n_agents, device=self.device)
        self.world_reset = torch.zeros(self.n, dtype=torch.bool,
                                       device=self.device)
        # Kept at 0 forever so the trainer's metric schema does not change and a
        # regression to the throw-in shows as a number moving, not a key vanishing.
        self.throw_ins = torch.zeros(self.n, device=self.device)
        self.ball_escapes = torch.zeros(self.n, device=self.device)
        self._ball_vcols = torch.arange(self.bv, self.bv + 6,
                                        device=self.device).unsqueeze(0)
        self.t = 0

        self.reward = reward if reward is not None else SoccerReward()
        self.reward.bind(self)

    # -- state accessors ----------------------------------------------------
    def root_frames(self, k):
        rb = self.root_body[k]
        return self.xpos[:, rb, :], self.xmat[:, rb]

    def root_vel(self, k):
        qr = self.qvel_root[k]
        return self.qvel[:, qr:qr + 3]

    def root_xy(self, k):
        return self.xpos[:, self.root_body[k], :2]

    def ball_xyz(self):
        return self.qpos[:, self.bq:self.bq + 3]

    def ball_xy(self):
        return self.qpos[:, self.bq:self.bq + 2]

    def ball_vel_xyz(self):
        return self.qvel[:, self.bv:self.bv + 3]

    def ball_vel_xy(self):
        return self.qvel[:, self.bv:self.bv + 2]

    def speed_to_ball(self):
        """[n, A] each player's speed along its own direction to the ball."""
        ball = self.ball_xy()
        out = []
        for k in range(self.n_agents):
            d = ball - self.root_xy(k)
            u = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp(min=1e-6)
            out.append((self.root_vel(k)[:, :2] * u).sum(-1).clamp(min=0.0))
        return torch.stack(out, -1)

    def upright(self):
        """[n, A] 1 = torso z-axis up, 0 = inverted (dm_control's mapping)."""
        return torch.stack([((1.0 + self.root_frames(k)[1][:, 2, 2]) / 2.0)
                            .clamp(0.0, 1.0) for k in range(self.n_agents)], -1)

    # -- observation --------------------------------------------------------
    def _player_obs(self, k):
        """Player k's full observation, [n, obs_dim]."""
        pos, rot = self.root_frames(k)
        pro = proprio_obs(self.qpos, self.qvel, self.xpos, self.xmat,
                          self.sensordata, self.pidx[k])

        def pt3(v3):        # a fixed world POINT [3] -> ego [n, 3]
            return to_ego3(pos, rot, v3.unsqueeze(0).expand(self.n, 3))

        def pt2(v2):        # a fixed world ground point [2] -> ego xy [n, 2]
            fwd, left = rot[:, :2, 0], rot[:, :2, 1]
            d = v2.unsqueeze(0) - pos[:, :2]
            return torch.stack([(d * fwd).sum(-1), (d * left).sum(-1)], -1)

        blocks = [to_ego3(pos, rot, self.ball_xyz()),
                  vec_to_ego3(rot, self.ball_vel_xyz()),
                  pt3(self.opp_goal_mid[k]),
                  pt2(self.opp_post_left[k]),
                  pt2(self.opp_post_right[k]),
                  pt3(self.own_goal_mid[k])]
        # Team-mates first, then the opponents in slot order. `n_per_team == 1`
        # (a 1v1 variant) has no team-mate and the block simply shortens --
        # task_dim is 6 * (n_agents - 1) either way.
        others = ([int(self.mate[k])] if self.n_per_team > 1 else []) \
            + [int(j) for j in self.opps[k]]
        for j in others:
            blocks.append(to_ego3(pos, rot, self.xpos[:, self.root_body[j], :]))
            blocks.append(vec_to_ego3(rot, self.root_vel(j)))
        return torch.cat([pro] + blocks, -1)

    def obs(self):
        """[n * n_agents, obs_dim], world-major (row = w * n_agents + k)."""
        return torch.stack([self._player_obs(k) for k in range(self.n_agents)],
                           1).reshape(self.n * self.n_agents, self.obs_dim)

    # -- spawning -----------------------------------------------------------
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _write_root(self, k, idx, xy, yaw):
        qr = self.qpos_root[k]
        self.qpos[idx, qr + 0] = xy[:, 0]
        self.qpos[idx, qr + 1] = xy[:, 1]
        self.qpos[idx, qr + 2] = self.spawn_z
        self.qpos[idx, qr + 3] = torch.cos(yaw / 2)
        self.qpos[idx, qr + 4] = 0.0
        self.qpos[idx, qr + 5] = 0.0
        self.qpos[idx, qr + 6] = torch.sin(yaw / 2)

    def _write_ball(self, idx, xy, z=None):
        self.qpos[idx, self.bq + 0] = xy[:, 0]
        self.qpos[idx, self.bq + 1] = xy[:, 1]
        self.qpos[idx, self.bq + 2] = self.ball_radius if z is None else z
        self.qpos[idx, self.bq + 3] = 1.0
        self.qpos[idx, self.bq + 4] = 0.0
        self.qpos[idx, self.bq + 5] = 0.0
        self.qpos[idx, self.bq + 6] = 0.0
        self.qvel[idx.unsqueeze(-1), self._ball_vcols] = 0.0

    def _mirror_spawn(self, m):
        """Home positions/yaws for `m` worlds; away is their exact mirror.

        Home player i takes its own third of the half and its own flank, so two
        team-mates never spawn inside each other, and the away team is
        M(x, y) = (-x, -y) with yaw + pi. The whole kickoff state is therefore
        invariant under the mirror that swaps the teams: neither side starts
        with an advantage, which is what makes a self-play win rate mean
        something.
        """
        fx, fy = self.field_half
        xy, yaw = [], []
        for i in range(self.n_per_team):
            lo = -0.85 + 0.75 * i / max(1, self.n_per_team)
            hi = -0.85 + 0.75 * (i + 1) / max(1, self.n_per_team)
            x = (lo + (hi - lo) * self._rand(m)) * fx
            flank = 1.0 if i % 2 == 0 else -1.0
            y = flank * (0.15 + 0.45 * self._rand(m)) * fy
            xy.append(torch.stack([x, y], -1))
            yaw.append(self._rand(m) * (2 * np.pi))
        return xy, yaw

    def _uniform_spawn(self, m):
        """dm_soccer's `UniformInitializer`: every player uniform in
        +/- spawn_ratio * pitch_half with a uniform yaw, re-drawn while any two
        are closer than `min_separation` (dm_control's own collision-avoidance
        retry, vectorised and bounded)."""
        px, py = self.pitch_half
        bound = torch.tensor([px, py], device=self.device) * self._spawn_ratio
        xy = [(self._rand(m, 2) * 2 - 1) * bound for _ in range(self.n_agents)]
        yaw = [self._rand(m) * (2 * np.pi) for _ in range(self.n_agents)]
        for _ in range(8):
            bad = torch.zeros(m, dtype=torch.bool, device=self.device)
            for a in range(self.n_agents):
                for b in range(a + 1, self.n_agents):
                    bad |= torch.linalg.norm(xy[a] - xy[b], dim=-1) < self._min_sep
            if not bool(bad.any()):
                break
            for a in range(self.n_agents):
                fresh = (self._rand(m, 2) * 2 - 1) * bound
                xy[a] = torch.where(bad.unsqueeze(-1), fresh, xy[a])
        return xy, yaw

    def _spawn_worlds(self, idx):
        """Kick off (or re-kick-off) the worlds in `idx`. Full per-world reset:
        joints, velocities and the ball, not just the root poses -- a creature
        teleported mid-stride keeps a stride it can no longer use."""
        m = int(idx.numel())
        if m == 0:
            return
        self.qpos[idx] = 0.0
        self.qvel[idx] = 0.0
        if self._spawn == "uniform":
            xy, yaw = self._uniform_spawn(m)
        else:
            home_xy, home_yaw = self._mirror_spawn(m)
            xy = home_xy + [-p for p in home_xy]
            yaw = home_yaw + [y + np.pi for y in home_yaw]
        for k in range(self.n_agents):
            self._write_root(k, idx, xy[k], yaw[k])
        if self.ball_qw_idx.numel():
            self.qpos[idx.unsqueeze(-1), self.ball_qw_idx.unsqueeze(0)] = 1.0
        ball_xy = torch.zeros(m, 2, device=self.device)
        if self._ball_jitter:
            ball_xy = (self._rand(m, 2) * 2 - 1) * self._ball_jitter
        self._write_ball(idx, ball_xy)
        self.world_reset[idx] = True

    # -- rules --------------------------------------------------------------
    def _in_goal_box(self, sign):
        """dm_control `PositionDetector._is_in_zone` on the goal at `sign`*x:
        strict on every bound, ball CENTRE (the geom xpos it binds)."""
        b = self.ball_xyz()
        x = sign * b[:, 0]
        return ((x > self.goal_x) & (x < self.pitch_half[0])
                & (b[:, 1].abs() < self.goal_half_width)
                & (b[:, 2] > 0.0) & (b[:, 2] < self.goal_height))

    def detected_goal(self):
        """0 = none, 1 = HOME scored, 2 = AWAY scored. `Pitch.detected_goal`:
        the ball in the -x (home) goal is a goal FOR away."""
        home_scores = self._in_goal_box(+1.0)     # ball in the away (+x) goal
        away_scores = self._in_goal_box(-1.0)
        return (home_scores.long() + 2 * (away_scores & ~home_scores).long())

    def detected_off_court(self):
        """dm_control's inverted `field` detector: outside => detected."""
        b = self.ball_xyz()
        return ~((b[:, 0].abs() < self.field_half[0])
                 & (b[:, 1].abs() < self.field_half[1]))

    def ball_escaped(self):
        """[n] bool: the ball is on the WRONG SIDE of the field box.

        Not a rule -- a smoke alarm. The field box is a physical wall, so this
        should be identically False, and the gate asserts it is over a long
        rollout. It is here because the one way it can fail is silent and
        permanent: the ball can only get out over the top (40 m) or through a
        solver blow-out, and once out it is TRAPPED in the 1.67 m strip between
        the field box and the hoardings for the rest of the match -- a dead ball
        that no reward signal distinguishes from a boring one.

        Per axis, because the two axes are not alike:
          * y: the boundary is unbroken, so any |y| past the line is an escape.
          * x: the goal mouth is a legitimate hole in the line, so a ball past
            the line THROUGH the mouth (|y| < goal_half_width, z < goal_height)
            is a goal in progress, not an escape.
        """
        b = self.ball_xyz()
        r = self.ball_radius + BALL_ESCAPE_MARGIN
        out_y = b[:, 1].abs() > self.field_half[1] + r
        in_mouth = ((b[:, 1].abs() < self.goal_half_width)
                    & (b[:, 2] < self.goal_height))
        out_x = (b[:, 0].abs() > self.field_half[0] + r) & ~in_mouth
        return out_x | out_y

    def _recover_escaped(self, mask):
        """Put an escaped ball back on the centre spot and count it.

        Deliberately NOT a throw-in: a throw-in is a rule the policy can learn
        to farm, and this is a fault. It is counted separately (`ball_escapes`)
        so that if it ever fires it shows up as a number rather than as a
        mysteriously well-behaved boundary.
        """
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        self._write_ball(idx, torch.zeros(int(idx.numel()), 2,
                                          device=self.device))
        self.ball_escapes[idx] += 1.0

    def _apply_rules(self):
        """One control step of match bookkeeping, in `match.py`'s order.

        `world_reset` is NOT cleared here: `_sanitize` runs before this and may
        already have re-spawned a diverged world, and the reward's potential
        must be masked on that teleport too. `step` clears it.
        """
        # -- goals: rising edge, exactly MatchSim._detect_goal.
        g = self.detected_goal()
        fresh = (g > 0) & ~self.goal_latch
        self.goal_latch = g > 0
        self.scored_now = torch.where(fresh, g, torch.zeros_like(g))
        home = (self.scored_now == 1).float()
        away = (self.scored_now == 2).float()
        self.score[:, 0] += home
        self.score[:, 1] += away
        # +1 to the scoring team, -1 to the conceding one (Task.get_reward).
        sign = home - away                                    # [n]
        team_sign = torch.where(self.team == 0, 1.0, -1.0)    # [A]
        self.goal_reward = sign.unsqueeze(-1) * team_sign.unsqueeze(0)
        # -- after a goal MultiturnTask re-spawns and does NOT terminate.
        if self._goal_respawn and bool(fresh.any()):
            self._spawn_worlds(fresh.nonzero(as_tuple=True)[0])
            self.goal_latch = self.goal_latch & ~fresh
            self._forward()
        # -- out of play: nothing to do. The ball bounces off the field box in
        # the physics, so there is no throw-in and no rule here at all. What
        # remains is the fault guard; see `ball_escaped`.
        esc = self.ball_escaped()
        if bool(esc.any()):
            self._recover_escaped(esc)
            self._forward()

    # -- physics ------------------------------------------------------------
    def _forward(self):
        self.backend.forward()

    def _sanitize(self):
        """Re-kick-off any world whose state went non-finite or explosive,
        BEFORE obs/reward see it -- the drills' guard, per world."""
        bad = ((~torch.isfinite(self.qvel).all(-1))
               | (~torch.isfinite(self.qpos).all(-1))
               | (self.qvel.abs().amax(-1) > 500.0))
        if not bool(bad.any()):
            return
        self.n_diverged += int(bad.sum().item())
        self._spawn_worlds(bad.nonzero(as_tuple=True)[0])
        self._forward()

    # -- API ----------------------------------------------------------------
    def reset(self):
        self.qpos.zero_()
        self.qvel.zero_()
        self._spawn_worlds(torch.arange(self.n, device=self.device))
        self.t = 0
        self.score.zero_()
        self.goal_latch.fill_(False)
        self.scored_now.zero_()
        self.goal_reward.zero_()
        self.throw_ins.zero_()
        self.ball_escapes.zero_()
        self.prev_ctrl.zero_()
        self._forward()
        self.reward.reset(self)
        return self.obs()

    def step(self, actions):
        """actions [n * n_agents, act_dim] in [-1, 1] -> (obs, reward, done).

        `done` is ONE bool for the whole batch (the match clock), as in every
        drill env here: goals and throw-ins are handled per world without ever
        ending the episode, which is what `terminate_on_goal=False` means in
        `match.py`.
        """
        a = actions.clamp(-1.0, 1.0)
        # [n*A, nu] -> [n, A*nu]: the model's actuators are creature-major in
        # slot order, so this reshape IS the slot->actuator routing.
        self.ctrl.copy_(a.reshape(self.n, self.n_agents * self.act_dim))
        self.backend.step()
        self.world_reset.fill_(False)
        self._sanitize()
        self._apply_rules()
        self.t += 1
        done = self.t >= self.episode_steps
        rew = self.reward(self)
        if self.energy_coef > 0:
            rew = rew - self.energy_coef * (a ** 2).mean(-1)
        if self.smooth_coef > 0:
            rew = rew - self.smooth_coef * ((a - self.prev_ctrl) ** 2).mean(-1)
        self.prev_ctrl = a
        rew = rew.clamp(self.rew_clip[0], self.rew_clip[1])
        return self.obs(), rew, done

    def fitness(self):
        return self.reward.fitness(self)

    def match_stats(self):
        """What a monitor should plot: goals per match per team, throw-ins, and
        how far the ball travelled from the centre spot."""
        return dict(
            home_goals=float(self.score[:, 0].mean()),
            away_goals=float(self.score[:, 1].mean()),
            throw_ins=float(self.throw_ins.mean()),
            ball_escapes=float(self.ball_escapes.mean()),
            ball_dist=float(torch.linalg.norm(self.ball_xy(), dim=-1).mean()),
            upright=float(self.upright().mean()),
            diverged=self.n_diverged)

    # -- mirror (the team symmetry, made executable) ------------------------
    def mirror_state(self):
        """Return (qpos, qvel) rotated 180 degrees about z WITH the home and
        away slots exchanged -- the transform the symmetry claim is about.

        Used by the gate to check `obs[M(s)][mirror_slot(k)] == obs[s][k]`, and
        usable as a free data augmentation for BC (unit 1c's mirroring note).
        """
        q, v = self.qpos.clone(), self.qvel.clone()
        out_q, out_v = q.clone(), v.clone()
        for k in range(self.n_agents):
            j = int(self.mirror_slot[k])
            qa, qb = self.qpos_root[k], self.qpos_root[j]
            va, vb = self.qvel_root[k], self.qvel_root[j]
            # root pose: (x, y, z) -> (-x, -y, z); quat (w,x,y,z) -> (-z,-y,x,w)
            # which is exactly the 180-degree z rotation qz (0,0,0,1) * q.
            out_q[:, qb + 0] = -q[:, qa + 0]
            out_q[:, qb + 1] = -q[:, qa + 1]
            out_q[:, qb + 2] = q[:, qa + 2]
            out_q[:, qb + 3] = -q[:, qa + 6]
            out_q[:, qb + 4] = -q[:, qa + 5]
            out_q[:, qb + 5] = q[:, qa + 4]
            out_q[:, qb + 6] = q[:, qa + 3]
            # joint angles are frame-independent.
            out_q[:, self.pidx[j].jq] = q[:, self.pidx[k].jq]
            out_v[:, self.pidx[j].jv] = v[:, self.pidx[k].jv]
            # free-joint qvel: linear is GLOBAL (negate x, y), angular is LOCAL
            # (unchanged -- the body frame rotated with the world).
            out_v[:, vb + 0] = -v[:, va + 0]
            out_v[:, vb + 1] = -v[:, va + 1]
            out_v[:, vb + 2] = v[:, va + 2]
            out_v[:, vb + 3] = v[:, va + 3]
            out_v[:, vb + 4] = v[:, va + 4]
            out_v[:, vb + 5] = v[:, va + 5]
        b, bv = self.bq, self.bv
        out_q[:, b + 0] = -q[:, b + 0]
        out_q[:, b + 1] = -q[:, b + 1]
        out_q[:, b + 3] = -q[:, b + 6]
        out_q[:, b + 4] = -q[:, b + 5]
        out_q[:, b + 5] = q[:, b + 4]
        out_q[:, b + 6] = q[:, b + 3]
        out_v[:, bv + 0] = -v[:, bv + 0]
        out_v[:, bv + 1] = -v[:, bv + 1]
        return out_q, out_v

    def mirror_actions(self, actions):
        """Permute a flattened [n*A, nu] action batch onto the mirrored slots.

        `mirror_slot` is an involution, so indexing with it is its own inverse.
        This is the other half of `mirror_state`: together they are the free
        data augmentation the BC builder wants (unit 1c's mirroring note) and
        the transform the gate's symmetry checks apply.
        """
        a = actions.view(self.n, self.n_agents, -1)
        return a[:, self.mirror_slot].reshape(actions.shape)

    def set_state(self, qpos, qvel, ctrl=None):
        """Write a full state and re-derive kinematics/sensors.

        `ctrl` is part of the state as far as the OBSERVATION is concerned: the
        accelerometer reads `qacc`, which includes the actuator forces of
        whatever is currently in `ctrl`. Restoring qpos/qvel and leaving a stale
        ctrl behind gives a creature the right pose and the wrong acceleration
        -- measured at ~3 m/s^2 on the ant, i.e. 0.03 in the /100-scaled obs.
        """
        self.qpos.copy_(qpos)
        self.qvel.copy_(qvel)
        if ctrl is not None:
            self.ctrl.copy_(ctrl.reshape(self.ctrl.shape))
        self._forward()
