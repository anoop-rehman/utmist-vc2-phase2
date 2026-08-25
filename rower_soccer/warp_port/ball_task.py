"""Shared machinery for the SEGMENTED ball-strike drills (`kick`, `shoot`).

`dribble` is a continuous task -- keep the ball near a moving target, scored every
step. `kick` and `shoot` are not: they are *events*. The creature walks up to a
ball at rest, strikes it, and the ball leaves. Everything interesting happens in
the half second around the strike, and everything before it is approach.

That mismatch is what this module exists for. It adds **strike segments** on top
of `WormEnv`'s world-synchronized episodes:

    episode (15 s, all worlds reset together -- graph-friendly, unchanged)
      |-- segment (<= 5 s, PER WORLD): ball is placed, a direction is commanded,
      |     the creature approaches, strikes, the ball departs -> credit banked,
      |     segment restarts in that world alone
      |-- segment ...

Per-world segments never call `env.reset()`, so the global episode boundary the
PPO trainer keys on (`done` is a single bool for all worlds, see ppo.collect) is
untouched. A segment restart is just a few qpos writes plus one `forward()`.

Detecting the strike
--------------------
Contact is NOT read out of the solver's contact array: `WormEnv` talks to physics
only through `PhysicsBackend`, which exposes qpos/qvel/xpos/sensordata and
nothing else, and both backends must agree. So the strike is detected
kinematically, from two facts that hold in every backend:

  * the creature is the only thing that can ADD energy to the ball (the floor and
    friction can only remove it), so a ball that was at rest and is now moving has
    been struck;
  * "close enough to touch" is a property of the body, so the contact radius is
    measured off the compiled model (`creature_reach`), never hardcoded.

  touched  := ||ball - root||_xy < contact_dist  AND  ball speed > 0.05 m/s
  released := touched AND ||ball - root||_xy > contact_dist + 0.5   (hysteresis)

What gets paid
--------------
The credited strike value is the PEAK of the ball's velocity component along the
commanded direction since the strike began, clipped to `speed_clip`. That is the
same number as "ball speed toward the command at contact-break" -- once the
creature stops touching it, nothing accelerates the ball, so its along-command
speed only decays -- but taking the peak makes the number independent of which
exact 25 ms step the release test happens to fire on, and it survives the case
where the creature chases the ball it just struck (release never fires; the
segment times out and the peak is banked anyway).

Credit is banked at most once per segment (`banked`), so a creature cannot farm
one strike by dancing in and out of the release radius.
"""

import mujoco
import numpy as np
import torch

from rower_soccer.warp_port.worm_env_base import CONTROL_DT, RewardStrategy

# A ball moving slower than this counts as "at rest": it separates a real strike
# from merely standing next to a stationary ball. The ball's rolling friction
# (condim 6, 0.075) brings it to rest well below this.
CONTACT_SPEED_EPS = 0.05      # m/s
# Hysteresis between "touching" and "released". Without a gap the two tests
# chatter on the same threshold and a single strike banks nothing.
RELEASE_MARGIN = 0.5          # m


def creature_reach(model, meta):
    """Max distance from the creature root's origin to any creature geom's
    bounding sphere, at the model's rest pose.

    Creature-generic on purpose: the ant's legs reach 1.2 m, the worm's body
    0.8 m, and a contact radius that is right for one is wrong for the other.
    Measuring it off the compiled model is the same discipline follow_env applies
    to the proprio width -- derive it from the body, never hardcode it.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = data.xpos[meta.root_body]
    own = set(meta.body_ids)
    reach = 0.0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in own:
            continue
        d = float(np.linalg.norm(data.geom_xpos[g] - root))
        reach = max(reach, d + float(model.geom_rbound[g]))
    return reach


class SegmentedBallTask:
    """Mixin over `WormEnv` adding strike segments. Subclasses must implement
    `_spawn_worlds(idx, root_xy=None, yaw=None)`; everything else is here.

    Buffers it owns (all `[n]` unless noted):
        cmd_dir [n,2]  world-frame UNIT vector the strike is scored along
        target_xy[n,2] the aim point (also what render.py draws as the marker)
        seg_t          steps elapsed in this world's segment
        touched        this segment's ball has been struck
        banked         this segment's credit has been paid
        seg_best       peak ball velocity along cmd_dir since the strike
        credit         value paid THIS step (0 on every non-banking step)
        seg_reset      worlds whose segment restarted THIS step -- reward
                       strategies must mask their potentials on it, else the
                       teleport reads as a huge free "progress"
        credit_sum / credit_count / n_segments   per-episode accumulators
    """

    # -- construction -------------------------------------------------------
    def _init_segments(self, segment_seconds=5.0, speed_clip=8.0,
                       contact_pad=0.0, release_margin=RELEASE_MARGIN):
        n, dev = self.n, self.device
        self.segment_steps = max(1, int(round(segment_seconds / CONTROL_DT)))
        self.speed_clip = float(speed_clip)
        self.reach = creature_reach(self.model, self.meta)
        self.contact_dist = self.reach + self.ball_radius + contact_pad
        self.release_dist = self.contact_dist + release_margin

        def z():
            return torch.zeros(n, device=dev)

        def b():
            return torch.zeros(n, dtype=torch.bool, device=dev)

        self.cmd_dir = torch.zeros(n, 2, device=dev)
        self.cmd_dir[:, 0] = 1.0
        self.target_xy = torch.zeros(n, 2, device=dev)
        # Where the ball was placed at the start of the current segment. The
        # ball itself moves the instant it is struck; this does not, which is
        # what makes it usable as "the spot to kick from" (see
        # TimedKickReward's w_anchor). Every spawn path must write it.
        self.ball_spawn_xy = torch.zeros(n, 2, device=dev)
        self.seg_t, self.seg_best, self.credit = z(), z(), z()
        self.touched, self.banked, self.seg_reset = b(), b(), b()
        # Cumulative since reset_stats(); see _close_segments.
        self.seg_started = 0.0
        self.seg_started_flat = 0.0
        # Worlds the divergence sanitizer teleported. _sanitize runs BEFORE
        # _update_task, which then overwrites seg_reset, so the flag has to be
        # parked here and folded in when the segment bookkeeping runs -- else a
        # potential-based reward reads the sanitize teleport as free progress.
        self._pending_reset = b()
        self.credit_sum, self.credit_count, self.n_segments = z(), z(), z()
        # Last episode's strike accumulators. PPOTrainer samples env.fitness()
        # at whatever point in the episode its rollout happens to end, and a
        # freshly reset episode has banked nothing -- without a fallback the
        # monitor logs fitness=0.000 at random and best.pt selection reads noise.
        self.prev_credit_sum, self.prev_credit_count = z(), z()
        # Column indices of the ball freejoint's 6 velocity dofs, cached: the
        # per-step spawn path would otherwise rebuild them on every restart.
        self._ball_vcols = torch.arange(self.bv, self.bv + 6,
                                        device=dev).unsqueeze(0)

    def _clamped_spawn_range(self, rng):
        """Push a spawn distance band outside the contact radius.

        The drills inherit dm_control's 1-3 m ball band, which was written for a
        worm whose whole body reaches 0.8 m. The ant reaches 1.2 m, so a 1.5 m
        spawn puts the ball INSIDE contact_dist (1.56 m) before the creature has
        moved -- the drill would be scoring a body that starts already touching.
        Derived from the body, like every other geometric constant here.
        """
        lo = max(rng[0], self.contact_dist + 0.5)
        return lo, max(rng[1], lo + 0.5)

    # -- per-segment / per-episode bookkeeping ------------------------------
    def _reset_segments(self, idx):
        self.seg_t[idx] = 0.0
        self.seg_best[idx] = 0.0
        self.touched[idx] = False
        self.banked[idx] = False

    def _reset_episode_stats(self):
        self._pending_reset.fill_(False)
        self.credit.zero_()
        self.prev_credit_sum = self.credit_sum.clone()
        self.prev_credit_count = self.credit_count.clone()
        self.credit_sum.zero_()
        self.credit_count.zero_()
        self.n_segments.zero_()
        self.seg_reset.fill_(False)

    # -- strike detection ---------------------------------------------------
    def _strike_update(self):
        """Update touched/seg_best from the current state. Returns
        (dist_player_ball [n], released [n] bool). Zeroes `credit` for the step."""
        self.credit.zero_()
        pos, _ = self._root_frames()
        ball_v = self._ball_vel_xy()
        dist_pb = torch.linalg.norm(self._ball_xy() - pos[:, :2], dim=-1)
        speed = torch.linalg.norm(ball_v, dim=-1)
        v_cmd = (ball_v * self.cmd_dir).sum(-1)
        self.touched |= (dist_pb < self.contact_dist) & (speed > CONTACT_SPEED_EPS)
        self.seg_best = torch.where(self.touched,
                                    torch.maximum(self.seg_best, v_cmd),
                                    self.seg_best)
        released = self.touched & (dist_pb > self.release_dist)
        return dist_pb, released

    def _bank(self, mask):
        """Pay the strike credit in `mask`-selected worlds that have struck the
        ball and not yet been paid. Accumulates into `credit` (never overwrites,
        so a subclass may call this more than once per step)."""
        pay = mask & self.touched & ~self.banked
        value = self.seg_best.clamp(0.0, self.speed_clip)
        self.credit = torch.where(pay, value, self.credit)
        self.banked |= pay
        self.credit_sum += torch.where(pay, value, torch.zeros_like(value))
        self.credit_count += pay.float()
        return pay

    @property
    def flat_start_frac(self):
        """Share of segments that began with the creature already down.

        The quantity DRILL_V4_NOTES section 22 identified as the actual cause
        of the kick plateau. Returns nan before any segment has closed, rather
        than 0.0, so an empty window is not mistaken for a perfect one.
        """
        if self.seg_started <= 0:
            return float("nan")
        return self.seg_started_flat / self.seg_started

    def take_flat_start_frac(self):
        """`flat_start_frac` for the window since the last call, then reset.

        Windowed on purpose. A cumulative average over a billion steps barely
        moves when the gait changes, which is the opposite of what a leading
        indicator is for.
        """
        v = self.flat_start_frac
        self.seg_started = 0.0
        self.seg_started_flat = 0.0
        return v

    def _close_segments(self, end):
        """Restart the segment in every world flagged in `end`."""
        self.seg_reset = end | self._pending_reset
        self._pending_reset.fill_(False)
        if not bool(end.any()):
            return
        idx = end.nonzero(as_tuple=True)[0]
        self.n_segments[idx] += 1.0
        # THE LEADING INDICATOR (DRILL_V4_NOTES section 22). A segment inherits
        # the creature's pose, and one that starts with upright < 0.30 scores
        # 0.117 -- the do-nothing floor -- against 0.450 for one that starts
        # standing. 49.5% of segments started flat in the run that plateaued.
        # Sampled HERE because this is the instant before `_spawn_worlds`, so
        # it is the posture carried IN, which is the quantity that predicts the
        # outcome (r = +0.674). Fitness takes ~100M steps to move; this moves
        # in ~10M, which is what makes a screening run cheap.
        u = upright(self)[idx]
        self.seg_started += float(idx.numel())
        self.seg_started_flat += float((u < 0.30).sum())
        self._spawn_worlds(idx)
        self._reset_segments(idx)
        # qpos was written directly; xpos/sensordata are stale until forward().
        self._forward()
        # The ball just teleported; re-baseline the contact diagnostic for
        # these worlds so it measures THIS segment's displacement.
        self._ball_track_respawn(idx)

    # -- spawn helpers (indexed; the base's _spawn_root writes ALL worlds) ---
    def _write_ball(self, idx, ball_xy):
        self.qpos[idx, self.bq + 0] = ball_xy[:, 0]
        self.qpos[idx, self.bq + 1] = ball_xy[:, 1]
        self.qpos[idx, self.bq + 2] = self.ball_radius
        self.qpos[idx, self.bq + 3] = 1.0     # identity quat (w, x, y, z)
        self.qpos[idx, self.bq + 4] = 0.0
        self.qpos[idx, self.bq + 5] = 0.0
        self.qpos[idx, self.bq + 6] = 0.0
        self.qvel[idx.unsqueeze(-1), self._ball_vcols] = 0.0
        # Every ball placement is a segment spawn, so this is the one place the
        # anchor can be recorded without a future spawn path being able to skip
        # it.
        self.ball_spawn_xy[idx] = ball_xy

    def anchor_excess(self, free_radius=1.0, cap=5.0):
        """How far past `free_radius` the creature has strayed from the spot the
        ball was spawned on, in metres, clipped to `cap`.

        Zero inside the radius: the creature has to be able to stand beside the
        ball and swing at it, and paying it to stand on the exact spawn point
        would fight the strike. Clipped above so a world that has wandered off
        (or one segment's leftover geometry) cannot dominate the return.
        """
        pos, _ = self._root_frames()
        d = torch.linalg.norm(pos[:, :2] - self.ball_spawn_xy, dim=-1)
        return (d - free_radius).clamp(min=0.0, max=cap)

    def _write_root(self, idx, xy, yaw):
        qr = self.meta.qpos_root
        self.qpos[idx, qr + 0] = xy[:, 0]
        self.qpos[idx, qr + 1] = xy[:, 1]
        self.qpos[idx, qr + 2] = self._spawn_z
        self.qpos[idx, qr + 3] = torch.cos(yaw / 2)
        self.qpos[idx, qr + 4] = 0.0
        self.qpos[idx, qr + 5] = 0.0
        self.qpos[idx, qr + 6] = torch.sin(yaw / 2)
        if self.ball_qw_idx.numel():
            self.qpos[idx.unsqueeze(-1), self.ball_qw_idx.unsqueeze(0)] = 1.0

    def _root_yaw(self, idx):
        """World heading of the creature root, from its rotation matrix."""
        _, rot = self._root_frames()
        fwd = rot[idx, :2, 0]
        return torch.atan2(fwd[:, 1], fwd[:, 0])

    @staticmethod
    def _unit(v):
        return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-6)

    # -- obs helper ---------------------------------------------------------
    def _ball_ego6(self):
        """ego ball position (3) + ego ball linear velocity (3), the SAME block
        dribble uses and the same shape the 2v2 game gives
        (`ball_ego_position` + `ball_ego_linear_velocity`), so it survives
        distillation into a drill prior."""
        return torch.cat([self._to_ego3(self._ball_xyz()),
                          self._vec_to_ego3(self._ball_vel_xyz())], -1)

    def _xy_ego3(self, xy):
        """A ground-plane world point [n,2] as a 3-D egocentric position."""
        z = torch.zeros(self.n, 1, device=self.device)
        return self._to_ego3(torch.cat([xy, z], -1))

    def _dir_ego3(self, d):
        """A ground-plane world DIRECTION [n,2] as a 3-D egocentric vector."""
        z = torch.zeros(self.n, 1, device=self.device)
        return self._vec_to_ego3(torch.cat([d, z], -1))


# ---------------------------------------------------------------------------
# Reward strategies
# ---------------------------------------------------------------------------
class _StrikeReward(RewardStrategy):
    """Common shaping for both strike drills.

    `paper` mode mirrors DribbleReward's velocity shaping (the terms that made
    dribble trainable): reward closing on the ball, and reward the ball moving
    along the command. `progress` mode replaces the first with the unhackable
    potential `prev_dist - dist` (Ng et al. 1999), masked on segment restarts --
    a restart teleports the ball, and an unmasked potential would read that
    teleport as several metres of free progress every few seconds.

    Both are multiplied by the mutable `env.shaping_scale`, so the trainer's
    --shaping-anneal-steps parks them and leaves the strike credit alone.
    """

    def __init__(self, mode="paper", w_strike=0.5, w_player_to_ball=0.15,
                 w_ball_to_cmd=0.1, approach_scale=0.5):
        self.mode = mode
        self.w_strike = w_strike
        self.w_p2b = w_player_to_ball
        self.w_b2c = w_ball_to_cmd
        self.approach_scale = approach_scale
        self.prev_pb = None

    def reset(self, env):
        pos, _ = env._root_frames()
        self.prev_pb = torch.linalg.norm(self._approach_xy(env) - pos[:, :2],
                                         dim=-1)

    def _approach_xy(self, env):
        """The point the me->ball approach shaping pulls the creature toward.

        The live ball by default. Subclasses override to anchor it (see
        TimedKickReward's w_anchor): once the ball is rolling, "move toward the
        ball" and "chase the ball" are the same instruction.
        """
        return env._ball_xy()

    def _shaping(self, env):
        pos, _ = env._root_frames()
        root_xy, root_vel = pos[:, :2], env._root_vel_xy()
        ball_vel = env._ball_vel_xy()
        d_pb = self._approach_xy(env) - root_xy
        dist_pb = torch.linalg.norm(d_pb, dim=-1)
        v_b2c = (ball_vel * env.cmd_dir).sum(-1).clamp(min=0.0)
        if self.mode == "progress":
            approach = self.prev_pb - dist_pb
            # A segment restart teleports the ball (and, in shoot, the creature).
            # That is not progress the policy earned.
            approach = torch.where(env.seg_reset, torch.zeros_like(approach),
                                   approach)
            self.prev_pb = dist_pb.detach()
            return self.approach_scale * approach + self.w_b2c * v_b2c
        self.prev_pb = dist_pb.detach()
        v_p2b = (root_vel * (d_pb / dist_pb.clamp(min=1e-6).unsqueeze(-1))).sum(-1)
        return self.w_p2b * v_p2b.clamp(min=0.0) + self.w_b2c * v_b2c


class KickReward(_StrikeReward):
    """reward = w_strike * (banked strike speed along the command)
               + shaping_scale * shaping

    The first term is zero on every step except the one where a segment banks --
    which is exactly the brief: "ball speed toward a commanded direction,
    measured at contact-break".

    fitness = mean banked strike speed (m/s) over this episode's strikes,
    falling back to the previous episode's in any world that has not struck yet.
    Unshaped: neither velocity-shaping term can inflate it, because it counts
    only what the ball did after being struck. The fallback matters because
    PPOTrainer reads fitness wherever its 64-step rollout happens to land -- a
    third of the time that is inside a freshly reset episode, and without it the
    monitor logs 0.000 at random and best.pt is selected on noise.
    """

    def __call__(self, env):
        return self.w_strike * env.credit + env.shaping_scale * self._shaping(env)

    def fitness(self, env):
        cur = env.credit_sum / env.credit_count.clamp(min=1.0)
        prev = env.prev_credit_sum / env.prev_credit_count.clamp(min=1.0)
        return torch.where(env.credit_count > 0, cur, prev)


def upright(env):
    """+1 torso z-axis up, 0 inverted -- dm_control's _upright_reward, mapped to
    [0,1] the same way fetch_env does ((1 + cos)/2).

    Every ball drill here multiplies by this, because DeepMind's fetch reward
    does and ours did not. Without it nothing prices posture: a creature lying
    flat scores exactly as well as one standing, and the ant duly learned to
    operate at 0.489 m against a 0.612 m passive stand -- splayed out, shoving
    the ball with its body. Multiplying (not adding) is the point: no amount of
    ball progress compensates for being on your side.
    """
    _, rot = env._root_frames()
    return ((1.0 + rot[:, 2, 2]) / 2.0).clamp(0.0, 1.0)


def linear_tolerance(d, bound, margin):
    """dm_control rewards.tolerance(sigmoid='linear', value_at_margin=0).

    1 inside `bound`, falling linearly to 0 at `bound + margin`. Bounded in
    [0,1] with a defined zero, unlike exp(-c*d) which has a long tail and whose
    scale (2 m at c=0.5) is easy to misjudge.
    """
    return (1.0 - (d - bound).clamp(min=0.0) / margin).clamp(0.0, 1.0)


class KickToPointReward(_StrikeReward):
    """Kick graded on WHERE THE BALL ARRIVED, not how fast it left.

    reward = w_arrive * exp(-c * closest ball-to-target distance)   [at segment end]
             + w_strike * (banked strike speed along the command)   [small]
             + shaping_scale * shaping

    Why this exists. `KickReward` scores `max(ball_velocity . command)`, a
    PROJECTION, and a projection cannot tell a hard wild kick from a gentle
    accurate one: 7.6 m/s at 60 degrees off and 3.8 m/s dead on both score 3.8.
    Since "hit harder" is a far easier gradient than "aim better", RL takes it.
    Measured on kick_ant_v1 over 2243 strikes: median aim error 35 deg, mean 48,
    only 24% of strikes inside 15 deg, 16% sent BACKWARDS, 37% of ball speed
    thrown away -- all while fitness rose monotonically for 446M steps.

    Distance to a point 4 m away is savage about aim where a cosine is not: 35
    degrees off is a 2.4 m miss. shoot already scores this way (exp(-d) to the
    goal mouth) and is the drill whose videos actually look right, which is the
    corroborating evidence for the diagnosis.

    w_strike is kept small rather than dropped. Arrival alone is satisfiable by
    walking the ball to the target -- that is dribble, not kick -- so the strike
    term (paid only at contact-break) keeps a genuine strike in the objective.
    The env's release gate does the structural work; this term just prices it.
    """

    def __init__(self, w_arrive=3.0, reward_coef=0.5, w_upright=1.0, **kw):
        super().__init__(**kw)
        self.w_arrive = w_arrive
        self.reward_coef = reward_coef
        # Exponent on the uprightness factor. 1.0 = dm_control's fetch weighting;
        # 0.0 disables it, which is what every run before 2026-08-09 effectively
        # used.
        self.w_upright = w_upright

    def __call__(self, env):
        # env.last_arrival, NOT env.arrival(): the env has already respawned the
        # ball by the time the reward is computed, so arrival() would price the
        # NEW segment's spawn distance instead of the kick that just happened.
        # last_arrival is the value snapshotted before the respawn, and is zero
        # on every step except the one a segment closes -- so it needs no
        # further gating.
        pay = self.w_arrive * env.last_arrival
        r = (pay + self.w_strike * env.credit
             + env.shaping_scale * self._shaping(env))
        # Multiplied by uprightness, as dm_control's fetch reward is. See
        # upright()'s docstring: without it, posture is unpriced and the ant
        # learns to splay flat and shove the ball with its torso.
        return r * upright(env) ** self.w_upright

    def fitness(self, env):
        """Mean arrival over this episode's completed segments, plus the one in
        flight -- the same shape as ShootReward.fitness, and directly comparable
        across the two drills because both are exp(-c*d) in [0, 1]."""
        cur = env.arrival()
        n = env.n_segments + 1.0
        live = (env.target_fit_sum + cur) / n
        prev = env.prev_target_fit_sum / env.prev_n_segments.clamp(min=1.0)
        return torch.where(env.n_segments > 0, live,
                           torch.where(env.prev_n_segments > 0, prev, cur))


class TimedKickReward(_StrikeReward):
    """A PASS: the ball at the target AT A GIVEN TIME (drill v4).

    reward = w_arrive * exp(-c * ||ball(T) - target||_3D)   [paid once, at T]
             + w_strike * (banked strike speed)             [w_strike = 0 by default]
             + shaping_scale * (me->ball approach only)
             all multiplied by upright

    What changed from KickToPointReward and why. That reward scored the CLOSEST
    the ball came to the target at ANY moment in a 2-6 s window, which a
    creature satisfies perfectly well by walking the ball over -- the small
    w_strike term was a patch discouraging exactly that, i.e. a property of the
    weights rather than of the objective. Here the segment ends at a deadline
    `T = d_spawn / v_pace` and the only thing measured is WHERE THE BALL IS
    THEN. Dribbling is now excluded by arithmetic, not by a penalty: the
    slowest pace in the band still needs the ball moving faster than the ant
    can run. Striking too hard is punished as symmetrically as striking too
    soft, because the ball sails past the target before T -- v3 punished
    neither, and rolling friction (~4 m/s^2 on this ball) means the required
    strike is genuinely a modulated one, not "as hard as possible".

    DENSE, never a ring test. Grading "did the ball pass within r of the target
    at T" is a gate random exploration essentially never opens, so the gradient
    is flat everywhere and the run learns nothing; exp(-c * d_at_T) always
    points somewhere.

    Shaping: the me->ball approach term is kept (reaching the ball is a
    prerequisite and it is what makes early exploration work at all), the
    ball->cmd_dir velocity term is DROPPED -- it pays monotonically for "faster
    toward the target", which is precisely the pace modulation this objective
    exists to teach. It is forced to zero here rather than left to a flag, so
    no launch command can quietly reintroduce it.

    fitness = mean over the episode's COMPLETED segments of exp(-c * d_at_T),
    falling back to the previous episode's mean in a world that has not closed
    one yet. Same shape and scale as shoot's, so the two drills stay
    comparable. The segment in flight is deliberately excluded: its arrival is
    undefined until the deadline.
    """

    def __init__(self, w_arrive=3.0, reward_coef=0.5, w_upright=1.0,
                 w_anchor=0.0, anchor_free_radius=1.0, strike_offset=0.0, **kw):
        super().__init__(**kw)
        # -- v8: aim the approach at the STRIKE POINT, not at the ball -----
        # Measured on v7/best.pt over 55,303 ball-moving samples, against a
        # random baseline of median 90 deg / 16.7% within 30 deg:
        #
        #   positioning (ant->ball vs ball->target)  median 103.9 deg, 13.1%
        #   aim         (ball vel  vs ball->target)  median  93.2 deg, 15.7%
        #
        # Both at or slightly WORSE than random: the ant has no positioning
        # skill, and past 90 deg it tends to stand BETWEEN ball and target, so
        # contact shoves the ball away (mean gain -0.12 m).
        #
        # The cause is this class's own shaping. To send a ball somewhere you
        # must first get to the far side of it, and the me->ball term pays
        # w_p2b * (speed TOWARD the ball) on every step -- so the circling
        # manoeuvre the task requires is penalised the whole way round. The
        # policy is paid to charge straight at the ball from wherever it
        # happens to be, which makes the strike direction whatever the approach
        # direction happened to be, i.e. random.
        #
        # strike_offset > 0 moves the approach target to
        #     ball + strike_offset * unit(ball - target)
        # i.e. `strike_offset` metres behind the ball on the ball->target line.
        # Walking to THAT point and then continuing forward IS the kick. Note
        # this is not another outcome-reward retune (v4->v6->v7 all were, and
        # all were null); it removes shaping that opposes the required
        # behaviour.
        self.strike_offset = strike_offset
        self.w_arrive = w_arrive
        self.reward_coef = reward_coef
        self.w_upright = w_upright
        self.w_b2c = 0.0   # see docstring: fights pace modulation
        # -- v7: the SPAWN ANCHOR ----------------------------------------
        # Strike the ball from where it lies; do not travel with it. Two parts,
        # deliberately in different channels:
        #
        #   1. the me->ball approach shaping is re-aimed at the ball's SPAWN
        #      point instead of the live ball (_approach_xy below). Before
        #      contact the two are identical, so approach is learned exactly as
        #      before. After contact they diverge, and the old term was paying
        #      w_p2b * (speed toward the rolling ball) -- i.e. paying for the
        #      dribble this drill exists to rule out. Anchoring deletes that
        #      payment rather than adding a second term to cancel it.
        #   2. a penalty on how far past `anchor_free_radius` the creature has
        #      strayed from that spawn point, OUTSIDE shaping_scale so
        #      --shaping-anneal-steps cannot quietly switch it off. It is part
        #      of the objective ("kick from there"), not a training aid.
        #
        # w_anchor is per step. A segment is T/CONTROL_DT = 50-200 steps, so a
        # creature walking the ball 2-3 m downfield for a whole segment pays
        # roughly w_anchor * 2.5 * 125 ~ 3 at w_anchor=0.01 -- the same order as
        # w_arrive * 1.0 = 3, a perfect pass. Approaching the ball REDUCES this
        # term (the anchor is where the ball was), so it never opposes reaching.
        self.w_anchor = w_anchor
        self.anchor_free_radius = anchor_free_radius

    def _approach_xy(self, env):
        base = env.ball_spawn_xy if self.w_anchor else env._ball_xy()
        if not self.strike_offset:
            return base
        # Behind the ball, on the far side from the target. Degenerate only if
        # the ball is exactly on the target, where any direction is as good.
        away = base - env.target_xy
        n = torch.linalg.norm(away, dim=-1, keepdim=True).clamp(min=1e-6)
        return base + self.strike_offset * (away / n)

    def __call__(self, env):
        # env.last_arrival is the deadline snapshot taken before the respawn,
        # zero on every other step -- see kick_env._update_task.
        r = (self.w_arrive * env.last_arrival + self.w_strike * env.credit
             + env.shaping_scale * self._shaping(env))
        r = r * upright(env) ** self.w_upright
        if self.w_anchor:
            # Not multiplied by upright: a penalty scaled by uprightness is
            # cheaper to incur while tipped over, which is a discount on
            # falling and has nothing to do with what this term measures.
            r = r - self.w_anchor * env.anchor_excess(self.anchor_free_radius)
        return r

    def fitness(self, env):
        cur = env.target_fit_sum / env.n_segments.clamp(min=1.0)
        prev = env.prev_target_fit_sum / env.prev_n_segments.clamp(min=1.0)
        return torch.where(env.n_segments > 0, cur,
                           torch.where(env.prev_n_segments > 0, prev,
                                       torch.zeros_like(cur)))


class ShootReward(_StrikeReward):
    """kick's reward with the command pinned at the goal, plus a TIMED goal bonus.

    reward = w_strike * (banked strike speed toward the goal)
             + goal_bonus * exp(-k * t_score) * (a goal was scored this step)
             + shaping_scale * shaping

    The `exp(-k * t_score)` factor (drill v4) is what makes this drill *shoot*
    rather than *walk the ball in*. A flat bonus pays the same for a goal at
    0.5 s and a goal at 5 s, so the cheapest policy is to escort the ball over
    the line -- which in a match is intercepted every time. `t_score` is the
    time within the SEGMENT (the creature is respawned in front of the mouth at
    every segment start, so it is "time since this attempt began"). At the
    default k=0.4: 1 s -> 0.67 of the bonus, 3 s -> 0.30, 5 s -> 0.14.

    fitness (drill v4) = mean over the episode's segments of

        scored ?  0.5 + 0.5 * exp(-k * t_score)  :  0.5 * exp(-c * d_mouth_best)

    computed by the env (`env.seg_fitness`, which is also where the deviation
    from the spec's exact scored branch is argued) so the reward and the
    accumulator can never drift apart. Before v4 this was
    `exp(-c * d_mouth_best)` alone -- accuracy only -- so best.pt was SELECTED
    on something the reward did not pay for, and a timid ant that trickled the
    ball to the line outranked one that buried it. Goals live in the upper half
    of [0, 1] and misses in the lower, so any goal outranks any miss.

    `d_mouth_best` is the closest the ball came to the GOAL MOUTH RECTANGLE, not
    to a point: 0 anywhere between the posts and under the bar, so a goal scores
    at the top of the miss branch whether it goes in dead centre or off the
    post. Measuring to the mouth centre would score a 7 m-wide goal as a miss.
    """

    def __init__(self, goal_bonus=5.0, reward_coef=0.5, goal_time_coef=0.4,
                 w_upright=1.0, **kw):
        super().__init__(**kw)
        self.goal_bonus = goal_bonus
        self.reward_coef = reward_coef
        # k in exp(-k * t_score). The env holds the same number (it needs it for
        # the fitness accumulator); shoot_env passes one value into both.
        self.goal_time_coef = goal_time_coef
        self.w_upright = w_upright

    def __call__(self, env):
        # env.last_score_t, NOT env.seg_score_t: a goal ENDS the segment, so by
        # the time the reward runs the env has already respawned and zeroed the
        # per-segment clock -- reading it here would pay exp(0) = the full flat
        # bonus for every goal, silently undoing the urgency term. Same
        # snapshot discipline as kick's last_arrival.
        urgency = torch.exp(-self.goal_time_coef * env.last_score_t)
        r = (self.w_strike * env.credit
             + self.goal_bonus * urgency * env.scored_now.float()
             + env.shaping_scale * self._shaping(env))
        return r * upright(env) ** self.w_upright

    def fitness(self, env):
        # Includes the segment in flight, which is why the denominator is
        # n_segments + 1: PPOTrainer samples fitness wherever its rollout lands.
        return (env.goal_fit_sum + env.seg_fitness()) / (env.n_segments + 1.0)
