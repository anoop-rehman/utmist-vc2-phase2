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
        self.seg_t, self.seg_best, self.credit = z(), z(), z()
        self.touched, self.banked, self.seg_reset = b(), b(), b()
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

    def _close_segments(self, end):
        """Restart the segment in every world flagged in `end`."""
        self.seg_reset = end | self._pending_reset
        self._pending_reset.fill_(False)
        if not bool(end.any()):
            return
        idx = end.nonzero(as_tuple=True)[0]
        self.n_segments[idx] += 1.0
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
        self.prev_pb = torch.linalg.norm(env._ball_xy() - pos[:, :2], dim=-1)

    def _shaping(self, env):
        pos, _ = env._root_frames()
        root_xy, root_vel = pos[:, :2], env._root_vel_xy()
        ball_xy, ball_vel = env._ball_xy(), env._ball_vel_xy()
        d_pb = ball_xy - root_xy
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


class ShootReward(_StrikeReward):
    """kick's reward with the command pinned at the goal, plus the goal bonus.

    reward = w_strike * (banked strike speed toward the goal)
             + goal_bonus * (a goal was scored this step)
             + shaping_scale * shaping

    fitness = mean over this episode's segments (including the one in flight) of
    exp(-reward_coef * d), where d is the closest the ball got to the GOAL MOUTH
    RECTANGLE, not to a point: d is 0 anywhere between the posts and under the
    bar, so a goal scores 1.0 whether it goes in dead centre or off the post.
    Measuring to the mouth centre instead would score a 10 m-wide goal as a miss.
    """

    def __init__(self, goal_bonus=5.0, reward_coef=0.5, **kw):
        super().__init__(**kw)
        self.goal_bonus = goal_bonus
        self.reward_coef = reward_coef

    def __call__(self, env):
        return (self.w_strike * env.credit
                + self.goal_bonus * env.scored_now.float()
                + env.shaping_scale * self._shaping(env))

    def fitness(self, env):
        cur = torch.exp(-self.reward_coef * env.seg_goal_best)
        return (env.goal_fit_sum + cur) / (env.n_segments + 1.0)
