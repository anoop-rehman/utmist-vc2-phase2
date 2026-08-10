"""Batched GPU shoot-drill env: `kick` with a real goal to aim at.

"Kick specialization with goal geometry + scoring termination"
(ANT_SPRINT_PLAN 3). The command direction is no longer drawn at random -- it is
pinned at the opponent goal -- and a segment ends when the ball crosses the goal
line, not when the creature stops touching it. That is the "scoring termination":
episodes stay world-synchronized (the trainer's `done` is one bool for all
worlds), so the per-world thing that terminates on a goal is the strike SEGMENT.
See ball_task.SegmentedBallTask.

THE GOAL IS THE PITCH'S OWN. `scene.py` compiles dm_soccer's 2v2 pitch into every
drill scene already -- 20 goal capsules, posts at x = +/-42.6667, y = +/-11.88,
crossbar at z = 5.3333 -- and this env reuses those coordinates verbatim
(`self.goal_x`, `self.goal_half_width`, `self.goal_height`) rather than inventing a toy goal.
The scene comment says exactly why they were put there: "so `shoot` has a real
goal to aim at". Consequence: this env does NOT use the small walled arena the
other drills default to; `_base_xml` returns None so `build_creature_scene` falls
through to the pitch. The creature spawns a few metres out from the away goal
mouth, because a 1 m/s ant would need eleven minutes to walk there from the
centre spot.

Obs (proprio-first, contiguous task block):

    proprio(P) | ball_ego(6) | goal_mid_ego(3)
               | post_left_ego(2) | post_right_ego(2)              = P + 13

  * proprio is byte-identical to follow/dribble/kick -- the frozen decoder's
    entire input contract (see follow_env's docstring).
  * ball_ego = ego position(3) + ego linear velocity(3), matching the game's
    `ball_ego_position` + `ball_ego_linear_velocity`.
  * goal_mid_ego(3) is the game's `opponent_goal_mid`: the mouth centre at half
    crossbar height, egocentric.
  * post_left_ego / post_right_ego(2 each) are the game's
    `opponent_goal_back_left` / `opponent_goal_front_right` corner form -- the
    two mouth posts as egocentric ground-plane xy. STAGE2_MULTITASK is explicit
    that shoot's goal obs must be the game's `opponent_goal_mid` plus a corner
    representation, because this obs survives distillation into the shoot prior
    and the prior is evaluated on GAME observations. A single centre point would
    also be the wrong thing to aim at: this goal is 23.8 m wide.
"""
import numpy as np
import torch

from rower_soccer.warp_port.ball_task import SegmentedBallTask, ShootReward
from rower_soccer.warp_port.scene import BallSpec, base_xml, goal_geometry
from rower_soccer.warp_port.worm_env_base import WormEnv


class WarpShootEnv(SegmentedBallTask, WormEnv):
    def __init__(self, num_worlds=2048,
                 creature_xml="creature_configs/ant.xml",
                 episode_seconds=15.0, segment_seconds=5.0,
                 shoot_dist_range=(2.0, 5.0), ball_spawn_range=(1.5, 3.0),
                 shoot_y_frac=0.4, spawn_cone=np.pi / 3, out_of_play_dist=20.0,
                 speed_clip=8.0, w_strike=0.5, goal_bonus=5.0,
                 w_player_to_ball=0.15, w_ball_to_cmd=0.1, approach_scale=0.5,
                 reward_coef=0.5, reward_mode="paper", device=None, seed=0,
                 use_graph=True, ball: BallSpec = None, nconmax=64, njmax=512,
                 energy_coef=0.0, smooth_coef=0.0, rew_clip=(-10.0, 10.0),
                 fixed_start=False, reward=None, use_gpu=True,
                 backend_cls=None, arena="fenced", pitch_scale=0.3125, w_upright=1.0):
        self._ball = ball
        self.shoot_dist_range = shoot_dist_range
        self.ball_spawn_range = ball_spawn_range
        self.shoot_y_half = shoot_y_frac * goal_geometry(pitch_scale)[1]
        self.spawn_cone = spawn_cone
        self.out_of_play_dist = out_of_play_dist
        self._segment_seconds = segment_seconds
        self._speed_clip = speed_clip
        self._reward_coef = reward_coef
        # Goal geometry must follow the PITCH SCALE. These were module constants
        # (42.6667 / 11.88 / 5.3333), which are the goal's position on the
        # unscaled 96x72 m pitch. At pitch_scale 0.3125 the real goal sits at
        # x=13.33, so a shoot env using the constants would aim 29 m past it and
        # score nothing -- with everything else looking perfectly healthy.
        self.goal_x, self.goal_half_width, self.goal_height = \
            goal_geometry(pitch_scale)
        self.shaping_scale = 1.0
        self.fixed_start = fixed_start
        reward = reward or ShootReward(w_upright=w_upright, 
            mode=reward_mode, w_strike=w_strike, goal_bonus=goal_bonus,
            reward_coef=reward_coef, w_player_to_ball=w_player_to_ball,
            w_ball_to_cmd=w_ball_to_cmd, approach_scale=approach_scale)
        super().__init__(num_worlds=num_worlds, creature_xml=creature_xml,
                         episode_seconds=episode_seconds, use_gpu=use_gpu,
                         device=device, seed=seed, use_graph=use_graph,
                         nconmax=nconmax, njmax=njmax, reward=reward,
                         energy_coef=energy_coef, smooth_coef=smooth_coef,
                         rew_clip=rew_clip, backend_cls=backend_cls, arena=arena, pitch_scale=pitch_scale)

    # -- scene --------------------------------------------------------------
    def _ball_spec(self):
        return self._ball or BallSpec()

    def _base_xml(self):
        # The scaled pitch WITH the goals -- shoot cannot use the fenced arena,
        # because the thing it aims at only exists on the pitch.
        # The other drills override this with the small walled arena; shoot
        # cannot, because the thing it aims at only exists on the pitch.
        return base_xml(self._pitch_scale)

    # -- task ---------------------------------------------------------------
    def _task_dim(self):
        return 13  # ball_ego(6) + goal_mid(3) + post_left(2) + post_right(2)

    def _task_init(self):
        self._init_segments(segment_seconds=self._segment_seconds,
                            speed_clip=self._speed_clip)
        self.ball_spawn_range = self._clamped_spawn_range(self.ball_spawn_range)
        n, dev = self.n, self.device
        # Away goal (+x). The pitch is mirror-symmetric, so training on one goal
        # and mirroring at deployment is exact; carrying both would only halve
        # the data per configuration.
        self.goal_mid = torch.tensor(
            [self.goal_x, 0.0, self.goal_height / 2.0], device=dev).expand(n, 3)
        self.post_left = torch.tensor(
            [self.goal_x, self.goal_half_width], device=dev).expand(n, 2)
        self.post_right = torch.tensor(
            [self.goal_x, -self.goal_half_width], device=dev).expand(n, 2)
        self.scored_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self.goals = torch.zeros(n, device=dev)
        self.goal_fit_sum = torch.zeros(n, device=dev)
        self.seg_goal_best = torch.full((n,), float(self.goal_x), device=dev)

    def _task_obs(self):
        return torch.cat([self._ball_ego6(),
                          self._to_ego3(self.goal_mid),
                          self._to_ego(self.post_left),
                          self._to_ego(self.post_right)], -1)

    # -- goal geometry ------------------------------------------------------
    def _goal_mouth_dist(self):
        """Distance from the ball to the goal MOUTH RECTANGLE (0 inside it).

        A point distance to the mouth centre would call a 10 m-wide goal a miss;
        the mouth is 23.8 m across and 5.3 m tall, and any part of it scores.
        """
        b = self._ball_xyz()
        dx = (self.goal_x - b[:, 0]).clamp(min=0.0)
        dy = (b[:, 1].abs() - self.goal_half_width).clamp(min=0.0)
        dz = (b[:, 2] - self.goal_height).clamp(min=0.0)
        return torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)

    def _scored(self):
        b = self._ball_xyz()
        return ((b[:, 0] > self.goal_x) & (b[:, 1].abs() < self.goal_half_width)
                & (b[:, 2] < self.goal_height))

    def _update_task(self):
        _, released = self._strike_update()
        self.seg_t += 1.0
        # Credit the strike the moment the ball leaves the creature -- the same
        # measurement kick makes. Unlike kick, that does NOT end the segment:
        # the ball still has to travel, and whether it goes in is the point.
        self._bank(released)

        d_goal = self._goal_mouth_dist()
        self.seg_goal_best = torch.minimum(self.seg_goal_best, d_goal)
        self.scored_now = self._scored()
        self.goals += self.scored_now.float()

        timeout = self.seg_t >= self.segment_steps
        # Out of play: past the goal line (scored or wide), off the touchline, or
        # rolled so far the segment cannot be rescued.
        out = ((self._ball_xyz()[:, 0] > self.goal_x)
               | (self._ball_xyz()[:, 1].abs() > 34.0)
               | (d_goal > self.out_of_play_dist))
        end = timeout | out
        # A segment that ends without the release test ever firing (creature
        # still standing over the ball) still banks its strike.
        self._bank(end)
        if bool(end.any()):
            idx = end.nonzero(as_tuple=True)[0]
            self.goal_fit_sum[idx] += torch.exp(
                -self._reward_coef * self.seg_goal_best[idx])
        self._close_segments(end)

    # -- spawning -----------------------------------------------------------
    def _spawn_worlds(self, idx, root_xy=None, yaw=None):
        """Stand a fresh shooting attempt up in worlds `idx`.

        Ball `shoot_dist` out from the goal mouth, laterally within the posts;
        creature behind the ball on the ball->goal line (up to +/-spawn_cone off
        it), facing the ball. Unlike kick, the creature IS repositioned: a shot
        is defined relative to a goal that does not move, so an attempt that
        left the ant behind the goal line has nothing left to shoot at.
        """
        k = int(idx.numel())
        d0, d1 = self.shoot_dist_range
        dist = d0 + (d1 - d0) * self._rand(k)
        y = (self._rand(k) * 2.0 - 1.0) * self.shoot_y_half
        ball_xy = torch.stack([self.goal_x - dist, y], -1)

        goal_xy = torch.stack([torch.full_like(y, self.goal_x),
                               torch.zeros_like(y)], -1)
        goal_dir = self._unit(goal_xy - ball_xy)
        self.cmd_dir[idx] = goal_dir
        self.target_xy[idx] = goal_xy

        # Creature behind the ball, i.e. on the far side from the goal.
        off = (self._rand(k) * 2.0 - 1.0) * self.spawn_cone
        c, s = torch.cos(off), torch.sin(off)
        back = torch.stack([-(goal_dir[:, 0] * c - goal_dir[:, 1] * s),
                            -(goal_dir[:, 0] * s + goal_dir[:, 1] * c)], -1)
        p0, p1 = self.ball_spawn_range
        pdist = p0 + (p1 - p0) * self._rand(k)
        root = ball_xy + back * pdist.unsqueeze(-1)
        face = torch.atan2(-back[:, 1], -back[:, 0])   # look at the ball
        root_yaw = face if self.fixed_start else self._rand(k) * (2 * np.pi)

        # Full per-world reset: a repositioned creature keeps neither its joint
        # state nor its velocity, or it lands mid-stride at the new pose.
        self.qpos[idx] = 0.0
        self.qvel[idx] = 0.0
        self._write_root(idx, root, root_yaw)
        self._write_ball(idx, ball_xy)
        self.seg_goal_best[idx] = torch.linalg.norm(
            torch.stack([(self.goal_x - ball_xy[:, 0]).clamp(min=0.0),
                         (ball_xy[:, 1].abs() - self.goal_half_width).clamp(min=0.0)],
                        -1), dim=-1)

    def _reset_state(self):
        idx = torch.arange(self.n, device=self.device)
        self._spawn_worlds(idx)
        self._reset_segments(idx)
        self._reset_episode_stats()
        self.goals.zero_()
        self.goal_fit_sum.zero_()
        self.scored_now.fill_(False)

    def _sanitize_task(self, idx):
        self._spawn_worlds(idx)
        self._reset_segments(idx)
        self._pending_reset[idx] = True
