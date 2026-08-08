"""Batched GPU kick-drill env, a sibling of `dribble_env` on the same base.

"Impart velocity to the ball toward a commanded direction" (ANT_SPRINT_PLAN 3).
Where dribble asks the creature to *shepherd* the ball to a moving point, kick
asks it to *strike* the ball once, hard, along a direction it is told. The scored
event is the strike, so the episode is cut into per-world strike segments -- see
ball_task.SegmentedBallTask for the segment/contact/credit machinery.

Obs (proprio-first, contiguous task block -- the contract every drill shares):

    proprio(P) | ball_ego(6) | target_ego3(3) | cmd_dir_ego3(3)     = P + 12

  * P is derived from the creature, never hardcoded (9 bodies / 8 joints / 9
    touch => 65 for the ant and the rower; 29 for the worm). See follow_env's
    module docstring: proprio is the frozen decoder's entire input contract, so
    it must be byte-identical across follow / dribble / kick / shoot or a
    transferred decoder is being fed a permuted vector.
  * ball_ego is ego position(3) + ego linear velocity(3), the same block dribble
    uses and the same shape the 2v2 game hands over, so it survives distillation.
  * target_ego3 is the aim POINT (ball spawn + command * target_dist) -- the
    thing a human clicks in the play server, and the same representation as
    dribble's target_ego3, so WS3's SkillController builds it the same way.
  * cmd_dir_ego3 is the commanded direction as a unit vector in the root frame.
    It is redundant with (target_ego3 - ball_ego) up to a normalisation, and it
    is here anyway: the direction is what the reward scores, and asking a
    2-layer expert to recover a normalised difference of two of its own inputs
    is free accuracy thrown away.

Task width is 12, identical to dribble's, so a dribble checkpoint warm-starts
kick with its task encoder AND critic input layer intact -- not just the decoder.
"""
import numpy as np
import torch

from rower_soccer.warp_port.ball_task import (KickReward, KickToPointReward,
                                              SegmentedBallTask)
from rower_soccer.warp_port.scene import BallSpec
from rower_soccer.warp_port.worm_env_base import WormEnv


class WarpKickEnv(SegmentedBallTask, WormEnv):
    def __init__(self, num_worlds=2048,
                 creature_xml="creature_configs/ant.xml",
                 episode_seconds=15.0, segment_seconds=5.0,
                 ball_spawn_range=(1.5, 3.0), target_dist=4.0,
                 speed_clip=8.0, w_strike=0.5, w_player_to_ball=0.15,
                 w_ball_to_cmd=0.1, approach_scale=0.5, reward_mode="paper",
                 device=None, seed=0, use_graph=True, ball: BallSpec = None,
                 nconmax=64, njmax=512, energy_coef=0.0, smooth_coef=0.0,
                 rew_clip=(-10.0, 10.0), fixed_start=False, target_cone=0.0,
                 reward=None, floor_half=10.0, use_gpu=True, backend_cls=None,
                 reward_coef=0.5, out_of_play_dist=12.0,
                 reward_kind="direction", w_arrive=3.0):
        self._ball = ball
        self.ball_spawn_range = ball_spawn_range
        self.target_dist = target_dist
        self._segment_seconds = segment_seconds
        self._speed_clip = speed_clip
        # Public mutable knobs the trainer / eval set at runtime, same names as
        # dribble's so the trainers stay interchangeable.
        self.shaping_scale = 1.0
        self.fixed_start = fixed_start
        self.target_cone = target_cone
        self._reward_coef = reward_coef
        self.out_of_play_dist = out_of_play_dist
        if reward is None:
            common = dict(mode=reward_mode, w_strike=w_strike,
                          w_player_to_ball=w_player_to_ball,
                          w_ball_to_cmd=w_ball_to_cmd,
                          approach_scale=approach_scale)
            reward = (KickToPointReward(w_arrive=w_arrive,
                                        reward_coef=reward_coef, **common)
                      if reward_kind == "point" else KickReward(**common))
        super().__init__(num_worlds=num_worlds, creature_xml=creature_xml,
                         episode_seconds=episode_seconds, use_gpu=use_gpu,
                         device=device, seed=seed, use_graph=use_graph,
                         nconmax=nconmax, njmax=njmax, reward=reward,
                         floor_half=floor_half, energy_coef=energy_coef,
                         smooth_coef=smooth_coef, rew_clip=rew_clip,
                         backend_cls=backend_cls)

    # -- scene --------------------------------------------------------------
    def _ball_spec(self):
        return self._ball or BallSpec()

    # -- task ---------------------------------------------------------------
    def _task_dim(self):
        return 12  # ball_ego(6) + target_ego3(3) + cmd_dir_ego3(3)

    def _task_init(self):
        self._init_segments(segment_seconds=self._segment_seconds,
                            speed_clip=self._speed_clip)
        self.ball_spawn_range = self._clamped_spawn_range(self.ball_spawn_range)
        # Arrival tracking, mirroring shoot's seg_goal_best. `seg_target_best` is
        # the CLOSEST the ball has come to the commanded point this segment, so a
        # kick is graded on where the ball actually got, not on how fast it left.
        dev = self.device
        self.seg_target_best = torch.full((self.n,), float(self.target_dist),
                                          device=dev)
        self.target_fit_sum = torch.zeros(self.n, device=dev)
        self.prev_target_fit_sum = torch.zeros(self.n, device=dev)
        self.prev_n_segments = torch.zeros(self.n, device=dev)

    def _target_dist_now(self):
        return torch.linalg.norm(self._ball_xy() - self.target_xy, dim=-1)

    def _task_obs(self):
        return torch.cat([self._ball_ego6(),
                          self._xy_ego3(self.target_xy),
                          self._dir_ego3(self.cmd_dir)], -1)

    def _update_task(self):
        _, released = self._strike_update()
        self.seg_t += 1.0

        # Bank the strike at release, but DO NOT end the segment there -- the
        # ball still has to travel, and where it ends up is the whole point.
        # This is the structural fix behind the aim problem: the old code ended
        # the segment the instant the ball left the creature, so nothing after
        # contact could be measured and the only gradient available was "hit it
        # harder". Measured on kick_ant_v1: median aim error 35 deg, 37% of ball
        # speed thrown away, while fitness rose the whole run. shoot never had
        # this bug because it keeps its segment open and scores arrival.
        self._bank(released)
        self.seg_target_best = torch.minimum(self.seg_target_best,
                                             self._target_dist_now())

        timeout = self.seg_t >= self.segment_steps
        # Out of play: the ball has rolled so far past the target that the
        # segment cannot be rescued, so holding it open only wastes steps.
        out = self._target_dist_now() > self.out_of_play_dist
        end = timeout | out
        # A segment that ends without release ever firing (creature still
        # standing over the ball) still banks whatever strike it made.
        self._bank(end)
        if bool(end.any()):
            idx = end.nonzero(as_tuple=True)[0]
            self.target_fit_sum[idx] += torch.exp(
                -self._reward_coef * self.seg_target_best[idx])
        self._close_segments(end)

    # -- spawning -----------------------------------------------------------
    def _spawn_worlds(self, idx, root_xy=None, yaw=None):
        """Place the ball and draw a fresh command for worlds `idx`.

        The CREATURE IS NOT MOVED on a mid-episode segment restart: it keeps
        walking from wherever the last strike left it and a new ball appears
        1.5-3 m away. Teleporting it every few seconds would make the drill a
        sequence of unrelated freeze-frames and would hand the shaping potential
        a discontinuity on every restart. (`shoot` does move it -- it has to be
        in front of a goal.)

        `root_xy`/`yaw` let the caller supply a root pose it has just written but
        not yet run forward() on (episode reset, divergence sanitize); otherwise
        the creature's live pose is read from the model.
        """
        k = int(idx.numel())
        if root_xy is None:
            root_xy = self._root_frames()[0][idx, :2]
        if yaw is None:
            yaw = self._root_yaw(idx)

        # fixed_start (curriculum stage 1): ball straight ahead of the creature
        # and the command colinear with it, so simply walking forward strikes the
        # ball toward the target. The world yaw is invisible to an egocentric obs,
        # so this constrains the TASK, not the pose.
        bang = yaw if self.fixed_start else self._rand(k) * (2 * np.pi)
        b0, b1 = self.ball_spawn_range
        bdist = b0 + (b1 - b0) * self._rand(k)
        ball_xy = root_xy + torch.stack([bdist * torch.cos(bang),
                                         bdist * torch.sin(bang)], -1)
        # Keep the ball inside the arena. Segments restart wherever the creature
        # happens to be, so late in an episode it can be against a wall -- and a
        # ball spawned 3 m past it lands on the far side of the barrier (MuJoCo
        # planes collide as INFINITE planes regardless of their visual size, so
        # it rests there quite happily) and the segment can only time out.
        lim = max(1.0, self._floor_half - 1.0)
        ball_xy = ball_xy.clamp(-lim, lim)
        self._write_ball(idx, ball_xy)

        if self.fixed_start:
            # Stage 2+: the command may sit up to +/- target_cone off the line
            # the creature is already walking, so it must learn to steer.
            cang = bang + (self._rand(k) * 2.0 - 1.0) * self.target_cone
        else:
            cang = self._rand(k) * (2 * np.pi)
        cmd = torch.stack([torch.cos(cang), torch.sin(cang)], -1)
        self.cmd_dir[idx] = cmd
        self.target_xy[idx] = ball_xy + cmd * self.target_dist
        # Baseline arrival at the spawn separation, so a segment that never
        # touches the ball scores exp(-c * target_dist) rather than a free 1.0.
        if hasattr(self, "seg_target_best"):
            self.seg_target_best[idx] = torch.linalg.norm(
                ball_xy - self.target_xy[idx], dim=-1)

    def _reset_state(self):
        n = self.n
        idx = torch.arange(n, device=self.device)
        yaw = self._rand(n) * (2 * np.pi)
        self._spawn_root(xy=torch.zeros(n, 2, device=self.device), yaw=yaw)
        # Pass the pose explicitly: qpos has been written but forward() has not
        # run yet, so xpos/xmat still hold the PREVIOUS episode's pose.
        self._spawn_worlds(idx, root_xy=torch.zeros(n, 2, device=self.device),
                           yaw=yaw)
        self._reset_segments(idx)
        # Same fallback shape as credit_sum: a rollout that lands inside a fresh
        # episode reports the PREVIOUS episode's arrival rather than 0/0. Both
        # snapshots must be taken BEFORE _reset_episode_stats, which zeroes
        # n_segments.
        self.prev_target_fit_sum = self.target_fit_sum.clone()
        self.prev_n_segments = self.n_segments.clone()
        self.target_fit_sum.zero_()
        self._reset_episode_stats()

    def _sanitize_task(self, idx):
        # The base has already zeroed this world and stood the creature at the
        # origin with identity orientation; re-place the ball around it and start
        # a fresh segment rather than leaving a stale command pointing nowhere.
        k = int(idx.numel())
        zeros = torch.zeros(k, device=self.device)
        self._spawn_worlds(idx, root_xy=torch.zeros(k, 2, device=self.device),
                           yaw=zeros)
        self._reset_segments(idx)
        self._pending_reset[idx] = True
