# Drill → Soccer observation bridge

Reference for [`rower_soccer/soccer_bridge.py`](../soccer_bridge.py): exactly what the drill-trained
**follow** policy expects, what the **soccer** env actually provides, how the two differ, and what the
bridge synthesizes vs. drops.

All dims below are for the **worm** (`three_seg_worm`, action dim 2). Rower deltas are noted where they
differ: `bodies_pos` 9→27, `joints_pos`/`joints_vel` 2→8, `touch_sensors` 3→9, `prev_action`/action 2→8.
The proprioception *keys* and the bridge logic are identical for both.

---

## 1. What the follow model consumes (drill contract)

The follow policy was trained on the flattened output of
[`drills/gym_wrap.py`](../drills/gym_wrap.py) (`DrillGymEnv`): observables concatenated in
**sorted-key order**, with `strip_singleton_obs_buffer_dim=True`. Proprio keys carry a `creature/`
prefix (because `composer.Environment` fully-qualifies walker observables); the task keys do not.

| # | key (sorted) | dim | group | source |
|---|---|---|---|---|
| 1 | `creature/absolute_root_mat` | 9 | proprio | world 3×3 xmat of `seg0` ([creature.py:210](../../creature.py#L210)) |
| 2 | `creature/absolute_root_pos` | 3 | proprio | world xpos of `seg0` |
| 3 | `creature/bodies_pos` | 9 *(27)* | proprio | ego framepos of each body vs `seg0` |
| 4 | `creature/joints_pos` | 2 *(8)* | proprio | actuated-joint qpos |
| 5 | `creature/joints_vel` | 2 *(8)* | proprio | actuated-joint qvel |
| 6 | `creature/sensors_accelerometer` | 3 | proprio | `seg0` accelerometer |
| 7 | `creature/sensors_gyro` | 3 | proprio | `seg0` gyro |
| 8 | `creature/sensors_velocimeter` | 3 | proprio | `seg0` velocimeter |
| 9 | `creature/touch_sensors` | 3 *(9)* | proprio | per-segment touch, ÷10000 |
| 10 | `target_ego` | 2 | **task** | target in root frame, now |
| 11 | `target_ego_future` | 2 | **task** | target in root frame, +1.0s lookahead |

**Total = 41** (proprio 37 + task 4). *(Rower: 73 + 4 = 77.)* Dribble adds one task key
`ball_ego` (4) → 45 *(rower 81)*.

The checkpoint bakes the proprio/task split as index buffers (`p_idx = 0..36`, `t_idx = 37..40`), so the
bridge **must** reproduce this exact order and length — see
[`play_interactive.load_latent_policy`](../play_interactive.py) (dimension guard).

---

## 2. What the soccer env provides (per player)

Soccer obs are a **list**, one dict per player (`timestep.observation[player_index]`), keys **unprefixed**,
and each value keeps a leading singleton buffer dim (no `strip_singleton_obs_buffer_dim`), e.g.
`absolute_root_pos` has shape `(1, 3)`. A single home worm, no away team, yields 28 keys / 74 scalars:

| group | keys (dim) |
|---|---|
| **Shared proprio** (same as drill, minus the prefix) | `absolute_root_mat` (9), `absolute_root_pos` (3), `bodies_pos` (9), `joints_pos` (2), `joints_vel` (2), `sensors_accelerometer` (3), `sensors_gyro` (3), `sensors_velocimeter` (3), `touch_sensors` (3) |
| **Previous action** *(soccer-only)* | `prev_action` (2) |
| **Ball, egocentric** *(soccer-only)* | `ball_ego_position` (3), `ball_ego_linear_velocity` (3), `ball_ego_angular_velocity` (3) |
| **Goals / field, egocentric** *(soccer-only)* | `team_goal_back_right` (2), `team_goal_front_left` (2), `team_goal_mid` (3), `opponent_goal_back_left` (2), `opponent_goal_front_right` (2), `opponent_goal_mid` (3), `field_front_left` (2), `field_back_right` (2) |
| **Game stats** *(soccer-only)* | `stats_vel_to_ball`, `stats_closest_vel_to_ball`, `stats_veloc_forward`, `stats_vel_ball_to_goal`, `stats_home_avg_teammate_dist`, `stats_teammate_spread_out`, `stats_home_score`, `stats_away_score` (all 1) |

With opponents/teammates present (e.g. 1v1), each other player adds a prefixed block plus the walker's own
`end_effectors_pos`: `opponent_0_ego_position` (3), `opponent_0_ego_orientation` (9),
`opponent_0_ego_linear_velocity` (3), `opponent_0_ego_end_effectors_pos`, `opponent_0_end_effectors_pos`,
and `end_effectors_pos` (teammate blocks use the `teammate_i_` prefix). The bridge uses **0 away players**,
so none of these appear.

---

## 3. Differences that matter

| # | Difference | Drill (model expects) | Soccer (provides) | Bridge action |
|---|---|---|---|---|
| 1 | **Key prefix** | `creature/joints_pos` | `joints_pos` (unprefixed) | re-prefix each proprio key |
| 2 | **Buffer dim** | stripped, `(N,)` | leading singleton, `(1, N)` | `.ravel()` before use; `build_obs` ravels on flatten |
| 3 | **Task obs present?** | `target_ego`, `target_ego_future` | **absent** | **synthesize** from a hardcoded target + root pose |
| 4 | **Extra soccer obs** | none | `prev_action`, `ball_ego_*`, goal/field, stats, teammate/opponent | **drop** (model never saw them) |
| 5 | **Ordering** | sorted-key; proprio 0..36, task 37..40 | dict (order irrelevant) | assemble in drill sorted order |
| 6 | **Ball encoding** (dribble only) | `ball_ego` = 4 = `_to_ego`(ball_xy) ⊕ ball planar vel | `ball_ego_position` (3, body-frame) + `ball_ego_linear_velocity` (3) | not handled yet (follow ignores the ball) |
| 7 | **Physics timestep** | 0.0025 (10 substeps) | 0.005 (5 substeps) by default | control rate identical (40 Hz); the bridge **matches the drill's 0.0025 by default** (`--no-match-physics-dt` keeps 0.005) |
| 8 | **Arena** | flat `Floor(30×30)`, no walls | `RandomizedPitch(40×30)` + walls + goalposts + ball | ignored by design (egocentric policy; wall/goal collisions unhandled) |

Key insight: rows 1–2 are pure re-keying/reshaping — the proprio **values are byte-identical** because both
envs use the same `Creature` walker. Only row 3 (the task signal) requires real computation.

---

## 4. What the bridge converts

[`soccer_to_drill_follow_dict(soccer_obs0, target_xy, proprio_bases, task_keys)`](../soccer_bridge.py)
builds the drill FOLLOW dict; `drill_follow_obs(...)` flattens it via
[`play_interactive.build_obs`](../play_interactive.py) into the trained sorted order.

- **Proprio (37 dims): copied verbatim.** For each drill key `creature/X`, pull soccer's `X`, ravel, and
  store under `creature/X`. No value change.
- **Task (4 dims): synthesized.** The root frame is read from the soccer obs
  (`root_xy = absolute_root_pos[:2]`, `R = absolute_root_mat.reshape(3,3)`, `fwd = R[:,0]`, `left = R[:,1]`)
  and the hardcoded world target is projected with the **exact** egocentric transform from
  [`drills/follow.py:_to_ego`](../drills/follow.py#L109-L115): `to_ego(w) = [ (w−root)·fwd, (w−root)·left ]`.
  Because the target is **static**, `target_ego_future == target_ego` (mirrors a stopped drill target,
  `_target_vel = 0`).
- **Everything else: dropped** — `prev_action`, all `ball_ego_*`, goals, field, stats, and any
  teammate/opponent blocks.

Result: a 41-dim vector *(rower 77)* in exactly the layout the checkpoint's `p_idx`/`t_idx` expect, so the
policy runs unchanged. The layout list, proprio bases, and task keys are read from a throwaway drill env at
startup ([`reference_follow_layout`](../soccer_bridge.py)), keeping this correct if the creature changes.

---

## 5. Extending to dribble (not yet implemented)

A dribble policy additionally needs `ball_ego` (4). Soccer's native `ball_ego_position` (3, MuJoCo framepos
in the **root body** frame) and `ball_ego_linear_velocity` (3) use a different frame and dim than the drill's
`ball_ego` (2D planar `_to_ego` of the ball + 2D planar velocity, [dribble.py:32-35](../drills/dribble.py#L32-L35)).
The bridge must recompute the drill-style `ball_ego` from the soccer ball's world pose (via `_to_ego` on the
ball body), **not** reuse soccer's ball observables directly.
