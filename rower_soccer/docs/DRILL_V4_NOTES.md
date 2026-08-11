# Drill v4 implementation notes

*Written 2026-08-11 alongside the code. The design is `DRILL_V4_SPEC.md`; this
records what was MEASURED, what was built, the two places the implementation
deliberately departs from the spec, and how the runs were launched.*

Scope: `warp_port/{ball_task,kick_env,shoot_env,train_kick_warp,train_shoot_warp}.py`,
the new `warp_port/probe_strike_speed.py`, and `tests/test_drill_v4.py`.
`skills/` and `game/` are untouched — see "What still has to change" at the end.

## 1. Achievability: what the ant can actually do

The spec's own warning is that a `v_pace` band above the body's ceiling gives a
flat gradient and a run that learns nothing. So the band was measured off
`kick_ant_v3/best.pt` (snapshotted mid-run to `/tmp`, the live run untouched)
in v3's exact env — pitch, `pitch_scale 0.3125`, 0.15 m / 0.045 kg ball,
`target_dist_range 3 6` — with 256 worlds, 937 segments, 95.7% of them
containing a strike:

    reproduce: MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m \
      rower_soccer.warp_port.probe_strike_speed --checkpoint runs_v2/kick_ant_v3/best.pt

| quantity                          | min  | p10  | median | mean | p90   | max   |
|-----------------------------------|------|------|--------|------|-------|-------|
| ball speed at contact-break (m/s) | 0.95 | 7.93 | **15.23** | 14.60 | 20.43 | 23.67 |
| time to first touch (s)           | 0.03 | 0.70 | **1.35**  | 1.37  | 2.12  | 3.50  |

**The strike speed is not the number the deadline cares about.** A 45 g,
0.15 m ball is nearly a balloon, so the ant launches it at a median 15 m/s —
but the deadline `T` starts at SEGMENT SPAWN, and two things eat it: the ant
spends a median 1.35 s walking to the ball, and rolling friction (~4.4 m/s²
on this ball, i.e. 4 m/s dies inside 2 m) bleeds the ball off fast. The
quantity `v_pace` actually names is `d / t_reach(d)`, measured from segment
start:

| pace over…      | min  | p10  | median | mean | p90  | max  | reached at all |
|-----------------|------|------|--------|------|------|------|----------------|
| 3 m             | 0.64 | 1.12 | **1.64** | 1.81 | 2.67 | 6.32 | 90.4% |
| 4 m             | 0.84 | 1.44 | **2.11** | 2.29 | 3.27 | 7.62 | 86.6% |
| 5 m             | 1.03 | 1.69 | **2.53** | 2.70 | 3.92 | 8.33 | 84.3% |
| 6 m             | 1.13 | 1.94 | **2.89** | 3.09 | 4.44 | 9.23 | 79.7% |

### Chosen band: `v_pace ~ U(1.5, 3.0) m/s`, deadline clamp [0.5, 4.0] s

The spec's default `U(2, 6)` is rejected on this evidence. At 6 m/s a 3 m pass
is due in 0.5 s and the ant needs 1.35 s just to REACH the ball — the entire
upper half of that band is unreachable at every distance, which is precisely
the flat-gradient failure the spec warns about. `U(1.5, 3.0)` spans, over the
3-6 m target band, deadlines of 1.0-4.0 s:

- easiest corner (6 m at 1.5 m/s, T = 4.0 s): comfortable to reach, and the
  challenge becomes NOT overshooting — a full-power 15 m/s strike would roll
  25 m. This is the pace-modulation case.
- hardest corner (3 m at 3.0 m/s, T = 1.0 s): at the p10 of v3's realised pace,
  so reachable but demanding — and the way to reach it is to hurry the
  approach, which is a gradient the policy can climb.
- dribbling is excluded everywhere: even the slowest corner needs the ball to
  average 2.3 m/s over its own travel, above the ant's ~1 m/s top speed.

The [0.5, 4.0] s clamp is kept as the spec specifies, but with this band it
only binds at the top (6 m / 1.5 m/s = exactly 4.0 s) — the pace, not the
clamp, is what sets difficulty.

### The reward is not flat over what the ant controls

A 5-minute smoke cannot show learning, so the objective was probed directly:
ball velocity injected at a realistic strike time, arrival-at-T read from the
env, target 4.5 m out.

| pace (T)        | strike at | flight budget | best arrival | at v0    | worst | span |
|-----------------|-----------|---------------|--------------|----------|-------|------|
| 1.50 (3.00 s)   | 0.70 s    | 2.30 s        | 0.808        | 6 m/s    | 0.002 | 0.81 |
| 1.50 (3.00 s)   | 1.35 s    | 1.65 s        | 0.601        | 7 m/s    | 0.002 | 0.60 |
| 2.25 (2.00 s)   | 0.70 s    | 1.30 s        | 0.775        | 6 m/s    | 0.003 | 0.77 |
| 2.25 (2.00 s)   | 1.35 s    | 0.65 s        | 0.924        | 9 m/s    | 0.075 | 0.85 |
| 3.00 (1.50 s)   | 0.70 s    | 0.80 s        | 0.903        | 8 m/s    | 0.030 | 0.87 |
| 3.00 (1.50 s)   | 1.35 s    | 0.15 s        | 0.326        | 17 m/s   | 0.137 | 0.19 |

Three things this establishes. The surface has a **span of 0.6-0.87** against a
do-nothing baseline of ~0.105, so there is signal for both the reward and
checkpoint selection. It has an **interior optimum at 6-9 m/s** — inside the
measured strike range but well BELOW the current median of 15.2, so the ant
must learn to strike softer, and both overshooting and undershooting are
punished. And the one hard row (pace 3.0 reached with a leisurely 1.35 s
approach, 0.33) becomes the best row in the table (0.90) when the same strike
happens at 0.70 s — the deadline pressure has a reachable answer.

## 2. kick — the timed reward (`--reward-kind timed`)

At segment spawn: target distance `d ~ U(3, 6)`, pace `v_pace ~ U(1.5, 3.0)`,
`T = clamp(d / v_pace, 0.5, 4.0)` discretized to whole 25 ms control steps. The
segment ends at exactly `T` — nothing else ends it, not even an out-of-play
ball, because grading position at any other instant is not grading position at
`T`.

    reward  = w_arrive * exp(-c * ||ball(T) - target||_3D)   [paid once, at T]
              + w_strike * banked strike speed                [w_strike = 0]
              + shaping_scale * (me->ball approach ONLY)
              all * upright

    fitness = mean over the episode's CLOSED segments of exp(-c * d_at_T),
              falling back to the previous episode's mean in a world that has
              not closed one yet

`w_strike` now defaults to **0** under `timed` (resolved in the trainer so the
value lands in `config.json`): power is a consequence of the deadline, and
paying for it separately prices the same thing twice and in one direction only.
The `ball->cmd_dir` velocity shaping term is **forced to zero in the reward
class**, not merely defaulted off — it pays monotonically for "faster toward
the target", which is the exact behaviour pace modulation has to override. The
`me->ball` approach term is kept; reaching the ball is a prerequisite and it is
what makes early exploration work.

The in-flight segment is excluded from fitness: its arrival is undefined until
the deadline.

### Obs change — kick task block 12 → 14

    proprio(65) | ball_ego(6) | target_ego3(3) | cmd_dir_ego3(3)
                | t_remaining(1) | required_pace(1)              = 79 total

- `t_remaining` = `seg_T - seg_t * CONTROL_DT`, seconds, ≥ 0. `seg_T` is stored
  already discretized, so this hits exactly 0.0 on the step the segment ends —
  the clock the policy sees is the clock the env enforces.
- `required_pace` = `||ball - target|| / t_remaining`, capped at
  `REQ_PACE_CAP = 10.0` (it diverges at the deadline, and an unbounded obs
  trips `ppo.OBS_SANITY_LIMIT`).
- **The first 12 columns and their order are unchanged**, so a v3 or dribble
  checkpoint still transfers its decoder and only the two new columns of the
  task encoder re-initialise.
- The widening is **conditional on `reward_kind == "timed"`** (`_task_dim`
  returns 14 or 12). It is not the new width for everybody, because
  `skills/registry.py` pins kick to `kick_ant_v3/best.pt`, whose task encoder is
  12 wide — widening unconditionally would break the checkpoint the game loads
  today.

## 3. shoot — urgency

- Goal bonus is now `goal_bonus * exp(-k * t_score)`, `k = --goal-time-coef`,
  default **0.4**: a 1 s goal keeps 0.67 of the bonus, 3 s 0.30, 5 s 0.14.
  `t_score` is time within the segment, and the creature is respawned in front
  of the mouth at every segment start, so it is "time since this attempt began".
- `t_score` is read from `env.last_score_t`, a snapshot taken BEFORE the
  respawn. A goal ends its segment, so by the time the reward runs the live
  per-segment clock is already back at 0 — reading it there would pay the full
  flat bonus for every goal and the urgency term would be dead code that still
  looked right. `tests/test_drill_v4.py` pins this.
- fitness per segment, defined once in `shoot_env.seg_fitness` and read by both
  the episode accumulator and `ShootReward.fitness`:

      scored ?  0.5 + 0.5 * exp(-k * t_score)  :  0.5 * exp(-c * d_mouth_best)

Verified against the trained v3 policy (shoot's obs width is unchanged, so
`shoot_ant_v3/best.pt` loads straight into the new env): over **1095 goals**,
`t_score` median 1.43 s (range 0.85-4.90), so the bonus paid spans **0.70-3.56**
where v3 paid a flat 5.00 every time. That is a 5x spread of real gradient on
speed where there was none.

### Deviation from the spec, deliberate

The spec writes the scored branch as bare `exp(-k * t_score)` and justifies the
miss branch's 0.5 ceiling with "any goal outranks any miss" — but at k = 0.4
the bare form crosses 0.5 at `ln(2)/k = 1.73 s`, so a goal scored in 3 s (0.30)
would rank BELOW a shot that merely grazed the post (0.5). Given v3's measured
median goal time of 1.43 s, a large fraction of real goals would land on the
wrong side of that line. Since the stated invariant is the whole reason the
formula has two branches, the scored branch is mapped affinely into the upper
half instead: goals occupy (0.5, 1], misses [0, 0.5]. Ordering within each
branch, the [0,1] bound and the calibration of k are all unchanged. This was
caught by a test asserting the invariant, not the formula.

## 4. Tests

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m tests.test_drill_v4
    ... --no-physics        # reward algebra only, ~2 s, no GPU

9/9 pass. Five algebra checks drive the reward objects against a stub env (goal
bonus discounting and its calibration; the pre-respawn clock regression; the
goal-outranks-miss invariant; `w_b2c` really absent from the timed shaping, not
just zero in a field; timed fitness aggregation and its fallback). Four
physics checks build the real Warp envs: task width 12/14, deadlines matching
`clamp(d/pace)` to within a control step with segments ending exactly on them
and never past them, `t_remaining` in the obs agreeing with the env's own
clock, and arrival paid on the closing step and nowhere else.

## 5. Launches

Both runs: frozen `runs_v2/_decoder_ant_final.pt`, pitch at 0.3125, 0.15 m /
0.045 kg ball, 2048 worlds, `--max-hours 48 --steps 5000000000`, syncing to
`gs://vc2-2026-checkpoints`.

```
MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.warp_port.train_shoot_warp \
  --run-name shoot_ant_v4 --creature-xml creature_configs/ant.xml \
  --arena pitch --pitch-scale 0.3125 \
  --init-from runs_v2/_decoder_ant_final.pt --freeze-decoder \
  --gcs-bucket vc2-2026-checkpoints --max-hours 48 --steps 5000000000 \
  --ball-radius 0.15 --ball-mass 0.045 --w-upright 1.0 --goal-time-coef 0.4 --no-wandb

MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.warp_port.train_kick_warp \
  --run-name kick_ant_v4_timed --creature-xml creature_configs/ant.xml \
  --arena pitch --pitch-scale 0.3125 \
  --init-from runs_v2/_decoder_ant_final.pt --freeze-decoder \
  --gcs-bucket vc2-2026-checkpoints --max-hours 48 --steps 5000000000 \
  --ball-radius 0.15 --ball-mass 0.045 \
  --reward-kind timed --w-arrive 3.0 --w-upright 1.0 \
  --target-dist-range 3.0 6.0 --pace-range 1.5 3.0 --deadline-range 0.5 4.0 --no-wandb
```

Two notes on those command lines.

**The kick run is `kick_ant_v4_timed`, not `kick_ant_v4`.** That name is
already taken by a real 450M-step run from 2026-08-09 (`reward_kind point`,
`target_dist_range 4-8`) with local artifacts, a `gs://vc2-2026-checkpoints/
kick_ant_v4/` prefix and wandb history under that id. Reusing it would have
resumed that wandb run's curves and interleaved two different objectives'
checkpoints in one GCS prefix, so the one-name-one-run rule forced a new name.

**`--no-wandb`.** The `WANDB_API_KEY` is present in the environment of the
already-running v3 processes but not in a form this session could legitimately
read. To turn wandb on, export the key and relaunch with `--resume` — kick and
shoot carry no curriculum state, so the only cost is the steps taken so far.

## 6. What still has to change before v4 is adopted

Not done here, deliberately — `skills/` and `game/` are owned elsewhere and
`kick_ant_v3`/`shoot_ant_v3` remain the registry pins and the fallback.

1. `skills/registry.py`: the kick entry's `fields` and checkpoint must move to
   the 14-wide contract (`PROPRIO_V1 + ball_ego, strike_target_ego3,
   cmd_dir_ego3, + two new deadline fields`) when a timed checkpoint is adopted.
   Shoot's width is unchanged (13) — only its checkpoint would move.
2. `skills/fields.py`: two new fields for `t_remaining` and `required_pace`.
3. The game's kick command must supply a pace/deadline. Simplest, per the spec:
   fix `v_pace` to a mid-band constant (≈2.25 m/s given the band above), so a
   human click means "pass it there, briskly", and derive `t_remaining` on the
   client from the click time.

## 7. The anchor arm (v7) is a clean NULL result — dribbling was not the problem

*Measured 2026-08-11.*

v7 added the spawn anchor (see `TimedKickReward.w_anchor`): the me->ball approach
shaping re-aimed at the ball's spawn point instead of the live ball, plus a
per-step penalty for straying from it. v6 is its exact control — identical flags,
`--w-anchor 0` — so the pair isolates the anchor.

**The mechanism works.** The reported `anchor=` stat fell 0.76 -> 0.29 m: the ant
really does stay next to where the ball lay and strike from there, instead of
travelling with it.

**It changes nothing.** Compared over the SAME step window (0 - 35.4M, since v7
is younger):

| | n | mean fitness | median |
|---|---|---|---|
| v6 (control) | 54 | 0.1254 | 0.1240 |
| v7 (anchor) | 54 | 0.1251 | 0.1245 |

Last quarter of that window: 0.1189 vs 0.1224. Indistinguishable.

A methodological note worth keeping, because it nearly produced a false positive:
comparing the two runs' LATEST monitor lines gave "v7 0.127 vs v6 0.104", which
looks like a win and is not one — those are different step counts on a noisy
curve. Control arms must be compared at matched steps, never at matched
wall-clock.

**What this rules out.** Three timed arms — v4 (reward_coef 0.5), v6 (gentler
0.2, closer spawn, longer deadline), v7 (v6 + anchor) — all sit at ~0.12 against
a 0.105 do-nothing baseline. The reward-shape hypothesis (v4->v6) and the
dribbling hypothesis (v6->v7) have both now been tested and neither moved it. The
remaining suspect is the timed formulation itself, not its shaping terms.

Do NOT read v3's 0.376-0.395 as "the untimed kick works better": v3's fitness is
`exp(-c * closest approach over the window)` while v4/v6/v7 grade distance AT the
deadline. Different measures; the numbers are not comparable.

## 8. What is actually wrong with kick: it contacts the ball and cannot aim

*Measured 2026-08-11 on `kick_ant_v7_anchor/best.pt`, deterministic actions,
~1800 closed segments per probe.*

Three hypotheses were tested and two died.

**"The task is unreachable."** No. Over 1837 closed segments the median
ball-to-target distance at the deadline is 4.40 m -- exactly the do-nothing
outcome, since targets spawn 3-6 m from the ball (mean 4.5). But the tail is
0.26 m at best, 1.53 m at pct99, 3.09 m at pct90, and a ball that never moves
can NEVER score below 3.0 m. So real passes do happen; the distribution is
bimodal, not flat. (Sanity check that probe and trainer agree:
`exp(-0.5 * 4.40) = 0.111`, which is the reported fitness.)

**"Most segments end with no contact."** No. **93.7% of segments are touched.**

**"Longer deadlines are worse."** This one LOOKED true and was a confound worth
recording. Raw medians by deadline fell monotonically -- 3.60 / 4.66 / 5.08 /
5.71 m as T went 1.5-2.5 / 2.5-3.5 / 3.5-4.5 / 4.5-6 s, i.e. fitness 0.165 down
to 0.058 -- which reads as "more time to touch the ball is worse". But
`T = target_dist / pace`, so long-T buckets have farther targets BY
CONSTRUCTION and each bucket was simply sitting at its own do-nothing baseline.
Normalising to gain = (start distance) - (distance at T) flattens it completely:

| T | start | gain | % closer |
|---|---|---|---|
| 1.5-2.5 s | 3.63 m | +0.00 m | 44.7% |
| 2.5-3.5 s | 4.55 m | -0.01 m | 46.3% |
| 3.5-4.5 s | 5.09 m | -0.08 m | 44.6% |
| 4.5-6.0 s | 5.57 m | -0.00 m | 49.2% |

**What is left is the answer: the strike direction is uncorrelated with the
target.** Median gain -0.00 m; the ball ends closer in 45.8% of segments and
farther in 54.2%. A coin flip, marginally worse than chance. The policy has
learned to locomote to the ball and make contact, and has learned nothing about
where to send it.

This is the same deficit `shoot` shows. Shoot's goal subtends +/-37 to 62 deg
from 2-5 m, so its post hits imply ~40 deg strike-direction error, and kick v1
measured a median aim error of 35 deg. One underlying problem, both drills:
**contact is solved, direction is not.**

Consequences for the arms already run: v4 -> v6 (reward curve) and v6 -> v7
(anchor) were both tuning the wrong thing. No reward-shaping variant can fix a
policy whose strike direction carries no information about the target; shaping
changes what is rewarded, not what the controller is able to express.

The open question -- whether the ant fails to POSITION itself on the far side of
the ball, or positions correctly and mis-strikes -- is what to measure next, and
it splits the fix. Bad positioning is a task/exploration problem. Correct
positioning with a bad strike points at the FROZEN DECODER: it was trained by
`follow` to track a target velocity, so its action repertoire is "walk in
direction X at speed Y", and a directed strike may simply be outside what
154,632 frozen parameters can express. That would predict an unfrozen kick run
gains aim where every frozen variant has not -- a cheap and decisive experiment.
