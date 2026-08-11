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

## 9. v8 = approach the STRIKE POINT (the code landed in cd02a8d)

*Housekeeping note: the v8 source changes were swept into commit `cd02a8d` by a
`git add -A`, whose message describes only the D2 retraction. The rationale
lives here instead. `ball_task.TimedKickReward.strike_offset`,
`kick_env`/`train_kick_warp` plumbing, and the geometry test are all in that
commit.*

Splitting section 8's finding one level further, on v7/best.pt over 55,303
ball-moving samples, against a random baseline of median 90 deg / 16.7% within
30 deg / 50% within 90 deg:

| | median | within 30 deg | within 90 deg |
|---|---|---|---|
| positioning (ant->ball vs ball->target) | 103.9 deg | 13.1% | 42.4% |
| aim (ball velocity vs ball->target) | 93.2 deg | 15.7% | 48.2% |

Both at or slightly WORSE than random, so this is not "positions well, strikes
badly" -- there is no positioning skill at all. Worse, being biased PAST 90 deg
means the ant tends to stand between the ball and the target, so contact pushes
the ball away from where it should go. That is the -0.12 m mean gain of section
8, explained.

**The approach shaping is the cause.** To send a ball somewhere you must first
reach the far side of it. `_StrikeReward._shaping` pays
`w_p2b * (speed TOWARD the approach point)` on every step, so with the approach
point ON the ball, the circling manoeuvre the task requires is penalised the
whole way round. The policy is paid to charge straight at the ball from
wherever it happens to be, and the strike direction is then whatever the
approach direction happened to be -- i.e. random, which is exactly what is
measured.

`--strike-offset 0.5` moves the approach point to 0.5 m behind the ball on the
ball->target line, reusing the `_approach_xy` hook v7 introduced. Walking there
and continuing forward IS the kick.

This is deliberately NOT another outcome-reward retune. v4->v6 (reward curve)
and v6->v7 (spawn anchor) both re-priced the outcome and both were null at
matched steps. Shaping cannot make a controller express a behaviour it is
simultaneously being paid to avoid.

Arms now live: v4 (original timed), v6 (control), v8 (strike point). v7 retired
as null. v6 differs from v8 only in `--strike-offset`, so the pair is a clean
comparison -- at MATCHED STEPS, per section 7's lesson.

## 10. `best.pt` is a running max over SINGLE-EPISODE evals, so it selects noise

*Measured 2026-08-11 across all four live drills.*

> **Correction (see section 14).** The `typical` column below is the TRAINING
> monitor's fitness -- stochastic actions, sampled mid-episode, averaged over
> 2048 worlds -- while `best.pt fitness` is the DETERMINISTIC end-of-episode
> eval. They are different quantities and the ratios are therefore inflated.
> Re-scoring `dribble_ant_v3/best.pt` on the correct comparison gives a
> deterministic single-episode mean of **0.895**, not 0.60, against the pinned
> 0.980. The conclusion is unchanged and in fact sharper: ten fresh
> single-episode draws of that exact checkpoint peaked at 0.973, so the pin
> really is the top of its own draw distribution -- but the effect is ~1.1x on
> the right scale, not the 1.6x claimed here.

The saved `best.pt` fitness sits far above each run's typical fitness:

| run | best.pt fitness | typical (training monitor -- see correction) | ratio |
|---|---|---|---|
| dribble_ant_v3 | 0.980 | ~0.60 | 1.6x |
| kick_ant_v4_timed | 0.312 | ~0.12 | 2.6x |
| kick_ant_v6_timed | 0.300 | ~0.10 | 3.0x |
| shoot_ant_v4 | 0.689 | ~0.40 | 1.7x |

That is the signature of a running maximum over a noisy estimator, in every
run at once.

The mechanism is NOT a train/eval mix-up -- the code and its comment agree, and
`fit` there really is the deterministic `eval_video` number. The problem is that
the eval is **one world, one 15 s episode**, with the target/spawn band drawn
once. `best.pt` is then `max` over every such draw taken during the run
(dribble: ~136 of them). Max-of-N over a noisy statistic grows with N, so the
longer a run goes the more certainly `best.pt` is a lucky draw rather than a
better policy.

Consistency check: dribble's monitor fitness has mean 0.605 and stdev 0.191, so
0.980 is ~2 sigma out -- exactly where the max of ~136 draws should land. That
stdev is itself computed on a 2048-world average, so a single-episode stdev is
LARGER and the estimate is conservative.

Two consequences, both live:

1. **Every registry pin is such a checkpoint**, and so is every probe in
   sections 7-9 above (they all load `best.pt`). The kick diagnosis is not
   invalidated -- "positioning is at the random baseline" is far outside what an
   episode draw can manufacture, and the same result appears across v6/v7 -- but
   any FINE comparison between two `best.pt` files is comparing draw luck.
2. **A genuinely improving policy can stop being saved.** Once `best_score`
   ratchets to a 2-sigma outlier, later and better policies lose to it unless
   they also draw well.

Fix (not yet applied): score the checkpoint on a BATCHED deterministic eval --
the same one-world env is used only because it is the render env, and fitness
does not need the renderer. 64 worlds would cut the standard error ~8x for
negligible cost. Failing that, a rolling mean/median of the last K evals rather
than an instantaneous max.

Reporting rule that follows: quote drill fitness as a mean over recent samples,
never as a single monitor line. Doing the latter is how dribble got reported as
"0.77-0.79" for hours when its mean was 0.605 -- those were the tops of an
oscillation caused by the monitor's 320-step-per-world cadence aliasing against
the 600-step episode.

## 11. v8 first read: the strike point works, and the old shaping was anti-learning

*Measured 2026-08-11. Each row is 40k-126k ball-moving samples from a
deterministic rollout of that arm's `best.pt`. Random baseline: median 90 deg,
16.7% within 30 deg, 50% within 90 deg.*

| arm | steps | positioning median | within 30 | within 90 | aim median |
|---|---|---|---|---|---|
| v6 (control, no offset) | 88M | **123.9 deg** | 9.3% | 31.3% | 124.6 deg |
| v7 (spawn anchor) | 35M | 103.9 deg | 13.1% | 42.4% | 93.2 deg |
| **v8 (`--strike-offset 0.5`)** | **9.2M** | **84.0 deg** | **17.8%** | **53.5%** | 94.4 deg |

Two things, and the second is the more important one.

**v8 is the only arm better than random.** Positioning 84.0 deg and 53.5% within
90 deg: the creature is now on the correct side of the ball more often than not.
Aim has NOT moved yet (94.4 vs v7's 93.2), which is what the causal story
predicts -- aim is downstream of positioning, and v8 is 9.2M steps old.

**Under the old shaping, more training makes positioning WORSE.** v6 at 88M is
worse than v7 at 35M on near-identical treatments, and both are worse than
random. At 123.9 deg the creature systematically stands on the TARGET side of
the ball and shoves it away, and v6's aim (124.6 deg) has converged to its
positioning -- the signature of "the ball goes wherever the creature happened to
be walking".

This inverts the obvious objection rather than merely answering it. v8 is the
YOUNGEST arm and the best positioned; the arm with 10x more training is the
worst. Training time alone predicts the opposite ordering, so the step mismatch
cannot explain the result -- it works against it.

It also explains why sections 7-9's arms were flat. The drill was not failing to
learn; it was learning an anti-skill, because `w_p2b * (speed toward the ball)`
pays for charging straight at the ball from wherever the creature happens to
stand, and the straight line is on the wrong side of the ball half the time.
Every step of training reinforced it.

**Not yet established:** whether v8 HOLDS. The correct next measurement is to
re-probe v8 at ~35M and ~88M and compare to v7 and v6 at those same step counts,
because the whole point of this section is that the old arms degraded with
training. Fitness has not moved yet either (v8 ~0.125, still the do-nothing
baseline) -- positioning is the leading indicator, not the result.

Caveat carried from section 10: all three rows load `best.pt`, which is a max
over single-episode evals. That is fine here -- an angular median over 40k-126k
samples is not something one lucky episode can manufacture, and the ordering is
monotone across three arms -- but it is another reason not to read fine
differences between two `best.pt` files.

## 12. The control arm UNLEARNS, monotonically -- and v8's falsifiable prediction

*Measured 2026-08-11. Fitness quoted as block means, per section 10's rule.*

v6 (the old shaping, no strike offset) over its whole life, in 20M-step blocks:

| steps | fitness mean |
|---|---|
| 0-20M | 0.1288 |
| 20-40M | 0.1204 |
| 40-60M | 0.1129 |
| 60-80M | 0.1045 |
| 80-100M | **0.0974** |

Five consecutive blocks, each lower than the last, passing straight through the
0.105 do-nothing baseline and out the bottom. **The old kick drill makes the
policy worse the longer it trains.** This is the outcome-metric shadow of
section 11's angular measurement (v7 103.9 deg at 35M -> v6 123.9 deg at 88M):
two independent metrics, same direction, same story.

That reframes every earlier "the arms are flat at ~0.12" observation in sections
7-9. They were not flat. They were DECLINING, and averaging over a whole run
hid it.

**Where v8 actually stands.** At matched steps (0-19M, v8's full life so far):

| arm | n | mean | median |
|---|---|---|---|
| v6 control | 29 | 0.1290 | 0.1290 |
| v8 strike point | 29 | 0.1232 | 0.1240 |

Indistinguishable, if anything marginally behind. So v8 has fixed POSITIONING
(84.0 deg vs v6's 123.9, the only arm better than random) and has NOT yet
improved the OUTCOME. A leading indicator is not a result, and v8 is not yet a
win.

**The prediction, recorded before the data exists so it cannot be
rationalised afterwards:** if the strike point genuinely removes the
anti-learning, v8's fitness should stay flat or rise while v6's continues to
fall. Concretely, at ~98M steps v6 sits at 0.0974; v8 at ~98M should be
materially above that, and above the 0.105 baseline. If instead v8 also decays
to ~0.097, the strike offset has fixed an angle that does not matter and the
frozen-decoder hypothesis (section 8) becomes the live one.

Both arms run on. v8 needs ~80M more steps, a few hours at ~7k fps.

## 13. v8 is declining too, faster than its control -- v9 tests the decoder

*Measured 2026-08-11 at v8 = 27.5M steps.*

Section 12 predicted v8 would stay flat or rise while v6 fell. By 10M-step
block, at matched steps:

| block | v8 (strike point) | v6 (control) |
|---|---|---|
| 0-10M | 0.1263 | 0.1287 |
| 10-20M | 0.1194 | 0.1290 |
| 20-30M | **0.1100** | 0.1222 |

v8 is declining FASTER than v6 (-0.0163 over 30M against -0.0065), not slower.
The prediction stays formally open until ~98M, where it was set -- but the
interim data runs against it and saying so now is the point of having written
it down.

So: the strike point fixed POSITIONING (123.9 -> 84.0 deg, the only arm ever
better than random) and that did not stop the decline. Positioning was a real
deficit and correcting it was not sufficient, which is a genuine result about
the drill even though it is not the one hoped for.

That leaves the hypothesis from section 8 as the live one:

> **the frozen decoder cannot express a directed strike.**

It was trained by `follow` to track a target VELOCITY, so its action repertoire
is "walk in direction X at speed Y", and 154,632 parameters are held fixed. If
aiming a ball lies outside that span, then no shaping over the expert's z can
recover it -- which is precisely the pattern four arms (v4, v6, v7, v8) have now
shown: contact solved, direction random, outcome decaying.

`kick_ant_v9_unfrozen` launched to test it: identical to v8 except
`--freeze-decoder` is dropped. The three live kick arms now form a clean design:

| arm | shaping | decoder |
|---|---|---|
| v6 | old | frozen |
| v8 | strike point | frozen |
| v9 | strike point | UNFROZEN |

v8 vs v6 isolates the shaping; v9 vs v8 isolates the decoder.

v9 deliberately breaks the NPMP arrangement (one gait shared by all four
drills), which is what makes it a clean test and also why it is an EXPERIMENT,
not a candidate checkpoint. If it works, the finding is about the decoder, and
adapting the shared-decoder setup -- e.g. including a striking task when the
decoder is trained -- is a separate design question.

## 14. `best.pt` is now selected on a BATCHED deterministic score

*Measured 2026-08-11. Code: `warp_port/score.py`, wired into all four drill
trainers. Tests: `tests/test_batched_score.py` (9 checks, all passing).*

Section 10's fix, applied. The render eval is unchanged -- same one world, same
cadence, same videos -- but it no longer selects anything. Selection now runs a
SEPARATE scoring env: N worlds (`--score-worlds`, default 64), no renderer, one
full deterministic episode, `best.pt` saved on the mean. The one-world number is
still computed and still logged, under its existing `eval/fitness_warp`, purely
so the two can be compared; the new one is `eval/fitness_batched`.

### The measurement

One fixed checkpoint (`runs_v2/dribble_ant_v3/best.pt`), dribble's own
`config.json`, 15 s episodes, deterministic actions, nothing changing but the
seed. Reproduce:

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m tests.test_batched_score \
        --k 10 --repeats 5

| estimator | samples | mean | sd | min | max |
|---|---|---|---|---|---|
| single episode, 1 world | 10 seeds | 0.8948 | **0.0832** | 0.730 | 0.973 |
| single episode, 1 world | 640 (pooled over the batched runs' worlds) | -- | **0.1758** | -- | -- |
| **batched, 64 worlds** | 10 seeds | 0.8801 | **0.0231** | 0.846 | 0.923 |
| batched, 64 worlds, SAME seed | 5 repeats | 0.8585 | 0.0375 | 0.808 | 0.894 |

**7.6x less noise, exactly as predicted.** sigma1/sqrt(64) = 0.1758/8 = 0.0220
against a measured 0.0231. The estimator is doing precisely what averaging 64
independent draws should do, with no hidden correlation between worlds.

**Same quantity, not a different metric.** Means 0.8948 (single) vs 0.8801
(batched), difference 0.0148 against a combined 3-sigma of 0.1682. This is a
variance fix.

Two notes on how sigma1 was estimated, because the obvious way is wrong. The
direct K=10 sd is **0.0832**, less than half the pooled 0.1758, and it is not
stable: two earlier runs of this same file at the same seeds measured 0.1132 and
0.0315. The single-episode fitness distribution is bounded above (fitness is
`exp(-c*d)` at the final step, so it piles up near 1) with a long lower tail, so
whether one bad draw lands in a 10-sample window swings the sd 3x. The pooled
figure uses every world of every batched rollout -- each world IS an independent
single-episode draw -- for 640 episodes at no extra GPU cost, and it is the
number the tests assert against. Anyone quoting a single-episode spread off
K~10 samples will under-report it.

### This directly confirms section 10's diagnosis

`dribble_ant_v3`'s saved `best.pt` recorded a fitness of **0.980**. Re-scoring
**that exact checkpoint** ten times, one episode each, gave mean 0.895 and a max
of **0.973**. The pin is the top of its own draw distribution. Nothing about it
was a better policy; 136 evals of an estimator with sd 0.18 against a ceiling of
1.0 arrive at ~0.98 whatever the weights are doing.

Correction to section 10 while we are here: that section compared `best.pt`'s
0.980 against a "typical ~0.60" taken from the training monitor. Those are two
different quantities -- the monitor is the STOCHASTIC policy's fitness sampled
wherever a 64-step rollout lands mid-episode over 2048 worlds, while the eval is
the DETERMINISTIC policy's fitness at the end of the episode, and dribble's
fitness rises through an episode as the ball is shepherded in. The deterministic
single-episode mean is 0.895, not 0.60. Section 10's conclusion is unaffected
and in fact strengthened -- the gap it needed to explain is smaller, and the
draw spread that explains it is larger than assumed.

### What the seed does and does not buy

`--score-seed` (default 12345) is re-applied before every rollout, so every
evaluation in a run faces the same 64 task draws: verified bitwise on `qpos` and
`target_xy`. What it cannot buy is a reproducible number. **mujoco_warp is not
bitwise deterministic run to run** -- its solver accumulates with atomics, so
reduction order varies -- and 600 chaotic steps amplify the last bits. Repeating
the identical call five times moved the score with sd 0.0375 (0.808-0.894),
statistically indistinguishable from the 0.0231 measured across different seeds
at this sample size. So the residual noise floor of the batched score is the
SIMULATOR, not the task draw, and the paired design is close to free rather than
load-bearing. It was kept because it costs nothing and makes score-vs-score
comparisons well defined; it is not what delivers the 7.6x.

Practical consequence: the batched score resolves policy differences of roughly
0.02-0.04 in fitness, not 0.005. Do not read finer differences between two
`best.pt` files than that, and do not expect a re-run of a scoring command to
reproduce to three decimals.

### Cost, and the flags

The scoring env is built ONCE next to the render env and reused, with
`use_graph=True` -- uncaptured it is ~16x slower per step (1462 vs 92 ms/step
measured on the dribble eval env in section 10's follow-up), which would make
one scoring call a 15-minute stall. Captured, a 64-world 600-step rollout is
~23 s idle and ~50 s with six trainers sharing the card: about the same as the
render eval it sits next to, so an evaluation now costs roughly twice what it
did. At `--video-secs 300` that is a few percent of wall clock.

    --score-worlds N   worlds in the scoring env (default 64; 0 restores the
                       old single-episode selection exactly)
    --score-secs S     scoring cadence; 0 (default) reuses --video-secs so the
                       two numbers in the log describe the same weights
    --score-seed S     re-applied before every rollout (default 12345)

New wandb keys: `eval/fitness_batched`, `eval/fitness_batched_sem`,
`eval/fitness_batched_std` (the spread ACROSS worlds -- the single-episode
sigma, logged every eval), `eval/ep_rew_batched`. The old `eval/fitness_warp`
and `eval/ep_rew_warp` keep their names and their one-world meaning.

### Per-trainer differences

- **kick and shoot** share `train_kick_warp.run`, which already had a
  `make_env_fn(args, num_worlds, seed, use_graph)`; the scoring env is one more
  call to it. One change covers both.
- **dribble and follow** built their eval env inline inside `make_eval`. That
  body is now `make_eval_env(args, num_worlds, seed)`, with `make_eval` calling
  it for the render env and `make_score_env` for the scoring env, so the two
  cannot drift apart. Their TRAINING env construction was deliberately left
  alone -- follow's, in particular, does not pass `arena`/`pitch_scale` while
  its eval env does, a pre-existing discrepancy this change does not touch.
- **follow selects on `ep_rew`, not fitness**, and still does -- now on the
  batched `ep_rew`. Its fitness is `exp(-c*dist)` read at the final step only,
  which grades where the creature happened to be standing when the clock ran
  out, while `ep_rew` integrates the episode. Which statistic selects is a
  separate question from how noisily it is measured, and only the second one
  was in scope here.
- The scoring env tracks the curricula: `target_cone` on the cone anneal
  (dribble, kick) and `speed_range` on the speed curriculum (dribble, follow).
  Without that it would grade `best.pt` at a difficulty the run had outgrown.

### 14a. Independent replication of the batched score

The section-14 numbers were produced by the agent that built the fix. Re-run
independently (`--k 6 --repeats 3`, different background load), 9/9 PASS:

| | agent's run | independent re-run |
|---|---|---|
| single-episode sd (pooled) | 0.1758 (640 eps) | 0.2148 (384 eps) |
| batched sd, 64 worlds | 0.0231 | 0.0314 |
| noise reduction | 7.6x | **6.8x** |
| theoretical `sqrt(64)` | 8.0x | 8.0x |
| same-seed repeat sd | 0.0375 | 0.0362 |

Both land under the theoretical 8.0x and bracket each other, so quote the win
as **~7x**, not 7.6x -- the ratio itself has sampling error, and reporting the
single most favourable measurement of a variance-reduction claim would repeat
in miniature exactly the error the fix exists to remove.

Means agree in both runs (0.8071 vs 0.8599 here, |diff| 0.0528 against a
0.2658 tolerance), confirming this is a variance fix and not a change of
metric.

## 15. v9 instrumentation check: the freeze works, the unfreeze is live

*Measured 2026-08-11 at v9 = 15.7M steps, comparing each arm's `best.pt`
against the shared warm-start `runs_v2/_decoder_ant_final.pt`.*

Before reading anything into v9, the prerequisite question: has the unfrozen
decoder actually moved? Relative Frobenius change, split by which parameters
`--freeze-decoder` is supposed to hold:

| arm | DECODER + action head | expert / z_proj |
|---|---|---|
| v8 (frozen) | **0.00e+00** over 8 tensors | 3.55e-01 |
| v9 (unfrozen) | **1.15e-01** | 7.41e-02 |

Two things confirmed. `--freeze-decoder` really does hold those 154,632
parameters bit-identical -- worth knowing, because every frozen-arm conclusion
in sections 7-13 rests on it and nothing had checked. And v9's decoder HAS
moved, 11.5% in relative norm by 15.7M steps, so the experiment is live rather
than stalled.

(First attempt at this got it wrong by lumping `expert.*` and `z_proj.*` in with
the decoder. Those are trainable in BOTH arms -- `--freeze-decoder` holds the
decoder and action head only -- which made the frozen arm look like it had
drifted 0.21. Splitting by what the flag actually freezes is what produced the
0.00e+00.)

**So the early v9 result carries more weight than its step count suggests.** At
matched steps (0-15.7M) v8 is 0.1249 and v9 is 0.1232, and by 5M block v9 runs
0.1253 / 0.1250 / 0.1203 against v8's 0.1257 / 0.1268 / 0.1233. The decoder is
changing substantially and the outcome is not. That is a point AGAINST the
frozen-decoder hypothesis, not merely an absence of evidence for it.

One confound to keep honest: v9 carries v8's strike-point shaping, and section
13 showed that shaping is not the fix either. If the shaping actively
mis-trains, unfreezing simply hands PPO more parameters to move in the wrong
direction, and v9 would fail for a reason that says nothing about decoder
capacity. The clean follow-up if v9 dies is `old shaping + unfrozen` -- the
missing cell of the 2x2 -- or better, testing decoder capacity directly by
supervised-fitting a known-good striking controller rather than inferring it
from an RL curve.

## 16. v9's AIM leaves the random baseline -- the first arm ever to do so

*Measured 2026-08-11, v9 at 21M steps, 17,776 ball-moving samples.*

| arm | decoder | steps | positioning | **aim** | aim within 30 deg |
|---|---|---|---|---|---|
| v6 | frozen | 88M | 123.9 deg | 124.6 deg | 9.2% |
| v7 | frozen | 35M | 103.9 deg | 93.2 deg | 15.7% |
| v8 | frozen | 9.2M | 84.0 deg | 94.4 deg | 16.5% |
| **v9** | **UNFROZEN** | **21M** | **74.5 deg** | **74.1 deg** | **22.3%** |
| random | -- | -- | 90 deg | 90 deg | 16.7% |

Every frozen arm sat at 93-125 deg aim -- at or worse than chance. v9 is at
74.1 deg with 22.3% of strikes inside 30 deg against a 16.7% baseline. **This is
the first time any arm has aimed better than random.**

The frozen-decoder hypothesis of section 8 is SUPPORTED. The decoder that
`follow` trained to track a target velocity apparently could not represent a
directed strike, and unfreezing it let the aim move within 21M steps -- fewer
steps than three of the four frozen arms ever ran.

**Correction to section 15.** That section read v9's fitness tracking v8 as "a
point AGAINST the decoder hypothesis". That inference was wrong. Aim moves
BEFORE fitness -- which is the same leading/lagging relationship argued for in
section 11 when v8's POSITIONING improved ahead of its outcome. The principle
was applied to v8 and then forgotten for v9. Fitness still tracks: v9 0.1158 vs
v8 0.1160 over 15-20M. Aim leads; the outcome has not followed yet.

**Confound not yet excluded.** v9's aim is measured at 21M and v8's at 9.2M, so
part of the gap could be training time rather than the decoder. v8 is now at
64M -- three times v9's age -- and is being re-probed. If v8 at 64M is still at
~94 deg while v9 at 21M is at 74 deg, training time is excluded and the decoder
is the only remaining explanation. If v8 has also drifted toward 74, the credit
belongs to the strike-point shaping and v9 gets none of it.

Until that lands this is a strong signal, not a conclusion.

## 17. RESOLVED: the frozen decoder is the cause

*Measured 2026-08-11. v8 re-probed at 64M, 109,139 ball-moving samples.*

Section 16 left one confound: v9's aim was measured at 21M and v8's at 9.2M, so
the gap might have been training time. v8 is now at 64M -- three times v9's age
-- and v8/v9 differ in exactly one flag, `--freeze-decoder`. Re-probing settles
it:

| arm | decoder | steps | positioning | aim | aim within 30 deg | aim within 90 deg |
|---|---|---|---|---|---|---|
| v8 | frozen | 9.2M | 84.0 deg | 94.4 deg | 16.5% | 48.2% |
| **v8** | **frozen** | **64M** | **115.0 deg** | **134.2 deg** | **6.3%** | **23.4%** |
| **v9** | **UNFROZEN** | **21M** | **74.5 deg** | **74.1 deg** | **22.3%** | **59.9%** |
| random | -- | -- | 90 deg | 90 deg | 16.7% | 50% |

**Training time is excluded, and in the strongest possible way: it has the
OPPOSITE sign.** With the decoder frozen, more training makes aim dramatically
worse -- 94.4 -> 134.2 deg, with within-30 collapsing to 6.3%, far BELOW the
16.7% of chance. The frozen arm is not failing to learn to aim; it is learning
to aim away.

The strike-point shaping's positioning gain was **transient**: 84.0 deg at 9.2M,
reverted to 115.0 deg at 64M. Good positioning cannot be held when the
controller underneath cannot convert it into a directed strike, which is also
why v8's fitness declined (section 13) rather than plateauing.

### The whole kick story, in order

1. The drill's approach shaping pays for speed TOWARD the ball, so the creature
   charges the ball from wherever it stands and the strike direction is whatever
   the approach direction happened to be (section 8).
2. Correcting that with a strike point moved POSITIONING off the random baseline
   (section 11) but not the outcome, and the gain did not hold (this section).
3. The reason is the frozen decoder. `follow` trained it to track a target
   VELOCITY -- "walk in direction X at speed Y" -- and a directed strike is
   outside what those 154,632 parameters can express. No shaping over the
   expert's z can recover a behaviour the controller cannot represent.
4. Unfreezing it produced the first better-than-random aim in six arms, in 21M
   steps, fewer than three of the four frozen arms ever ran.

Resolves section 12's prediction as well: v8 was to sit above 0.0974 and above
the 0.105 baseline at ~98M. Its recent mean is 0.097 at 64M and falling. **The
prediction failed, as recorded, and the reason is now known.**

### What this costs, and the design question it raises

v9 breaks the NPMP arrangement on purpose: one decoder, learned once by
`follow`, shared frozen by all four drills so no skill can degrade another. If a
directed strike genuinely lies outside that decoder's span, the arrangement
cannot hold as designed, and the options are not equivalent:

- **Train the decoder on a striking task too**, not just `follow`, so one shared
  decoder spans both walking and striking. Keeps shared-z; costs a redesign of
  the decoder-training stage and a re-run of everything downstream of it.
- **Per-skill decoders.** Cheapest to implement, and it discards the shared-z
  property the whole pipeline is built on -- the 2v2 controller would be
  switching between four unrelated controllers rather than steering one.
- **Unfreeze during drills, with a KL anchor to the follow decoder.** A middle
  path: lets the strike develop while bounding gait drift. Untested here.

This is a real fork in the pipeline design and belongs to the user, not to an
overnight loop. Recorded rather than chosen.

## 18. Aim improved and the OUTCOME did not -- six matched blocks

*Measured 2026-08-11, v9 at 28M.*

Section 16 predicted, on the aim-leads-fitness principle, that v9's outcome
would follow its aim. Over six matched 5M blocks it has not:

| block | v9 (UNFROZEN, aim 74.1 deg) | v8 (frozen, aim 134.2 deg) |
|---|---|---|
| 0-5M | 0.1253 | 0.1257 |
| 5-10M | 0.1250 | 0.1268 |
| 10-15M | 0.1203 | 0.1233 |
| 15-20M | 0.1158 | 0.1160 |
| 20-25M | 0.1121 | 0.1124 |
| 25-30M | 0.1068 | 0.1049 |

Two arms whose aim differs by 60 degrees produce indistinguishable fitness, both
declining monotonically. So section 17's conclusion needs a boundary drawn
around it: **unfreezing the decoder fixed AIM, and aim alone does not produce
the outcome.** The decoder finding stands -- it is the only thing that has ever
moved aim off chance -- but it is not by itself the fix for the drill.

The likely reason is that this is a TIMED pass and aim is only half of it. The
ball has to be AT the target at T, so the strike SPEED has to be modulated too,
and section 1 already measured the ant striking at a median 15.2 m/s where the
reward surface's interior optimum is 6-9 m/s. A 15 m/s strike rolls the ball
~10.6 m past a 3-6 m target no matter how precisely it is aimed. Perfect
direction with triple the required speed still scores the do-nothing value.

That predicts v9's strike speed is still ~15 m/s, unchanged by the decoder
unfreeze, and that is being measured now. If so, kick needs BOTH an expressive
controller (the decoder) and a reason to modulate power -- and the arms so far
have supplied neither together.

Method note: this is the second time the aim-leads-fitness principle has been
invoked, once correctly (section 11, v8's positioning led nothing and the arm
died) and once as a premature promise (section 16). A leading indicator earns
its name only after the lag actually resolves; until then it is a hypothesis
about a correlation, not a forecast.

## 19. The timed arms do not strike AT ALL -- and section 2 caused it

*Measured 2026-08-11, v9 at 28M, 2703 closed segments, per-segment PEAK ball
speed (the hardest the ball was hit that segment).*

| v9 peak ball speed | |
|---|---|
| median | **0.00 m/s** |
| mean | 1.15 |
| p90 | 3.97 |
| max | 8.69 |
| **never really struck (<1 m/s)** | **63.6%** |
| soft (1-6) | 35.1% |
| in the 6-9 optimum | 1.3% |
| **overshoot (>9)** | **0.0%** |

**The prediction in section 18 was wrong, and wrong in the opposite direction.**
It expected ~15 m/s overshoot. In fact nothing overshoots at all and in almost
two thirds of segments the ball never moves. Section 1's whole "the ant must
learn to strike softer" framing describes v3, not the timed arms.

### Why: section 2 removed the only thing paying for a strike

v3 used `reward_kind=point` with `w_strike=0.1` -- an explicit payment for ball
speed -- and struck at a median 15.2 m/s. Every timed arm has `w_strike` forced
to **0**, by a decision recorded in section 2 of this document:

> "power is a consequence of the deadline, and paying for it separately prices
> the same thing twice and in one direction only."

That reasoning does not survive contact with the measurement. The arrival term
is paid ONCE, at T. The approach shaping is paid EVERY step. A per-step payment
for walking at the ball outweighs a single terminal payment for where the ball
ended, so the policy converges on standing near the ball without hitting it --
which is exactly the "contact solved, direction random" picture of section 8,
now with a mechanism: there was never a gradient toward striking in the first
place.

It also explains the monotone DECLINE rather than a plateau (sections 12-13).
Not striking is an attractor: the approach shaping keeps paying, the arrival
term is nearly indifferent between "ball did not move" and "ball moved a little
in a random direction", and every step of training walks further into it.

### What this means for the decoder result

Section 17 stands but its scope narrows again. Unfreezing the decoder is still
the only thing that has ever moved aim off chance (74.1 vs 134.2 deg). But an
arm that strikes the ball in 36% of segments and reaches the useful power band
in 1.3% cannot express its aim as an outcome no matter how good that aim is.
Aim was necessary; a reason to strike is also necessary; the arms so far have
never had both at once.

### Recommendation, NOT applied

Restore a strike-speed term under `timed` -- i.e. reverse section 2's
`w_strike = 0` -- and re-run the decoder comparison with it. This is a
recommendation rather than a launched arm because it reverses a documented
design decision and because the NPMP fork (section 17) is already waiting on the
user; adding a seventh arm before that fork is resolved risks answering a
question about a pipeline we may be about to change.

The cheap sanity check first: v3 (point reward, w_strike=0.1) reached fitness
0.376 on ITS metric while every timed arm sits near 0.10 on its own. Those two
numbers are not comparable (section 7), but "the arm that paid for strikes
struck, and the arms that did not, did not" is not a subtle signal.
