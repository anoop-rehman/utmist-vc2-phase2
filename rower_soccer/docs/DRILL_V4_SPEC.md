# Drill v4: timed kick, urgent shoot, uncapped follow

*Spec written 2026-08-10, queued for execution. Decided with the user; implement
from this doc.*

## Why

`kick` v3 grades **closest approach at any moment in a 2-6 s window**. That is
still satisfiable by nudging the ball along for six seconds — the `w_strike=0.1`
term exists only to discourage it, which is a patch on the objective rather than
a property of it. And `shoot` pays for power but SELECTS checkpoints on accuracy
alone, so `best.pt` can prefer a timid ant. `follow` is pinned at a cap we chose
(3.0 m/s) while tracking with 0.45 m error — the ceiling is ours, not the body's.

## 1. kick — "put the ball THERE, at THIS pace"

**Objective.** At segment start, sample a target (3-6 m) and a pace
`v_pace ~ U(2, 6) m/s`. Set `T = d_spawn / v_pace`, clamped to a sane band
(suggest [0.5 s, 4 s]). The segment ENDS at exactly `T`. Reward, paid once:

    arrival = exp(-c * || ball_pos(T) - target ||)      # 3-D, c = reward_coef = 0.5
    reward  = w_arrive * arrival + shaping ; all * upright

**Dense, not binary** (user's explicit call): grade WHERE THE BALL IS at time T,
never "did it pass through a ring at T". A ring is unreachable by random
exploration and gives a flat gradient; position-at-T always has a gradient.

**Why this is better than v3**
- Power becomes a CONSEQUENCE, not a priced term: far target ⇒ must strike hard;
  near target ⇒ striking hard overshoots and is punished at T. So set
  `w_strike = 0` and delete the anti-dribble patch — the objective now forbids
  dribbling by itself (pace > the ant's ~1-3 m/s top speed is unwalkable).
- Symmetric punishment of early (overshoot) and late (short); v3 punished neither.
- Rolling friction makes it genuinely hard: must kick FASTER than d/T since the
  ball decelerates. That is the control skill we want.

**Physics sanity — do not use a fixed 0.1 s.** 3 m in 0.1 s is 30 m/s; the env
clips ball speed at 8 m/s and real ant strikes are a few m/s. A fixed short
deadline is unreachable at every distance ⇒ flat gradient ⇒ nothing learned.
Distance-scaled T via `v_pace` is the fix. Verify the sampled band is achievable
by measuring actual strike speeds from a v3 checkpoint BEFORE launching a long run.

**Obs change (load-bearing).** The policy cannot hit a timed target without
knowing the deadline. Add to kick's task obs: remaining time `(T - t)` — and,
because the ant should reason about required speed, optionally `d_target / (T-t)`.
This widens kick's task block (12 → 13/14) which means:
- a new checkpoint contract, so `rower_soccer/skills/registry.py`'s kick entry and
  a matching `fields.py` field must be updated when the run is adopted;
- the game's kick command must supply a pace/deadline. Simplest: the game fixes
  `v_pace` to a mid-band constant so a human click means "pass it there, briskly".

**fitness** = mean over segments of `exp(-c * d_at_T)` — same shape as before and
still directly comparable to shoot.

**Shaping.** Keep the small `me→ball` approach term (getting to the ball is a
prerequisite, and it is what makes early exploration work). DROP the
`ball→cmd_dir` velocity term: it rewards "faster toward target" monotonically,
which now actively fights pace modulation.

## 2. shoot — "in the goal, fast"

Keep the current structure; two changes.

- **Time-discount the goal bonus**: `goal_bonus * exp(-k * t_in_segment)` instead
  of a flat 5.0, so a quick goal beats a slow one. Suggest k ≈ 0.3-0.5 /s
  (tune so a 1 s goal ≈ 0.7-0.75 of the max, a 5 s goal ≈ 0.1-0.2).
- **Fix the fitness/reward mismatch** (flagged since 2026-08-09): fitness is
  `exp(-0.5 * closest distance to mouth)`, i.e. accuracy only, so checkpoint
  selection ignores everything the reward pays for. Replace with something that
  ranks a fast goal above a slow goal above a near miss, bounded [0,1]:

      per segment:  scored ? exp(-k * t_score) : 0.5 * exp(-c * d_mouth_best)
      fitness    :  mean over the episode's segments

  (the 0.5 ceiling on misses guarantees any goal outranks any miss.)

Rationale for the kick/shoot split, which this spec makes real: **kick is a PASS**
(placement in space AND time, because a teammate has to be there), **shoot is
maximum urgency at a 7.4 m mouth** (accuracy barely matters, speed does — a slow
shot gets intercepted). This is the distinction the 2v2 BC stage needs.

## 3. follow — raise the cap and find the real ceiling

`tgt_spd` has been pinned at `--speed-max 3.0` while tracking error stays
0.45-0.51 m, i.e. the ant is comfortably keeping up. Relaunch with
`--speed-max 8`. The tracking gate self-limits (it bumps only when the creature
is BOTH near the target AND not falling behind), so the run finds the body's
physical ceiling on its own. The old 2.15 m/s figure was measured on the
abandoned 2.7x ant and does not apply.

Needs a `--resume` relaunch (`speed_max` is read at launch; curriculum state is
not checkpointed, so the only cost is re-climbing the band).

## Execution order

1. follow relaunch (one flag, zero risk) — do first.
2. shoot: time-discount + fitness fix — small, self-contained; relaunch as
   `shoot_ant_v4` (fresh name: reward change ⇒ new run, per the one-name-one-run
   rule).
3. kick: the real work. New env timing machinery + obs + reward, a short
   achievability measurement, then launch `kick_ant_v4`.
4. Keep `kick_ant_v3` / `shoot_ant_v3` checkpoints — they are the current game
   registry pins and the fallback if v4 regresses.
