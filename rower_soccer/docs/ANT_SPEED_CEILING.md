# The ant's real top speed, and how the curriculum sailed past it

*Measured 2026-08-11. Corrects a change I made the same day.*

## What happened

`follow` had sat at `tgt_spd = 3.00` for a whole run — exactly our
`--speed-max 3.0` — with tracking error 0.45-0.51 m. Reading that as "the cap is
ours, not the body's", I raised it to 8.0. The curriculum then climbed three
bands in minutes to **1.51-7.53 m/s** while `ep_rew` fell 541 -> 381.

## The measurement

Fixed target speed (no band), 256 worlds, deterministic policy,
`follow_ant_final_frozen/best.pt`:

| target m/s | ant speed mean / p90 | tracking err mean / p90 |
|---|---|---|
| 1.0 | 1.02 / 1.56 | **0.14** / 0.17 |
| 2.0 | 1.91 / 2.81 | **0.45** / 0.65 |
| 3.0 | 2.19 / 3.81 | 2.01 / 5.21 |
| 4.0 | 1.88 / 3.71 | 3.01 / 6.26 |
| 6.0 | 1.37 / 3.21 | 3.71 / 6.66 |
| 7.5 | 1.22 / 2.91 | 3.92 / 6.79 |

**The ant tracks cleanly to ~2 m/s and comes apart by 3.** Above that it does not
sprint and fail — it gets SLOWER (1.22 m/s when chasing 7.5). Sprint p90 is
3.42 m/s. So the original 3.0 cap was already just under the ceiling: raising it
found no headroom, it only pushed the task past the body.

Now running at `--speed-max 3.5` with `--speed-near-frac 0.25`.

## Why the gate allowed it (third failure of this gate, different mechanism)

Two independent reasons, both worth remembering:

1. **`near_frac 0.5` is too loose.** 0.5 x spawn 3.22 = a 1.61 m "close enough"
   threshold, on a body that parks within **0.15 m** when the task is sane. A
   metre of error read as success.
2. **The target BOUNCES off the bounds** (`worm_env_base`: velocity reverses at
   +/-`bounds`, position clamps). A 7.5 m/s target ping-pongs across a 20 m box
   every ~2.7 s instead of running away, so mean distance stays bounded no matter
   how badly the ant is doing. `curriculum.py`'s header already warned that
   "distance not growing" is satisfiable inside a bounded arena; `near_frac` was
   the fix, and it was set too loose to work.

## Why the eval video looked fine (it did — the human was right)

An eval episode draws ONE target speed from the band. The video that looked
healthy had drawn **2.44 m/s**, comfortably inside the tracking range. Pinning
the target at 7.5 m/s gives `ep_rew 157 / fitness 0.123`, against `556 / 0.947`
at the slow band.

**The lesson for evals**: a single episode sampled from a WIDE randomized band
is not evidence about the band's hard end. Where a range matters, eval at the
range's extremes, not at a draw from it. Both observations here were correct and
they looked contradictory only because nobody had checked which speed the video
actually drew.
