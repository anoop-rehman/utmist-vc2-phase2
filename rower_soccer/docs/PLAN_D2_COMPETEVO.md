# Direction 2 — CompeteEvo: reproduce, GPU-port, then 2v2 soccer co-evolution

*Written 2026-08-10. Priority direction when compute conflicts (user's call).*

Paper: https://arxiv.org/pdf/2405.18300 · Code: https://github.com/KJaebye/competevo
Local clone: `/workspace/competevo` (own venv — NEVER shares an env with our repo).

## Motivation

CompeteEvo co-evolves morphology AND control in **competitive** two-agent settings
(the Bansal-style run-to-goal / sumo family): the opponent is part of the fitness
landscape, so body plans are shaped by adversarial pressure, not a static task.
That is exactly the mechanism our research track wants — except we want it for
**teams**: 2v2, eventually heterogeneous (two different morphologies per team,
evolving to complement each other), eventually on our dm_control soccer pitch.
This is the paper track (CompeteEvo → multi-agent), distinct from Direction 1's
video sprint.

## Goal

Extend CompeteEvo from 1v1 competitive co-evolution to 2v2 team co-evolution,
culminating in 2v2 soccer, with the intermediate reproductions proving each rung
of the ladder before we stand on it.

## Approach / milestones

- **M1 — Reproduce, their way (smoke/sanity scale).** Their exact code, their
  pins, their configs, CPU (that's what the code targets). NOT chasing full paper
  numbers here — a smoke run proving the whole loop (env, morph evolution, PPO,
  logging), then one longer sanity run whose curve shape can be compared to the
  paper. Full-scale number matching is deferred to M2 where it's cheap.
- **M2 — GPU port (MJX or mujoco_warp).** Port their envs + training loop onto
  batched GPU physics like our drill stack. Validation IS the paper numbers: at
  GPU speed, paper-scale runs cost hours, so match their reported results here.
  This is where the work moves into OUR repo, on a branch/worktree
  (`competevo-port`), and where their multiprocessing PPO becomes our batched PPO.
- **M3 — Team extensions on their tasks.** Keep their arenas, go 2v2:
  homogeneous teams first (cheapest delta), then heterogeneous. Expect the
  research decisions to start here: credit assignment across teammates,
  shared-vs-separate evolution populations, opponent sampling for 2v2.
- **M4 — 2v2 soccer.** Their co-evolution machinery on our pitch/ball/goals.
  Design tasks that actually reward morphological division of labor — sumo
  probably doesn't; a keeper/striker split might.

## Parallelized implementation plan

| unit | depends on | owner-shape |
|---|---|---|
| 2a. Clone + install + smoke (M1) | — | agent RUNNING now |
| 2b. Longer sanity run, curve vs paper shape | 2a | background after 2a |
| 2c. Read their code deeply; write PORT_MAP.md (env API, morph genome, PPO loop) | 2a | agent, parallel with 2b |
| 2d. GPU port of ONE task end-to-end | 2c | worktree `competevo-port` |
| 2e. Paper-number validation at GPU speed | 2d | after 2d |
| 2f. 2v2 homogeneous on one ported task | 2d | after 2e |
| 2g. Heterogeneous teams + team-credit design doc | 2f | research decisions here |
| 2h. Soccer arena integration | 2f + D1's warp pitch | last |

We're lucky: the repo is given, so M1 is mostly dependency archaeology, and 2c
(reading + mapping) can overlap everything.

## Eval method

- **M1**: does the loop run; does the sanity curve match the paper's *shape*.
- **M2**: match their reported table numbers (win rates / returns per task) at
  paper scale, GPU-fast. This is the reproduction gate.
- **M3+**: win-rate matrices (evolved 2v2 vs fixed-morphology 2v2 baselines,
  round-robin across seeds), plus morphology audits — did heterogeneous teams
  actually diverge in body plan, or collapse to twins? Behavioral probes for
  division of labor (who takes the ball, who blocks).
- Always: watch the rendered matches. Metrics lie, videos don't.

## Notes / risks

- Their results stay within a creature class (ant-ish stays ant-ish) — expected;
  Transform2Act (Direction 3) is the answer to open-endedness, not this track.
- Competitive co-evolution is famously unstable (cycling, forgetting). Their
  opponent-sampling scheme is load-bearing; port it faithfully before innovating.
- 2v2 quadruples sim cost per match and squares the opponent space — the GPU port
  (M2) is not optional, it is what makes M3/M4 affordable.
- CPU allocation: ≤30 cores while Direction 1's GPU runs are live.

## Measured 2026-08-11: the GPU port IS a ~19x speedup

*This section previously claimed the opposite. That claim was wrong; the
retraction and its cause are kept below because the mistake is instructive.*

| | env-steps/s | 1000-epoch config |
|---|---|---|
| Their CPU reference (24 workers, REPRO_NOTES.md) | 185 | ~3 days |
| **Our GPU port** (dev_smoke_v2, median iteration 18.7 s) | **3,505** | **~4 hours** |
| Bare `RunToGoalDevEnv.step()`, 1024 worlds, isolated | ~10,200 | — |

M2's premise -- "at GPU speed, paper-scale runs cost hours" -- **holds**: 5e7
steps at 3,505 env-steps/s is about four hours against their three days.

Where the remaining ~3x sits, measured by the stage-3 agent with interleaved
medians of 7 at 1024 worlds:

| | one learner | two learners |
|---|---|---|
| total iteration | 17.20 s | 28.40 s |
| `env.step` (64 calls) | 10.07 s (**59%**) | 10.50 s (**37%**) |
| `policy.act` | 1.19 s (64 calls) | **11.02 s** (640 calls) |
| `policy.value` | 0.71 s | 2.25 s |
| update + host work | 5.23 s | 4.63 s |
| `DesignWriter.write` | 0.42 s (~2%) | 0.43 s (~2%) |

So physics DOMINATES a one-learner iteration, and the tractable target is
`policy.act`'s call count under self-play (640 vs 64: `blocks=4` means 2 ego +
8 opponent forwards per step, ~17 ms each for a 38k-parameter MLP -- pure
launch overhead, fixable by stacking slot params into one `bmm` or lowering
`blocks`).

### The retraction, and why it matters

The earlier claim was "168 env-steps/s, slower than their CPU, physics only
1.6% of the iteration". It came from reading `log.json`'s `sec` field as
per-iteration when it is CUMULATIVE elapsed seconds. The median of a
monotonically increasing cumulative series is about half its total, so a 758 s
run over 27 iterations yielded a "median iteration" of 391 s -- which is
758/2, not a duration. Everything downstream inherited the 20x error.

Two lessons worth keeping:

- **The isolated measurement was right and the ratio was wrong.** The bare-env
  timing (95.6 ms/step) matched the agent's independent 100.0 ms. The error was
  entirely in the denominator, which is exactly where a plausible-looking
  derived number hides.
- **A sanity check was available and skipped.** 27 iterations x 391 s is 2.9
  hours; the run demonstrably took 12.6 minutes. Any wall-clock cross-check
  would have caught it immediately.

