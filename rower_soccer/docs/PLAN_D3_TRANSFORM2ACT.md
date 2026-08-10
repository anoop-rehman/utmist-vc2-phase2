# Direction 3 — Transform2Act: the GNN design-and-control policy, toward soccer

*Written 2026-08-10.*

Paper: https://arxiv.org/pdf/2110.03659 · Code: https://github.com/Khrylx/Transform2Act
Local clone: `/workspace/Transform2Act` (own venv; 2021 codebase — expect mujoco-py
/ old torch archaeology).

## Motivation

Two reasons, one strategic and one honest:

- **Strategic**: CompeteEvo's evolved morphologies stay within their creature
  class. Transform2Act's design space is far more open: the agent *builds* the
  body — skeleton transform actions, attribute transform actions, then control —
  all under ONE GNN policy over the morphology graph. If we want optimization to
  find genuinely different bodies for different roles (Direction 2 M3's
  heterogeneous-team dream), this is the machinery that could.
- **Honest**: a single GNN that reads the creature-as-graph and outputs both
  design edits and motor torques is a beautiful object. Implementing and playing
  with it is worth doing for its own sake; expect exploration here, not just
  milestone-chasing. Diversity per se is NOT the goal — optimized creatures are —
  but a policy that generalizes across morphologies is valuable to every other
  track (including Direction 1's creature swap).

## Goal

Working Transform2Act reproduction; GPU port; then its design-and-control agent
dropped into our soccer setting (drills first, 2v2 later), where the body plan
itself becomes part of what training optimizes.

## Approach / milestones

- **M1 — Reproduce, their way (smoke scale).** Their code, their pins, smallest
  task (2D locomotion class) for reduced iterations: prove the graph policy
  forward pass, the transform-stage/execution-stage switch, PPO update, logging.
  Table-number matching deferred to M2.
- **M2 — GPU port.** Batched envs (MJX/warp) under their agent. Two hard parts we
  should scope honestly in a PORT_MAP.md: (a) morphology changes mid-episode mean
  a CHANGING model — batched GPU physics wants one compiled model per batch, so
  design stages likely batch by generation/population instead of by env step;
  (b) the GNN needs padded/variable graph batching. Validate against their
  reported numbers at GPU speed.
- **M3 — Soccer.** Start with our drills (follow/dribble on the warp pitch) as
  the task suite for design+control; then 2v2. Clarity on the exact shape comes
  when we get there — the paper's tasks are single-agent, so the multi-agent
  bridge design belongs to whichever of D2/D3 gets there first.

## Parallelized implementation plan

| unit | depends on | owner-shape |
|---|---|---|
| 3a. Clone + install + smoke (M1) | — | agent RUNNING now |
| 3b. Deep read → PORT_MAP.md (graph repr, transform actions, JSRC details) | 3a | agent, parallel |
| 3c. Standalone GNN-policy playground (their nets, our tensors, no sim) | 3b | fun-sized, parallel |
| 3d. GPU port of one locomotion task | 3b | worktree `t2a-port` |
| 3e. Paper-number validation at GPU speed | 3d | gate |
| 3f. Design+control on our drill tasks | 3e | later |
| 3g. Merge insights with D2 (transform actions inside competitive co-evolution) | 3f + D2-M3 | the interesting endgame |

## Eval method

- **M1**: loop runs end-to-end; reduced-iteration curve is sane.
- **M2**: their Table 1 numbers (final return across seeds per environment) —
  the reproduction gate, run at GPU speed.
- **M3**: drill fitnesses (we already have the harnesses) with design freedom on
  vs off — does letting the body change beat the fixed ant at equal compute?
  Plus: cross-morphology generalization of the GNN controller (zero-shot on a
  body it never trained — the property Direction 1's creature swap would love).
- Always: watch the creatures. A "better" body that exploits sim physics will
  show up in video long before it shows up in a metric.

## Notes / risks

- Oldest codebase of the three: mujoco-py + MuJoCo 2.1 binaries + old gym +
  torch 1.x. M1's real work is installation; deviations from pins must be logged
  in REPRO_NOTES.md or M2 debugging becomes archaeology-squared.
- Mid-episode morphology mutation is fundamentally at odds with compiled batched
  physics — the M2 design decision (batch-by-generation) should be written down
  BEFORE porting, not discovered during.
- Keep D3 fun. If the GNN playground (3c) spawns side experiments, that's the
  point of this track, not scope creep — timebox it rather than kill it.
- CPU allocation: ≤12 cores while D1 GPU runs + D2 repro are live.
