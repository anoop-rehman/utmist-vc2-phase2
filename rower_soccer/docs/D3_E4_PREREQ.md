# D3 M3 E4 prerequisite — what can Transform2Act's design head actually see?

*2026-09-05. `PLAN_D3_M3.md`'s E4 rung records the requirement: D2 found a
shared design head cannot condition on role or opponent at all until
`--role-in-design` was added (SMD 0.110 → 0.833), and **"the equivalent question
here is what Transform2Act's design head can see. It must be checked before
this rung, not after."** This is that check. **No E4 code has been written and
nothing has been launched.***

## The answer: it sees nothing about the simulation. At all.

`design_opt/models/transform2act_policy.py` lines **170** (attribute stage) and
**194** (skeleton stage) both begin with the same slice:

```python
obs = torch.cat((obs[:, :self.attr_fixed_dim], obs[:, -self.attr_design_dim:]), dim=-1)
```

and the dimensions confirm what that discards:

```
control_state_dim = attr_fixed_dim + sim_obs_dim + attr_design_dim
skel_state_dim    = attr_fixed_dim +                attr_design_dim
attr_state_dim    = attr_fixed_dim +                attr_design_dim
```

On our run-to-goal ant the node row is **25 columns = 4 attr_fixed + 16 sim_obs
+ 5 attr_design**. The control head receives all 25. **The design heads receive
9** — the 4-column body-depth one-hot and the body's own 5-column attribute
genome. **The entire 16-column `sim_obs` slice is dropped**, and that slice is
where every appended task column lives: `(opp_dx, opp_dy, goal_dx)`, plus all
qpos and qvel.

**Measured, not just read.** Moving the opponent 4 metres (x = +1.0 → −3.0) and
recomputing the observation:

| | max \|Δ\| |
|---|---|
| **design-head input** (the 9 columns) | **0.000e+00** |
| dropped `sim_obs` slice | 4.000e+00 |
| the 3 appended columns | `(+2.00, ~0, +5.00)` → `(−2.00, ~0, +5.00)` |

> **The design head is blind to the opponent — and to the goal, to its own
> joint angles, and to its own velocity. It sees only its own body's structure
> and parameters.**

## What this means for E4 as written

E4's headline question is *"does co-evolution produce an arms race in
morphology, or do both sides converge on the same body?"* **As posed, it is
unanswerable on this architecture**, and for a stronger reason than D2's:

* D2's design head *could* be given role information — `--role-in-design` was
  an addition that worked, moving SMD 0.110 → 0.833.
* Transform2Act's design head has **no simulation input to condition on in the
  first place**. Two agents with identical bodies produce **identical
  design-head inputs**, so they must produce identical design *distributions*.
  **Convergence would be guaranteed by construction, not discovered by
  dynamics** — and an "arms race" is impossible, because neither side can
  perceive that it has an opponent.

Any morphological difference between two self-play agents could only arise from
(a) different weights, or (b) the reward signal reaching the design heads
through PPO advantages. **Neither is the design head responding to the
opponent**, which is what the question asks about.

## A retrospective consequence for E3 and E3.1

**The three opponent/goal columns E2 added to the observation never reached the
design head.** `D3_E2_RTG.md` §2 introduced them as *"CompetEvo's information
content in a translation-invariant frame"* and noted "both arms are fed exactly
these columns" — true of the **control** head, and false of the design heads,
which had already sliced them away.

So E3's and E3.1's design searches optimised body plans **blind to the task**:
no knowledge of the opponent, the goal line, or the distance remaining. The
only channel from task to morphology was the PPO advantage. **That the search
nevertheless found three distinct bodies that beat the fixed ant by 1.7-3.3x is
a stronger result than it looked** — and it also explains why the design heads
are slow (§3g-ii's unexplained 2-3x σ gap is a different quantity, but the same
blindness limits what gradient the design heads can exploit).

## Proposed E4 shape

**Not launched. Proposed for a decision.**

The prerequisite fails, so E4 cannot be run as written. Three options, in the
order I would rank them:

### Option 1 — fix the design head first, then run E4 (recommended)

Add the sim-obs slice to the design heads' input, gated by a cfg flag so every
prior rung is byte-identical with it off. The change is two lines — the slice
at 170 and 194 — plus `skel_state_dim`/`attr_state_dim`. Then E4 asks its
original question of an architecture that can express the answer.

**But it needs its own prerequisite rung**, because it changes the design head
on a task we have just measured: **E4.0, the same E3.1 primary arm with the
design head sighted, 3 seeds.** If sighted design does not at least match
E3.1's 2-of-3, the fix is not free and E4 inherits a confound.

### Option 2 — run E4 as a convergence study only, question restated

Keep the architecture and drop the arms-race half of the question. E4 then asks
*"do two co-evolving agents converge on the same body when neither can perceive
the other?"* — a real question with a **predicted answer of yes**, which makes
it a weak experiment but a cheap one. Its value is as a negative control for
Option 1.

### Option 3 — role-in-design, matching D2

Give the design head **only** a role/side scalar rather than the full sim obs.
Closest to what D2 did, smallest change, and it directly tests "can the design
head specialise by role at all" — which is E5's real prerequisite (the back
agent evolving a different body). Cheaper than Option 1 and answers less.

### Fixed regardless of option

* **≥ 3 seeds per arm from the start.** E3.1 would have read as an outright
  failure at n = 1 had we drawn s3, and we measured a 1-in-3 controller-failure
  rate on this task. Any self-play claim needs to survive one dead controller.
* **`control_log_std = −1.5`** — E3.1's derived value (§3f), without which the
  design search deletes its actuators.
* **The same instrument**, plus a frozen-body diagnostic arm per seed, which is
  what separated "the body failed" from "the controller failed" here.
