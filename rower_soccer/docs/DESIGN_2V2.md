# Unit 2f research precursor — extending the CompetEvo GPU port to 2v2 run-to-goal

*Branch `competevo-2v2-design`, worktree `/workspace/vc2-2v2-design`, written
2026-08-14. **Nothing here was trained.** Every quantitative claim carries the
command that produced it; §10 lists what could not be measured in this budget,
stated as gaps rather than answered by guesses.*

Code written for this document:

| file | what it is |
|---|---|
| `rower_soccer/competevo_port/team_scene.py` | the N-agent dev-ant scene: spawn table, contact bitmask, role-symmetric observation layout |
| `rower_soccer/competevo_port/team_env.py` | `TeamRunToGoalDevEnv` — `down_rule` x `win_rule` x `goal_credit` as flags, so the arguments below are run, not asserted |
| `rower_soccer/competevo_port/probe_2v2.py` | the five probes every number here comes from |
| `rower_soccer/competevo_port/tests/test_team2v2.py` | 13-check gate, three negative controls with a `--break` flag that makes each actually fail |
| `rower_soccer/competevo_port/dev_env.py` | two subclass hooks added (`_build_scene`, `_mask_motors`). No behaviour change: `test_design_parity` is still 6/6. |

The policy used in every rollout probe is the **1v1 pair from
`runs/competevo_port/m2e_fixed/policies.pt`** (M2E_VALIDATION §10: 83.9% goal
rate, 15.6% falls, 0% timeouts, 175.7-step episodes, 4.71 m travel — i.e. a
gait of **1.79 m/s**), transplanted into the 2v2 scene. "Trained" below always
means "trained at 1v1", never "trained at 2v2".

---

## 0. Two bugs the naive port would have shipped

Both are in code that looks like it generalises and does not. Both are now
covered by the gate.

### 0a. The contact bitmask is a two-agent trick

`competevo/evo_envs/evo_utils.py:88-89` writes, per agent, `conaffinity=i,
contype=1-i`. For `i in {0,1}` that is exactly "collide with the other agent
and the floor, never with yourself". For `i in {2,3}` it is `contype=-1,
conaffinity=2` and `contype=-2, conaffinity=3`. `-1` is all bits set. Measured
by enumerating every geom pair and applying MuJoCo's
`(contype1 & conaffinity2) || (contype2 & conaffinity1)`:

```
$ PYTHONPATH=. .venv/bin/python -m rower_soccer.competevo_port.tests.test_team2v2
[PASS] n=4: their formula is broken, ours is not
       naive self-collides [(2, 2), (3, 3)], misses [(0, 2), (1, 3)];
       ours == complete graph: True
```

So under the naive extension **teammates pass through each other** and
**agents 2 and 3 self-collide**. `team_scene.py` replaces it with one bit per
agent (`contype = 1<<i`, `conaffinity = ALL ^ (1<<i)`, world geoms `ALL`), and
the gate asserts that at `n=2` the resulting set of colliding pairs is
*identical* to theirs — the integers differ, the physics does not.

### 0b. `goal_rewards` pays the scorer's own teammate −1000

`multi_dev_agent_env.py:218-234` is

```python
for i in range(self.n_agents):
    goal_rews[i] = GOAL_REWARD if touchdowns[i] else -GOAL_REWARD
```

At two agents that is "winner +1000, loser −1000". At four agents, transcribed
unchanged and measured:

```
$ .venv/bin/python -c "... win_rule='exactly_one' ... agent 0 crosses ..."
exactly_one parse [1000.0, -1000.0, -1000.0, -1000.0]
team_first  parse [1000.0, -1000.0, 1000.0, -1000.0]
```

Agent 2 is agent 0's **teammate** and the naive rule fines it 1000 for its own
team scoring. Any 2v2 built by copying `goal_rewards` optimises against itself.

---

## 1. Spawn geometry

Implemented in `team_scene.team_init_pose`. Agent order is `(A1, B1, A2, B2)`
deliberately: agents 0 and 1 keep the 1v1 spawn, euler, goal assignment and
qpos slice offsets bit for bit, so a 2v2 scene truncated to its first two
agents *is* the validated 1v1 scene (gate: `test_first_two_agents_are_the_1v1_pair`).

```
$ PYTHONPATH=. .venv/bin/python -m rower_soccer.competevo_port.probe_2v2 geometry
```

| agent | team | spawn (x, y) | attacks | to target line | to own line | d(A1) | d(B1) | d(A2) | d(B2) |
|---|---|---|---|---|---|---|---|---|---|
| 0 = A1 | A | (−1.00, 0) | +4 | **5.00** | 3.00 | — | 2.00 | 3.00 | 5.00 |
| 1 = B1 | B | (+1.00, 0) | −4 | **5.00** | 3.00 | 2.00 | — | 5.00 | 3.00 |
| 2 = A2 | A | (−4.00, 0) | +4 | **8.00** | 0.00 | 3.00 | 5.00 | — | 8.00 |
| 3 = B2 | B | (+4.00, 0) | −4 | **8.00** | 0.00 | 5.00 | 3.00 | 8.00 | — |

Their existing goal-assignment rule needs no change: `MultiAgentEnv.__init__`
gives an agent spawning at `x < 0` the right goal, so A2 at x = −4 correctly
attacks +4.

**The goal-line spawn is physically unremarkable.** The worry was that a torso
at exactly x = ±4 straddles the goal-line cylinder (radius 0.03, lying on the
floor). It does not: the ant's four feet sit at x-offsets ±0.4 and never at 0,
so nothing touches the rod. Measured, `mj_forward` then free fall:

```
contacts per agent, at qpos0:       {}          (vs a goal rod: {})   torso z [0.75]*4
contacts per agent, after 0.3 s:    {}          (vs a goal rod: {})   torso z [0.622]*4
contacts per agent, after 3.0 s:    {0:4,1:4,2:4,3:4}  (vs a goal rod: {})  torso z [0.546]*4
```

All four torso heights agree to three decimals at every time, i.e. the
back-line ant settles exactly as the front one does.

### Is the goal-line spawn recoverable, or is defending forced?

Two numbers decide it.

**Travel budget.** At the measured 1v1 gait of 1.79 m/s, 8 m takes 4.47 s =
**298 control steps** of the 500-step budget, against 186 steps for the 5 m
front spawn. So P2 *can* physically reach the far line inside an episode; it is
not out of range.

**But it cannot win a race.** Under any first-crossing-ends-it rule, the
opposing P1 reaches its line ~112 steps before our P2 reaches ours. A back
agent that simply runs forward is, against a competent opponent, always second.
Its only positive-value plays are (i) intercept the enemy P1, (ii) score after
the front pair has stalled each other, (iii) topple. **That is the asymmetry the
brief wants, and the geometry does produce it.**

**The catch, and it is the main risk in the whole design.** For the first ~90%
of CompetEvo's curriculum the reward is ~90% dense, and the dense term is
`survive + forward-velocity-toward-my-goal − ctrl`. That term pays the back
agent to run *straight past* the enemy attacker. Defending is only rational once
`(1−α)·1000` beats the forward reward it forgoes, and at their
`termination_epoch: 1000` that is α = 0.90 at epoch 100 (M2E_VALIDATION §6). So
role differentiation is competing against an explicit shaping term that
actively discourages it for most of training.

**A geometric fact that limits defence even in principle.** The goal is
`com_x > 4` with no y constraint, and the goal line spans 10 m in y while an
ant's silhouette is ~1.30 m: a keeper standing on its line covers **13.0%** of
it. There is nothing to guard. Defence in this task can only be *interception*,
never *goal-tending*.

### Recommendation, item 1

Ship the user's layout as specified — `back_x = 4.0`, `front_x = 1.0`,
`back_y = 0.0` — because it is physically clean and it does produce the
intended asymmetry. Add two things:

1. **Randomise the asymmetry** per episode: which side the back agent is on,
   a y-offset in ±1 m, and the back x in [3.0, 4.0]. Baker et al. (arXiv
   1909.07528) Table A.1 is the reason: their emergent-phase count drops from
   6 to 4 to 2 as randomisation is removed, and a fixed asymmetric spawn is
   exactly a removal of randomisation. `team_init_pose` already takes
   `back_x`, `front_x`, `back_y` for this.
2. **Keep a y-gated goal in your back pocket.** If defence turns out to be
   unrepresentable, the cheapest fix by far is to require `|y| < 1.5` at the
   crossing, which turns 13% into 43% and makes a keeper post meaningful. It
   *is* a deviation from CompetEvo and should be declared as one.

---

## 2. Termination and downed-player semantics

`TeamRunToGoalDevEnv.down_rule` implements five rules. `"any"` is theirs
(`np.any(dones)`) extended naively; `"ignore"` is the other extreme and exists
as a control; `"frozen"` latches a fallen agent out of the episode (torque
zeroed via `_mask_motors`, dense reward exactly 0, cannot score, **body stays as
a collidable obstacle**); `"recover"` un-freezes it after N steps by teleporting
it upright in place; `"team_down"` is `"frozen"` plus "a team with both members
down loses immediately".

Three of those are load-bearing enough to be gated:

```
[PASS] one agent down: 'any' ends the episode, 'frozen' does not
       {'any': (True, True), 'frozen': (False, True), 'ignore': (False, False), 'team_down': (False, True)}
[PASS] frozen agent: torque zeroed and dense reward exactly 0
       dense [-2.24, -0.0, -2.24, -2.24]  |ctrl| down=0.000 alive=0.900
[PASS] a downed body is still a collidable obstacle
       agent0 vs fallen opponent: 1 contacts at 0.30 m, 0 at 10 m; vs fallen TEAMMATE: 1
```

The second of those is not bookkeeping. If a downed agent keeps its +1 survive
bonus it collects up to +500 per episode for doing nothing, and lying down
becomes a *strategy*. `--break payslacker` restores that behaviour and the check
fails, so the guard is real.

### Measured

```
$ PYTHONPATH=. .venv/bin/python -u -m rower_soccer.competevo_port.probe_2v2 \
      downed --worlds 512 --games 200 --rules any,frozen,team_down,recover
```

512 worlds, >= 200 finished games per cell, mean actions, `win_rule="team_first"`,
`goal_credit="team"`, `back_x = 4.0`. `1st down` is the mean step at which the
first agent in an episode goes down. **Trained (1v1 transplant):**

| rule | games | mean len | median | P(any agent down) | #down at end | 1st down | goal | wipeout | fall | timeout | reach rate A1 / B1 / A2 / B2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `any` (theirs) | 200 | **160.4** | 166 | 0.470 | 0.47 | 150.7 | **0.535** | — | **0.465** | 0.000 | 0.190 / 0.355 / **0.000** / **0.000** |
| `frozen` | 203 | 178.7 | 180 | 0.177 | 0.18 | 151.9 | **1.000** | — | 0.000 | 0.000 | 0.404 / 0.606 / 0.000 / 0.000 |
| `team_down` | 201 | 178.7 | 180 | 0.194 | 0.22 | 156.8 | **0.980** | **0.020** | 0.000 | 0.000 | 0.383 / 0.602 / 0.005 / 0.000 |
| `recover` (N=50) | 200 | 178.2 | 180 | 0.150 | 0.15 | 152.0 | **1.000** | — | 0.000 | 0.000 | 0.365 / 0.640 / 0.000 / 0.000 |

A second, independent run at 300 games agrees within sampling noise (`any`
169.8 steps / 0.588 goal / 0.412 fall; `frozen` 191.7 / 1.000; plus an
`ignore` control at 190.8 / 1.000), so none of the separations below is a
one-run artifact.

**Untrained**, which is the regime training actually starts in. One trap first:
an untrained `DevActorCritic` at MEAN actions emits ~0 torque (their
`control_action_mean.weight.data.mul_(0.1)` init) and simply **stands still for
500 steps** — measured, `P(any down) = 0.000`, 100% timeout, 0.00 m/s travel.
That row measures the initialisation, not the task. The row that matters is with
**sampled** actions at `log_std = 0`, which is what the first rollouts of a run
look like:

```
$ PYTHONPATH=. .venv/bin/python -u -c "... rollout_stats(env, drv, 200, mean=False) ..."
```

| rule | games | mean len | P(any down) | #down at end | 1st down | goal | wipeout | fall | timeout |
|---|---|---|---|---|---|---|---|---|---|
| `any` | 201 | **193.1** | **1.000** | 1.00 | — | 0.000 | — | **1.000** | 0.000 |
| `frozen` | 256 | **500.0** | 0.832 | 1.37 | 295.0 | 0.000 | — | 0.000 | **1.000** |
| `team_down` | 258 | **466.4** | 0.837 | 1.34 | 286.0 | 0.000 | **0.240** | 0.000 | 0.760 |

(256 worlds; `n_diverged = 0` in all three.)

**This is the sharpest result in the document.** Early in training:

* under **`any`**, 100% of episodes end in a fall at ~193 steps and **0% produce
  any sparse signal at all**;
* under **`frozen`**, episodes run the full 500 steps with 1.37 of 4 agents
  already out, and **still 0% produce a sparse signal** — 2.6x the physics per
  episode for the same nothing;
* under **`team_down`**, **24.0% of episodes end in a whole team going down**,
  which pays ±1000. A rule that terminates on a team wipe-out is the only one of
  the three that gives a *fresh* policy a sparse learning signal in this task.

### What the table says

* **Their rule throws away 46.5% of episodes.** Under `any`, nearly half of all
  games end because *somebody* fell, and the goal rate is 0.535. Under any rule
  that lets the game continue, the goal rate is **1.000** — with 15-19% of
  episodes still containing a fallen agent. So the falls are not the game
  ending; they are one player leaving it. `np.any(dones)` is discarding
  half the outcomes for no reason and, worse, it makes toppling an opponent a
  way to *stop* the game rather than to win it.
* **Episodes get 11% longer, not 3x longer.** 160.4 -> 178.7 steps. Nothing
  times out under any rule, so the worry that "frozen" would leave four corpses
  lying in an empty arena for 500 steps does not materialise at this policy —
  the survivors finish the game.
* **`recover` buys nothing.** 30 recoveries over 200 games, and every number in
  its row is inside noise of `frozen`'s. It costs an unphysical teleport and an
  extra `mj_forward` per step for a 0.03 shift in P(any down).
* **`team_down` costs 2% of episodes and adds the strategy we care about.**
  Both members of a team going down happens in 2.0% of games under this policy
  — rare enough that it is not a shortcut, present enough that it is a
  reachable terminal state, and it is *exactly* the "coordinate to flip both
  opponents and win" outcome the brief asks about. Everything else in its row
  matches `frozen`.
* **The back agents essentially never score.** Reach rate is 0.000-0.005 for
  A2/B2 against 0.19-0.64 for A1/B1, in every rule. This is the §1 race
  arithmetic showing up as a measurement: 8 m against 5 m, with the first
  crossing ending the game.
* **Falls are concentrated on the front pair** (per-agent down rate under `any`:
  0.260 / 0.095 / 0.085 / 0.030 for A1/B1/A2/B2) — they are what happens when
  the two attackers meet in the middle, which is also where a toppling strategy
  would have to operate.


### Recommendation, item 2

**`down_rule = "team_down"`**: a fallen agent is out for the rest of the
episode — torque zeroed, no dense reward, cannot score, body left in the arena
as a collidable obstacle — and a team whose *both* members are down loses
immediately, paying the goal reward to the survivors.

The measurement that justifies it, in one line: **`team_down` is the only rule
of the three that gives an untrained policy any sparse signal (24.0% of
episodes) and the only one that makes toppling both opponents a win, and it
costs 2.0% of episodes at a competent policy where the alternative, `frozen`, is
identical on every other number.**

The rest of the case, briefly:

* **against `any` (theirs)**: it discards 46.5% of competent-policy episodes and
  turns "topple an opponent" into "stop the game", which makes the behaviour the
  brief is about literally unrepresentable. It is also the fastest rule early
  on (193 vs 500 steps per episode, 2.6x more episodes per unit of physics) —
  that is its only advantage and it buys nothing, because 0% of those episodes
  carry a sparse signal either.
* **against `frozen`**: identical on every trained-policy number, but 0% sparse
  signal from a fresh policy against `team_down`'s 24%, and unbounded episodes
  when all four are down.
* **against `recover`**: 30 recoveries per 200 games moved nothing; it needs an
  unphysical teleport (an ant cannot right itself, so "recover" without a
  re-pose is just "frozen" with a delay), and it removes the point of toppling.
* **against teleporting the corpse out**: not implemented, and it deletes the
  obstacle. "Flip them over and walk *past* them" needs something to walk past.
  The gate asserts a fallen body still generates contacts with both an opponent
  and a teammate.

Two details that are part of the recommendation, not decoration:

* **A downed agent must earn exactly zero**, including the survive bonus. This is
  a `--break payslacker` control in the gate because getting it wrong turns
  lying down into a +500/episode strategy.
* **Keep the `bad`/non-finite exit** exactly as the 1v1 env has it. It is
  orthogonal to all of this and `n_diverged` was 0 in every run here.


### 2b. The renders, which were looked at

```
$ ... probe_2v2 render --worlds 8 --episodes 3 --down-rule team_down \
      --out runs/competevo_port/2v2_probe/trained_teamdown.mp4      # 932 frames, 23.3 s
$ ... probe_2v2 render --worlds 8 --episodes 2 --down-rule any --untrained \
      --out runs/competevo_port/2v2_probe/team2v2_untrained_any.mp4 # 1000 frames, 25.0 s
```

Frame grids (`*_grid.png`) were extracted and inspected. What they confirm, and
none of it was visible in a number:

* **The layout on screen is the layout in the table.** Four ants, left to right:
  beige A2 straddling the left red goal line, beige A1, dark B1, dark B2
  straddling the right line. Teams are colour-coded (`team_scene._team_rgb`)
  because a 2v2 clip in which all four ants are the same beige is unreadable.
* **The goal-line ants sit on their line without visible interference** — the
  rod passes between their feet, matching the zero-contact measurement.
* **The trained transplant's ants converge on the middle and tangle.** Several
  frames show three or four bodies in contact around x = 0, with ants climbing
  over and flipping each other. That is the physical substrate the "topple them
  and walk past" strategy would have to use, and it exists.
* **The untrained clip shows the ants collapsing in place** and the episode
  restarting repeatedly with no net travel — the visual form of the
  `travel ≈ 0.00 m/s`, `100% fall` row.
* No clipping, no ghosting, no exploded states.

---

## 3. Win condition with four agents

`win_rule="team_first"`, `goal_credit="team"`:

* a team scores the moment **any** of its members' `com_x` crosses that team's
  target line; the episode ends;
* **both** members of the scoring team are paid +1000 and both opponents −1000;
* if both teams cross on the same step, nobody is paid and the episode ends —
  their `num_reached != 1 → all zeros, game_done=True` draw rule, lifted from
  agents to teams;
* a downed agent cannot score (it is out of the game, so its body drifting over
  a line is not a goal).

Gated:

```
[PASS] both teammates cross: 'exactly_one' pays nobody, 'team_first' wins
[PASS] both teams cross on the same step: draw, episode ends
[PASS] goal_credit team/scorer/split pay what they say
       {'team': [1000,-1000,1000,-1000], 'scorer': [1000,-1000,0,-1000], 'split': [500,-500,500,-500]}
```

### What it does to the reward scale the curriculum anneals against

The honest answer is: **nothing, provided the trainer keeps per-agent rewards.**
Each agent still receives ±1000 exactly once, and its dense term is still its
own. `alpha * dense + (1 - alpha) * parse` is unchanged term for term, so
`DEV_CURRICULUM_STEPS` does not need retuning and the 1v1 α schedule stays
comparable epoch for epoch.

Four things *do* move, and only two of them are about the sparse term:

1. **Episode length moves the dense side.** The sparse term is paid once; the
   dense term accumulates per step. Any rule that lengthens episodes therefore
   lowers the sparse term's share at fixed α. This is the real coupling between
   items 2 and 3, and it is measured in §2's table — read the `len` column as a
   reward-scale column. Measured for the recommended rule (§4): per-agent dense
   return **460 ± 124** over 566 episodes, against a one-off ±1000, so at
   α = 0.90 the curriculum weighs **0.90 x 460 = 414 of dense against 0.10 x
   1000 = 100 of sparse** — the sparse term is ~24% of the dense one at the
   epoch where the 1v1 reference's win rate first leaves zero.
2. **`down_rule="team_down"` adds a second ±1000 event** (the wipe-out). It is
   the same magnitude and mutually exclusive with the goal payout in the same
   step (`terms` fires it only when `game_done` is false), so it does not change
   the scale — it changes the *frequency* of a sparse event, from 0% to 24% of
   episodes at a fresh policy. That is the point of it.
3. **`goal_credit="split"` is a curriculum change in disguise.** Halving the
   goal reward is exactly equivalent to halving `(1−α)` at every epoch. If you
   want that, change α, not the payout.
4. **`goal_credit="scorer"` breaks the strategy the brief is about.** A blocker
   that topples both opponents so its teammate can walk through is paid zero for
   it. Do not use it.

If you later decide to sum teammates' rewards into one team scalar (§4 says do
not), *both* terms double, the ratio survives, PPO's advantage standardisation
absorbs most of it — but the value targets double and the critic's `l2_reg`
(already a declared half-strength deviation, M2E §3c) is calibrated at the 1v1
scale. That is a second reason to keep rewards per-agent.

### Recommendation, item 3

`win_rule="team_first"`, `goal_credit="team"`, per-agent rewards, curriculum
constants unchanged from 1v1. Log the dense-return distribution in the first
smoke run and compare it to the 1v1 run's; if episodes are much longer, that —
not the goal payout — is what has shifted the curriculum.

---

## 4. Team credit assignment

Two measurements, one per-step and one per-episode.

```
$ PYTHONPATH=. .venv/bin/python -u -m rower_soccer.competevo_port.probe_2v2 \
      credit --worlds 128 --k 24 --warmup 60 --ep-steps 1200
```

**Per step — a one-step counterfactual on frozen states.** 128 states, 24
resamples of one agent's motor action (sd 0.5, everyone else held at the
deterministic mean), the state restored exactly between draws.

| agent | `own` = Var over own action of own dense | `team` = ... of the team dense | `mate` = Var over the TEAMMATE's action of the team dense | **mate/own** | `cross` (control) | cross/own |
|---|---|---|---|---|---|---|
| 0 (A1) | 1.776e−1 | 1.779e−1 | 1.667e−1 | **0.94** | 1.14e−4 | 6.4e−4 |
| 1 (B1) | 1.878e−1 | 1.878e−1 | 1.762e−1 | **0.94** | 0.00 | 0.0 |
| 2 (A2) | 1.698e−1 | 1.695e−1 | 1.692e−1 | **1.00** | 1.45e−4 | 8.5e−4 |
| 3 (B2) | 1.726e−1 | 1.726e−1 | 1.826e−1 | **1.06** | 2.16e−14 | 1.3e−13 |

Mean `mate/own` = **0.98**, range [0.94, 1.06].

**NEGATIVE CONTROL, and it can fail.** `cross` is the variance that resampling
agent *i*'s action induces in its **teammate's own** dense reward. Physically
that can only travel through contact, so on states where the two are metres
apart it must be ~0; measured max **8.5e−4** of `own`. This is the check that
the state restore is real — if the snapshot/restore leaked, the teammate's
reward would drift between draws and `cross` would be the same order as `own`.
`probe_2v2 credit --break-restore` skips the restore's `forward()` and moves it.
(Note `team ≈ own` in the table: resampling *i*'s action moves the team total by
essentially exactly what it moves *i*'s own reward by — same statement, seen
from the other side.)

**Per episode — 566 completed episodes.**

| agent | mean dense return | sd(R_i) | sd(R_i + R_mate) | **Var ratio** | corr(R_i, R_mate) |
|---|---|---|---|---|---|
| 0 (A1) | 419.0 | 110.6 | 178.8 | **2.61** | 0.030 |
| 1 (B1) | 443.4 | 94.3 | 170.7 | **3.27** | 0.101 |
| 2 (A2) | 484.0 | 137.2 | 178.8 | **1.70** | 0.030 |
| 3 (B2) | 489.4 | 133.1 | 170.7 | **1.65** | 0.101 |

**What these say, in one sentence each.**

* Teammates' episode returns are nearly **uncorrelated** (r = 0.03-0.10), so
  `Var(R_i + R_mate) ≈ Var(R_i) + Var(R_mate)`: summing the team's dense
  rewards into one scalar buys **1.65x to 3.27x the variance for no additional
  controllable signal**. Per step the same statement reads `mate/own ≈ 1`:
  every unit of variance an agent can steer comes with one unit it cannot.
* The **back agent is not a freeloader in dense terms — it is the better paid
  one** (484/489 against 419/443). It travels further before the episode ends
  and is knocked over less. That inverts the naive lazy-agent story here: the
  danger is not that the back ant does nothing, it is that running forward
  *already pays it well*, so nothing in the dense reward pushes it toward a
  defensive or interfering role.
* Scale: per-agent dense return ~460 ± 124 against a one-off ±1000. At α = 0.90
  the curriculum weighs them 0.90 x 460 = 414 against 0.10 x 1000 = 100, i.e.
  the sparse term is ~24% of the dense one in magnitude at the epoch where the
  1v1 reference's win rate first left zero.


### Recommendation, item 4

**Individual dense + shared sparse.** Each agent's reward is its own
`dense` plus the team's `parse`; no team-summed scalar, no counterfactual
baseline. Three independent reasons:

1. **Theory says the penalty is at its floor at n = 2.** Kuba et al. (arXiv
   2108.08612, NeurIPS 2021) bound the excess variance of a CTDE policy
   gradient over a decentralised one by `(n−1)(εB_i)²/(1−γ²)`, and prove COMA's
   counterfactual baseline reduces but does not eliminate it and is *not* the
   optimal baseline. At n = 2 the factor is 1 — the smallest nonzero credit
   penalty a team can have.
2. **The measurement agrees** (table above): summing the teammate's dense
   reward into the signal adds variance an agent cannot influence, for no added
   signal, and the amount it adds is close to the amount the agent already
   controls. Whereas the *sparse* term is a team event and is not decomposable
   at all — "shared sparse" is not a design choice, it is what the game is.
3. **The empirical literature is one-sided.** MAPPO (arXiv 2103.01955) and IPPO
   (arXiv 2011.09533) with a plain shared reward beat COMA on SMAC by wide
   margins; COMA is discrete-action by construction (its baseline sums over the
   action set) and its continuous extension is sampling-based and unsolved. Liu
   et al.'s 2v2 soccer (arXiv 1902.07151) uses exactly this split: sparse team
   score plus two *individual* dense shaping channels.

**COMA was not implemented and this document does not claim to have tested it.**
What the counterfactual probe bounds is how much a perfect COMA critic could
buy: at best it removes the teammate-induced variance from the dense term,
which is the `mate/own` column — and it would need a joint-action critic over
4 x 28 action dimensions to do it.

Two things to carry into the trainer that the literature is firm about:

* **Death masking.** MAPPO Appendix C.3: when an agent is dead, feed the
  centralised critic an all-zeros state carrying only that agent's ID one-hot,
  not the (still evolving) true global state. Their ablation finds this
  "significantly outperforms" both alternatives and lowers value loss on most
  maps. Our `frozen` rule creates exactly this situation and the port has no
  death-masking today.
* **Team spirit as a fallback, not a default.** OpenAI Five's
  `r_i = (1−τ)ρ_i + τρ̄` with τ annealed 0.3 → 0.8 exists precisely because
  "lower team spirit reduces gradient variance in early training". If the
  measured lazy-agent failure shows up, that is the knob — it is one line on top
  of the recommended split.

---

## 5. Population and genome structure

### The finding that decides this item: **the design head cannot see the world at all**

`DevActorCritic.dists` (and their `custom/models/dev_actor.py:88-96`, which it
transcribes) feeds the scale head **only the 20-dim `scale_vector`** — the
i.i.d. `U(−1, 1)^20` their env redraws per agent per episode. `sim_obs` reaches
the *control* head and nothing else. Their `use_entire_obs` flag does not change
this: it is read at `dev_actor.py:36` and `:97`, both inside the **control**
branch. There is no configuration of their code in which morphology is
conditioned on position, teammate, or opponent.

Measured, on the trained `ac_0`, 512 worlds:

```
$ PYTHONPATH=. .venv/bin/python -u -m rower_soccer.competevo_port.probe_2v2 \
      roles --worlds 512 --warmup 120 --ep-steps 800

spread of the design across random scale vectors        sd = 0.1809
|design(agent 0) - design(agent 2)|, mean abs              = 0.2017
... holding the scale vector fixed, changing only the spawn = 0.0000  (0.0% of the spread)
```

So the two teammates *do* get different bodies — and **100% of that difference
is the random scale vector**; the contribution of being at x = −1 versus
x = −4 is exactly zero, to fp32. "One policy already gives two genomes" is true
and worthless: the two genomes are two draws from the same distribution.

### What the control policy can see

```
CPD on the opponent-position channel, after 120 steps:
  per agent [0.0095, 0.0185, 0.0158, 0.0155]   (mean |motor action| = 0.165)
  CPD on the TEAMMATE channel is 0 by construction for a transplanted 1v1 net.

role separability by com_x over 51,200 agent-steps:
  front mean +0.51, back mean -1.91, distribution overlap 30.4%
```

Counterfactual policy divergence (shuffle the other-agent `(x, y)` across worlds,
hold own state fixed) is **6-11% of the mean action magnitude** — the control
head does use the opponent channel, weakly. The teammate channel does not exist
in a 52-dim net; in the 56-dim team layout it is `obs[..., 50:52]` and is the
first thing a role-capable policy needs.

Role separability by absolute `com_x` is real but leaky: the front and back
agents' position distributions **overlap by 30.4%**, so a shared memoryless
policy that must infer "which of the two am I" from position is wrong about a
third of the time.


### The other structural facts

* A policy shared across **teams** is not possible without work. The scene is
  not mirror-canonicalised: the observation is world-frame absolute `qpos`, and
  agent 1 differs from agent 0 by a 180° yaw and a sign flip on the goal. The
  1v1 port sidesteps this by training two independent learners, one per side.
  Sharing *within* a team is fine — both teammates attack the same direction.
* Cost. M2E measured a two-learner iteration at ~1.65x a one-learner one. One
  policy per team is 2 learners, i.e. the 1v1 cost model unchanged. One policy
  per *role* per team is 4 learners, ~2.7x, and halves the ego transitions per
  learner at a fixed step budget.

### Recommendation, item 5

**One policy per team (two learners total, the validated `CoEvoPPO` shape), with
three input changes — and the design-head one is not optional if morphological
division of labour is a goal of 2f.**

1. **Feed the role bit and the sim state to the SCALE head**, not just the
   control head. Today the genome is a function of an i.i.d. random vector and
   nothing else, measured at exactly 0.0 dependence on position. A team whose
   two bodies differ only by noise is not a team with roles; it is a team with
   two random ants. Minimal change: `scale_mlp` input becomes
   `[scale_vector (20) | role one-hot (2)]`, or the full observation if you want
   spawn-conditioned bodies too. **This is a declared deviation from CompetEvo**
   — their code has no such path, `use_entire_obs` notwithstanding — and it
   should be an ablation arm, because "role-conditioned morphology helps" is a
   claim 2f can actually test.
2. **Add an explicit role/index one-hot to the control observation** (2 dims).
   VDN (arXiv 1706.05296) is blunt: parameter-shared agents cannot specialise
   unless given an identifier — "when specialized roles are required… we provide
   each agent with role information, or an identifier." The measurement above
   says the only role-carrying channel today is absolute `com_x`, and its two
   distributions overlap by **30.4%**. Note this trades against §1's
   randomisation: randomise *which spawn the indexed agent gets*, so the bit
   means "player 2" and not "the ant at x = −4".
3. **Add the teammate to the observation.** It already exists in the
   role-symmetric layout (`obs[..., 50:52]`); the 1v1 net simply has no slot for
   it. Without it, teammate-conditioned behaviour — the *definition* of a role —
   is not representable, and the CPD metric the soccer papers use to detect
   roles is identically zero by construction.

Against **two policies per team**: it costs ~2.7x an iteration (4 learners
against 2 at M2E's 1.65x-per-extra-learner) and halves each learner's ego data
at a fixed step budget, and items 1-3 above buy the same specialisation for
about 24 input dimensions. Against **one enforced genome per team**: it is
strictly less expressive than the above and nothing in the brief wants it.

Heterogeneous teams (two policies *and* two genomes per team) is 2g, not 2f. The
thing 2f must not do is foreclose it: key the ring by team so a heterogeneous
pair is checkpointed and resampled as a unit (§6).

---

## 6. Opponent sampling for teams

**Sample a whole past team, not a mix.** Under the §5 recommendation a team is
one checkpoint, so whole-team sampling is also the *cheapest* option — mixing
would require deliberately drawing two indices.

The evidence against mixing is strong and comes from outside self-play. PSRO
(arXiv 1711.00832) measures *joint policy correlation*, the return lost when
agents from independent runs are paired: **34.2%** on laser-tag small2 and
**71.7%** on small4, in a physical domain. Hanabi cross-play collapses 23.97 →
2.52 (Other-Play, arXiv 2003.02979). Team-PSRO (NeurIPS 2023) makes the
structural version of the point: the best-response oracle must be a *joint*
team best response, not per-agent best responses. Mixing teammates from
different epochs is precisely cross-play, and its cost is a metric to measure,
not a diversity trick to assume is free.

**Preserve `delta` exactly.** `OpponentRing.sample_epoch` transcribes their
rule — `start = max(1, floor(delta*epoch))`, `randint(start, epoch)`
**high-exclusive**, so the opponent is uniform on `[start, epoch−1]` and is
strictly past (selfplay.py's docstring, and M2E §1). Nothing about teams
changes that rule; the checkpoint index is a team index instead of an agent
index and everything downstream is unchanged, including `n_clamped` and the
`opp_lag ≈ epoch/4` invariant the 1v1 run checks against.

One measured note on `delta` itself, which the port should record rather than
inherit blindly: Bansal et al. (arXiv 1710.03748) Table 1 finds δ = 0.5 best for
*humanoids* and **δ = 0.0 best for ants** (E[Win] 0.50 vs 0.34-0.36 for
δ ∈ {0.5, 0.8, 1.0}). CompetEvo's devants config uses 0.5. We should keep 0.5
for reproduction fidelity and add δ = 0 as the first sampling ablation, because
our creature is the one Bansal measured δ = 0 to be better for.

**Cost.** `blocks = 4` slots per side is unchanged in launch count: one net now
drives two agents, so the batched `StackedDevActors` forward sees 2x the rows
and the same number of kernel launches. M2E measured `blocks=4` at ~7% of
iteration wall time; expect that to stay in single digits.

### Recommendation, item 6

Keep `OpponentRing` and `delta = 0.5` verbatim, re-keyed from agent to team;
push one entry per team per epoch; sample one index per opposing team per slot.
Add cross-play (a team's agents drawn from two different epochs) as a *metric*
in the 2g ablation set, not as the sampling scheme.

---

## 7. Literature: what is solved, what is open, what we would walk into

**Solved, and directly usable.**

* *Ant toppling in run-to-goal already exists at 1v1, from a pure sparse
  signal.* Bansal et al. (arXiv 1710.03748) report ants "using legs to topple
  the opponent and running towards the goal", with no interference term in the
  reward. In you-shall-not-pass the blocker is paid only for a successful block
  while standing. **Interference does not have to be rewarded to emerge** — at
  1v1, on this creature, in this task.
* *The dense→sparse anneal is load-bearing and ablated.* Without it, agents
  "optimize for a particular component of the dense reward" — in sumo they
  stand still at the centre; in kick-and-defend the kicker carries the ball.
  Their α is annealed to 0 over 500 iterations, ~10-15% of training.
* *Shared team reward is not the blocker.* Every reported case of emergent
  within-team differentiation — hide-and-seek, both Liu soccer papers — uses a
  fully shared, permutation-invariant team reward.
* *Death masking* (MAPPO Appendix C.3) is the settled treatment of a dead agent
  in a centralised critic.
* *Episode-level termination with in-episode recovery is the universal
  convention.* dm_control soccer terminates only on `detected_goal()`; falls do
  nothing and Liu et al. 2022 report 80% fall recovery after six hours of
  training. Google Research Football returns a scalar `done` even when
  controlling multiple players. **Nobody terminates the episode because one
  agent fell.** Their `np.any(dones)` is the outlier, not our departure from it.

**Genuinely open.**

* Whether spawn asymmetry alone induces stable attacker/defender roles in a 2v2
  continuous-control game. Untested anywhere I can find.
* Whether a pure win/lose signal suffices for *coordinated* interference at 2v2.
  The 1v1 existence proof does not transfer for free.
* Whole-team vs per-agent checkpoint sampling: no direct ablation exists.

**The three failure modes this plan would walk into.**

1. **A compute shortfall reported as a negative result.** This is the big one.
   Liu et al. 2021 (arXiv 2105.12196) need **1.5×10⁹ steps** before division of
   labour even stops *decreasing*, and 8×10¹⁰ for it to reach 0.85; Liu 2019
   sees teammate-conditioning only after 5-20×10⁹ steps; hide-and-seek's fourth
   phase took 132.3M episodes / 31.7B frames and **never converged** at batch
   16k or 8k. CompetEvo's run-to-goal budget is 1000 epochs x 50k = **5×10⁷
   steps** — three orders of magnitude below every regime in which role
   emergence has been reported. Worse, both soccer papers report that division
   of labour gets *worse* first: expect both ants to crowd the objective for a
   long time and do not read that as failure. **Plan the 2f eval so that "no
   roles emerged" is reported as "not reached at this budget", with the budget
   quoted.**
2. **The lazy agent.** VDN's own worked example is literally this setup: "imagine
   training a 2-player soccer team using RL with the number of goals serving as
   the team reward signal… one agent learns a useful policy, but a second agent
   is discouraged from learning because its exploration would hinder the first."
   The asymmetric spawn actively invites it — the front ant is closer to winning
   alone, and the back ant's exploration mostly *costs* the shared return early.
   Mitigations, all in the recommendations above: per-agent dense (§4), an
   index one-hot (§5), team spirit τ as the fallback knob (§4). The n = 2
   relative-overgeneralization pathology (Wei & Luke, JMLR 17:2914-2955) is a
   closely related risk that policy gradients are specifically prone to.
3. **A "win" that is an observation attack rather than a skill.** Gleave et al.
   (arXiv 1905.10615) trained adversaries for 20M steps — under 3% of the
   victim's budget — that beat Bansal's frozen victims **without ever standing
   up**: 86% win rate in you-shall-not-pass. Blinding the victim to the
   adversary's position took the victim from 14% to **99%**. With ±1000 and two
   ants able to converge on one opponent, the cheapest 2v2 policy may be to
   induce a fall through off-distribution contact. Ants help a little — Gleave
   reports adversaries do markedly worse in the low-dimensional Sumo Ants than
   in Sumo Humans — but do not immunise. **The probe is cheap and should be in
   2f's eval: mask the opponent's view of the winner and re-measure. If the
   advantage collapses, it is an exploit, not the behaviour we wanted.**

Two more findings worth carrying, both cheap:

* **Measure roles the way the soccer papers do**, not by eyeballing: *division
  of labour* (one but not both teammates within radius r of the objective) and
  *counterfactual policy divergence* on the teammate channel. CPD directly
  answers "does this policy depend on where my teammate is?", which is the
  necessary condition for any role claim, and `probe_2v2 roles` already
  computes it.
* **Value normalisation is genuinely contested for locomotion.** MAPPO says it
  "never hurts"; Andrychowicz et al. (arXiv 2006.05990, 250k+ runs) find it
  "crucial for HalfCheetah and Humanoid" but that it "significantly hurts the
  performance on Walker2d". Ablate rather than adopt.

---

## 8. Ranked implementation plan for 2f

Ordered so the cheapest thing that could falsify the whole approach runs first.

**1 — DONE HERE, and it half passed.** "Does the task survive at four bodies and
8 m?" Under `team_down` + `team_first` a transplanted 1v1 pair scores in
**98.0%** of episodes in **178.7** steps — the task is intact. But the back
agents' reach rate is **0.000-0.005** against the front pair's 0.38-0.60. So the
falsifier fired *partially*: the game works, and the second player is decorative
under a first-crossing rule. **The decision this forces, before training:** if
2f wants the back agent to be a player rather than a spectator, the goal
condition or the geometry has to give it something to do — a y-gated goal
(§1), a shorter `back_x`, or accepting that its only value is interference and
letting the sparse team reward find that. This document recommends the third,
with the first held in reserve, but it is a decision and not a default.

**2 — DONE HERE.** Rendered and inspected: `probe_2v2 render` under
`team_down` (trained transplant, 932 frames) and `any` (untrained, 1000
frames). Both are in §2b below. Nothing visually wrong; the layout on screen is
the layout in the table.

**3 — (half a day) Observation and network plumbing, with a regression.** Widen
`DevActorCritic` to the 56-dim obs (+2 for the role one-hot = 58). **The
regression that matters:** a 1v1 net loaded into the widened one, with the
teammate and role channels zeroed, must reproduce its 1v1 behaviour to fp32 on
the same states. Without that check a silent input-permutation bug is
indistinguishable from "2v2 is hard".

**4 — (hours) Single-learner 2v2 smoke.** One net driving all four ants (both
teams — legitimate here only because it is a smoke, and it will be bad at one
side), no ring, 20 iterations. Checks: 0 NaNs, 0 diverged worlds, `alpha`
schedule intact, dense reward per step in the same ballpark as the 1v1 −3.0
(M2E §5), design head moving.

**5 — (a day) Two-learner co-evolution with the team ring.** `CoEvoPPO` with
`n_agents=4`, ego split by team instead of by agent, one ring entry per team.
Checks: `opp_lag ≈ epoch/4`, `ring_clamped = 0`, both learners' `train_ret`
climbing. **Note `CoEvoPPO.__init__` asserts `env.n_agents == 2` today** — that
assert, `_LaneEnv`, and the `act[w, e]` lane indexing are the three places the
trainer is 2-agent-shaped.

**6 — (a day, GPU) The 200-epoch run**, at the config M2E validated, against
the 1v1 run as the control on every shared quantity (train return per step,
eval length, ending histogram). The comparison is *not* to the paper; it is to
our own 1v1 run, which is the only thing that makes a 2v2 number interpretable.

**7 — (a day) Role metrics, not vibes.** Division of labour, CPD on the teammate
channel, topple counts (`newly_down` caused within 0.5 m of an opponent),
and the masked-opponent exploit probe from §7.

**8 — (ablations, cheapest first)** `down_rule` (team_down vs frozen vs any);
`goal_credit` (team vs scorer) — this one directly tests whether the blocker
being paid is what makes blocking appear; `delta` 0.5 vs 0.0 (Bansal says 0.0
for ants); shared-vs-summed reward; spawn randomisation on/off.

**9 — (2g, out of scope here)** heterogeneous teams: two policies and two
genomes per team, ring keyed by pair.

### What 2v2 costs, measured

```
$ ... python -c "... 60 timed steps, interleaved, after 10 warmup ..."
W=128: 46.3 ms/step,  2,766 world-steps/s
W=512: 60.7 ms/step,  8,440 world-steps/s
```

Against the 1v1 port's ~10,200 world-steps/s at 1024 worlds (M2E §"the GPU port
IS a ~19x speedup"), a 2v2 world costs about **1.2x** a 1v1 world at comparable
batch — the 4-agent model is 2x the bodies but the step is launch-bound, not
arithmetic-bound. Per *agent-transition* 2v2 is therefore **cheaper**. Budget a
2f 200-epoch run at roughly the same wall clock as the 1v1 one (~48 h at their
PPO settings), and note that 512 worlds is well past the point where the batch
stops being launch-bound: do not run 2f at 128.

**Operational note for whoever runs these.** Several long probe processes were
killed silently mid-run in this session when launched detached; the runs that
completed were foreground or tool-tracked. Nothing in the port caused it
(`n_diverged = 0` everywhere, no tracebacks), but a 2f run should write
per-iteration to `log.json` as the existing trainer does rather than trusting a
process to survive to its final print.

---

## 9. Summary of recommendations

| # | question | recommendation |
|---|---|---|
| 1 | spawn geometry | user's layout as specified (5 m / 8 m); randomise side, y ±1 m, back_x ∈ [3, 4]; y-gated goal held in reserve |
| 2 | downed player | `team_down`: out for the episode, torque zeroed, unpaid, body left as an obstacle; both members down = the team loses |
| 3 | win condition | `team_first` + `goal_credit="team"`; per-agent rewards; curriculum constants unchanged |
| 4 | credit assignment | individual dense + shared sparse; no team sum, no COMA; add death masking; team spirit τ as the fallback |
| 5 | population / genome | one policy per team, two emergent genomes; +role one-hot; +teammate in the obs |
| 6 | opponent sampling | whole past teams; `delta=0.5` semantics verbatim; δ=0 as the first ablation |
| 7 | literature | interference is a reported 1v1 result; the budget gap to reported role emergence is ~3 orders of magnitude |
| 8 | plan | falsify the geometry first, then watch it, then plumb, then train |

---

## 10. What was NOT tested

Stated plainly, because an honest gap is worth more than a plausible guess.

* **Nothing was trained.** Every behavioural number comes from a 1v1-trained
  pair transplanted into 2v2. That policy has never seen a teammate, cannot see
  a teammate (its 52-dim input has no slot for one), and was optimised against a
  5 m spawn. It is a *scripted competent walker*, not a 2v2 agent, and it is a
  lower bound on nothing in particular.
* **COMA / difference rewards were not implemented.** §4's argument is theory
  (Kuba's bound), literature (MAPPO/IPPO vs COMA), and a variance decomposition
  that bounds what a perfect counterfactual critic could recover. It is not an
  ablation.
* **No 2v2 trainer exists.** `CoEvoPPO` still asserts two agents. The plan's
  step-5 estimate of what has to change is a code reading, not a build.
* **The `recover` teleport is unphysical** and was implemented only so the
  option could be measured. If it is chosen it needs a real fall-recovery
  mechanism, which is a research project of its own (Liu et al. get 80% fall
  recovery, but out of a humanoid with a motor-control prior).
* **No cross-play measurement.** §6's recommendation rests on PSRO and Hanabi
  numbers from other domains, not on ours.
* **Solver fidelity at four agents was not gated.** The 1v1 parity gate compares
  our scene against their checked-in merged 2-agent scene; there is no 4-agent
  scene of theirs to compare against, so `test_first_two_agents_are_the_1v1_pair`
  (same masses, slices, spawns, actuator order) is the closest available check.
  Contact counts were measured — max 12 per world at 4 agents against 7 at 2, so
  `nconmax=64, njmax=512` (both *per world* in mujoco_warp) are not close to
  binding.
* **Episode-length effects on the curriculum were measured but not tuned.** §3
  says episode length is what moves the sparse/dense balance; nobody has decided
  what to do about it if 2v2 episodes turn out to be twice as long.
* **`win_rule="exactly_one"` was only exercised on hand-set states**, not in a
  rollout. §0b's "it fines your own teammate" is exact and gated; what fraction
  of *rollout* episodes it would corrupt is not measured (it is all of the ones
  that score, but the number is not in this document).
* **The 1v1 dense-return distribution was not re-measured** for the §4 scale
  comparison, so "sparse is ~24% of dense at α = 0.90" is stated for 2v2 only.
  Comparing it to 1v1 needs the same probe run against `RunToGoalDevEnv`, which
  is ~10 minutes of work nobody did.
* **`goal_credit="split"` and `"scorer"` were gated but never rolled out.** The
  argument against them in §3 is structural, not empirical.
All four negative controls in this document were run and all four failed on
demand: `--break bitmask`, `--break nomask`, `--break payslacker` in the gate,
and `probe_2v2 credit --break-restore`, which moves `cross/own` from
**8.5e−4 to 1.03** — i.e. with the restore broken, resampling one agent's action
appears to move its distant teammate's own reward by as much as its own.

---

## 11. First training results (2026-08-24) — and a correction to how they were read

`runs/competevo_port/t2v2_cold`: step 6, cold start, 512 worlds, 200 epochs, the
config M2E validated. `runs/competevo_port/m2e_fixed` is the 1v1 control.

### The logged goal rate is not the goal rate

`train_team_selfplay`'s per-iteration eval reported "goal 0.595" at epoch 59.
**That number is wipeouts.** Its classifier derived the ending from
`info["winner"]`, and under `down_rule="team_down"` the env sets `winner` on a
wipeout as well as on a crossing (`team_env.py:226`,
`winner = torch.where(fire, (~lose), winner)`). So "one team knocked the other
team over" was being counted as a goal.

Fixed to use the env's own `last_end`; the run in flight predates the fix, so
**`end_goal` and `end_fell` in that log.json are not to be quoted.**
`score_policies.py` and `role_metrics.py` read `last_end` and are authoritative.

### What actually happened

Measured with `role_metrics.py`, 192-241 games, mean actions:

| epoch | goal | wipeout | timeout | down events per game |
|---|---|---|---|---|
| 60 | 0.8% | **66.4%** | 32.8% | 2.5 |
| 80 | 1.6% | 7.8% | 90.6% | 0.95 |

The apparent "collapse from 59.5% to 7.8%" was the **wipeout rate** falling.
What the agents learned between epochs 60 and 80 is to STAY UP: down events per
game drop 2.5 → 0.95, wipeouts 66% → 8%, and episodes lengthen back toward the
timeout because nothing ends them early any more.

Real scoring is 0.8% → 1.6%. **The 1v1 control is at 1.5% at epoch 80** and did
not take off until ~90-120, so 2v2 is neither ahead nor behind — it is at the
same place on a slower clock.

### The back agent is still a spectator, and this is now a training result

**Back pair: 0.0% of all crossings, at both epochs.** The design doc's step-1
falsifier measured 0.000-0.005 for a transplanted 1v1 pair and asked whether
training would move it. Through 80 epochs of native 2v2 training, it has not.

It is not that the back agent does nothing — its mean root x moves off the goal
line (agent 3: +4.0 at spawn, +1.47 by epoch 80) so it does advance. It simply
never arrives first.

**This re-opens decision 1 of section 9.** The options were: accept interference
as the back agent's only value (recommended, and what this run tests), a y-gated
goal, or a shorter `back_x`. The evidence so far says the recommended option
leaves the second player decorative on the scoring metric. It does not yet say
interference is absent — see the CPD below — and 200 epochs is early.

### What the policy actually attends to (corrected)

Counterfactual policy divergence, each channel group perturbed by **one
standard deviation of that channel**, taken from the policy's own
`control_norm` statistics:

| channel group | epoch 60 | epoch 80 |
|---|---|---|
| own state | 0.0373 | 0.0407 |
| far opponent | 0.0057 | 0.0058 |
| role one-hot | 0.0051 | 0.0053 |
| teammate | 0.0052 | **0.0035** |
| near opponent | 0.0017 | 0.0025 |

**This table replaces a wrong one.** The first version perturbed every group by
the same ABSOLUTE jitter. The policy whitens its input by the running std, and
those stds are not equal — 0.5 for the role one-hot against ~0.85-1.75 for the
position channels — so a fixed jitter handed the role bit roughly twice the
normalised perturbation of everything it was being compared against. It scored
0.0111 and ranked second; scaled properly it scores 0.0053 and ranks third.
Two claims made on the bad table do not survive:

* ~~"the role one-hot is the most-used input after the agent's own state"~~ —
  it is comparable to the far opponent, not above it.
* ~~"the teammate channel is the least used of the three"~~ — the NEAR opponent
  is, at both epochs.

What does survive, and it is the thing worth watching:

* **Own state dominates by ~7x.** At 80 epochs this is still mostly a
  locomotion policy that happens to have other agents in its observation.
* **Teammate influence is FALLING** — 0.0052 to 0.0035 over those 20 epochs,
  the only group that moved down, while the near opponent rose 0.0017 to
  0.0025. The policy is being handed teammate position and is progressively
  attending to it less. On this evidence there is no coordination developing.
* The role one-hot is used at all, which is what section 5 needed: one policy
  per team is producing two differentiated agents rather than the same agent
  twice.

Two epochs of one run. The direction is a hint, not a trend.

### Not measured yet

The run is unfinished. Everything above is epoch 60-80 of 200. Topple
attribution stays uninformative at these counts (2-3 credited events), and no
render of a trained team has been looked at yet.

### 11b. What it looks like: a four-ant scrum

`render_team.py` on `policies_ep0080.pt`, 1,000 frames, looked at rather than
counted. The clip is unambiguous and the numbers alone did not say it:

* **step 40** — four ants near their spawns, spread across the pitch;
* **step 200** — all four converged into one tight interleaved cluster around
  x ≈ −3, orange/dark/orange/dark;
* **step 400** — still clustered, tangled, one agent visibly toppled;
* **step 700** — still clustered, now sitting on the left goal line.

So the behaviour behind "84.8% timeout, 1.5% goal" is not four ants milling
about failing to find the goal. It is **all four converging and shoving**, and
the episode ending on the clock because neither side can get through.

**That is interference, and it is the behaviour this design was hoping for** —
the user's "coordinating to flip over the opponents and render them useless".
Team A wipes team B out in 13.6% of episodes, which is the shoving occasionally
working. What is missing is the second half of the sentence: *and then passing
them by*. Nobody passes.

It also explains the epoch 60→80 numbers. Wipeouts 66% → 8% is not "less
contact", it is the scrum stabilising: the ants got good enough at staying
upright that shoving no longer knocks anyone over, so the same pile-up now ends
on the clock instead of in a wipeout.

Two cautions. This is epoch 80 of 200 and the 1v1 control did not begin scoring
until ~90-120, so the stalemate may simply be the phase before someone learns to
break out. And a scrum is also what a *degenerate* equilibrium looks like — if
it persists to 200, the honest reading is that `team_first` + `goal_credit=team`
pays interference well enough that scoring is not worth the risk, which is
decision 3 of section 9 coming back.

### 11c. Section 8 step 1 re-measured on the regenerated policies

The step-1 falsifier was measured with `m2e_fixed/policies.pt` as it existed on
the old pod. That file was destroyed; the re-run regenerated it, and it is a
STRONGER pair (M2E §12: 96.9% against the old 83.9%). So the measurement was
repeated rather than assumed to carry over.

`probe_2v2 downed`, 256 games per rule, transplanted trained pair:

| | as documented | re-measured |
|---|---|---|
| goal rate | 98.0% | **100.0%** (every rule) |
| episode length | 178.7 | 185.0 |
| front pair reach rate | 0.38-0.60 | 0.543 / 0.445 |
| **back pair reach rate** | **0.000-0.005** | **0.0 / 0.027** |

The conclusion is unchanged and now rests on a policy that exists: the task
survives at four bodies and 8 m, and the back agent is decorative under a
first-crossing rule. The untrained arm still reads 100% timeout and zero travel,
so the probe still has a floor.

### 11d. Epochs 100 and 120: a stable stalemate, and two claims withdrawn

| epoch | goal | wipeout | timeout |
|---|---|---|---|
| 60 | 0.8% | 66.4% | 32.8% |
| 80 | 1.6% | 7.8% | 90.6% |
| 100 | 1.6% | 1.6% | 96.9% |
| 120 | 1.6% | 1.6% | 96.9% |

**The stalemate is stable, not transitional.** Goal rate has sat at 1.6% for
three consecutive measurements 20 epochs apart while wipeouts decayed to
nothing. The hope in §11b — that this was the phase before someone breaks out,
as the 1v1 control did at ~90-120 — is not supported at 120. The 1v1 control
was at 0.114 by epoch 90 and climbing; this is flat.

The agents are also getting spatially closer, not further apart: mean root x
spans 3.9 m at epoch 80 (−2.39 to +1.47) and 2.4 m at 120 (−1.37 to +1.01).
The scrum is tightening.

**Withdrawn: "teammate influence is falling."** Across four checkpoints the
teammate CPD reads 0.0052, 0.0035, 0.0036, 0.0038 — it dropped once and then
flattened, and the far opponent went the other way (0.0058 → 0.0036) over the
same span. Two points looked like a trend; four do not. There is no direction
here, only noise, and the honest statement is the weaker one: **teammate,
near-opponent and far-opponent influence are all of the same small size, and
all are ~10x below the agent's own state.**

**Not a finding: "the back agent now takes 50% of crossings."** At epochs 100
and 120 the crossing split reads 50/50, against 0/100 at 60 and 80. That is
computed over **two crossings in 128 games**. One front and one back agent
crossing produces exactly 50/50. The split is uninformative at this goal rate
and should not be quoted until scoring is common enough to divide.

---

## 12. 2f step 6 COMPLETE — 2v2 is solved, and roles emerged

`runs/competevo_port/t2v2_cold`, 200 epochs, cold start, ~2 h. Scored with
`score_policies.py`, 321 games, mean actions.

| | goal | wipeout | fell | timeout | length | win, per team |
|---|---|---|---|---|---|---|
| **2v2 @200** | **96.6%** | 0.3% | 0.0% | 3.1% | 278.6 | [0.430, 0.539] |
| 1v1 control @200 | 96.9% | — | 2.4% | 0.0% | 179.5 | 0.969 summed |

**The 2v2 task is solved to the same degree as 1v1** — 96.6% against 96.9% —
on a task with twice the bodies, an 8 m pitch, and an opponent pair actively in
the way. Episodes take 55% longer (278.6 against 179.5), which is what the
obstruction costs.

### Section 11d was wrong: the stalemate WAS transitional

§11d recorded 1.6% goals flat across epochs 100 and 120 and concluded "stable,
not transitional", explicitly rejecting the hope that a breakout was coming. It
came, between epoch 120 and 174. **That conclusion was drawn from three
measurements over 60 epochs of a 200-epoch run and it was wrong.** The lesson is
the ordinary one and worth writing down anyway: a flat stretch in a self-play
curve is not evidence of convergence, and the 1v1 control's own curve was flat
from epoch 1 to 74.

### Division of labour emerged — this is the result 2f was for

Crossings, `role_metrics.py`, 256 worlds:

| | transplanted 1v1 pair | epoch 60-80 | **epoch 200** |
|---|---|---|---|
| front pair | ~100% | 100% | **64.2%** |
| **back pair** | **0.0-0.5%** | **0.0%** | **35.8%** |

The back agent was decorative for the first 80 epochs and is now responsible for
**more than a third of all goals**. Section 8's step-1 falsifier fired against a
transplanted pair and asked whether training would move it. It does. **Decision
1 of section 9 is settled in favour of the recommended option:** the sparse team
reward found a use for the second player without needing a y-gated goal or a
shorter `back_x`, both of which can now stay in reserve.

### The teammate channel is now the most-attended other-agent input

CPD, std-scaled jitter, at epoch 200 against epoch 120:

| channel group | epoch 120 | **epoch 200** |
|---|---|---|
| own state | 0.0407 | 0.0476 |
| **teammate** | 0.0038 | **0.0075** |
| role one-hot | 0.0042 | 0.0057 |
| near opponent | 0.0023 | 0.0045 |
| far opponent | 0.0036 | 0.0037 |

Teammate influence **doubled** and is now the largest of the three other-agent
groups, having been the joint-smallest at epoch 120. §11b's "the policy was
handed teammate position and is declining it" described epoch 80 and does not
survive to 200 — as §11d already half-conceded when it withdrew the "falling"
claim. Coordination did develop; it developed late, alongside the breakout.

### What it looks like, and the honest caveat

The clip is NOT four ants running past each other. They still converge into a
pack by step ~120 — but the pack now **migrates**, travelling to one goal line
with someone crossing at the end. It is a shoving match with a winner rather
than a stalemate, and in the recorded episodes the pack drags toward whichever
side is pushing harder (team B here, 0.589 against 0.377).

So "one defending and one attacking" is not what emerged. What emerged is a
maul that both teams contest and one wins, with the back agent contributing a
third of the finishes. Falls are essentially gone (0.23 down events per game
against 2.5 at epoch 60).

### Still not established

One run, one seed, one config. The `goal_credit="scorer"` ablation is running
to test whether paying the non-scoring teammate is what produces the pack, and
that is the single most informative follow-up.
