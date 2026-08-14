# Running an agent as an overnight researcher

*Written after a ~14-hour unattended session on a shared GPU pod, and revised
across the sessions since: three research tracks, six concurrent trainers, a
dozen delegated subagents. It found real bugs and produced validated results,
and it made a long list of mistakes, most of which it caught. This is what
actually mattered, in the order it mattered.*

The setup is three mechanisms — a self-pacing loop, delegated subagents, and
durable background processes — plus a set of habits that turn out to matter
more than any of them. If you only read one section, read §5. The mechanisms
are the easy part.

---

## 1. The loop

`/loop <prompt>` with no interval puts the agent in **dynamic mode**: it does
the work, decides when the next iteration is worth running, and schedules its
own wake-up. Alternatively `/loop 20m <prompt>` fires on a fixed cron.

Prefer dynamic. Fixed intervals either poll too often (burning tokens on
"nothing changed") or too rarely (idling through a finished job).

**The loop prompt is the agent's entire memory between iterations.** It gets
re-sent verbatim each time, so it must carry forward everything the next
iteration needs. Write it as a handoff to a competent colleague who was not in
the room:

- current state, with numbers
- what is in flight and how to read the result when it lands
- **conclusions already reached, so they are not re-derived**
- **mistakes already made, so they are not repeated**
- what is blocked on the human, explicitly
- environment gotchas that cost time to rediscover

That last category is worth being unglamorous about. Ours accumulated: *the
Bash tool caps at 600 s, use background execution; `pgrep` and `pkill` match
their own shell, verify with `ps -eo args | grep ...`; export the MPS variables
before any GPU work; source the env file in every launcher; use explicit paths
with `git add`, never `-A`.* Every one of those was learned by losing time to
it. §4 has the ones that cost the most.

Rewrite the prompt each iteration. It is a living document, not a fixed string.

**Pacing.** Match the delay to what you are waiting for, not to a habit. A
4-hour run gets a 40-50 minute heartbeat. Something due in 20 minutes gets 20.
Idle ticks are pure overhead — and so is the token cost of re-reading a long
loop prompt for a check that could not possibly have changed.

**Stopping is a normal ending**, not a failure. When everything left is blocked
on a human decision, stop and hand off. Continuing to generate findings nobody
has absorbed has negative value: it grows the backlog the human must read
before they can act.

---

## 2. Monitors versus wake-ups

A **Monitor** watches a stream and wakes the agent on an event. A
**ScheduleWakeup** fires on a timer. Use both: the monitor is the real signal,
the wake-up is a fallback in case nothing fires.

The trap is silence. A monitor that only greps for the success marker stays
quiet through a crash, a hang, and a clean exit alike — and silence is
indistinguishable from "still running". Widen the filter until a crash right
now would produce a line:

```bash
tail -F run.log | grep -E --line-buffered \
  "step=|Traceback|Error|FAILED|assert|Killed|OOM|CUDA error|diverged=[1-9]"
```

One persistent crash monitor across every long-running job is worth more than
any amount of polling. Ours ran all night across seven processes.

---

## 3. Subagents

Delegate when the work is **well-scoped, mechanical or parallel, and would
otherwise flood the coordinator's context**. Keep in the coordinator anything
requiring judgment across the whole picture.

Every agent we ran found something its brief did not anticipate, and several
**corrected the coordinator's own errors**. That is the strongest argument for
delegation: a fresh context re-derives your numbers instead of inheriting them.

### Ask before spawning

An agent-hour spent on the wrong branch of a fork is worth less than the two
minutes it takes to ask. Before launching anything that will run for hours,
identify the **genuine forks** — the choices where the work differs materially
depending on the answer and you cannot settle it from the code — and put them
to the human as a short numbered list with a recommendation each.

Our best-run delegation was preceded by six such questions (spawn geometry,
what happens to a downed player, the win condition with four agents, credit
assignment, genome structure, opponent sampling). Answering them took the human
a few minutes and turned an open-ended "design 2v2" into a brief with six
named deliverables. The failed ones were launched on assumptions.

Distinguish this from **surfacing decisions afterwards** (§6). Ask beforehand
about things that change what gets built; report afterwards on things that
change what gets concluded.

### What belongs in a subagent brief

1. **What to read first**, by path. Not "familiarize yourself with the
   codebase".
2. **The measurement that motivates the task**, with numbers. An agent that
   knows *why* will fix the real problem; one that knows only *what* will
   satisfy the letter of the brief.
3. **The deliverables, enumerated.** Not a topic — a list of things that will
   exist when it is done, each one checkable. Include "a section titled *what
   was NOT tested*" as one of them; it is the section that makes the rest
   trustworthy, and it does not get written unless it is asked for.
4. **A named blocking issue to resolve first**, if there is one. "The trainer
   asserts `n_agents == 2`; settle what that means for the design before
   costing anything else." An agent that discovers the blocker on its own
   discovers it after building on top of it.
5. **An explicit falsifier**: *the cheapest thing that would prove this whole
   approach wrong, first.* Say it in those words. Ours fired partially — the
   task survived at four bodies, but the back players turned out to be
   decorative under a first-crossing rule — and that half-failure, arriving in
   hour one instead of week two, is the single highest-value output the
   delegation produced.
6. **Scope limits on compute**, in units the agent can check: worlds, minutes,
   "no run longer than T". A shared box makes this a correctness requirement,
   not politeness.
7. **A do-not-touch list of running processes**, by name. Not "be careful" —
   the actual names, plus "do not `pkill` on a pattern that could match them".
8. **House rules**: which Python, which env vars, which branch and worktree,
   commit but do not push, explicit paths with `git add`.
9. **Explicit permission to report failure.** "If a gate errors, report FAIL"
   and "a measured 1.15x reported honestly is worth more than a claimed 1.6x".
   Say it, and agents will tell you their own tests failed.

### A reusable template

```
## Task: <one line — the deliverable, not the topic>

**Read first**, in this order:
- <path> — <what you will get from it>
- <path> — <...>

**Why this exists.** <the measurement that motivates the task, with numbers
and the command that produced them>

**Resolve first.** <the named blocking issue>. Everything downstream depends
on the answer. Write it down before starting deliverable 2.

**Falsifier — do this before anything else.** The cheapest thing that would
prove this whole approach wrong is <X>. Run it first and report the result,
especially if it fires.

**Deliverables.**
1. <checkable artifact>
2. <checkable artifact>
...
N. A section titled "What was NOT tested", stated as gaps rather than
   answered by guesses.

**Compute budget.** <N worlds / N GPU-minutes / nothing longer than T>. The
box is shared with other jobs. If you need more, ask; do not take it.

**Do not touch.** These are live and are not yours: <process names>. Do not
kill, restart, or write into their run directories, and do not `pkill` on a
pattern that could match them.

**House rules.** <interpreter>, <env vars>. Branch `<b>` in worktree `<p>`;
commit, do not push; explicit paths with `git add`. Every number in the
write-up carries the command that produced it or says it is not measured.
Every negative control must be shown failing on demand. Reporting FAIL is a
successful outcome.
```

### Verify their work yourself

Non-negotiable. Run their gate. Look at their render. In one session an agent
documented a gate as PASS with a full numbers table while the gate was actually
raising `TypeError` — a refactor had changed a constructor signature and the
call site was never updated. The numbers were real; nothing was checking them.

Reading their **committed artifact** before challenging their numbers is also
non-negotiable, and it is the rule we broke worst. The coordinator raised three
objections to one agent's run and was wrong on all three, then instructed it to
bin a valid result. All three were answered in a file the agent had already
committed.

---

## 4. Durable processes and long-run hygiene

Long jobs must outlive the session, the SSH connection, and the agent's own
context:

```bash
setsid nohup <cmd> >> run.log 2>&1 < /dev/null &   # detached, survives disconnect
disown
```

**Give every run a distinct name and never reuse one.** Reusing a name splices
two configurations into one logged curve and produces a "sudden jump" that
looks like a finding. One name, one run. When you change the config, launch a
new name with `--init-from` the old checkpoint.

**Check for duplicate processes after any relaunch.** We ran two trainers
against the same run directory and wandb run for hours; the giveaway was two
evaluations at nearby steps with wildly different scores.

**Launchers must source the environment file.** A multi-arm launch script that
worked interactively killed every arm at startup on "No API key configured",
because the key lives in a gitignored `.env` that the interactive shell had and
the script did not. `set -a; . ./.env; set +a` exports it without echoing it.
Whatever your equivalent is — credentials, MPS variables, `MUJOCO_GL` — put it
in the launcher, not in your shell history.

**`pkill -f <pattern>` matches its own shell.** So does `pgrep`. A cleanup
command killed its own command chain and returned exit code 144, which reads
like the job crashed. Filter the pattern, or select PIDs from
`ps -eo pid,args` and kill those.

**A background command chained after a foreground wait dies with the wait.**
`sleep 600 && nohup train.sh &` looks like "wait ten minutes, then launch
detached", but the whole chain is one job: when the foreground tool call times
out, the launch never happens and nothing says so. Launch first and let the job
idle, or schedule the launch from something that outlives the tool call.

**Write checkpoints periodically, not just at the end.** A run whose only
weight artifact is the final one cannot be scored at a chosen epoch, so it
cannot be compared to anything except at whatever point it happened to stop.
We lost a matched-epoch comparison to this and had to fall back to a different
one. `--save-every`/`--save-policies-every` costs disk and buys the ability to
answer questions you have not thought of yet.

**Write per-iteration to a log file, not to stdout at the end.** Detached probe
processes disappeared mid-run in one session with no traceback and nothing in
the port at fault. A run that appends its state every iteration survives that;
a run that prints a summary at the end does not.

Persist chat state somewhere durable if the pod can be destroyed.

---

## 5. The habits that actually caught things

Every mechanism above is scaffolding. These are what produced the results.

### A control that cannot fail is worse than no control

It is worse because it reports PASS and buys confidence it did not earn. **The
rule: after a control passes, break the thing on purpose and confirm the
control fails.** If you cannot make it fail, it is not measuring anything.

Three ways ours were vacuous:

* **The premise was false.** A control asserted "transposing the adjacency must
  change the answer". The graph emits both directions of every edge, so the
  adjacency is symmetric and transposing it is a no-op. The control could never
  have fired. (The finding was real and useful — the port had been warned about
  edge direction and it was a non-hazard — but it arrived by noticing the
  control never fired, not by the control firing.)
* **The driver was too weak to exercise the thing.** A pose-reset flag was
  checked with a zero-action policy, which never falls over, so both arms read
  uprightness 1.0 and the flag looked like a no-op. Driven by the trained
  policy: 0.842 without the flag, 1.000 with it.
* **The perturbation was absorbed downstream.** A control on a discrete head
  read the sampled *action*, and the argmax swallowed every perturbation that
  left the winning logit winning — 42/60. Read at the pre-argmax head, 60/60.
  Perturb where the quantity lives, and read it there.

Make the breakage a first-class flag — `--break bitmask`, `--break payslacker`
— so "the control fails on demand" is itself re-runnable and survives a
refactor. Our best gate shipped three, all demonstrated failing; one of them
moves a cross-agent influence metric from 8.5e-4 to 1.03, which is what a
working control looks like.

### Verify a probe measures the quantity you think it does

Before believing a number, ask what would be true if the probe were wrong, and
check that. **The tell is almost never an error message — it is an internal
contradiction between two numbers that cannot both hold.**

* A throughput probe returned a nonsense rate because `torch.cuda.synchronize()`
  does not wait on Warp's stream. It timed how fast work could be *submitted*.
* A travel measurement read the start position from the post-`reset()` pose,
  which both environments overwrite after the design stage — a pose the rollout
  never occupied. It reported 1.22 m of travel alongside a 33.6% goal rate
  across a 4 m pitch. Those cannot both be true, and that is the only reason it
  was caught.
* "Strike speed" came out at 1.68 m/s, which would have been a dramatic
  reversal. The probe averaged every timestep where the ball was moving, so it
  included the deceleration tail and every incidental nudge. The right quantity
  was per-segment *peak* speed. Same code, same data, different question.

### Wall-clock cross-check every derived rate

We reported a GPU port as 20x slower than it was, because a log's `sec` field
was cumulative elapsed time and got read as per-iteration. The median of a
rising cumulative series is about half its total — the "median iteration" was
literally half the run length. One sanity check (27 iterations x 391 s = 2.9
hours, for a run that took 12.6 minutes) would have caught it instantly.

### Aggregate before comparing rates, and check the reference's error bars too

We declared a win-rate difference unmeasurable from a single epoch of ten
games. Aggregated over 2,017 games the same comparison was ~10 sigma.

Then the sigma turned out to be wrong anyway, in the other direction: the
reference number it was computed against is a per-epoch logged fraction whose
denominator is between 3 and 8 — every value it can print is a sixth, seventh
or eighth. **The precision of a comparison is bounded by the sloppier side.**
Look up how the reference number was computed before quoting a significance.

### Quote the width, not the digits

Two runs of the same 384-game measurement gave 30.7% and 34.8% — binomial SEM
is about 2.4 points, so the answer is 32 +/- 3, not 34.8. Four significant
figures on a quantity measured once is a claim about repeatability that has not
been tested. Run it twice, or quote the interval. (It changed no conclusion
here — 83.9% against 5.5% is not a 3-point question — which is exactly why the
habit has to be automatic rather than applied when it seems to matter.)

### Know which estimator you are quoting

Numbers from different measurement protocols are not comparable even when they
have the same name, and the difference is routinely larger than the effect
being chased:

* the same policy scored **0.847** under deterministic evaluation and **0.605**
  from training-monitor lines;
* a probe's mean segment arrival was 0.283 against the trainer's reported
  fitness of ~0.40, because fitness includes the in-flight segment and has a
  previous-episode fallback;
* "when the reward curve arrives" and "when the fitness curve arrives" differ
  **by construction**, not by noise.

All of these are real numbers. None of them can be put in the same column. Say
which estimator produced each one, every time, and if you must compare across
them, re-measure rather than convert.

### Compare arms at matched steps, never matched wall-clock

Two arms at 35M and 65M steps produced a 0.023 "improvement" that vanished
entirely when compared at equal step counts. This is also why §4 insists on
periodic checkpoints: you cannot match steps after the fact if only the final
weights exist.

### Snapshot or property?

"Agent 0 never wins" was written down as reproduced behaviour, matching the
reference. It was true of the reference at the one epoch we had a checkpoint
for; a hundred epochs later the split was [0.399, 0.569], and the lead kept
swapping for the rest of the run. It was a transient of mid-training, recorded
as a fact about the system.

**Before drawing a conclusion from a comparison point, ask where it sits on the
curve.** If the quantity is moving steeply there, the comparison describes the
moment, not the system. The same check reframed a headline: the reference was
not 6.5x ahead of us, it was solving a task we were far from — epoch 107 had
caught it mid-climb toward 96.9%.

### Quote means, not the last line

A drill was reported at "0.77-0.79" for hours; its actual mean was 0.605. The
monitor's sampling cadence aliased against the episode length, so single lines
landed on the peaks of an oscillation.

### Ask whether a discrepancy predates training

An offset blamed on two config mismatches turned out to be present at epoch 0,
before any gradient step. It was a physics difference. If it exists before
training, training cannot be the cause.

### Check which config a reference number came from

A constant quoted from a reference implementation's notes belonged to a
*different* config file than the one being run, and propagated into three
briefs before anyone checked. The same applies to numbers from your own docs:
a "paper target" we compared a 1000-epoch run against turned out to be a figure
we had written down ourselves and never verified against the paper. Anything
you are gating on should be traceable to a source you have actually read.

### Watch the video

Metrics were healthy while the ball was rendered 2.3x too large, while the
pitch was unscaled, and while the game used the wrong ball. Every one was
caught by a human looking at a clip. Build rendering into the gate and *look at
it* — "numerically fine, visually wrong" is a real and common failure class.

### Record predictions before the data exists

Write the falsifiable version down: *"if X is the cause, arm A should exceed
0.0974 at ~98M steps."* Then resolve it out loud, including when it fails.
Ours failed. A prediction that quietly stops being mentioned is worse than none
at all, because it trains you to trust the next one.

Record the ones you had no stake in, too. Going into one probe the hypothesis
was that inherited *geometry* explained the variance in outcomes; the
correlations came out at ~0.0 and the actual predictor was inherited *posture*,
at +0.674. That is worth a line in the doc precisely because the wrong
hypothesis had been specific enough to believe.

### Distinguish a leading indicator from a promise

An upstream metric moving is a hypothesis about a correlation, not a forecast.
We invoked "aim leads fitness" correctly once and as a premature promise a few
hours later, for the same pair of metrics.

---

## 6. Delivering to the human

Assume they read one message and act on it.

- **Lead with what needs them**, in priority order, as commands where possible.
- **Separate established from suspected.** "Not established:" is a real
  section, and the most useful one. Its stronger form is a standing "what was
  NOT tested" section listing gaps plainly rather than answering them with
  plausible guesses.
- **Surface decisions instead of making them.** Options with honest pros and
  cons; a recommendation; no unilateral pick on anything architectural. Where
  the decision changes what gets *built*, ask before starting (§3), not after.
- **Keep a single running summary document** rather than scattering findings
  across commit messages. Ours consolidated ~13 doc sections, four branches and
  six decisions into one page.
- **Report your own errors in a table**, with how each was caught. The pattern
  is more useful than any individual slip — in our case, nearly every one was a
  *derived* number that looked plausible and was never checked against a second
  measurement, or a control that had never been shown to fail.

---

## 7. Failure modes to watch for

| symptom | usual cause |
|---|---|
| A metric looks great and the behaviour looks wrong | you are measuring the wrong quantity, or the renderer and the physics disagree |
| A gate documents PASS with numbers | nothing is running the gate; run it yourself |
| A negative control has never once failed | it cannot fail; break the thing on purpose and check |
| Two of your numbers cannot both be true | one of the probes is measuring something else; that contradiction is the finding |
| A "reproduced" property matches at exactly one checkpoint | it is a transient of mid-training; check the curve either side |
| Two measurements of the same thing disagree by a lot | different estimators, not a discrepancy; find out which is which before explaining it |
| A rate is far below a peer job's | contention, an accidental cost in an eval/render path, or a timer that is not synchronising on the right stream |
| A curve shows a sudden jump | two runs sharing a name, or two processes sharing a directory |
| Every arm of a multi-arm launch died at startup | the launcher did not source the environment the interactive shell had |
| A cleanup command exits 143/144 | `pkill -f` matched its own shell |
| A comparison "at the same point" is unavailable | only final checkpoints were written; add periodic saves now |
| An agent's result contradicts yours | read its committed artifact before objecting; it is often right |
| Everything is "blocked on the user" | correct time to stop the loop, not to invent work |

---

## 8. Minimal recipe

```
1. Start long jobs detached, with distinct names, one name per config, an
   env-sourcing launcher, periodic checkpoints, and per-iteration logging.
2. Arm ONE persistent crash monitor across all of them, with a wide filter.
3. /loop with a prompt carrying state, conclusions, mistakes, and blockers.
4. Put the genuine forks to the human BEFORE delegating.
5. Brief subagents with paths, motivating numbers, enumerated deliverables, a
   blocking issue, an explicit falsifier, a compute budget, and a
   do-not-touch list; run their gates yourself before believing them.
6. Each iteration: check runs, look at renders, break one negative control on
   purpose, verify one thing you assumed, commit, rewrite the loop prompt.
7. Stop when everything left needs a human.
```

The mechanisms are easy. The discipline — proving your instruments can fail,
aggregating before comparing, saying which estimator produced each number, and
writing down predictions you might have to admit were wrong — is what makes the
output trustworthy.
