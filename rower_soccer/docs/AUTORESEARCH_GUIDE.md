# Running an agent as an overnight researcher

*Written after a ~14-hour unattended session on a shared GPU pod: three
research tracks, six concurrent trainers, five delegated subagents. It found
four real bugs, produced two validated results, and made seven mistakes that it
caught and corrected. This is what actually mattered, in the order it mattered.*

The setup is three mechanisms — a self-pacing loop, delegated subagents, and
durable background processes — plus a set of habits that turn out to matter
more than any of them.

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
Bash tool caps at 600 s, use background execution; `pgrep` matches its own
shell, verify processes with `ps -eo args | grep ...`; export the MPS variables
before any GPU work; use explicit paths with `git add`, never `-A`.* Every one
of those was learned by losing time to it.

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

Five agents ran here. Every one of them found something the brief did not
anticipate, and **two of them corrected the coordinator's own errors**. That is
the strongest argument for delegation: a fresh context re-derives your numbers
instead of inheriting them.

### What belongs in a subagent brief

1. **What to read first**, by path. Not "familiarize yourself with the
   codebase".
2. **The measurement that motivates the task**, with numbers. An agent that
   knows *why* will fix the real problem; one that knows only *what* will
   satisfy the letter of the brief.
3. **The gate**, stated as the headline deliverable. "Numerical equivalence is
   the gate, not the speedup" changes what gets built.
4. **Non-negotiables, spelled out.** Do not kill processes you did not start.
   Do not push. Do not scrape credentials. Which Python, which env vars.
5. **Explicit permission to report failure.** "If a gate errors, report FAIL"
   and "a measured 1.15x reported honestly is worth more than a claimed 1.6x".
   Say it, and agents will tell you their own tests failed.
6. **A note that the shared machine is busy** and smokes must be small.

### Verify their work yourself

Non-negotiable. Run their gate. Look at their render. In this session an agent
documented a gate as PASS with a full numbers table while the gate was actually
raising `TypeError` — a refactor had changed a constructor signature and the
call site was never updated. The numbers were real; nothing was checking them.

Reading their **committed artifact** before challenging their numbers is also
non-negotiable, and it is the rule we broke worst. The coordinator raised three
objections to one agent's run and was wrong on all three, then instructed it to
bin a valid result. All three were answered in a file the agent had already
committed.

---

## 4. Durable processes

Long jobs must outlive the session, the SSH connection, and the agent's own
context:

```bash
nohup <cmd> >> run.log 2>&1 &      # detached, survives disconnect
```

Give every run a distinct name and **never reuse one**. Reusing a name splices
two configurations into one logged curve and produces a "sudden jump" that
looks like a finding. One name, one run. When you change the config, launch a
new name with `--init-from` the old checkpoint.

Check for **duplicate processes** after any relaunch. We ran two trainers
against the same run directory and wandb run for hours; the giveaway was two
evaluations at nearby steps with wildly different scores.

Persist chat state somewhere durable if the pod can be destroyed.

---

## 5. The habits that actually caught things

Every mechanism above is scaffolding. These are what produced the results.

### Verify a probe measures the quantity you think it does

We measured "strike speed" and got 1.68 m/s, which would have been a dramatic
reversal. The probe sampled every timestep where the ball was moving, so it
averaged the deceleration tail and every incidental nudge. The right quantity
was per-segment *peak* speed. Same code, same data, different question.

### Wall-clock cross-check every derived rate

We reported a GPU port as 20x slower than it was, because a log's `sec` field
was cumulative elapsed time and got read as per-iteration. The median of a
rising cumulative series is about half its total — the "median iteration" was
literally half the run length. One sanity check (27 iterations x 391 s = 2.9
hours, for a run that took 12.6 minutes) would have caught it instantly.

### Aggregate before comparing rates

We declared a win-rate difference unmeasurable from a single epoch of ten
games. Aggregated over 2,017 games the same comparison was ~10 sigma. Never
compare single samples; and check the error bars on the *reference*, not just
your own.

### Compare arms at matched steps, never matched wall-clock

Two arms at 35M and 65M steps produced a 0.023 "improvement" that vanished
entirely when compared at equal step counts.

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
briefs before anyone checked.

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

### Distinguish a leading indicator from a promise

An upstream metric moving is a hypothesis about a correlation, not a forecast.
We invoked "aim leads fitness" correctly once and as a premature promise a few
hours later, for the same pair of metrics.

---

## 6. Delivering to the human

Assume they read one message and act on it.

- **Lead with what needs them**, in priority order, as commands where possible.
- **Separate established from suspected.** "Not established:" is a real
  section, and the most useful one.
- **Surface decisions instead of making them.** Options with honest pros and
  cons; a recommendation; no unilateral pick on anything architectural.
- **Keep a single running summary document** rather than scattering findings
  across commit messages. Ours consolidated ~13 doc sections, four branches and
  six decisions into one page.
- **Report your own errors in a table.** The pattern is more useful than any
  individual slip — in our case, every one was a *derived* number that looked
  plausible and was never checked against a second measurement.

---

## 7. Failure modes to watch for

| symptom | usual cause |
|---|---|
| A metric looks great and the behaviour looks wrong | you are measuring the wrong quantity, or the renderer and the physics disagree |
| A gate documents PASS with numbers | nothing is running the gate; run it yourself |
| Throughput is far below a peer job's | contention, or an accidental cost inside an eval/render path |
| A curve shows a sudden jump | two runs sharing a name, or two processes sharing a directory |
| An agent's result contradicts yours | read its committed artifact before objecting; it is often right |
| Everything is "blocked on the user" | correct time to stop the loop, not to invent work |

---

## 8. Minimal recipe

```
1. Start long jobs with nohup, distinct names, one name per config.
2. Arm ONE persistent crash monitor across all of them, with a wide filter.
3. /loop with a prompt carrying state, conclusions, mistakes, and blockers.
4. Delegate scoped work to subagents; brief them with measurements and gates;
   run their tests yourself before believing them.
5. Each iteration: check runs, look at renders, verify one thing you assumed,
   commit, and rewrite the loop prompt.
6. Stop when everything left needs a human.
```

The mechanisms are easy. The discipline — verifying instruments, aggregating
before comparing, and writing down predictions you might have to admit were
wrong — is what makes the output trustworthy.
