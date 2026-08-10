# Are our drills the drills the paper describes?

*2026-08-09. Prompted by watching eval videos and not believing the numbers.
Three drills checked against Liu et al. 2022 Tables S2/S3; two were wrong in
ways their own fitness curves could not show.*

## Table S2, verbatim

| Drill | Paper's definition |
|---|---|
| Follow | follow a moving target at fixed velocity, variable directions; agent observes current AND future target position |
| Dribble | "similar to the 'follow' drill but the agent must keep the ball close to the **moving** target" |
| Shoot | ball initialized randomly on the pitch; agent has a **budget of three ball contacts** to score |
| Kick-to-target | "a small window of time (**randomized between two and six seconds**) in which to manoeuvre the ball and kick it to a **distant** fixed target" |

Table S3 gives the rewards: kick-to-target and dribble are both
`exp(-c ||x_ball - x_target||)` (c = 1/2 for kick, 1/5 for dribble); follow is
`exp(-1/2 ||x_player - x_target||)`; shoot is velocity-ball-to-goal + scoring.

## What we found

### kick — WRONG, fixed (kick_ant_v3_paper)
Ours scored `max(ball_velocity . command)`, a projection. A projection cannot
separate a hard wild kick from a gentle accurate one: 7.6 m/s at 60 deg off and
3.8 m/s dead on both score 3.8. "Hit harder" is the easier gradient, so that is
what RL learned. `probe_kick.py` over 2243 strikes:

    median aim error 35 deg, mean 48    only 24% within 15 deg
    16% sent the ball BACKWARDS         37% of ball speed lost to aim

all while fitness rose monotonically for 446M steps. Scored on arrival that
policy manages 0.255 against a 0.135 floor for never touching the ball.

Fixed by adopting the paper's shape: `exp(-c*d)` to a distant (4-8 m)
randomized target, inside a randomized 2-6 s window. The window is what
separates kick from dribble, and it constrains nothing about the body -- which
matters, because players kick on the move. The three-contact budget belongs to
SHOOT, not kick; proposing it for kick was a misreading.

### dribble — WRONG, retrain queued (dribble_ant_v2_movingtgt)
The code has a `target_vel` field, which is not the same as the target moving.
Measured target path over a 15 s episode:

    dribble_ant_v1 (--target-speed 0.05 0.22)   median 1.77 m, slowest world 0.78 m
    follow's band  (--target-speed 0.07 0.6)    median 4.21 m

0.78 m is about one ball diameter on a 20 m arena. Much of training was
effectively static-target dribbling, so v1's 0.991 is a score on an easier task
than the paper's. v2 retrains at follow's band.

### shoot — CORRECT as trained, no retrain
`shoot_env` never implemented the three-contact budget, but the policy
satisfies it anyway. Measured over 1681 segments of `shoot_ant_v1/best.pt`:

    contacts per segment   median 1.0, mean 1.1, p90 2, max 4
    within the 3 budget    99.9%
    segments that score    81.4%
    contacts when scoring  median 1.0

It scores with a single touch -- genuinely shooting, not walking the ball in.
Adding the budget would be a non-binding constraint; worth doing as regression
insurance, but it changes nothing today.

## The transferable lesson

Both failures were invisible to the thing we were watching. Kick's fitness rose
for 446M steps while its aim did not improve; dribble's fitness reached 0.991 on
a task that was quietly easier than intended. Neither was found by reading code
-- the `target_vel` field and the strike-credit machinery both look correct in
isolation. Both were found by **watching the eval video, disbelieving the
number, and then measuring the specific thing the number could not see**.

So: for every drill, keep a probe that measures the task property directly
(`probe_kick.py` for aim, the contact counter for shoot, target-path length for
dribble), and run it before trusting a checkpoint. A fitness curve is evidence
that *something* is being optimized, never evidence that it is the right thing.
