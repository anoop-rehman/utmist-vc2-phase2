# Direction 1 — Ant skills → human demos → BC → RL: the 2v2 video

*Written 2026-08-10. The continuation of the post-pivot sprint (STATUS_2026-08-08.md);
supersedes nothing, but pins the road ahead so we don't get lost.*

## Motivation

The club promised a 2v2 creature-soccer video three years ago. The NPMP/SMP style-
transfer track died (NPMP_SMP_POSTMORTEM.md); what survived the pivot is the
DeepMind humanoid-football recipe (Liu et al. 2022, Science Robotics): train drill
skills, collect human demonstrations, behavior-clone them into a prior, then
RL-fine-tune 2v2 self-play anchored to that prior. Motion style no longer matters —
task competence does. The ant is the validation body: every architecture dimension
(65 proprio / 8 act / z=16) matches the rower byte-for-byte, so everything ports by
swapping the XML.

## Goal

A watchable 2v2 creature-soccer match — first with ants, then with the evolved
creatures (rower + worm) — driven by policies that learned from human play.
Deliverable: the video, plus the pipeline that made it.

## Approach

DeepMind's pipeline, one shared frozen decoder per body:

1. **Drills** (DONE, training deeper now): follow / dribble / kick / shoot, all on
   `_decoder_ant_final.pt`. Checkpoints + GCS links in the skills registry.
2. **Human demos**: the 2v2 game server records per tick: exact expert obs vector,
   z latent, action, skill label, target — already verified bit-exact-replayable
   (`replay.py --mode controller`).
3. **BC**: train a policy `game obs → (skill, target)` (high level) and/or
   `obs → z` (low level, reusing the frozen decoder) from the demo corpus.
4. **RL fine-tune**: 2v2 self-play in the warp soccer env, KL-anchored to the BC
   prior so play stays human-flavored while win-rate climbs.
5. **Creature swap**: retrain drills on rower/worm (same trainers, same registry),
   re-collect a small demo set, repeat 3-4. The 0.52 ball/torso proportion rule
   (scale the BALL and PITCH to the creature, never the creature to the ball)
   carries over.

## Implementation plan (parallelizable units)

| unit | depends on | status |
|---|---|---|
| 1a. Drill training to saturation (48 h ceiling, we interrupt) | — | RUNNING |
| 1b. Demo-collection sessions (4 humans, play_online.sh) | 1a good-enough | ready |
| 1c. BC dataset builder (demos → training tensors) | demo format (fixed) | can start NOW |
| 1d. BC trainer + eval harness | 1c | after 1c |
| 1e. Warp 2v2 self-play env (4 creatures + ball on one pitch) | — | can start NOW |
| 1f. RL fine-tune w/ KL anchor | 1d + 1e | later |
| 1g. Creature swap (rower drills) | 1a validates recipe | later |

1c and 1e are independent of everything running and of each other — good agent-
sized chunks on separate branches/worktrees.

## Eval method

- **Drills**: per-skill warp fitness + the CPU-soccer transfer evals
  (`demo_follow_soccer` gate, `eval_dribble_soccer`) — both already scripted.
  Rule learned five times over: **watch the eval videos**; metrics lie, videos don't.
- **BC**: held-out demo action agreement + controller-replay determinism check +
  "does a BC-driven 2v2 look like play" (human eval).
- **RL**: win rate vs scripted baseline and vs BC-only; KL to prior as the
  style-drift gauge.
- **End**: the video itself, judged by humans.

## Notes / risks

- Shoot trained on the +x goal only; the −x mirror transform is still TODO in the
  game (documented in `fields._post_ego`).
- Shoot's fitness selects on accuracy only while its reward pays for power — fine
  so far, revisit if best.pt looks timid at saturation.
- Demo volume: DeepMind used ~hours of human play; we'll have far less. BC may need
  heavy augmentation (mirroring exploits the pitch symmetry) — design 1c with that in mind.
- GPU is shared with nothing (drills own it); CPU yields to Direction 2 on conflict.
