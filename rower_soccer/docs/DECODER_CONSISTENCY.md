# Which decoder is *the* decoder

*Short note, written 2026-08-08 while queueing kick/shoot. Records a real
inconsistency in the current ant checkpoints, why it does not break the game,
and why it must be fixed before the BC sprint.*

## The architectural claim

The ant sprint replaces DeepMind's distillation stage with: train `follow` with
a z-bottleneck, **freeze that decoder**, and train every other skill as a
z-emitting expert on top of it. All skills then share one low-level controller
and one z-space by construction — no distillation needed. That shared z-space is
what next sprint's BC trains *into*: human demos are recorded as latent motor
intentions, which only means anything if every skill decodes z the same way.

## The inconsistency

`follow_ant_v1` produced two checkpoints with **different decoders** (max
per-weight difference 0.35, since `best.pt` is the 55.8M-step weights and
`final.pt` the 147M-step ones):

| artifact | decoder it carries / froze |
|---|---|
| `follow` skill in the game registry | `follow_ant_v1/final.pt` |
| `dribble_ant_v1` (`--init-from ... best.pt --freeze-decoder`) | `follow_ant_v1/best.pt` |
| `kick_ant_v1`, `shoot_ant_v1` (queued 2026-08-08) | `follow_ant_v1/best.pt` |

So the game currently runs **two** low-level controllers depending on which
skill is active. `final.pt` is pinned for `follow` because `best.pt` has a
symmetric-state fixed point that makes the ant sit still when the target is dead
ahead (see the registry comment); dribble was trained before that was known.

kick/shoot were deliberately pointed at `best.pt` **to match dribble** — better
three skills sharing one decoder than each on its own.

## Why the game still works

Every exported checkpoint is self-contained: it carries its own decoder weights
alongside its expert head. `SkillController` loads a whole policy per skill, so
each skill is internally consistent and produces valid actions. Nothing is
silently mismatched at runtime — the mismatch is *between* skills, not inside
any one of them.

## Why it must be fixed before BC

BC in z-space assumes one decoder. With two, a recorded z means different things
depending on which skill was active when it was recorded, and a BC policy
trained on that mixture is decoding a blend of two controllers. The failure
would be quiet: training converges, the video looks vaguely right, and the
z-space claim is simply false.

## The fix (folded into the planned follow retrain, task #13)

The follow retrain is already required for the 71-obs contract drift. It should
also settle this:

1. Retrain follow with entropy annealing + near-zero-bearing targets in the
   curriculum (kills the sit-down fixed point, so `best.pt` and `final.pt` stop
   disagreeing about which is usable).
2. Nominate exactly ONE artifact as the canonical decoder — write its path into
   the registry and this doc — and publish it under a stable name
   (`runs_v2/_decoder_ant_v2.pt`), the way `train_track_warp --publish-decoder`
   does for the rower.
3. Retrain dribble/kick/shoot from that one decoder.
4. Add a check to `SkillController`: on construction, compare the decoder
   tensors of every loaded skill and refuse (or warn loudly) if they differ.
   This is the only step that makes the guarantee structural rather than a
   convention someone has to remember.

Until then, treat cross-skill z as untrustworthy. Per-skill behaviour, the
playable game, and the demo recording are all unaffected.
