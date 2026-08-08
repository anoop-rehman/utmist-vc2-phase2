# NPMP / SMP postmortem — motor-prior work, parked 2026-08-08

*Technical record of the motor-prior line: what was built, what was measured,
what failed and precisely why, and how to resume any of it. Parked because the
project dropped locomotion-style mimicry as a goal (see
[STATUS_2026-08-08.md](STATUS_2026-08-08.md) decision record) — not because the
work dead-ended.*

## 1. NPMP (Liu et al. 2022) on the rower — the full arc

### What was built
- **Reference gait**: `tools/rower_ref.py` — phase-averaged canonical cycle from
  the Unity Karl-Sims brain port, Froude-corrected to 0.890 Hz, with a torque
  gate (`check`) that refuses training if the body can't drive the reference
  (worst margin 1.98x, byte-stable across mujoco 3.1.3 → 3.11.0).
- **Tracking trainer**: `warp_port/train_track_warp.py` + `track_env.py`.
  obs=91 (proprio 65 + reference lookahead 26), act=8, z=16. Decoder sees
  proprio+z only — that layout, not the network, is what makes it reusable.
- **Transfer**: `train_follow_warp.py --init-from --freeze-decoder` (25 tensors,
  154,632 params frozen, verified bit-identical after 150M steps of task
  training).

### Decoders trained
| run | steps | joint err | style (track env) | where |
|---|---|---|---|---|
| `npmp_rower_track` | 448M | 67° flat — **gear_scale bug**, physically impossible reference | — | failure, pre-fix |
| `npmp_rower_v2` (July) | 94.9M | 11.7° best | never measured | **weights lost** with pod `simple_maroon_skunk` (`gcs_bucket: null`); only wandb metrics/videos survive |
| `npmp_rower_v3` (Aug 8) | 150M | **8.0°** | **0.931** | `gs://vc2-2026-checkpoints/npmp_rower_v3/` + published `runs_v2/_init_rower_npmp.pt` |

Retraining a decoder costs ~62 min on the RTX 4000 Ada. The export path is now
run-scoped with an explicit `--publish-decoder` promotion, and the decoder
syncs to GCS on every new-best (both were failure modes that cost the July
artifacts).

### The headline negative result

**Freezing the NPMP decoder does not preserve gait style in downstream task
training.** The July run `follow_rower_npmp` (fitness 0.960, visibly rows —
video at `videos/compare/WITH_PRIOR_final_135.7M_fitness0.960.mp4`, pulled
from wandb) was the anomaly. Five controlled reproduction attempts, all with
config identical on every substantive key:

| arm | isolates | style (16 worlds) | fitness |
|---|---|---|---|
| `follow_rower_npmp_v2` | reference reproduction | 0.379 ± 0.042 | 0.972 |
| `follow_npmp_seedrep` | seed variance | 0.413 ± 0.087 | ~0.92 |
| `follow_npmp_middec` | decoder @75M vs 150M | 0.378 ± 0.040 | — |
| `follow_npmp_frozenstd` | inherited per-joint log_std held | 0.336 ± 0.044 | — |
| `follow_npmp_zar` | AR(1) latent prior in follow | 0.356 ± 0.055 | — |
| (`follow_rower_baseline`, no prior) | control | 0.301 ± 0.033 | 0.950 |

All weights in GCS under those run names. Every arm nails the *task* (fitness
0.92-0.97); none rows.

### The mechanism, measured

Latent-probe (`z(t)` recorded during deterministic rollouts):

| policy | z dominant freq | z amplitude |
|---|---|---|
| tracking expert (decoder's training regime) | 1.15 Hz | 1.39 |
| follow expert, no AR prior | **12.67 Hz** | 4.67 |
| follow expert, AR prior 0.01 | 1.67 Hz | 2.13 |

Two-layer finding:
1. Without the AR(1) latent prior (which `train_track_warp` defaults ON at
   0.01 but `train_follow_warp` never passed — now added, default 0.0), the
   task expert thrashes the frozen decoder ~11x outside its training
   distribution. Fixing this restores the latent *regime*…
2. …and the gait still does not return (style 0.356). **A rhythmic gait
   requires z to oscillate at gait frequency, and nothing in a task objective
   asks for that.** The tracking expert cycles z only because its observations
   contain the reference phase. A task expert has no phase signal and no reward
   for rhythm; smooth-but-flat z glides to the target without stroking.

### The one untested hypothesis
The July decoder (`v2`) had near-uniform log_std (~0.61 all joints); `v3`'s is
bimodal (0.07-0.14 on the four gait-carrying arm joints, ~1.0 elsewhere), and
that structure transfers. Uniform arm noise in early follow training may be
what let the July expert *discover* that rowing pays. v2's weights are gone, so
the only proxy: rerun with log_std reset to uniform ~0.6 on init (~50 min).
Never run — the pivot landed first.

### If style is ever wanted again — the conclusion to build on
Style must live in the **objective** (a style/motion-prior reward during task
training), not in a frozen module. That is exactly the AMP/SMP design, and why
the SMP work below was started.

## 2. The style metric — `tools/style.py` (kept, active)

`python -m rower_soccer.tools.style score --checkpoint <pt> --env follow|track`
Grades HOW the creature moves against `runs_v2/rower_ref_gait.npz`. Four terms,
geometric mean:
- **amp** — per-joint amplitude ratio (reference-still joints graded on stillness)
- **freq** — per-joint dominant-frequency ratio (live joints)
- **shape** — waveform correlation at the best **common** circular shift
  (per-joint shifts would score 8 independent oscillators as a coordinated
  stroke; one shared shift is the design decision that keeps inter-joint timing
  inside the measurement)
- **pose** — mean joint angle vs reference; the only offset-sensitive term;
  catches folded-arms posture and the paddle joints pinned at ±74°

`selftest` = 13 adversarial cases (twitcher, frozen limbs, phase-scramble,
global shift, half-frequency, quarter-amplitude, still-joint thrash, folded
posture…) — run it after any edit. Anchors: reference ceiling 0.999, tracking
policy 0.931, follow arms 0.34-0.41, twitcher baseline 0.301. Score tracks
tracking joint error monotonically. Also wired into `train_follow_warp`
(`eval/style` in wandb; `--no-style` to disable). Costs nothing; keep it.

## 3. SMP / MimicKit on Newton — infrastructure standing, run parked

**Motivation**: SMP (Score-Matching Motion Priors, SIGGRAPH 2026) puts the
motion prior in the *reward* (frozen diffusion model as score-distillation
reward) — the architecture the NPMP finding says is necessary. MimicKit
(xbpeng) ships SMP + AMP/ASE/DeepMimic/etc. with pretrained models and
training logs, and supports **Newton** (Warp + MuJoCo-Warp — our exact stack).

**Setup that works** (at `/workspace/MimicKit`, venv `/workspace/mimickit-venv`):
- PyPI `newton-physics` is a tombstone → install **`newton==1.0.0`**
- Newton 1.0.0 requires **`mujoco==3.5.0` + `mujoco-warp==3.5.0.2`** (3.11
  fails spec-API type checks in Newton's model conversion)
- mujoco-warp 3.5.0.2 fails warp 1.16 codegen despite loose pins → **`warp-lang==1.13.0`**
- Keep this venv separate from the repo's `.venv` (warp 1.16 / mujoco 3.11)
- Data bundle (assets/motions/models/logs, 402 MB) downloads **without auth**:
  `https://1sfu-my.sharepoint.com/personal/xbpeng_sfu_ca/_layouts/15/download.aspx?share=EclKq9pwdOBAl-17SogfMW0Bved4sodZBQ_5eZCiz9O--w`
- Test mode needs `--test_episodes N` (default is infinite → looks like a hang)

**Sim2sim finding (measured)**: Isaac-trained pretrained models collapse on the
Newton backend — SMP spinkick 41/300 mean episode length, DeepMimic control
34/300, vs ~300 in their logs. Cross-engine weight transfer is dead; the valid
reproduction is *training from scratch on Newton* against `data/logs/*.txt`.

**Reproduction status at kill** (2026-08-08, pivot): spinkick prior-policy
training on Newton, their config verbatim (`num_envs 4096`, fits in 5.8/20 GB):

| iter | their Smp_Reward | ours (Newton) |
|---|---|---|
| 0 | 0.062 | 0.051 |
| ~300 | 0.176 | 0.150 |
| ~539 | (0.176-0.18 band) | **0.176 — on-curve** |

Their final target: 0.187 @ iter 2900, 380M samples, 3.96 h (our projection
~6.2 h). Killed at iter ~570.

**To resume**: relaunch the exact command in `MimicKit/output_smp_train.log`
header (or: `mimickit/run.py --mode train --num_envs 4096 --engine_config
data/engines/newton_engine.yaml --env_config data/envs/smp_humanoid_env.yaml
--agent_config data/agents/smp_humanoid_agent.yaml --out_dir
output/smp_spinkick_newton`); partial checkpoints in
`MimicKit/output/smp_spinkick_newton/`. Next steps were: match their curve to
2900 iters, then train a prior on OUR one-cycle rower gait
(`tools/diffusion_model/train_tinymdm.py`, single-clip config) and use the SMP
reward alongside task reward in our own trainers — that last step is the real
port and was never started.
