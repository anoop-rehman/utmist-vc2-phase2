# The GPU was never saturated: MPS gives 3.4x for free

*Measured 2026-08-11 on the RTX 4000 Ada pod.*

## The symptom, and the wrong diagnosis

Five drill trainers were running and per-run fps had collapsed (27-46k each with
four runs, down to single-digit thousands with five). The obvious reading is "the
GPU is full, buy another one." It was wrong.

Three signals said capacity was not the wall:

- **Utilization 28-60%**, with real idle gaps between samples.
- **Memory 6.1 of 20 GB.**
- **Aggregate throughput FELL as runs were added.** This is the decisive one. A
  saturated resource *divides*: N runs each get 1/N and the total holds flat.
  A total that DROPS when you add work is the signature of overhead.

Corroborating: killing one unrelated process (a leftover game server holding a
CUDA context) roughly doubled utilization, 28% -> 60%. A capacity-bound GPU would
not care about one small consumer.

## The A/B

Both arms: the same five trainers, freshly relaunched (a relaunch resets each
trainer's fps clock, since `fps = steps_since_launch / seconds_since_launch` is a
cumulative average and would otherwise drag in old history), measured ~4 minutes
in.

| run | no MPS | with MPS | speedup |
|---|---|---|---|
| follow_ant_final_frozen | 5,049 | 15,853 | 3.1x |
| dribble_ant_v3 | 9,949 | 29,983 | 3.0x |
| kick_ant_v3 | 5,105 | 18,673 | 3.7x |
| kick_ant_v4_timed | 3,743 | 18,543 | 5.0x |
| shoot_ant_v4 | 6,608 | 20,413 | 3.1x |
| **aggregate** | **30,454** | **103,465** | **3.4x** |

**GPU utilization: 57% before, 58% after.** Same number, triple the work — which
is the whole lesson about that metric. `nvidia-smi`'s "utilization" is the
fraction of TIME at least one kernel was resident, not the fraction of the chip
doing useful work. It cannot distinguish one tiny kernel dribbling along from a
fully packed device.

## Why MPS wins here

Without MPS each process owns a CUDA context and the driver time-slices between
them, flushing state on every switch. Our kernels are small — 1-2k ants of
batched physics plus a small MLP — so the switch is a large fraction of the work.
MPS funnels every client through ONE server context, so kernels from different
trainers interleave on the device instead of taking turns.

The gain scales with how many processes share the card and how small their
kernels are, which is exactly our regime (many modest concurrent runs).

## Using it

```bash
bash scripts/mps_start.sh          # once per pod boot, BEFORE any trainer
```

Every shell that launches a trainer must export the same pipe directory or the
client silently runs outside MPS and gets the slow path:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
```

`runs_v2/relaunch_all.sh` already does. To stop: `echo quit | nvidia-cuda-mps-control`.
If the daemon dies, clients fall back to their own contexts — slower, not broken.

## What this changes strategically

The second pod is still worth buying, but **not** to relieve this card — Pod A
just got 3.4x roomier and is still only ~58% utilized. Buy it for the work that
has NO GPU at all: the Transform2Act port, whose paper-scale reproduction is
60-80 days on CPU because its float64 PPO update dominates.
