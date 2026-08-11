# Transform2Act — GPU profile (device vs. dtype), 2026-08-11

Question this answers: **how much faster is the training step on this pod's GPU, and how
much of that is the DEVICE vs. the hardcoded `float64` DTYPE?**

## Answer in one table

| | T_update (real cfg) | epoch (real cfg) | 1000-epoch ETA | vs. CPU |
|---|---|---|---|---|
| CPU float64 (what we run today) | 1911.6 s | 2118.4 s | **24.5 days** | 1.0x |
| GPU float64 (**device only, zero code change**) | 93.9 s | 300.7 s | **3.48 days** | **7.0x** |
| GPU float32 (device + dtype) | 93.4 s | 300.2 s | **3.47 days** | 7.1x |

**It is all device.** Moving the PPO update to the GPU cuts it 20x and the whole run 7x.
Flipping `float64 -> float32` on top of that buys **0.2%** — nothing. The RTX 4000 Ada's
1/64 fp64 throughput never bites, because this update is not FLOP-bound (see
"Why fp64 is free here"). The 88%-of-time PPO update stops being the bottleneck: after the
move, **sampling is 60% of the epoch** and the update only 31%.

Recommendation: run it on this pod's GPU in **float64, unmodified**. Do not port to fp32.
Do not buy pod B for this workload (see "Pod B" at the bottom).

---

## 1. What was measured, and how

Three arms, **identical config and batch size**, differing only in device/dtype:

* **(a) CPU float64** — re-measured here, not quoted from the full config.
* **(b) GPU float64** — device only. Their code already supports this (`--gpu_index`,
  `torch.cuda.set_device`); nothing but the venv changes.
* **(c) GPU float32** — device + dtype, via env-var overrides in a **copy** of the repo at
  `/workspace/T2A-bench`. The live CPU sanity run in `/workspace/Transform2Act/.venv` and
  `results/hopper/` was never touched.

Benchmark config `hopper_bench.yml` = `hopper.yml` with `min_batch_size: 50000 -> 7000`,
`eval_batch_size: 10000 -> 2000`, `max_epoch_num: 4`, checkpointing off.
**`mini_batch_size` (2048) and `num_optim_epoch` (10) are left at their real values**, so a
bench epoch is 3 minibatches x 10 = 30 optimizer steps against the real config's 24 x 10 =
240 — an exact 8x scale model of the real update, not a different workload.
8 sampler processes, `OMP_NUM_THREADS=1` (their default), MPS exported for every GPU run.

Three separate measurements were taken, because the pod is shared with a live CPU trainer,
a competevo run and six drill trainers, and the full training loop is consequently noisy:

1. **Full training loop, 4 epochs per arm** — end-to-end, what a real run feels like.
2. **Isolated `T_update` micro-benchmark** (`bench_update.py`) — samples one batch, freezes
   it, then times `agent.update_params(batch)` (the *actual* call the trainer makes) 3x per
   arm. Same frozen batch in every arm; no sampling noise.
3. **Direct measurement at the real 50 000-state batch** for the GPU arms, so the headline
   GPU numbers are measured, not extrapolated.

## 2. Full training loop — 4 epochs per arm (medians)

Reduced config, seconds:

| arm | T_sample | T_update | T_eval | epoch wall |
|---|---|---|---|---|
| (a) CPU float64 | 23.8 | 236.4 | 7.1 | 300.2 |
| (b) GPU float64 | 37.3 | 23.6 | 9.3 | 71.3 |
| (c) GPU float32 | 22.0 | 17.1 | 12.7 | 52.7 |

Per-epoch raw values (epochs 0-3):

```
(a) CPU fp64  T_sample 22.12 19.39 45.92 25.40 | T_update 298.20 190.78 136.79 281.98 | T_eval  5.48 75.60  6.90  7.29
(b) GPU fp64  T_sample 37.59 118.49 26.92 36.98 | T_update  19.21  48.11  21.01  26.20 | T_eval 13.30  8.38  9.39  9.22
(c) GPU fp32  T_sample 19.08 21.11 132.02 22.84 | T_update  13.32  23.11  20.90  11.09 | T_eval  7.67 20.99 17.80  6.28
```

Read `T_sample`/`T_eval` differences across arms as **pod noise, not signal**: khrylib wraps
sampling in `with to_cpu(*self.sample_modules)` (`khrylib/rl/agents/agent.py`), so rollouts
run on CPU in *every* arm regardless of `--gpu_index`. The 118 s and 132 s outliers are
other tenants. Only `T_update` is genuinely device-dependent.

## 3. Isolated T_update — the controlled comparison

Same frozen 7148-state batch, 30 optimizer steps, 3 reps. `min` is the least-contended
estimate and the one to trust:

| arm | min (s) | median | max | vs. CPU fp64 |
|---|---|---|---|---|
| CPU float64 | 281.69 | 284.89 | 300.78 | 1.00x |
| CPU float32 | 68.29 | 112.12 | 194.18 | 4.12x |
| GPU float64 | 11.42 | 13.01 | 16.80 | **24.66x** |
| GPU float32 (TF32 on, the default) | 9.80 | 11.79 | 15.83 | 28.75x |
| GPU float32 (TF32 off) | 10.32 | 12.00 | 13.40 | 27.31x |

**Decomposition:**

* device alone (CPU fp64 -> GPU fp64): **24.7x**
* dtype alone, on GPU (GPU fp64 -> GPU fp32): **1.11x-1.17x**
* dtype alone, on CPU (CPU fp64 -> CPU fp32): 4.12x — real, but irrelevant once you have
  the GPU

And at the **real** batch size (50 165 states, 240 optimizer steps, min of 2 reps):

| arm | T_update at real batch |
|---|---|
| GPU float64 | **93.88 s** |
| GPU float32 (TF32 off) | **93.40 s** |

At the real batch the dtype advantage vanishes entirely (0.5%). The 8x scale model held:
11.42 x 8 = 91.4 s predicted vs. 93.88 s measured.

### Why fp64 is free here

A 64x fp64:fp32 hardware ratio that produces a 1.0-1.17x wall-clock ratio means the update
is not spending its time in double-precision ALUs. It isn't FLOP-bound at all:

* the tensors are tiny — a 2048-state minibatch is ~10k graph nodes x 64 hidden dims, so
  every GEMM and every `scatter_add` is launch-latency- and memory-bound;
* `Transform2ActPolicy.forward` runs a **Python loop over every state in the minibatch**
  (`for i, x_i in enumerate(x)` building the per-stage masks) plus `batch_data`'s
  `torch.cat`/`np.concatenate` per stage, three times per policy call. That cost is pure
  host-side Python — identical in fp32, fp64, CPU and GPU, and it is a large part of the
  remaining 94 s.

Consequence: the ~94 s GPU update is close to a floor set by Python overhead, not by
arithmetic. fp32, TF32, or a faster card will not move it much; batching the per-state
Python work would.

## 4. Projection to the real config and ETA

`T_sample` and `T_eval` are taken from the **live full-config CPU run** (13 epochs,
`results/hopper/log/log_train.txt`): medians 179.49 s and 27.31 s. They are unchanged by
the GPU because sampling is CPU-side in all arms. `T_update` for CPU fp64 is that run's own
median, 1911.61 s (min 1251.94, max 2874.00 — the pod is busy); the micro-benchmark's
scaled estimate, 281.69 x 8 = 2253.5 s, sits inside that spread. GPU `T_update` values are
the directly-measured 50k-batch numbers.

| arm | T_sample | T_update | T_eval | epoch | **1000-epoch ETA** |
|---|---|---|---|---|---|
| (a) CPU float64 | 179.5 | 1911.6 | 27.3 | 2118.4 s | **24.5 days** |
| (b) GPU float64 | 179.5 | 93.9 | 27.3 | 300.7 s | **3.48 days** |
| (c) GPU float32 | 179.5 | 93.4 | 27.3 | 300.2 s | **3.47 days** |

Sanity check: the live run's own ETA field currently reads 25-28 days, matching row (a).

**The bottleneck moves.** On CPU the update is 90% of the epoch; on GPU it is 31% and
**sampling becomes 60%**. The live run uses only `--num_threads 8` on a 48-core box, so the
next win after the GPU is more sampler processes, not more GPU. That is a CPU purchase, not
a GPU purchase.

## 5. fp32 safety: a direct gradient comparison

Comparing episode rewards over a handful of epochs is close to vacuous — PPO's run-to-run
spread swamps any precision effect at that horizon. So instead (`bench_grad.py`): freeze one
sampled batch, and at each of the first 6 PPO minibatch steps compute the policy and critic
gradients **from identical weights and identical inputs** in fp64 (reference) and in fp32,
then measure how far apart the two gradient vectors are.

| arm | rel. L2 error (policy grad) | 1 - cos | rel. L2 (critic grad) |
|---|---|---|---|
| fp64 recomputed (control) | 1.4e-13 | 0.0 | 4.3e-17 |
| fp32, **TF32 on** (what you get by default) | 5.00e-04 | 1.1e-07 | 2.20e-04 |
| fp32, TF32 off (true IEEE single) | 5.28e-04 | 1.4e-07 | 2.19e-07 |
| *minibatch-to-minibatch disagreement (fp64)* | *1.003* | — | — |

Reading:

* The fp64 control reproduces to 1e-13, so the harness is clean and the GPU kernels in play
  (`scatter_add` etc.) are deterministic here. Whatever the fp32 rows show is precision.
* **fp32 policy gradients agree with fp64 to ~5e-4 relative, and in direction to 1e-7.**
  The gradient noise the algorithm already lives with — swapping one minibatch for another —
  is `1.003`, i.e. **~2000x larger**. On this evidence fp32 introduces nothing the optimizer
  can distinguish from its own sampling noise on a single step.
* **Turn TF32 off if you ever do go fp32.** `torch.backends.cuda.matmul.allow_tf32` defaults
  to `True` in torch 1.8 and TF32 keeps only a 10-bit mantissa. It degrades the **critic**
  gradient by 1000x (2.2e-4 vs. 2.2e-7) and buys ~5% of wall clock, which is inside the
  noise. Bad trade.
* The policy-gradient error (5e-4) is the same with TF32 on and off, so it is not coming
  from matmuls. The likely source is
  `Transform2ActPolicy.get_log_prob`, which does `torch.cumsum` over all node log-probs and
  then differences consecutive cumulative sums to recover per-episode sums — a cancellation
  pattern that loses ~4 decimal digits in fp32 once the running sum reaches ~1e4. If fp32 is
  ever pursued, that line is the first thing to fix (segment-sum instead of cumsum-diff).
* Worst single parameter tensor: 4.4e-2 (TF32 on) — but on a tensor holding 0.00% of the
  total gradient norm, so it does not matter. With TF32 off the worst is 6.0e-3 on
  `control_action_log_std` (1.9% of the norm).

**This does not validate fp32, and it is not claimed to.** It is a single-batch, 6-step,
untrained-policy measurement. It rules out gross breakage; it says nothing about 240 000
optimizer steps of accumulation, about whether a 5e-4 perturbation systematically flips PPO
clip decisions near the `1 +/- 0.2` boundary, or about the discrete skeleton-transform
`Categorical` head late in training when logits are large and saturated. To actually
validate fp32 you would need: paired fp64/fp32 runs at >= 3 seeds each carried to at least
100-200 epochs, compared as learning-curve distributions (not single runs, PPO seed variance
is large); the gradient check repeated at a *trained* checkpoint where log-probs and
advantages are far from initialisation; and clip-fraction / ratio-distribution / NaN
monitoring throughout. **None of that is worth doing, because fp32 is worth 0.2%.**

## 6. GPU memory

| | torch peak allocated | torch peak reserved |
|---|---|---|
| float64, bench batch (7 148 states) | 814.7 MB | 876.0 MB |
| float32, bench batch | 411.4 MB | 510.0 MB |
| **float64, real batch (50 165 states)** | **970.6 MB** | — |
| float32, real batch | 717.2 MB | — |

Whole-process footprint on the card as `nvidia-smi` sees it, at the real batch in float64:
**2278 MiB** (the extra ~1.3 GB is the CUDA context and cuBLAS workspaces, which every
process pays once).

Card: RTX 4000 Ada, **20 475 MiB total**. During this profiling the six drill trainers were
holding 4.3-6.1 GiB and the GPU was **8-12% utilised**. A Transform2Act run at ~2.3 GiB and
low occupancy fits alongside them with a very large margin — memory is not the constraint,
and neither is compute.

## 7. Install recipe that worked

`/workspace/Transform2Act/.venv-gpu` (built by the previous attempt; verified working here —
`torch.cuda.is_available() == True`, CUDA-built PyG extensions, full training loop passes in
both dtypes). Same as the CPU recipe in `REPRO_NOTES.md` except for the four marked lines:

```sh
uv venv --python 3.9 /workspace/Transform2Act/.venv-gpu
P=/workspace/Transform2Act/.venv-gpu/bin/python
$P -m pip install "setuptools==59.5.0" wheel "numpy==1.21.6" "Cython<3"

# (1) CUDA build instead of +cpu
$P -m pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# (2) same pins, but from the cu111 wheel index
$P -m pip install torch-scatter==2.0.8 torch-sparse==0.6.12 torch-cluster==1.5.9 \
    torch-spline-conv==1.2.1 --no-index -f https://data.pyg.org/whl/torch-1.8.0+cu111.html
# (3) CUDA 11 runtime that torch 1.8's wheels do NOT ship in a usable form (see below)
$P -m pip install nvidia-cuda-runtime-cu11 nvidia-cublas-cu11 nvidia-cusparse-cu11

$P -m pip install torch-geometric==1.6.1
$P -m pip install googledrivedownloader==0.4 rdflib pandas==1.3.5 h5py
$P -m pip install "gym==0.15.4" "opencv-python==4.5.5.64" glfw pyyaml \
    "tensorboard==2.11.2" "protobuf==3.20.3" lxml pillow "scipy==1.7.3"
$P -m pip install "mujoco-py==2.1.2.14"
```

Environment — `source /workspace/Transform2Act/env-gpu.sh` before every GPU run:

```sh
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
_T2A_SP=/workspace/Transform2Act/.venv-gpu/lib/python3.9/site-packages
export LD_LIBRARY_PATH=$_T2A_SP/nvidia/cusparse/lib:$_T2A_SP/nvidia/cuda_runtime/lib:$_T2A_SP/nvidia/cublas/lib:$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps     # MPS is worth ~3.4x on this shared pod
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
```

### Deviations from the authors' pins (beyond those already in REPRO_NOTES.md)

| theirs | ours | why |
|---|---|---|
| torch 1.8.0 + CUDA (unspecified) | torch **1.8.0+cu111** | the CUDA build the 2021 wheels shipped |
| PyG extensions, unversioned | 2.0.8 / 0.6.12 / 1.5.9 / 1.2.1 from the **cu111** index | same pins as the CPU venv; the cu111 wheels carry real CUDA kernels |
| — | **`nvidia-cuda-runtime-cu11` + `LD_LIBRARY_PATH`** | load-bearing. torch 1.8 ships its cudart under a mangled SONAME (`libcudart-6d56b25a.so.11.0`), which does not satisfy the plain `libcudart.so.11.0` that `torch_scatter/_scatter_cuda.so`, `torch_sparse/_spmm_cuda.so` and `torch_cluster/_grid_cuda.so` link against. Without it the PyG CUDA extensions fail to load. The system CUDA here is 12.4 — wrong major version, unusable for a cu11 build. |
| — | `nvidia-cublas-cu11`, `nvidia-cusparse-cu11` | installed and on the path, but no `.so` in the venv declares a dynamic dependency on them (torch 1.8 links cuBLAS/cuSPARSE statically). Harmless; keep for safety, ~900 MB of the venv is these three packages. |
| CUDA arch | sm_89 (Ada) running sm_86 cubins | torch 1.8/CUDA 11.1 predates Ada and emits no sm_89 code; it works via CUDA's binary compatibility guarantee **within** compute-capability major version 8.x. No recompilation needed. |
| Python 3.9.25, mujoco-py 2.1.2.14, gym 0.15.4, numpy 1.22.4 | unchanged from the CPU venv | — |

Disk cost: the GPU venv is ~5.5 GB (3.9 GB torch + 0.9 GB nvidia libs).

### Code changes needed for each arm

* **(b) GPU float64: none.** `design_opt/train.py` already picks CUDA when
  `torch.cuda.is_available()`. Just run with `.venv-gpu`.
* **(c) GPU float32: three small patches**, applied only in the copy at
  `/workspace/T2A-bench`, never in `/workspace/Transform2Act`:
  1. `design_opt/train.py:24` — `dtype` and device read from `T2A_DTYPE` / `T2A_DEVICE`
     env vars instead of being hardcoded to `torch.float64`.
  2. `design_opt/agents/transform2act_agent.py` — `tensorfy` builds the observation tensor
     with `dtype=torch.get_default_dtype()`. Without this the float64 numpy obs silently
     produces a float64 tensor and hits a dtype mismatch against fp32 weights. (The edge
     list, element 1, must stay int64.)
  3. `design_opt/models/transform2act_policy.py:238` — `skel_action.double()` ->
     `skel_action.to(action.dtype)`.

## 8. Pod B: recommendation

**Do not buy a second GPU pod for Transform2Act.** The profile says the opposite of what the
"88% of the time is the GNN update" framing suggests: the update is 20x faster on the GPU
and, once moved, is only 31% of the epoch — while the card sits at 8-12% utilisation and
under 1 GB of the 20.5 GB it has. Transform2Act simply does not want much GPU. All four
paper configs (`hopper`, `ant`, `swimmer`, `gap`) would fit on this one card simultaneously,
memory-wise, alongside the six drill trainers. What they would *not* fit is this pod's 48
CPU cores: each run wants 8+ mujoco sampler processes, sampling is now the dominant cost at
60% of the epoch, and the box already carries the drill trainers, the competevo run and the
live CPU job. So if a second pod is bought, buy it for **cores, not for a GPU** — and the
cheaper move first is to raise `--num_threads` on the existing box (the live run uses 8 of
48) and re-measure, which costs nothing and attacks the actual bottleneck. Concretely, for
the immediate goal: kill the CPU sanity run, restart the same `hopper` config on this card
with `.venv-gpu` in unmodified float64, and the paper run lands in **~3.5 days instead of
~25** on hardware you already own.

---

## Appendix: reproducing this

Everything lives in the scratch copy `/workspace/T2A-bench` (the pristine repo and the live
run were not modified):

| file | what it does |
|---|---|
| `run_arms.sh` | the three full-loop arms, sequentially |
| `design_opt/cfg/hopper_b_{cpu64,gpu64,gpu32}.yml` | the reduced config, one id per arm |
| `bench_update.py` | isolated `T_update` timing, device x dtype x TF32 |
| `bench_grad.py` | the fp64-vs-fp32 gradient comparison |
| `arms.log`, `upd.log`, `mem.log`, `grad.log` | raw output of the above |
| `results/hopper_b_*/log/log_train.txt` | per-epoch timings for each arm |

```sh
cd /workspace/T2A-bench && source /workspace/Transform2Act/env-gpu.sh
bash run_arms.sh                                          # 3 arms x 4 epochs, ~30 min
/workspace/Transform2Act/.venv-gpu/bin/python bench_update.py --reps 3
/workspace/Transform2Act/.venv-gpu/bin/python bench_update.py --cfg hopper_mem --reps 2 \
    --arms "GPU float64,TF32 off"                         # real 50k batch
/workspace/Transform2Act/.venv-gpu/bin/python bench_grad.py --steps 6
```

Caveat on every number here: this pod is shared (a live CPU Transform2Act run, a competevo
run, six drill trainers). Medians and mins are reported for that reason, and the spread is
shown wherever it is wide. The GPU-vs-CPU ratios are large enough that contention cannot
change the conclusion.

## Addendum 2026-08-11: run the configs SEQUENTIALLY, not in parallel

Both `hopper_gpu` and `ant_gpu` were launched together on the strength of the
memory headroom above (2.3 GB of 20 GB each, card ~34% utilised). Memory was
never the constraint; CPU sampling was.

Measured with both running:

| run | T_sample | T_update | T_eval | ETA |
|---|---|---|---|---|
| hopper_gpu alone (epoch 0) | 113.1 | 110.1 | 26.1 | **2 d 21 h** |
| hopper_gpu with ant alongside (epoch 3) | 168.2 | 273.6 | 38.4 | **5 d 13 h** |
| ant_gpu (epoch 0, 16 threads) | 248.6 | 452.0 | 135.6 | **9 d 16 h** |

Parallel finishes both at ~9.7 days. Sequential delivers hopper at ~2.9 days and
then ant at roughly 5-6 days uncontended — both done at about the same time, but
with the FIRST paper-number data point in hand three days sooner.

That earlier data point is the whole argument. M2's gate is reproducing their
Table 1; if our GPU run does not reproduce it, that needs to surface in 3 days,
not 10. Parallelism here buys nothing and delays the only signal that can
invalidate the approach.

`ant_gpu` was therefore stopped after epoch 0. Its cfg and log are kept; relaunch
it with `--num_threads 32` once hopper finishes.
