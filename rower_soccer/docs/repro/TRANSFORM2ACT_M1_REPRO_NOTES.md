# Transform2Act — smoke reproduction notes (2026-08-10)

Milestone 1 of the paper-reproduction track. Goal: run the official ICLR'22 code
(https://github.com/Khrylx/Transform2Act, arXiv 2110.03659) end-to-end their way,
CPU-only, on this shared pod. Status: **smoke run passed** (see "Smoke run" below).

## Environment

- Python 3.9.25 (via `uv python install 3.9`; repo says >=3.7, torch 1.8.0 wheels top out at cp39)
- venv: `/workspace/Transform2Act/.venv`
- MuJoCo 2.1.0 binary at `~/.mujoco/mujoco210` (mujoco.org/download/mujoco210-linux-x86_64.tar.gz)
- System packages: `apt-get install -y libosmesa6-dev libglew-dev patchelf libgl1-mesa-dev libglfw3`
- Required env vars for EVERY run — `source /workspace/Transform2Act/env.sh`:
  ```sh
  export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
  export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
  export MUJOCO_GL=osmesa      # headless; training never renders anyway
  export OMP_NUM_THREADS=1     # per README (khrylib/rl/agents/agent.py also sets it)
  export MKL_NUM_THREADS=1
  ```

## Working install (exact commands)

```sh
uv venv --python 3.9 /workspace/Transform2Act/.venv
P=/workspace/Transform2Act/.venv/bin/python
$P -m pip install "setuptools==59.5.0" wheel "numpy==1.21.6" "Cython<3"
$P -m pip install torch==1.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# PIN the PyG extension versions — see landmine #1
$P -m pip install torch-scatter==2.0.8 torch-sparse==0.6.12 torch-cluster==1.5.9 \
    torch-spline-conv==1.2.1 --no-index -f https://data.pyg.org/whl/torch-1.8.0+cpu.html
$P -m pip install torch-geometric==1.6.1
$P -m pip install googledrivedownloader==0.4 rdflib pandas==1.3.5 h5py   # PyG 1.6.1 imports these at import time
$P -m pip install "gym==0.15.4" "opencv-python==4.5.5.64" glfw pyyaml \
    "tensorboard==2.11.2" "protobuf==3.20.3" lxml pillow "scipy==1.7.3"
$P -m pip install "mujoco-py==2.1.2.14"     # cython ext builds lazily on first `import mujoco_py`
```

## Deviations from their pins

| theirs | ours | why |
|---|---|---|
| torch 1.8.0 + CUDA | torch **1.8.0+cpu** | pod constraint: 4 live GPU runs, CPU only |
| torch-scatter/cluster/spline-conv unversioned | 2.0.8 / 1.5.9 / 1.2.1 | the cp39 wheels on the torch-1.8.0+cpu index; unpinned pip resolves to modern PyPI sdists that fail to build against torch 1.8 (see landmine #1) |
| mujoco-py unspecified (2021-era = 2.0/2.1) | 2.1.2.14 + mujoco210 | current last release; all `mujoco_py` APIs used by the repo exist |
| Python >= 3.7 (2021 ⇒ 3.7/3.8 typical) | 3.9.25 | newest python with torch-1.8.0 wheels; no 3.9 incompatibilities hit |
| numpy unspecified | 1.21.6 → later bumped to 1.22.4 by a transitive dep | both work; keep <1.24 (`np.bool` removal breaks old gym) |
| tensorboard unversioned | 2.11.2 + protobuf 3.20.3 | protobuf 4.x breaks torch 1.8's tensorboard writer |
| torch-geometric 1.6.1 | same | but needs the extra runtime deps listed above |

Everything else (gym 0.15.4, torch-geometric 1.6.1, torch-sparse 0.6.12, algorithms,
configs) is exactly theirs. No source-code modifications were needed — the 2021 code
runs unpatched.

## Install landmines

1. **Do not install torch-scatter/torch-cluster/torch-spline-conv unpinned.** The README's
   `pip install torch-scatter -f <wheel index>` picks the *latest PyPI version* (2.1.x sdist),
   which compiles from source for ~40 min single-core and fails against torch 1.8 headers.
   Pin to the versions that have cp39 wheels on https://data.pyg.org/whl/torch-1.8.0+cpu.html
   and pass `--no-index` (the old `pytorch-geometric.com/whl` URLs redirect there).
2. torch-geometric 1.6.1 crashes at import (`google_drive_downloader`, `rdflib`, `pandas`,
   `h5py` missing) — install those four; `googledrivedownloader==0.4` specifically
   (0.5 renamed the module).
3. mujoco-py compiles its cython extension on first import; needs `patchelf`, OSMesa/GLEW
   dev headers, `LD_LIBRARY_PATH` including `~/.mujoco/mujoco210/bin`, and `Cython<3`.
4. `train.py` defaults to `--num_threads 20` sampling processes — set explicitly to respect
   core budgets. `--gpu 0` from the README does not exist; the flag is `--gpu_index`, and with
   CPU-only torch the code falls back to CPU automatically (`torch.cuda.is_available()`).
5. Training runs `torch.set_default_dtype(torch.float64)` — everything is double precision;
   checkpoints are ~157 MB each (`save_model_interval` matters for disk).
6. Config lookup is `design_opt/cfg/**/<id>.yml` relative to **cwd** — always run from repo root.

## Entrypoint / config map (paper experiments)

`python design_opt/train.py --cfg <id> --num_threads N`, run from `/workspace/Transform2Act`:

| cfg | paper experiment |
|---|---|
| `hopper` | 2D Locomotion |
| `ant` | 3D Locomotion |
| `swimmer` | Swimmer |
| `gap` | Gap Crosser |

All configs: 1000 epochs x 50,000 samples/epoch (= 50M env steps), PPO 10 epochs x 2048
minibatch, GraphConv GNNs (3x64) for all three policy stages (skeleton transform, attribute
transform, control) + GNN critic. Eval: `design_opt/eval.py --cfg <id> [--save_video]`
(needs a checkpoint under `results/<id>/models/`). Outputs land in
`results/<cfg>/{models,log/log_train.txt,tb}`.

The GNN core is exercised on every policy call: `design_opt/models/transform2act_policy.py`
routes each state through skel/attr/control `GNNSimple` stacks (torch-geometric `GraphConv`
over the morphology edge list), and episodes begin with `skel_transform_nsteps=5` skeleton-
transform steps followed by an attribute-transform step before control execution.

## Smoke run (passed)

Config `design_opt/cfg/hopper_smoke.yml` = `hopper.yml` with
`min_batch_size: 2000, mini_batch_size: 512, eval_batch_size: 500, max_epoch_num: 3, save_model_interval: 1`.

```sh
cd /workspace/Transform2Act && source env.sh
.venv/bin/python design_opt/train.py --cfg hopper_smoke --num_threads 8
```

Result (12 min total, 8 sampler procs + 1 learner, well under 12 cores):

```
0  T_sample 8.91  T_update 202.20  T_eval 2.10  train_R 0.84  train_R_eps 29.84  exec_R_eps 42.13
1  T_sample 7.11  T_update 261.60  T_eval 2.00  train_R 0.83  train_R_eps 29.39  exec_R_eps 42.21
2  T_sample 6.16  T_update 228.73  T_eval 1.80  train_R 0.89  train_R_eps 31.31  exec_R_eps 42.09
```

3 checkpoints + `best.p` written, tensorboard events written, `training done!` logged.
Rewards at ~0 training are naturally near the survival-bonus floor; the point here is
the full loop, not the number.

> **Correction, 2026-08-24.** This line originally read "paper's 2D locomotion converges
> to ~4000+ over 1000 epochs". That figure was never in the paper — it was ours, written
> down here and then cited back as if it were theirs. The real number, read off Figure 3
> of the paper directly (`assets/plot_baselines.png` of the ar5iv build), is **~9,000**.
> See "What the paper actually reports" below. Every claim that used ~4000 as the bar
> was measuring against a number half the size of the real one.

## Long sanity run (in progress)

Started 2026-08-10 22:04 UTC, full paper `hopper` config, CPU, 8 workers, nohup pid 2872061:

```sh
cd /workspace/Transform2Act && source env.sh
nohup .venv/bin/python design_opt/train.py --cfg hopper --num_threads 8 \
  > results/hopper_nohup.log 2>&1 &
```

- stdout: `/workspace/Transform2Act/results/hopper_nohup.log`
- structured log: `/workspace/Transform2Act/results/hopper/log/log_train.txt`
- NOTE: checkpoints save every 100 epochs at ~157 MB (`best.p` refreshes more often).

## Honest paper-scale cost estimate (this pod, CPU, <=12 cores)

Extrapolating from the smoke run: sampling 50k steps with 8 workers ~ 2.5-3.5 min;
the PPO update dominates at ~6 s per 512-minibatch step single-threaded (float64, CPU)
=> 50k-sample epoch = ~240 minibatches x 10 PPO epochs ~ 1.5-2 h/epoch.
**1000 epochs ~ 60-80 days per run — not feasible on CPU.** The long sanity run above
will yield ~10-15 epochs/day, enough to check the reward trend against the paper's
early learning curve. A faithful full run needs a GPU slot (the authors trained on GPU;
torch 1.8 + cu111 wheels install the same way) or porting the update loop to float32 +
larger OMP parallelism, both out of scope for milestone 1.

---

## What the paper actually reports (2026-08-24)

`D3_HANDOFF.md` listed this as the cheapest open item in D3: *"Read the paper's 2D
Locomotion number and settle whether M1 is met. Costs minutes; currently gates a claim we
would otherwise be making on our own arithmetic."* Settled here, and it does not settle
the way the handoff expected.

### There is no number in the text

The paper contains **two tables, both hyperparameters** (Appendix G, Table 1 for
Transform2Act and Table 2 for the NGE / RGS / ESS baselines). There is no results table,
no appendix table of returns, and **no sentence anywhere in the paper that quotes a
numeric return for any experiment**. All four environments' performance is communicated
only as learning curves in Figure 3.

So the earlier attempt to verify "~4000+" did not fail because the PDF was too large. It
would have failed anyway: the quantity does not exist in prose.

### Read off Figure 3

Figure 3 is `assets/plot_baselines.png` in the ar5iv build of arXiv 2110.03659 — a
figure, so these are read by eye off gridded axes, ±5% or so, not transcribed:

| environment | Transform2Act at 50 M simulation steps | y-axis top |
|---|---|---|
| **2D Locomotion** | **~9,000** (band ~8,000–10,500) | 10,000+ |
| 3D Locomotion | ~4,100, still climbing at 50 M | 4,000+ |
| Swimmer | ~750, flat from ~10 M | 800 |
| Gap Crosser | ~3,500 | 4,000 |

All four curves run to 50 M simulation steps, which matches Appendix G: batch size 50,000
for 1,000 epochs.

### What that does to M1

Our completed `hopper_gpu` run — 1,000/1,000 epochs, 50 M steps, the same config — reached
a final-20 mean `exec_R_eps` of **6,836**, max 7,452.

**6,836 against ~9,000 is about 76% of the reference, and below the lower edge of their
shaded band.** The handoff's reading — "a clean, monotone, fully-completed run that lands
well above a number we wrote down" — was true only of the number we wrote down. Against
the paper's actual curve the run lands *short*.

**M1 is therefore NOT met on the strength of that run.** Nobody should claim it is.

### Three caveats, none of which close the gap

* **One seed against their several.** Their band is over seeds; ours is a single run and
  has no band at all. The honest comparison is one sample against a distribution.
* ~~**`exec_R_eps` versus their "Reward" may not be the same quantity.**~~
  **CHECKED, and they do coincide.** `design_opt/utils/logger.py:19-22` accumulates
  `exec_episode_reward` only on steps where `info['stage'] == 'execution'`, and the two
  design stages return `reward = 0.0` literally (`design_opt/envs/hopper.py:120` for
  skeleton_transform, `:140` for attribute_transform). So the execution-stage return IS
  the whole episode return, and this axis is apples-to-apples. One caveat closed.
* **Read by eye.** ~9,000 is a figure reading. It is not 9,000.

The gap is large enough (24%) that none of these plausibly account for it, but the first
is the cheapest to attack.

Note what the re-run now in flight is and is not. `hopper_gpu.yml` carries `seed: 1`, the
same as the run it replaces, so it is **a replicate on different hardware, not an
independent second seed** — the GPU changed, so it will not be bit-identical, but nothing
about it was deliberately varied. It regenerates the checkpoints the pod destroyed and it
gives a second point on the curve; it does not give an error bar. A real answer to "is
M1 met" needs 3-5 seeds run to 1,000 epochs, which at ~16 h each is a two-day commitment
and a decision for whoever owns the milestone, not a thing to slip in.


---

## The released code does not implement the paper's reward (2026-08-24)

Found while closing the `exec_R_eps` caveat above, and it bears directly on the 24% gap.

**The paper, equation 17**, for 2D Locomotion:

> The reward function is defined as `r_t = |p^x_{t+1} − p^x_t| / δt + 1`, where `p^x_t`
> denotes the x-position of the agent and `δt = 0.008` is the time step. An alive bonus of
> 1 is also used inside the reward.

Note the **absolute value**.

**The released code**, `design_opt/envs/hopper.py:159-160`:

```python
reward = (posafter - posbefore) / self.dt
reward += alive_bonus
```

There is no `abs`. The same signed form appears in `ant.py:160` and `swimmer.py:158`; no
env in the repository takes an absolute value of displacement.

`δt` does match: the model timestep is 0.002 (`assets/mujoco_envs/hopper.xml:8`) and
`self.dt = frame_skip * timestep` with `frame_skip = 4`, so 0.008.

### Why it could matter, and why it might not

Under the paper's `|·|`, motion in EITHER direction is paid, so an agent that oscillates
rapidly in place accrues reward without net displacement. Under the code's signed form,
backward motion is penalised and only net forward progress pays. For a converged forward
runner the two coincide exactly — `posafter > posbefore` every step — so this is **not
automatically** an explanation for 6,836 against ~9,000.

Where it could bite is the shape of the optimum. A morphology-searching method rewarded on
`|Δx|` can find a fast-vibrating body that a signed reward would never select, and such a
body would score far above a runner. That is a plausible route to a materially higher
curve, and it is exactly the kind of thing Transform2Act's design stage is good at finding.

### What this is and is not

It is a **documented discrepancy between the paper's text and the released code**, verified
in both. It is not a demonstration that it caused the gap. Two readings survive:

1. The `|·|` is a write-up error and Figure 3 was produced by this code, in which case the
   gap is real and ours to close.
2. Figure 3 was produced with `|·|` and the released code differs, in which case our run
   is being compared against a number no run of this code can reach.

**Distinguishing them is cheap** — add the `abs` and train one hopper seed — and it should
be done before anyone spends days chasing the gap. It is not being done here: changing the
reward mid-flight would invalidate the two seeds now running, whose purpose is to measure
the spread of THIS code.

### First result: `|Δx|` learns 2.5-6.5x faster, and it is not seed noise

`hopper_gpu_abs` (seed 11, `reward_specs.abs_displacement: true`) against the
two unmodified seeds, `exec_R_eps` at matched epochs:

| epoch | seed 1 signed | seed 2 signed | **seed 11 \|Δx\|** | ratio |
|---|---|---|---|---|
| 10 | 211 | 206 | 272 | 1.3x |
| 20 | 335 | 311 | 397 | 1.2x |
| 30 | 371 | 421 | **2,565** | **6.5x** |
| 40 | 494 | 520 | **2,750** | 5.4x |
| 50 | 1,314 | 1,357 | **3,344** | 2.5x |

**The two signed seeds agree with each other throughout** — 211/206, 335/311,
371/421, 494/520, 1,314/1,357, 1,356/1,403 — so Transform2Act's seed variance on
this task is small and the gap is not sampling. It is the reward form.

**Why the size of the gap matters more than its sign.** `|Δx| ≥ Δx` pointwise,
so *some* increase is guaranteed by construction. But for an agent that runs
forward the two are **equal**, because `posafter > posbefore` every step. A
6.5x gap at epoch 30 therefore is not the same behaviour scored more generously
— it says the `|Δx|` agent is doing something the signed reward would not pay
for at all, which is exactly the oscillating-body failure mode predicted above.

Our completed signed run reached 6,836 over 1,000 epochs. The `|Δx|` run is at
3,344 by **epoch 50**.

### What this does to M1, provisionally

If the `|Δx|` curve continues, the reading that survives is the second of the
two in the previous section: **Figure 3 was produced with a reward the released
code does not implement, and our 6,836 was being compared against a number no
run of this code can reach.** Under that reading M1 was never failed; it was
mis-specified, and the bar for the released code is ~6,800, not ~9,000.

**This is not yet established.** Three things would settle it, none expensive:

1. **Let `hopper_gpu_abs` reach 300 epochs** (running) and see where it lands
   relative to the signed seeds at the same epoch. Currently at 56.
2. **Look at the `|Δx|` agent.** If it evolves a vibrating body with little net
   displacement, that confirms the mechanism and also says the paper's own
   number describes an agent that does not locomote. `render` on its checkpoint
   costs minutes.
3. **Measure net displacement, not reward.** The honest cross-check is metres
   travelled per episode, which is comparable across both reward forms and is
   what "2D Locomotion" is supposed to mean.

Until (2) is done, the alternative reading stays alive: `|Δx|` may simply be an
easier curriculum that reaches the same gait sooner, in which case the paper's
~9,000 is reachable and the gap is ours.

### RESOLVED: equation 17's `|·|` is a write-up error, and the M1 gap is real

`displacement_probe.py`, 12 mean-action episodes each, both at **epoch 50**:

| | signed (released code) | `\|Δx\|` (paper's eq. 17) |
|---|---|---|
| `exec_R_eps` (Figure 3's axis) | 1,321 | **3,324** |
| **net displacement** | **2.57 m** | **0.60 m** |
| path length | 2.57 m | 18.59 m |
| **net / path** | **0.999** | **0.032** |
| net speed | 0.32 m/s | 0.07 m/s |

The signed agent is a **pure runner**: net/path = 0.999, i.e. it essentially
never steps backward, and its path length equals its displacement to two
decimals. The `|Δx|` agent **oscillates**: 18.59 m of movement to achieve 0.60 m
of progress, and it scores 2.5x more reward while travelling **4x less far**.

That is the degenerate solution predicted when the discrepancy was found, and it
settles which of the two readings is right — **in the opposite direction to the
hint the reward curves gave.**

**The `|·|` in equation 17 is a write-up error.** The paper's own claims rule out
the alternative: it reports agents that "look plausible", shows their rendered
designs, and titles the environment 2D *Locomotion*. A body that vibrates 18 m
in place to travel 0.6 m is not that, and would not have been showcased. Figure
3 was produced by the signed code that was released.

**Consequences, and they are not comfortable:**

1. **~9,000 is a real locomotion number and our 6,836 is genuinely short.** M1 is
   not met, the bar is not mis-specified, and the gap is ours to close.
2. The previous section's provisional reading — "M1 was never failed, it was
   mis-specified" — is **withdrawn**. It was based on reward curves, and reward
   is exactly the quantity that is not comparable across the two forms. This is
   why the displacement cross-check was written before the reading was believed.
3. `hopper_gpu_abs` has answered its question at epoch 50 and was stopped rather
   than run to 300. The flag stays (default off) and the patch stays in
   `docs/t2a/`, because the measurement is worth being able to repeat.

**A note on the method, since it nearly went the other way.** The reward curves
said `|Δx|` learns 2.5-6.5x faster and that looked like support for the
mis-specification reading. It was support for nothing: a reward that pays for
oscillation is higher on an oscillating agent, which is a tautology, not
evidence. The number that discriminated was net displacement — chosen because it
is comparable across both reward forms, which reward is not.
