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
Rewards at ~0 training are naturally near the survival-bonus floor (paper's 2D locomotion
converges to ~4000+ over 1000 epochs); the point here is the full loop, not the number.

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
