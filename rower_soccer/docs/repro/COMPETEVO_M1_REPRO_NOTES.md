# CompetEvo reproduction notes (milestone 1: smoke/sanity, CPU-only)

Repo: https://github.com/KJaebye/competevo (IJCAI-2024, arXiv:2405.18300)
Date: 2026-08-10. Pod: 48-core CPU (we cap ourselves at <=30), 4 unrelated GPU jobs live -> everything here is CPU-only.

## Stack they assume (from docker/dockerfile + docker/requirements.txt)

- Python 3.8 (Ubuntu 20.04 image)
- torch 1.12.0 (+cu113 in their image; we use +cpu)
- **Modern MuJoCo bindings** (`mujoco==2.3.5`) + `gymnasium==0.28.1` — NOT mujoco-py, NOT old gym. No mujoco210 tarball / LD_LIBRARY_PATH / patchelf dance needed.
- torch_geometric (unpinned) for the GNN morphology policy
- No setup.py / pyproject: the repo is run in-place with `PYTHONPATH=./` (see `dev.env`).

## Install steps that WORKED (runnable)

```bash
git clone https://github.com/KJaebye/competevo /workspace/competevo
cd /workspace/competevo

# uv (any recent version) to get Python 3.8 without touching system python
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.8
uv venv --python 3.8 /workspace/competevo/.venv

# CPU-only torch, same version numbers as their dockerfile
VIRTUAL_ENV=/workspace/competevo/.venv uv pip install \
  torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu

apt-get install -y swig               # box2d-py builds from source and needs swig
VIRTUAL_ENV=/workspace/competevo/.venv uv pip install -r docker/requirements.txt
VIRTUAL_ENV=/workspace/competevo/.venv uv pip install six   # missing from their requirements
```

Verified: `torch 1.12.0+cpu, mujoco 2.3.5, gymnasium 0.28.1, torch_geometric 2.6.1, torch.cuda.is_available()==False`.

## Deviations from their pins (all of them)

| Their pin | Ours | Why |
|---|---|---|
| torch 1.12.0+cu113 | torch 1.12.0+cpu | 4 GPU jobs live on this pod; CPU-only mandated. Same version number. |
| Python 3.8 (apt, 3.8.10-ish) | Python 3.8.20 (uv) | System only has 3.10/3.11; uv standalone build. Same minor version. |
| torch_geometric (unpinned) | 2.6.1 | Resolver's choice for py3.8; works. |
| (not listed) | six | `gym_compete/new_envs/agents/agent.py` imports `six`; absent from requirements.txt (in their docker it rode in as a transitive dep). |
| swig (in dockerfile apt list) | apt-get install swig | Needed to build box2d-py sdist. box2d is not actually used by any competevo env (all MuJoCo); could be skipped entirely. |

Everything else in requirements.txt installed at their exact pins.

## Entrypoint map

- `train.py --cfg config/<name>.yaml [--use_cuda false] [--num_threads N]` — training. Env + runner are chosen by the yaml:
  - `runner_type: multi-agent-runner` -> fixed-morph baselines (`run-to-goal-{ants,bugs,spiders}-v0`, `robo-sumo-{ants,bugs,spiders}-v0`)
  - `runner_type: multi-evo-agent-runner` -> the paper's CompetEvo agents (`run-to-goal-dev*`, `robo-sumo-dev*`)
  - `runner_type: selfplay-agent-runner` -> self-play variants
- `display.py --cfg <yaml> --ckpt_dir <run>/models` — visualization (needs a display; skipped here. If ever needed: MUJOCO_GL=egl and note base_runner uses render_mode="human", so headless display would need a code tweak).
- Outputs land in `./tmp/<env_name>/<timestamp>/{models,log,tb}` (relative to cwd; the hardcoded `/root/ws/...` in config.py's `out_dir` is dead code — logger uses `./tmp`).
- `--num_threads N` = number of multiprocessing sampler workers; eval is hardcoded to 10 workers (`nthreads=10` in the runners). `OMP_NUM_THREADS=1` is forced inside the runner, so cores ~ N during sampling.
- One iteration ("epoch") = sample `min_batch_size` steps -> PPO update (`num_optim_epoch` x `mini_batch_size`) -> eval `eval_batch_size` steps -> checkpoint (every `save_model_interval` epochs + best-so-far).

## Smoke runs (both PASSED, 2026-08-10)

Reduced configs created for this: `config/smoke-run-to-goal-ants-v0.yaml` and
`config/smoke-run-to-goal-devants-v0.yaml` (copies of the originals with
`min_batch_size: 2500, mini_batch_size: 512, eval_batch_size: 1000, max_epoch_num: 3/2`).

```bash
cd /workspace/competevo
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python train.py \
  --cfg config/smoke-run-to-goal-ants-v0.yaml --use_cuda false --num_threads 8
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python train.py \
  --cfg config/smoke-run-to-goal-devants-v0.yaml --use_cuda false --num_threads 8
```

Observed (fixed-morph ants, 3 iterations, ~23 s/iter at 2500 steps / 8 workers):
- Iter 0: sample 12.8 s, policy update 4.4 s, eval 7.6 s; eval reward agent0/1 = 498.8 / 488.5, win rate 0.00
- Iter 2: eval reward 509.6 / 491.6; "training done!", best checkpoints saved each iter.

Observed (devants = CompetEvo evolution agents, 2 iterations, ~30 s/iter):
- Iter 0: sample 12.5 s, update 11.1 s, eval 7.9 s; eval reward 427.7 / 429.6
- Checkpoints under `tmp/run-to-goal-devants-v0/<ts>/models/agent_{0,1}/`. Full loop
  (env steps incl. morphology params, GNN policy update via torch_geometric, eval, ckpt) works.

Reward magnitudes at iter 0 (~400-500) are dominated by the dense move-to-goal shaping; win rate 0 is expected before any learning. Nothing to compare to the paper yet — that is what the sanity run below is for.

## Longer sanity run (in progress, backgrounded)

Their exact `config/run-to-goal-devants-v0.yaml` (min_batch_size 50000, max_epoch_num 1000 — will NOT finish; kill whenever, checkpoints are saved every epoch):

```bash
cd /workspace/competevo && PYTHONPATH=. MUJOCO_GL=egl OMP_NUM_THREADS=1 nohup \
  .venv/bin/python train.py --cfg config/run-to-goal-devants-v0.yaml \
  --use_cuda false --num_threads 24 > /workspace/competevo/sanity_run.log 2>&1 &
```

- Log: `/workspace/competevo/sanity_run.log`
- Run dir: `tmp/run-to-goal-devants-v0/20260810_211247/` (tb curves in `tb/`)
- Measured CPU footprint: ~5-6 cores effective (workers are I/O/queue-bound between bursts), well under the 30-core cap.
- First full iteration (measured): sampling 50,000 steps / 24 workers = 157.9 s, policy update 94.8 s, eval 17.0 s -> **4.5 min/iteration**; iter-0 eval rewards 428.0 / 428.5, win rate 0.00 (matches the smoke run's starting point).
- Compare `Agent_i gets eval reward` / `win rate` trajectory over the first ~50-100 epochs against paper Fig. curves (win-rate should leave 0 once the exploration curriculum (`termination_epoch: 200`) starts to bite).

## Paper-scale cost vs this pod

- Their configs: 50,000 env steps/iteration x 1000 iterations = 5e7 steps per 2-agent matchup, per seed. Paper runs every matchup in 3 arenas (run-to-goal, sumo/robo-sumo) x {ant,bug,spider} x {fixed,dev} pairings, plus asymmetric matchups.
- Their image trains on GPU (cu113); policy is small (MLP 128x128 + GNN), so sampling dominates either way.
- On this pod, CPU, 24 workers: measured 4.5 min/iteration -> a single 1000-epoch config is ~75 h (~3 days) CPU. Full paper grid (~12+ configs, multi-seed) is weeks-to-months on CPU — feasible only for 1-2 selected matchups; the rest needs a GPU window or more pods.

## Landmines for the next person

1. `six` missing from requirements.txt (ModuleNotFoundError deep in gym_compete).
2. box2d-py sdist needs `swig` on PATH (or drop box2d-py — unused).
3. Always run with `PYTHONPATH=.` from the repo root (no package install; `import competevo`, `import gym_compete`, `from config.config import ...` are all repo-relative).
4. torch 1.12 has no py3.11 wheels — you need py<=3.10; we matched their 3.8.
5. `train.py` defaults `--use_cuda true`; it falls back to CPU automatically if CUDA is unavailable, but pass `--use_cuda false` explicitly on shared GPU boxes.
6. Training never renders (gym.make without render_mode), so no GL setup is needed for training; MUJOCO_GL=egl is set only as insurance. `display.py` requires a real display as written.
7. `torch.set_default_dtype(torch.float64)` — everything runs in float64; that is intentional (their code), don't "fix" it.
