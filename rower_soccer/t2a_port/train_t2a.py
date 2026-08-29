"""D3 unit 3e: Transform2Act training, ported -- CPU design, batched execution.

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
           CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    cd /workspace/utmist-vc2-phase2
    PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.train_t2a \
        --cfg hopper_gpu_s2 --run port_s1 --seed 1 --worlds 64 --epochs 1000

With metrics and video (`set -a && . /workspace/.env && set +a` first, and
`MUJOCO_GL=egl` -- 2.2 ms/frame against osmesa's ~46):

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m \
        rower_soccer.t2a_port.train_t2a --cfg hopper_gpu_s2 --run port_s1 \
        --wandb --video-secs 900 --stop-file /tmp/stop_port_s1

**Both are decoration and neither may ever end a run.** A failed `wandb.init`,
a failed render, a full disk and a dead upload all print and continue; the
video draws from its own generator with every global stream snapshotted around
the event, so a seeded run with video is bit-identical to the same seed
without it. `gate_t2a_logging.py` asserts all of that, the last of it against
the pre-video file at git HEAD.

Every algorithmic choice is either theirs or is written down in `D3_HANDOFF.md`
under "M2 acceptance criterion, settled". The ones to re-read before changing
anything:

**Time limits (settled decision 3).** `hopper.py:179` folds
`control_nsteps < max_nsteps` into `done`, so the 1,000-step limit sets
`done = True` and their GAE bootstraps ZERO there -- the OPPOSITE of the
CompetEvo port's convention. This trainer bootstraps nowhere at all; see
`estimate_advantages`.

**Batch size (settled decision 4).** Their sampler delivers ~57,000-64,000
agent-steps per PPO iteration, not the nominal 50,000. One generation of
`--worlds` complete episodes at ~1,000 steps each is that batch, hence the
default of 64.

**Sampler shape (settled decision 5, ADJUSTED -- read this).** The decision as
written is "reset all worlds together, roll `T = k * max_ep_len` with per-world
auto-reset, so the batch is exactly `N*k` complete episodes and there are ZERO
rollout-boundary truncations." Two measurements stop that being literally
implementable:

1. **Auto-reset would restart the same BODY.** The design stages run on the CPU
   before the rollout, so a world that auto-resets mid-rollout begins a new
   execution episode on the morphology it already had -- whereas their
   `env.reset()` calls `reset_robot()` and draws a NEW design every episode
   (`hopper.py:310, 318`). Auto-reset is not a faithful sampler; it is a
   different algorithm.
2. **Episodes are not `max_ep_len` long, so `T = k * max_ep_len` does not yield
   `N*k` complete episodes.** At convergence they are 928 +/- 51
   (`gate_batched_exec.py`, 20 of their episodes); early in training they are
   tens of steps. With auto-reset a cut at fixed `T` truncates whatever episode
   each world is in.

So this trainer samples in GENERATIONS: design N worlds, roll them until every
one is `done`, stop. That delivers what decision 5 was FOR -- only complete
episodes, zero rollout-boundary truncations, and therefore no bootstrap
anywhere -- in every regime rather than only at convergence, and each episode
gets its own design as theirs does. The cost is the tail of a generation
running with some worlds already finished, logged every epoch as `gen_fill`.

**Eval (settled decision 6).** `n_eval` is logged beside every `exec_R_eps`,
because theirs is a mean over `num_threads` episodes and a different count is a
differently-biased curve.

**Where their wall-clock actually goes -- measured here, and it is not what
PORT_MAP section 6 says.** Over all 1,000 epochs of `hopper_gpu_s2`:

    block      T_sample   T_update   T_eval    total
    0-99          34.6       88.0      13.2    135.9
    500-599       28.5       92.3      11.0    131.8
    900-999       16.8       54.6       6.4     77.8

The PPO **update is 65-70%** of their wall-clock, not the 26% section 6
recorded from an early snapshot of a different run. The "~3.8x Amdahl ceiling
for a physics-only port" therefore describes a port that this one is not: the
update moves to dense fp32 on the GPU here too.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from rower_soccer.t2a_port.dense_policy import (DenseTransform2ActPolicy,
                                                DenseTransform2ActValue)
from rower_soccer.t2a_port.design_stage import DesignSpec, DesignWorld
from rower_soccer.t2a_port.two_stage_pipeline import (compile_design,
                                                       group_designs,
                                                       iter_groups)


# `hopper.py:if_use_transform_action` numbers the stages in this order, and
# their `get_perm_batch_design` sorts a minibatch permutation by it.
STAGE_RANK = {"skel_trans": 0, "attr_trans": 1, "execution": 2}

# The policy's three parameter groups, by attribute prefix. `_Tower` names them
# so a `named_parameters()` prefix match is exact.
TOWERS = ("skel", "attr", "control")


def stage_sorted_perm(perm, row_stage):
    """Their `get_perm_batch_design`: shuffle, then re-sort BY STAGE.

    `transform2act_agent.py:282` builds `np.array(inds[0] + inds[1] + inds[2])`
    by scanning the already-shuffled batch and bucketing each row by
    `state[2] = use_transform_action`, so the result is a stable sort of the
    shuffled order by stage. A 2,048-row minibatch is then a consecutive slice
    of that array and is stage-PURE except at the two stage boundaries.

    Gated in `gate_batch_design.py`.
    """
    return perm[torch.argsort(row_stage[perm], stable=True)]


# ---------------------------------------------------------------- video --
# `--video-secs` fires a best/median/worst clip on a wall-clock cadence, and
# `--wandb` sends the same numbers the text log already prints. BOTH are
# strictly optional decoration: every entry point below is wrapped so that a
# failed wandb init, a failed render, a full disk or a missing codec PRINTS and
# lets the run continue. Nothing added here may ever end a training run.


def _label(img, title, sub=""):
    """Burn a caption into a panel's top strip.

    A missing Pillow must cost the caption, not the video, so the bare image is
    returned rather than raising.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:                                   # noqa: BLE001
        return img
    im = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 25], fill=(0, 0, 0))
    d.text((4, 1), title, fill=(255, 255, 255))
    d.text((4, 13), sub, fill=(185, 185, 185))
    return np.asarray(im)


def _camera(model, name="track"):
    """Their `track` camera if the task's XML ships one, else a tracking free
    camera on the root body.

    `hopper.xml` and `ant.xml` both carry `track`, so this normally returns the
    name; the fallback exists so a task whose XML has no camera renders a
    creature running away rather than an empty frame.
    """
    import mujoco
    for i in range(model.ncam):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) == name:
            return name
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1 if model.nbody > 1 else 0
    cam.distance, cam.elevation, cam.azimuth = 4.0, -20.0, 90.0
    return cam


def render_panels(picks, path, panel=(320, 240), fps=40, pad=8):
    """Tile one panel per pick, left to right, and encode.

    **Each pick is rendered against its OWN design, recompiled from the XML the
    episode actually ran** (`pick["xml"]` is `DesignWorld.cur_xml_str` as it
    stood when the rollout was taken), not against a cached starting model.
    That is the whole point on D3 -- the body is what changes over training --
    and it is the failure this project has shipped twice, so the compiled
    model's body count and geom sizes are returned in `meta` for the gate to
    assert on rather than left to inspection.

    Recompiling is ~4 ms per design (`two_stage_pipeline`'s measurement) and
    the renderer is built per model because a model-bound `MjrContext` cannot
    be reused across topologies. Measured cost of that choice: 61
    create/`close()` cycles moved RSS by 34 MB total and left nothing on the
    card, so it is not the D1 leak -- there the object held a mujoco_warp
    `Data` and a captured CUDA graph, which `close()` has no counterpart for.

    `macro_block_size=1` because the tiled frame is `n*pw + (n+1)*pad` wide and
    that is not a multiple of 16 for most panel sizes; letting imageio resize
    instead would silently change what the labels say the frame is.

    Frames are STREAMED, one tiled row at a time, and the three renderers are
    held open together: buffering every panel first is 276 MB of host RAM at
    the 400-frame default and 2.7 GB at `--video-frames 1000 --video-panel 640
    480`. `eval_soccer2v2.render_grid` streams for the same reason.
    """
    import imageio
    import mujoco

    pw, ph = int(panel[0]), int(panel[1])
    ctx, meta = [], []
    try:
        for p in picks:
            model = compile_design(p["xml"])
            ren = mujoco.Renderer(model, ph, pw)
            ctx.append({"p": p, "model": model, "data": mujoco.MjData(model),
                        "ren": ren, "cam": _camera(model),
                        "q": np.asarray(p["qpos"]), "last": (-1, None)})
            meta.append({"bodies": int(model.nbody) - 1,
                         "geoms": int(model.ngeom), "nq": int(model.nq),
                         # A scalar fingerprint of the DESIGN's geometry. The
                         # gate asserts this moves when the design does.
                         "geom_size": round(
                             float(np.asarray(model.geom_size).sum()), 6)})
        if not ctx or min(c["q"].shape[0] for c in ctx) < 1:
            # Filtering here instead would desynchronise `meta` from `ctx` and
            # leak the dropped panel's renderer, so this is a hard error.
            raise RuntimeError("a pick carried no recorded states")
        n = len(ctx)
        T = max(c["q"].shape[0] for c in ctx)
        W = n * pw + (n + 1) * pad
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with imageio.get_writer(path, fps=fps, quality=8,
                                macro_block_size=1) as wr:
            for t in range(T):
                grid = np.full((ph + 2 * pad, W, 3), 20, np.uint8)
                for i, c in enumerate(ctx):
                    # A short episode HOLDS its last frame rather than going
                    # black, so "worst" reads as "fell over at step 12" and
                    # not as a bug. The held frame is rendered once and
                    # reused; re-rendering it every step would make the worst
                    # panel the most expensive one.
                    tt = min(t, c["q"].shape[0] - 1)
                    if c["last"][0] != tt:
                        c["data"].qpos[:] = c["q"][tt]
                        c["data"].qvel[:] = 0.0
                        mujoco.mj_forward(c["model"], c["data"])
                        c["ren"].update_scene(c["data"], camera=c["cam"])
                        c["last"] = (tt, _label(c["ren"].render(),
                                                c["p"]["title"],
                                                c["p"]["sub"]))
                    x = pad + i * (pw + pad)
                    grid[pad:pad + ph, x:x + pw] = c["last"][1]
                wr.append_data(grid)
    finally:
        for c in ctx:
            try:
                c["ren"].close()
            except Exception:                           # noqa: BLE001
                pass
    return T, meta


def _rng_snapshot(device):
    """Every global random stream the video path could touch."""
    st = {"np": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _rng_restore(st, device):
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


class Bucket:
    """One STAGE's transitions, padded to a common node count.

    Not one stage per topology, and not one stage per node count. Both earlier
    versions were tried and measured:

      * **by topology**: ~85 buckets in an untrained batch (17 topologies x 3
        stages, and the skeleton stage's topology set grows with `t`);
      * **by (stage, node count)**: 29 buckets measured on a real epoch-0 batch;
      * **by stage, padded**: 3.

    That matters only for the UPDATE, and it matters a lot there. A random
    2,048-row minibatch touches nearly every bucket, so a PPO gradient step
    costs one forward+backward per bucket: measured 382 ms per step at 29
    buckets against 27 ms for a single dense block of the same 2,048 rows.
    T_update was 130 s per epoch because of it -- against their 88 s.

    Padding is the hazard `dense_policy.py`'s docstring named: a zero row is
    not a neutral sample. It is kept out of `RunningNorm`'s statistics and out
    of the per-graph log-prob sum by `node_mask`, and it cannot reach a real
    node through the graph because its adjacency row and column are zero.
    `gate_dense_policy.py` checks that a padded block gives the same answers as
    the unpadded graphs, with a control that fails when the mask is ignored.
    """

    def __init__(self, stage):
        self.stage = stage
        self._blocks = []
        self.n = 0

    def add(self, obs, act, adj, ind):
        first = self.n
        self._blocks.append((obs, act, adj, ind))
        self.n += obs.shape[0]
        return first

    def finish(self, keep_rows):
        nmax = max(b[0].shape[1] for b in self._blocks)
        obs, act, adj, ind, mask = [], [], [], [], []
        for o, a, dj, nd in self._blocks:
            k, n = o.shape[0], o.shape[1]
            pad = nmax - n
            if pad:
                o = F.pad(o, (0, 0, 0, pad))
                a = F.pad(a, (0, 0, 0, pad))
                dj = F.pad(dj, (0, pad, 0, pad))
                nd = F.pad(nd, (0, pad))
            m = torch.zeros(k, nmax, device=o.device, dtype=torch.bool)
            m[:, :n] = True
            obs.append(o); act.append(a); adj.append(dj); ind.append(nd)
            mask.append(m)
        self.obs = torch.cat(obs)[keep_rows]
        self.act = torch.cat(act)[keep_rows]
        self.adj = torch.cat(adj)[keep_rows]
        self.ind = torch.cat(ind)[keep_rows]
        self.mask = torch.cat(mask)[keep_rows]
        self._blocks = None
        self.n = int(self.obs.shape[0])
        self.n_nodes = nmax

    def take(self, rows):
        return (self.obs[rows], self.adj[rows], self.ind[rows],
                self.act[rows], self.mask[rows])


class Batch:
    """One PPO iteration, flattened over worlds and time.

    Rows for worlds that had already finished their episode are recorded (the
    rollout steps every world in lockstep) and dropped in `finish()`; keeping
    them out of the loss without a per-step device sync is the reason for the
    two-phase build.
    """

    def __init__(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.buckets, self._key = [], {}
        self._bid, self._brow, self._keep = [], [], []
        self._r, self._m, self._lp, self._w, self._t = [], [], [], [], []

    def add(self, stage, adj, ind, obs, act, logp, reward, mask,
            worlds, times, keep):
        """`adj [K, n, n]` and `ind [K, n]` are per ROW; a caller that has one
        graph for the whole block must expand it, which is free (a view)."""
        key = stage
        if key not in self._key:
            self._key[key] = len(self.buckets)
            self.buckets.append(Bucket(stage))
        bid = self._key[key]
        first = self.buckets[bid].add(obs, act, adj, ind)
        k = obs.shape[0]
        self._bid.append(np.full(k, bid, dtype=np.int64))
        self._brow.append(np.arange(first, first + k, dtype=np.int64))
        self._r.append(reward)
        self._m.append(mask)
        self._lp.append(logp)
        self._w.append(np.asarray(worlds, dtype=np.int64))
        self._t.append(np.asarray(times, dtype=np.int64))
        self._keep.append(keep)

    def finish(self):
        bid = np.concatenate(self._bid)
        brow = np.concatenate(self._brow)
        keep = torch.cat(self._keep).cpu().numpy().astype(bool)
        new_row = np.full(bid.shape[0], -1, dtype=np.int64)
        for b_i, b in enumerate(self.buckets):
            sel = (bid == b_i) & keep
            rows = brow[sel]
            b.finish(torch.as_tensor(rows, device=self.device))
            new_row[sel] = np.arange(rows.shape[0])
        self.b_id = torch.as_tensor(bid[keep], device=self.device)
        self.b_row = torch.as_tensor(new_row[keep], device=self.device)
        kt = torch.as_tensor(keep, device=self.device)
        self.reward = torch.cat(self._r)[kt]
        self.mask = torch.cat(self._m)[kt]
        self.logp = torch.cat(self._lp)[kt]
        self.world = np.concatenate(self._w)[keep]
        self.time = np.concatenate(self._t)[keep]
        self.size = int(self.reward.shape[0])
        self.order = torch.as_tensor(
            np.lexsort((self.time, self.world)), device=self.device)
        self._r = self._m = self._lp = self._keep = None
        return self

    def _regroup(self, idx):
        bid = self.b_id[idx]
        for b_i in bid.unique().tolist():
            sel = (bid == b_i).nonzero(as_tuple=True)[0]
            yield self.buckets[b_i], sel, self.b_row[idx[sel]]

    def eval_value(self, value_net, idx):
        out = torch.zeros(idx.shape[0], device=self.device, dtype=self.dtype)
        for b, sel, rows in self._regroup(idx):
            obs, adj, _, _, m = b.take(rows)
            out = out.index_copy(0, sel,
                                 value_net(b.stage, obs, adj, m)[:, 0])
        return out

    def eval_logp(self, policy, idx):
        out = torch.zeros(idx.shape[0], device=self.device, dtype=self.dtype)
        for b, sel, rows in self._regroup(idx):
            obs, adj, ind, act, m = b.take(rows)
            out = out.index_copy(
                0, sel, policy.log_prob(b.stage, obs, adj, ind, act, m))
        return out


def estimate_advantages(batch, values, gamma, tau):
    """Their `estimate_advantages`, run in per-world time order.

    Theirs runs over one flat concatenation of every worker's memory and is
    safe only because each worker's memory ends on `mask = 0` (the corollary in
    the M1 notes). Ordering explicitly makes that structural instead of
    incidental. Every episode in a generation ends `done`, so `mask` is 0 at
    every episode end and **nothing is ever bootstrapped** -- decision 3.
    """
    o = batch.order
    r = batch.reward[o].double().cpu().numpy()
    m = batch.mask[o].double().cpu().numpy()
    v = values[o].double().cpu().numpy()
    adv = np.empty_like(r)
    prev_v = prev_a = 0.0
    for i in range(r.shape[0] - 1, -1, -1):
        delta = r[i] + gamma * prev_v * m[i] - v[i]
        prev_a = adv[i] = delta + gamma * tau * prev_a * m[i]
        prev_v = v[i]
    ret = v + adv
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    out_a = torch.zeros_like(values)
    out_r = torch.zeros_like(values)
    out_a[o] = torch.as_tensor(adv, device=values.device, dtype=values.dtype)
    out_r[o] = torch.as_tensor(ret, device=values.device, dtype=values.dtype)
    return out_a, out_r


class Trainer:

    def __init__(self, args):
        self.args = args
        self.cfg = yaml.safe_load(open(
            f"/workspace/Transform2Act/design_opt/cfg/{args.cfg}.yml"))
        self.spec = DesignSpec(self.cfg)
        self.device = torch.device(args.device)
        self.dtype = torch.float32 if args.fp32 else torch.float64
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.gen = torch.Generator(device=self.device).manual_seed(args.seed)

        self.init_xml = open(
            "/workspace/Transform2Act/assets/mujoco_envs/hopper.xml", "rb").read()
        probe = DesignWorld(self.spec, self.init_xml)
        self.attr_fixed_dim = self.spec.max_body_depth
        self.attr_design_dim = int(probe.design_cur_params.shape[-1])
        self.sim_obs_dim = 5
        self.state_dim = (self.attr_fixed_dim + self.sim_obs_dim
                          + self.attr_design_dim)

        self.policy = DenseTransform2ActPolicy(
            self.cfg["policy_specs"], self.attr_fixed_dim, self.sim_obs_dim,
            self.attr_design_dim, self.spec.skel_num_action,
            control_action_dim=1).to(self.device, self.dtype)
        self.value = DenseTransform2ActValue(
            self.cfg["value_specs"], self.state_dim).to(self.device, self.dtype)
        self.opt_p = torch.optim.Adam(self.policy.parameters(),
                                      lr=self.cfg.get("policy_lr", 5e-5))
        self.opt_v = torch.optim.Adam(self.value.parameters(),
                                      lr=self.cfg.get("value_lr", 3e-4))

        self.gamma = self.cfg.get("gamma", 0.995)
        self.tau = self.cfg.get("tau", 0.95)
        self.clip = self.cfg.get("clip_epsilon", 0.2)
        self.mini = self.cfg.get("mini_batch_size", 2048)
        self.n_opt = self.cfg.get("num_optim_epoch", 10)
        self.max_nsteps = self.cfg.get("done_condition", {}).get("max_nsteps",
                                                                 1000)
        # `agent_specs.batch_design` (TRUE in every hopper cfg they ship).
        # Their `update_policy` shuffles the batch and then RE-SORTS it by
        # stage (`transform2act_agent.py:282, get_perm_batch_design`), so a
        # 2,048-row minibatch is a consecutive slice of a stage-sorted array
        # and is therefore stage-PURE except at the two boundaries. This port
        # sliced a plain `randperm`, giving stage-MIXED minibatches -- which
        # is not a wash: it changes how many Adam steps each stage's tower
        # takes per epoch and how many rows each of those steps sees. See
        # D3_HANDOFF.md, "2026-08-27 (second): batch_design".
        _bd = getattr(self.args, "batch_design", None)
        self.batch_design = (_bd if _bd is not None else
                             bool(self.cfg.get("agent_specs", {})
                                  .get("batch_design", False)))
        # Sampling must not touch the RunningNorm statistics: theirs samples
        # under `to_test(*self.sample_modules)` and updates them only inside
        # the PPO forward passes (`agent.py:111`, `transform2act_agent.py:224`).
        # nn.Module defaults to train mode, so this line is load-bearing.
        self.policy.eval()
        self.value.eval()

        # Seeds the world-count heuristic. An untrained hopper survives a few
        # tens of steps; the estimate is replaced by measurement after the
        # first generation, so the constant only decides the first one.
        self.len_est = 40.0

        # Give memory back to the driver. mujoco_warp allocates a fresh
        # `Data` per topology group, and with a caching mempool the process's
        # resident set grows to the high-water mark over every group shape it
        # has ever seen -- measured climbing past 5 GB on a card shared with
        # three other jobs. A release threshold makes the pool return unused
        # blocks instead of hoarding them.
        if self.device.type == "cuda" and args.mempool_mb >= 0:
            import warp as wp
            wp.set_mempool_release_threshold("cuda:0",
                                             args.mempool_mb * 1024 * 1024)

        self.out = os.path.join(args.outdir, args.run)
        os.makedirs(os.path.join(self.out, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.out, "videos"), exist_ok=True)
        self.logf = open(os.path.join(self.out, "log_train.txt"), "a")
        self.best = -1e9

        # The video rollout draws from its OWN generator and every global
        # stream is snapshotted around the event (`_maybe_video`), so a run
        # with video is bit-identical to the same seed without it. That is not
        # a nicety: it is what lets `--video-secs 0` be a control rather than a
        # different experiment, and `gate_t2a_logging.py` asserts it against
        # the pre-video file at git HEAD.
        self._video_gen = torch.Generator(device=self.device).manual_seed(
            args.seed + 20259)
        self._next_video = None
        self._video_n = 0
        self.wb = None
        self._wb_fails = 0

    # ---------------------------------------------------------------- log --
    def log(self, s):
        print(s, flush=True)
        self.logf.write(s + "\n")
        self.logf.flush()

    # ------------------------------------------------------------- sample --
    def _graph(self, world):
        e = np.asarray(world.edges())
        n = len(world.robot.bodies)
        a = torch.zeros(1, n, n, device=self.device, dtype=self.dtype)
        a[0, e[0], e[1]] = 1.0
        i = torch.as_tensor(world.body_index(), device=self.device).unsqueeze(0)
        return a, i

    def design_phase(self, n_worlds, batch, mean_action, world_offset=0):
        worlds = [DesignWorld(self.spec, self.init_xml) for _ in range(n_worlds)]
        for t in range(self.spec.skel_transform_nsteps + 1):
            stage = ("skel_trans" if t < self.spec.skel_transform_nsteps
                     else "attr_trans")
            for key, idx in group_designs(worlds).items():
                obs = torch.as_tensor(np.stack([worlds[i].obs() for i in idx]),
                                      device=self.device, dtype=self.dtype)
                adj1, ind1 = self._graph(worlds[idx[0]])
                k = len(idx)
                with torch.no_grad():
                    act, lp = self.policy.act(
                        stage, obs, adj1.expand(k, -1, -1), ind1.expand(k, -1),
                        mean_action=mean_action, generator=self.gen)
                if batch is not None:
                    z = torch.zeros(k, device=self.device, dtype=self.dtype)
                    batch.add(stage, adj1.expand(k, -1, -1),
                              ind1.expand(k, -1), obs, act, lp,
                              z, z + 1.0, [world_offset + i for i in idx],
                              [t] * k,
                              torch.ones(k, dtype=torch.bool,
                                         device=self.device))
                a_np = act.detach().cpu().numpy()
                for r, i in enumerate(idx):
                    if stage == "skel_trans":
                        worlds[i].skel_step(a_np[r][:, -1])
                    else:
                        worlds[i].attr_step(a_np[r][:, 1:-1])
        return worlds

    def rollout(self, worlds, batch, mean_action, check_every=16,
                world_offset=0):
        """One generation on the GPU: roll every world until it is `done`.

        Groups are built LAZILY, one at a time. Building them all first put 112
        live mujoco_warp `Data` objects on the card (5.1 GB) for no benefit --
        the GPU rolls them sequentially anyway.
        """
        import time as _t
        seed = int(torch.randint(0, 2 ** 30, (1,), generator=self.gen,
                                 device=self.device).item())
        t_build0 = _t.time()
        it = iter_groups(
            worlds, self.spec, backend=self.args.backend,
            done_condition=self.cfg.get("done_condition"),
            reward_specs=self.cfg.get("reward_specs", {}),
            clip_qvel=self.cfg["obs_specs"].get("clip_qvel", False),
            seed=seed)
        offset = self.spec.skel_transform_nsteps + 1
        rets, lens, useful, rolled, bsteps = [], [], 0, 0, 0
        con_peak = 0.0
        n_groups = 0
        t_build = 0.0
        while True:
            t0 = _t.time()
            try:
                gi, g, idx = next(it)
            except StopIteration:
                break
            t_build += _t.time() - t0
            n_groups += 1
            K = g.n
            # `rows[r]` is the WORLD index sitting in row `r` of this group, so
            # per-transition bookkeeping stays attached to the world that
            # produced it rather than to a group row.
            rows = [world_offset + i for i in idx]
            # The execution env's dtype is the PHYSICS backend's, not the
            # trainer's: mujoco_warp is float32-only (its Data/Model arrays are
            # declared `float` = wp.float32, and `io.py:426` says so in as many
            # words), while `CompeteCpuBackend` is float64. So under `--fp64`
            # the policy is float64 and `obs`/`adj` arrive float32, and
            # `nn.Linear` refuses the mix -- which is why `--fp64` had never
            # run. Casting here makes the trainer's dtype the dtype of
            # everything downstream of the sim; `.to()` on a matching dtype is
            # a no-op, so the fp32 path is byte-identical to before.
            adj, ind = g.adj().to(self.dtype), g.ind()
            alive = torch.ones(K, dtype=torch.bool, device=self.device)
            ret = torch.zeros(K, device=self.device, dtype=self.dtype)
            ln = torch.zeros(K, device=self.device, dtype=self.dtype)
            obs = g.env.reset().to(self.dtype)
            for t in range(self.max_nsteps):
                with torch.no_grad():
                    act, lp = self.policy.act(
                        "execution", obs, adj, ind, mean_action=mean_action,
                        generator=self.gen)
                nobs, r, done, _ = g.env.step(act, auto_reset=False)
                nobs, r = nobs.to(self.dtype), r.to(self.dtype)
                live = alive.to(self.dtype)
                if batch is not None:
                    batch.add("execution", adj, ind, obs, act, lp,
                              r * live, (~done).to(self.dtype),
                              rows, [offset + t] * K, alive.clone())
                ret += r * live
                ln += live
                alive = alive & (~done)
                obs = nobs
                rolled += K
                bsteps += 1
                if (t + 1) % check_every == 0:
                    dead = (~alive).nonzero(as_tuple=True)[0]
                    if int(dead.numel()) == K:
                        break
                    if int(dead.numel()):
                        g.env._write_initial(dead, add_noise=False)
                        g.env.backend.forward()
                        obs = g.env.obs().to(self.dtype)
            rets += ret.tolist()
            lens += ln.tolist()
            useful += int(ln.sum().item())
            used, cap = g.check_contact_capacity()
            if used is not None:
                assert used < cap, (
                    f"contact buffer full ({used}/{cap}) on topology "
                    f"{g.key}: contacts were dropped, which makes the physics "
                    f"wrong without making it fail")
                con_peak = max(con_peak, used / cap)
            del g, adj, ind, obs, act
        return {"returns": rets, "lens": lens, "steps": useful,
                "rolled": rolled, "groups": n_groups,
                # The cost of a generation is NOT the agent-steps collected but
                # the number of BATCHED steps: each group is rolled until its
                # LONGEST-surviving world is done, so one lucky world in a
                # two-world group costs as many launches as a thousand-world
                # group. This is the number to watch when the port looks slow.
                "batched_steps": bsteps, "con_peak": con_peak,
                "compile_s": 0.0, "build_s": t_build}

    def sample(self, mean_action, record, budget=None, n_worlds=None):
        """One PPO iteration's worth of sampling, in GENERATIONS.

        `budget` is in AGENT-STEPS, because that is what their sampler
        collects (`while logger.num_steps < min_batch_size`) and what settled
        decision 4 fixes at ~57,000-64,000. The number of worlds per
        generation is derived from the episode length seen so far, so early in
        training -- when a hopper survives ~30 steps and their batch holds
        ~1,700 episodes -- one generation carries ~1,900 worlds, and at
        convergence it carries ~62. That is the whole reason the world count
        is not a constant: a fixed 64 worlds would have made the early batches
        1,800 steps instead of 57,000, which is a thirtieth of the gradient
        signal their run gets.

        Like theirs, this OVERSHOOTS and never truncates: a generation is
        always run out in full.
        """
        batch = Batch(self.device, self.dtype) if record else None
        rets, lens, gens = [], [], 0
        steps = rolled = groups = rolled_worlds = bsteps = 0
        con_peak = 0.0
        t_compile = t_build = 0.0
        while True:
            if n_worlds is not None:
                n = n_worlds
            else:
                left = budget - steps
                n = int(np.clip(np.ceil(left / max(self.len_est, 1.0)),
                                self.args.min_worlds, self.args.max_worlds))
            worlds = self.design_phase(n, batch, mean_action,
                                       world_offset=rolled_worlds)
            info = self.rollout(worlds, batch, mean_action,
                                world_offset=rolled_worlds)
            rolled_worlds += n
            rets += info["returns"]
            lens += info["lens"]
            steps += info["steps"]
            rolled += info["rolled"]
            groups += info["groups"]
            bsteps += info["batched_steps"]
            con_peak = max(con_peak, info["con_peak"])
            t_compile += info["compile_s"]
            t_build += info["build_s"]
            gens += 1
            # ONLY the training pass may move the estimate. The eval pass runs
            # mean actions, whose episodes are a different length entirely
            # (21.2 against a training 31.8, measured at epoch 14 of
            # `port_s1`), and letting it write here means the world count for
            # the next TRAINING generation is sized from the wrong
            # distribution. `record` is what distinguishes the two passes.
            if lens and record:
                self.len_est = float(np.mean(lens[-max(len(info["lens"]), 1):]))
            if n_worlds is not None or steps >= budget:
                break
        if batch is not None:
            batch.finish()
        return batch, {"returns": rets, "lens": lens, "steps": steps,
                       "rolled": rolled, "groups": groups, "gens": gens,
                       "batched_steps": bsteps, "con_peak": con_peak,
                       "compile_s": t_compile, "build_s": t_build}

    # ------------------------------------------------------------- update --
    def update(self, batch):
        idx_all = torch.arange(batch.size, device=self.device)
        self.policy.eval()
        self.value.eval()
        with torch.no_grad():
            values = batch.eval_value(self.value, idx_all)
        adv, ret = estimate_advantages(batch, values, self.gamma, self.tau)
        fixed_lp = batch.logp                # same weights, same norm stats
        self.policy.train()
        self.value.train()
        n_mb = batch.size // self.mini
        stats = {"v_loss": 0.0, "p_loss": 0.0, "n": 0}
        # Per-tower gradient norm and end-to-end parameter delta. Until
        # 2026-08-28 the port trained for 1,000 epochs with three towers whose
        # gradient was identically zero and nothing in any log said so; the
        # inference "the port's gains are design-side" could not be checked
        # because nobody was measuring per tower. Cost is one clone of the
        # policy's parameters per epoch and a norm per minibatch.
        p_before = {n: q.detach().clone()
                    for n, q in self.policy.named_parameters()}
        gsum = {t: 0.0 for t in TOWERS}
        gcnt = {t: 0 for t in TOWERS}
        row_stage = None
        if self.batch_design:
            rank = torch.as_tensor([STAGE_RANK[b.stage] for b in batch.buckets],
                                   device=self.device)
            row_stage = rank[batch.b_id]
        for _ in range(self.n_opt):
            perm = torch.randperm(batch.size, device=self.device)
            if row_stage is not None:
                perm = stage_sorted_perm(perm, row_stage)
            for i in range(n_mb):
                idx = perm[i * self.mini:(i + 1) * self.mini]
                v = batch.eval_value(self.value, idx)
                v_loss = (v - ret[idx]).pow(2).mean()
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                self.opt_v.step()

                lp = batch.eval_logp(self.policy, idx)
                ratio = torch.exp(lp - fixed_lp[idx])
                a = adv[idx]
                surr = -torch.min(ratio * a,
                                  ratio.clamp(1 - self.clip, 1 + self.clip) * a
                                  ).mean()
                self.opt_p.zero_grad(set_to_none=True)
                surr.backward()
                # BEFORE the clip: the clip is global, so a clipped norm hides
                # which tower produced it.
                for t in TOWERS:
                    g = sum(float(q.grad.pow(2).sum())
                            for n, q in self.policy.named_parameters()
                            if q.grad is not None and n.startswith(t + "."))
                    gsum[t] += g ** 0.5
                    gcnt[t] += 1
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.opt_p.step()
                stats["v_loss"] += float(v_loss)
                stats["p_loss"] += float(surr)
                stats["n"] += 1
        self.policy.eval()
        self.value.eval()
        stats["n_minibatch"] = n_mb
        for t in TOWERS:
            stats[f"g_{t}"] = round(gsum[t] / max(gcnt[t], 1), 5)
            stats[f"d_{t}"] = round(max(
                [float((q.detach() - p_before[n]).abs().max())
                 for n, q in self.policy.named_parameters()
                 if n.startswith(t + ".")] or [0.0]), 7)
            # The number the port_s1 bug would have shown as 0: how many of
            # this tower's tensors did not move at all this epoch.
            stats[f"frozen_{t}"] = sum(
                1 for n, q in self.policy.named_parameters()
                if n.startswith(t + ".")
                and float((q.detach() - p_before[n]).abs().max()) == 0.0)
        return stats

    # ------------------------------------------------------------- wandb --
    def wandb_init(self):
        """Attach the run to wandb, or print why not and carry on.

        `id = run name` with `resume="allow"` so restarting `--run port_s1`
        REATTACHES to the same wandb run instead of opening a second one beside
        it -- the same convention `scripts/wandb_ship.py` uses, which is what
        lets a natively-logged port run and a shipped reference log land in one
        workspace.
        """
        a = self.args
        if not getattr(a, "wandb", False):
            return
        try:
            import wandb
            cfg = dict(vars(a))
            cfg["batch_design_effective"] = self.batch_design
            cfg["dtype"] = str(self.dtype)
            cfg["source"] = "train_t2a"
            self.wb = wandb.init(
                project=getattr(a, "wandb_project", "creature-soccer"),
                name=a.run, id=a.run, resume="allow",
                tags=list(getattr(a, "wandb_tags", None) or ["D3", "port"]),
                config=cfg)
            # Same two lines as the shipper: the x-axis is the EPOCH, which is
            # what makes port and reference curves overlay.
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
            self.log(f"wandb: {self.wb.url}")
            # A REATTACHED run is not a resumed TRAINER: this trainer has no
            # `--resume`, so it always starts at epoch 0, and wandb refuses a
            # step behind the one the run already reached ("Tried to log to
            # step 0 that is less than the current step N ... this data will
            # be ignored"). Reusing a run name therefore drops every epoch up
            # to N -- silently, in a warning buried in the client's own
            # output. Said plainly here instead, because a half-populated
            # chart that nobody knows is half-populated is the failure this
            # whole change exists to stop.
            n = int(getattr(self.wb, "step", 0) or 0)
            if n > 0:
                self.log(f"wandb: this run id already reached step {n}; this "
                         f"process starts at epoch 0, so wandb will DROP "
                         f"epochs 0-{n} as non-monotonic. Use a new --run "
                         f"name for a new experiment.")
        except Exception as e:                          # noqa: BLE001
            self.log(f"wandb DISABLED ({e!r}) -- training continues")
            self.wb = None

    def wandb_log(self, epoch, payload):
        """Never raises. Gives up after five consecutive failures so a dead
        backend cannot turn every epoch into a stack trace."""
        if self.wb is None:
            return
        try:
            self.wb.log({**payload, "epoch": epoch}, step=epoch)
            self._wb_fails = 0
        except Exception as e:                          # noqa: BLE001
            self._wb_fails += 1
            self.log(f"wandb log FAILED at epoch {epoch} ({e!r}) -- "
                     f"training continues")
            if self._wb_fails >= 5:
                self.log("wandb: five consecutive failures -- disabling")
                self.wb = None

    @staticmethod
    def metric_payload(mon, side):
        """The text log's two lines, as wandb keys.

        The prefixes are `scripts/wandb_ship.py`'s: the monitor line's fields
        under `t2a/` and the JSON sidecar's under `port/`. A run logged here
        and the same run shipped from its text file therefore write the SAME
        keys, so they overlay instead of forming two half-populated panels.
        """
        d = {f"t2a/{k}": float(v) for k, v in mon.items()}
        for k, v in side.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            d[f"port/{k}"] = float(v)
        return d

    # ------------------------------------------------------------- video --
    @torch.no_grad()
    def video_rollout(self, n_worlds, max_frames, mean_action=False,
                      max_steps=0):
        """One generation with the CURRENT policy, recording qpos per world.

        Deliberately the SAME code path as `rollout` -- design on the CPU with
        the live policy, group by topology, roll on the GPU -- so the clip
        shows the design distribution the sampler is producing at this epoch
        rather than a re-derivation of it. Actions are SAMPLED by default, not
        mean: with mean actions every world draws the identical skeleton and
        the three panels would be one morphology three times, which on D3
        throws away the thing being studied.

        Only the first `max_frames` states are kept; the episode is run out in
        full regardless, because the ranking is by EPISODE reward and a rollout
        cut at the frame cap would rank a truncated return. `max_steps` exists
        for the operator who decides that trade is worth making anyway -- see
        `--video-max-steps`, which is off by default.
        """
        worlds = self.design_phase(n_worlds, None, mean_action)
        seed = int(torch.randint(0, 2 ** 30, (1,), generator=self.gen,
                                 device=self.device).item())
        it = iter_groups(
            worlds, self.spec, backend=self.args.backend,
            done_condition=self.cfg.get("done_condition"),
            reward_specs=self.cfg.get("reward_specs", {}),
            clip_qvel=self.cfg["obs_specs"].get("clip_qvel", False),
            seed=seed)
        recs = [None] * n_worlds
        for _gi, g, idx in it:
            K = g.n
            adj, ind = g.adj().to(self.dtype), g.ind()
            alive = torch.ones(K, dtype=torch.bool, device=self.device)
            ret = torch.zeros(K, device=self.device, dtype=self.dtype)
            ln = torch.zeros(K, device=self.device, dtype=self.dtype)
            obs = g.env.reset().to(self.dtype)
            qs = [g.env.backend.qpos.clone()]
            for t in range(min(max_steps, self.max_nsteps) if max_steps > 0
                           else self.max_nsteps):
                act, _ = self.policy.act("execution", obs, adj, ind,
                                         mean_action=mean_action,
                                         generator=self.gen)
                nobs, r, done, _ = g.env.step(act, auto_reset=False)
                nobs, r = nobs.to(self.dtype), r.to(self.dtype)
                live = alive.to(self.dtype)
                ret += r * live
                ln += live
                alive = alive & (~done)
                obs = nobs
                if len(qs) < max_frames:
                    # Kept on the device and stacked once at the end: a
                    # per-step `.cpu()` would sync the GPU 1,000 times and make
                    # the measured video cost an artefact of the measurement.
                    qs.append(g.env.backend.qpos.clone())
                if (t + 1) % 16 == 0:
                    dead = (~alive).nonzero(as_tuple=True)[0]
                    if int(dead.numel()) == K:
                        break
                    if int(dead.numel()):
                        g.env._write_initial(dead, add_noise=False)
                        g.env.backend.forward()
                        obs = g.env.obs().to(self.dtype)
            Q = torch.stack(qs).float().cpu().numpy()          # [T, K, nq]
            rl, ll = ret.tolist(), ln.tolist()
            for row, i in enumerate(idx):
                # Trim to the world's OWN episode: after a world dies the
                # batch keeps stepping it (and `_write_initial` snaps it back
                # to the start pose), so an untrimmed panel would show a second
                # phantom episode.
                keep = int(max(1, min(int(ll[row]) + 1, Q.shape[0])))
                recs[i] = {"ret": float(rl[row]), "len": int(ll[row]),
                           "qpos": Q[:keep, row].copy(),
                           "xml": worlds[i].cur_xml_str,
                           "bodies": len(worlds[i].robot.bodies)}
            del g, adj, ind, obs, act
        return [r for r in recs if r is not None]

    def render_best_median_worst(self, path):
        """Roll `--video-worlds` episodes, film the best, median and worst.

        Ranked by EPISODE REWARD. Unlike D1's self-play -- where the team
        reward is zero-sum and ranking by it would rank noise -- this task is
        single-agent, so reward is exactly the eval metric and is the right
        key.
        """
        a = self.args
        n = int(getattr(a, "video_worlds", 12))
        recs = self.video_rollout(
            n, int(getattr(a, "video_frames", 400)),
            mean_action=bool(getattr(a, "video_mean_action", False)),
            max_steps=int(getattr(a, "video_max_steps", 0)))
        if not recs:
            raise RuntimeError("video rollout produced no episodes")
        order = sorted(range(len(recs)), key=lambda i: recs[i]["ret"])
        picks, stats = [], {}
        for label, k in (("best", len(order) - 1),
                         ("median", len(order) // 2), ("worst", 0)):
            r = recs[order[k]]
            picks.append({**r,
                          "title": f"{label}  R={r['ret']:.1f}",
                          "sub": f"{r['len']} steps   {r['bodies']} bodies"})
            stats[f"{label}_R"] = round(r["ret"], 3)
            stats[f"{label}_len"] = r["len"]
            stats[f"{label}_bodies"] = r["bodies"]
        frames, meta = render_panels(
            picks, path, panel=tuple(getattr(a, "video_panel", (320, 240))),
            fps=int(getattr(a, "video_fps", 40)))
        stats["episodes"] = len(recs)
        stats["mean_R"] = round(float(np.mean([r["ret"] for r in recs])), 3)
        stats["mean_len"] = round(float(np.mean([r["len"] for r in recs])), 1)
        stats["mean_bodies"] = round(
            float(np.mean([r["bodies"] for r in recs])), 2)
        stats["frames"] = int(frames)
        stats["mb"] = round(os.path.getsize(path) / 1e6, 2)
        # The fingerprint of what was actually COMPILED and drawn. `gate_t2a_
        # logging.py` asserts it moves when the design does, which is the check
        # that a cached starting model would fail.
        for label, m in zip(("best", "median", "worst"), meta):
            stats[f"{label}_geom_size"] = m["geom_size"]
            # `compile_design` puts the world body at index 0, so this is the
            # rendered model's node count and must equal the DesignWorld's.
            # The gate asserts the two agree; a cached model would hold this
            # constant while `{label}_bodies` moved.
            stats[f"{label}_model_bodies"] = m["bodies"]
        return path, stats

    def maybe_video(self, epoch):
        """Fire a clip if the cadence is due. Never raises.

        The FIRST clip fires at the first epoch boundary rather than after a
        full `--video-secs`, because on D3 the starting body and how it changes
        is the experiment, and waiting 15 minutes to find out that the render
        is broken is 15 minutes wasted.

        **The cadence is a floor, not the schedule.** Measured on this task
        (RTX 4000 Ada, MPS, four other clients): one event costs 15-21 s while
        episodes are ~30 steps, and **464 s** once they reach the 1,000-step
        limit -- 52% of a 900 s cadence, and essentially all of it the rollout
        (455 s of the 464). The cost is neither the panels nor the world count
        but the SEQUENTIAL steps of one generation: designs differ, so 12
        worlds are ~12 topology groups and the GPU rolls them one after
        another. It grows ~30x between the start of a run and convergence,
        which no fixed cadence can be right for at both ends.
        So the interval is `max(--video-secs, cost / --video-budget-frac)`:
        the event is capped at a fixed FRACTION of wall clock, it stretches
        itself when it gets expensive, and the number it stretched to is
        printed. Nothing is truncated to achieve that -- `--video-worlds` and
        `--video-max-steps` are there for an operator who wants the cadence
        back instead.
        """
        a = self.args
        secs = float(getattr(a, "video_secs", 0.0) or 0.0)
        if secs <= 0:
            return
        if self._next_video is not None and time.time() < self._next_video:
            return
        # Stamped BEFORE the attempt so a render that fails every time costs
        # one attempt per cadence, not one per epoch.
        self._next_video = time.time() + secs
        path = os.path.join(self.out, "videos", f"epoch_{epoch:05d}.mp4")
        # Freeze every global stream: whatever the video draws, training's
        # sequence is unchanged. See `_video_gen` in `__init__`.
        snap = _rng_snapshot(self.device)
        saved_gen, self.gen = self.gen, self._video_gen
        try:
            t0 = time.time()
            _, st = self.render_best_median_worst(path)
            dt = time.time() - t0
            frac = float(getattr(a, "video_budget_frac", 0.1) or 0.0)
            interval = secs if frac <= 0 else max(secs, dt / frac)
            self._next_video = time.time() + interval
            self._video_n += 1
            # PRINTED, not estimated. The cadence knob is only useful if
            # someone can see what it is spending, and on this task the cost
            # grows ~30x with episode length, so the early number is not the
            # late one.
            self.log(f"  video {path} {json.dumps(st)} cost={dt:.1f}s "
                     f"({100 * dt / secs:.1f}% of the {secs:.0f}s cadence"
                     + (f"; next in {interval:.0f}s to stay under "
                        f"{100 * frac:.0f}% of wall clock)"
                        if interval > secs + 1 else ")"))
            if self.wb is not None:
                try:
                    import wandb
                    self.wandb_log(epoch, {
                        # No `fps=`: wandb warns (and ignores it) when the
                        # argument is a PATH -- the mp4 already carries its
                        # own frame rate, set by `--video-fps`.
                        "video/best_median_worst": wandb.Video(path,
                                                               format="mp4"),
                        "video/cost_s": dt,
                        "video/cost_frac_of_cadence": dt / secs,
                        "video/next_interval_s": interval,
                        **{f"video/{k}": v for k, v in st.items()
                           if isinstance(v, (int, float))}})
                except Exception as e:                  # noqa: BLE001
                    self.log(f"  video wandb upload FAILED ({e!r}) -- "
                             f"training continues")
        except Exception as e:                          # noqa: BLE001
            import traceback
            self.log(f"  video FAILED ({e!r}) -- training continues")
            self.log("  " + traceback.format_exc().rstrip().replace(
                "\n", "\n  "))
        finally:
            self.gen = saved_gen
            _rng_restore(snap, self.device)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    # -------------------------------------------------------------- train --
    def save(self, epoch, tag=None):
        path = os.path.join(self.out, "models",
                            tag or f"epoch_{epoch + 1:04d}.p")
        torch.save({"policy": self.policy.state_dict(),
                    "value": self.value.state_dict(),
                    "epoch": epoch, "cfg": self.cfg,
                    "attr_fixed_dim": self.attr_fixed_dim,
                    "attr_design_dim": self.attr_design_dim,
                    "sim_obs_dim": self.sim_obs_dim}, path)

    def train(self, epochs):
        """`--stop-file` is not a convenience.

        Under MPS, killing one CUDA client has already corrupted live
        survivors on this pod twice. A long run therefore needs a way to be
        ENDED rather than killed: `touch <stop-file>` and the trainer saves and
        exits at the next epoch boundary. Do not wrap this process in
        `timeout`, either -- `timeout` forwards its own SIGTERM to the child,
        so killing the wrapper kills the CUDA process too. (Learned the
        expensive way on 2026-08-27; the other four MPS clients survived that
        one, which is luck, not evidence.)
        """
        a = self.args
        # Positive evidence in the run's own log of which arm this is. The
        # A/B on `agent_specs.batch_design` is otherwise invisible after the
        # fact -- the argv does not carry it when the default follows the cfg.
        self.log(f"run {a.run}  cfg {a.cfg}  seed {a.seed}  "
                 f"batch_design {self.batch_design} "
                 f"(cfg agent_specs.batch_design "
                 f"{self.cfg.get('agent_specs', {}).get('batch_design', False)}"
                 f", --batch-design {getattr(a, 'batch_design', None)})  "
                 f"dtype {self.dtype}")
        self.wandb_init()
        for epoch in range(epochs):
            if a.stop_file and os.path.exists(a.stop_file):
                self.log(f"stop file {a.stop_file} present -- saving and "
                         f"exiting cleanly at epoch {epoch}")
                self.save(epoch - 1, "stopped.p")
                break
            t0 = time.time()
            batch, tr = self.sample(False, True, budget=a.batch_steps)
            t1 = time.time()
            st = self.update(batch)
            t2 = time.time()
            _, ev = self.sample(True, False, n_worlds=a.eval_worlds)
            t3 = time.time()

            train_R_eps = float(np.mean(tr["returns"]))
            exec_R_eps = float(np.mean(ev["returns"]))
            fill = tr["steps"] / max(tr["rolled"], 1)
            train_len = float(np.mean(tr["lens"]))
            eval_len = float(np.mean(ev["lens"]))
            train_R = train_R_eps / max(train_len, 1)
            exec_R = exec_R_eps / max(eval_len, 1)
            row = (f"{epoch}\tT_sample {t1 - t0:.2f}\tT_update {t2 - t1:.2f}\t"
                   f"T_eval {t3 - t2:.2f}\ttrain_R {train_R:.2f}\t"
                   f"train_R_eps {train_R_eps:.2f}\t"
                   f"exec_R {exec_R:.2f}\t"
                   f"exec_R_eps {exec_R_eps:.2f}\t{a.run}")
            self.log(row)
            side = {
                "epoch": epoch, "batch_steps": tr["steps"],
                "rolled_steps": tr["rolled"], "gen_fill": round(fill, 3),
                "n_train_eps": len(tr["returns"]), "n_eval": len(ev["returns"]),
                "gens": tr["gens"], "len_est": round(self.len_est, 1),
                "eval_len": round(float(np.mean(ev["lens"])), 1),
                "train_len": round(float(np.mean(tr["lens"])), 1),
                "groups": tr["groups"], "batched_steps": tr["batched_steps"],
                "eval_batched_steps": ev["batched_steps"],
                "buckets": len(batch.buckets),
                "contact_buf_peak": round(tr["con_peak"], 3),
                "compile_s": round(tr["compile_s"], 2),
                "build_s": round(tr["build_s"], 2),
                "minibatches": st["n_minibatch"],
                **{k: st[k] for k in st if k[:2] in ("g_", "d_")
                   or k.startswith("frozen_")},
                "v_loss": round(st["v_loss"] / max(st["n"], 1), 4),
                "p_loss": round(st["p_loss"] / max(st["n"], 1), 5),
                "steps_per_s_sample": round(tr["rolled"] / (t1 - t0)),
                "gpu_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20)
                if self.device.type == "cuda" else 0}
            self.log("  " + json.dumps(side))
            # Implied episode length, logged EXPLICITLY and on the port's
            # convention. Neither codebase prints length; both print the two
            # numbers whose ratio is it, and the ratio does not mean the same
            # thing on the two sides. The port's `train_R` divides by EXECUTION
            # steps only, so `train_R_eps / train_R` is the episode length as
            # this trainer counts it, and equals the `train_len` above.
            # THEIRS divides by every logged step, and
            # `khrylib/rl/agents/agent.py:70` logs the 5 skeleton and 1
            # attribute steps too (reward 0 each), so their ratio is longer by
            # `skel_transform_nsteps + 1`. `scripts/wandb_ship.py` subtracts
            # that from reference logs; `train_ep_len_all_stages` is what
            # theirs would print for this epoch, kept so the two conventions
            # are visible side by side rather than argued about. An agent lost
            # hours to this off-by-six chasing a phantom physics discrepancy.
            design_steps = self.spec.skel_transform_nsteps + 1
            mon = {"T_sample": t1 - t0, "T_update": t2 - t1,
                   "T_eval": t3 - t2, "train_R": train_R,
                   "train_R_eps": train_R_eps, "exec_R": exec_R,
                   "exec_R_eps": exec_R_eps,
                   "train_ep_len": train_R_eps / train_R if train_R else 0.0,
                   "train_ep_len_all_stages":
                       (train_R_eps / train_R if train_R else 0.0)
                       + design_steps,
                   "exec_ep_len": exec_R_eps / exec_R if exec_R else 0.0}
            self.wandb_log(epoch, self.metric_payload(mon, side))
            self.maybe_video(epoch)
            if exec_R_eps > self.best:
                self.best = exec_R_eps
                self.save(epoch, "best.p")
            if (epoch + 1) % a.save_interval == 0:
                self.save(epoch)
            del batch
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.log("training done!")
        if self.wb is not None:
            try:
                self.wb.finish()
            except Exception as e:                      # noqa: BLE001
                self.log(f"wandb finish FAILED ({e!r})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="hopper_gpu_s2")
    p.add_argument("--run", default="port_smoke")
    p.add_argument("--outdir", default="runs/t2a_port")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-steps", type=int, default=57344,
                   help="agent-steps per PPO iteration; settled decision 4 is "
                        "~57,000-64,000, what their 16-thread sampler delivers")
    p.add_argument("--min-worlds", type=int, default=32)
    p.add_argument("--max-worlds", type=int, default=2048)
    p.add_argument("--eval-worlds", type=int, default=16,
                   help="theirs averages exec_R_eps over num_threads episodes")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", default="warp", choices=["warp", "cpu"])
    p.add_argument("--batch-design", dest="batch_design",
                   action="store_true", default=None,
                   help="stage-sort each minibatch permutation, as their "
                        "`agent_specs.batch_design` does; default follows the "
                        "cfg, which sets it true for every hopper run")
    p.add_argument("--no-batch-design", dest="batch_design",
                   action="store_false")
    p.add_argument("--fp32", action="store_true", default=True)
    p.add_argument("--fp64", dest="fp32", action="store_false")
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--mempool-mb", type=int, default=512,
                   help="warp mempool release threshold; -1 leaves warp's "
                        "default, which hoards to the high-water mark")
    p.add_argument("--stop-file", default="",
                   help="touch this path to end the run cleanly at the next "
                        "epoch boundary; NEVER kill a CUDA process under MPS")
    p.add_argument("--wandb", action="store_true",
                   help="mirror the two log lines into wandb. OFF by default, "
                        "and the training path is bit-identical without it")
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--video-secs", type=float, default=900.0,
                   help="wall-clock cadence for the best/median/worst clip; "
                        "0 disables. The FIRST clip fires at the first epoch "
                        "boundary regardless, not after a full cadence")
    p.add_argument("--video-worlds", type=int, default=8,
                   help="episodes rolled and ranked per video event. This is "
                        "the cost knob: every design is its own topology group "
                        "and the GPU rolls groups SEQUENTIALLY, so the event "
                        "costs about (worlds x episode length) steps -- "
                        "measured 38 s per 1,000-step episode on this card. "
                        "Panels and frames are nearly free by comparison "
                        "(8.5 s of the 464 s measured at convergence). 8 is a "
                        "compromise: enough episodes for a meaningful median, "
                        "~310 s per event at convergence")
    p.add_argument("--video-budget-frac", type=float, default=0.1,
                   help="hard ceiling on the share of wall clock the video may "
                        "take: the next event is scheduled at "
                        "max(--video-secs, cost / this). Measured on hopper, "
                        "one event costs ~20 s early and ~464 s at 1,000-step "
                        "episodes, so a fixed cadence cannot be right at both "
                        "ends of a run. 0 disables the stretch and honours "
                        "--video-secs literally")
    p.add_argument("--video-max-steps", type=int, default=0,
                   help="cap the video rollout at this many execution steps "
                        "(0 = the env's own limit). This is the knob for "
                        "getting the cadence back at convergence, and it is "
                        "off by default because the ranking is by EPISODE "
                        "reward: a capped rollout ranks a truncated return")
    p.add_argument("--video-frames", type=int, default=400,
                   help="frames kept per panel (400 = 10 s at 40 fps). The "
                        "episode is still run out IN FULL -- the ranking is by "
                        "episode reward and must not see a truncated return")
    p.add_argument("--video-panel", type=int, nargs=2, default=(320, 240),
                   help="per-panel WIDTH HEIGHT; capped by the model's "
                        "offscreen framebuffer, 640x480 for the hopper XML")
    p.add_argument("--video-fps", type=int, default=40)
    p.add_argument("--video-mean-action", action="store_true",
                   help="film mean actions. Off by default because with mean "
                        "actions every world draws the SAME skeleton and the "
                        "three panels become one morphology three times")
    args = p.parse_args()
    Trainer(args).train(args.epochs)


if __name__ == "__main__":
    main()
