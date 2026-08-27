"""D3 unit 3e: Transform2Act training, ported -- CPU design, batched execution.

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
           CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    cd /workspace/utmist-vc2-phase2
    PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.train_t2a \
        --cfg hopper_gpu_s2 --run port_s1 --seed 1 --worlds 64 --epochs 1000

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
from rower_soccer.t2a_port.two_stage_pipeline import (group_designs,
                                                       iter_groups)


# `hopper.py:if_use_transform_action` numbers the stages in this order, and
# their `get_perm_batch_design` sorts a minibatch permutation by it.
STAGE_RANK = {"skel_trans": 0, "attr_trans": 1, "execution": 2}


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
        self.logf = open(os.path.join(self.out, "log_train.txt"), "a")
        self.best = -1e9

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
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.opt_p.step()
                stats["v_loss"] += float(v_loss)
                stats["p_loss"] += float(surr)
                stats["n"] += 1
        self.policy.eval()
        self.value.eval()
        stats["n_minibatch"] = n_mb
        return stats

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
            row = (f"{epoch}\tT_sample {t1 - t0:.2f}\tT_update {t2 - t1:.2f}\t"
                   f"T_eval {t3 - t2:.2f}\ttrain_R {train_R_eps / max(np.mean(tr['lens']), 1):.2f}\t"
                   f"train_R_eps {train_R_eps:.2f}\t"
                   f"exec_R {exec_R_eps / max(np.mean(ev['lens']), 1):.2f}\t"
                   f"exec_R_eps {exec_R_eps:.2f}\t{a.run}")
            self.log(row)
            self.log("  " + json.dumps({
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
                "v_loss": round(st["v_loss"] / max(st["n"], 1), 4),
                "p_loss": round(st["p_loss"] / max(st["n"], 1), 5),
                "steps_per_s_sample": round(tr["rolled"] / (t1 - t0)),
                "gpu_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20)
                if self.device.type == "cuda" else 0}))
            if exec_R_eps > self.best:
                self.best = exec_R_eps
                self.save(epoch, "best.p")
            if (epoch + 1) % a.save_interval == 0:
                self.save(epoch)
            del batch
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.log("training done!")


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
    args = p.parse_args()
    Trainer(args).train(args.epochs)


if __name__ == "__main__":
    main()
