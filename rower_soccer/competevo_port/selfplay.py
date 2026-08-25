"""Stage 3: two independent learners and the opponent checkpoint ring.

Stages 1-2 trained ONE `DevActorCritic` playing both ants. That is not the
setting the paper's result rests on. Theirs is competitive co-evolution: two
learners with no shared weights, and -- the load-bearing part -- each learner's
rollouts are collected against a *sampled past checkpoint* of the other, not
against the other's current weights. That is what stops the pair from chasing
each other into a cycle where each is only good against this week's opponent.

Everything below follows `runner/multi_evo_agent_runner.py` (the
`multi-evo-agent-runner` that `config/run-to-goal-devants-v0.yaml` selects).
Their rule, verbatim, `multi_evo_agent_runner.py:190-225`:

    if not self.cfg.use_opponent_sample or mean_action or self.epoch == 0:
        ckpt = self.epoch
        try:  ... load agent_0 and agent_1 at `ckpt` ...
        except: pass
    else:
        start = math.floor(self.epoch * self.cfg.delta)
        start = start if start > 1 else 1
        end = self.epoch
        ckpt = randomstate.randint(start, end) if start!=end else end
        ... opponent `1-idx` loads `ckpt`, ego `idx` loads `self.epoch` ...

Read that carefully, because it is NOT "half the time the current opponent,
half the time a uniform past one":

  * `randomstate` is a `np.random.RandomState` (`:397-404`), so `randint(start,
    end)` is HIGH-EXCLUSIVE. The opponent is uniform on the integers
    `[max(1, floor(delta*epoch)), epoch - 1]` -- a STRICTLY PAST checkpoint. The
    current opponent is drawn only in the degenerate `start == end` case, which
    for `delta = 0.5` happens exactly once, at epoch 1.
  * `delta` is a *window* parameter, not a mixing probability. `delta = 0.5`
    (dev) means "uniform over the most recent half of history"; `delta = 0`
    (their fixed-morph ants) means the whole history, Bansal-style; `delta = 1`
    would collapse to `start == end` and always play the current opponent.

So with delta=0.5 at epoch 100 the opponent is uniform on {50..99} and the
probability of facing the CURRENT opponent is 0, not 0.5. This module
implements THEIR rule; `OpponentRing.sample_epoch` is a direct transcription and
the gate measures the empirical distribution against it.

Other semantics taken from their runner:

  * **Two fleets, ego data only.** Per epoch they launch two worker fleets,
    `idx in {0, 1}`; in fleet `idx`, ego `idx` runs its current weights and
    opponent `1-idx` runs the sampled checkpoint, and the merge keeps only ego's
    half (`b = [ma_buffer_0[0], ma_buffer_1[1]]`, `:457`). Here that is a split
    of the world batch: worlds `[0, N/2)` are ego-0 worlds and worlds
    `[N/2, N)` are ego-1 worlds. Theirs pays `2 x min_batch_size` env steps to
    keep `min_batch_size` per learner; splitting disjoint worlds gets the same
    ego data for half the physics, which is the whole point of batching.
  * **Resampled per episode.** The `while ma_logger[0].num_steps <
    min_batch_size` loop body (`:179-303`) is ONE episode -- it `env.reset()`s
    at the top and `break`s on terminated/truncated -- and the samplers are
    rebuilt and reloaded at the top of it. (The port map's section 4.3 says
    "once per worker-batch, not per episode". That is wrong; the checkpoint IS
    redrawn every episode. Corrected here and in PORT_STATUS.)
  * **The dev opponent acts stochastically.** `noise_rate = 1.0`
    (`base_runner.py:27`, never reassigned) makes `use_mean_action` False for
    both agents in the evo runner. Their FIXED-morph runner is the one that
    forces the opponent to mean actions (`multi_agent_runner.py:243`). We are
    porting the dev runner, so the opponent samples.
  * **Cadence `save_model_interval: 1`** -- every epoch is kept, and after epoch
    E the file is named `epoch_{E+1}.p`, which is what epoch E+1 then reads as
    "current". So a ring tagged with "number of completed epochs" indexes their
    files exactly: at the start of epoch E the newest entry is E, and it is the
    learner's current weights.

### The one thing that cannot be ported literally: per-world opponents

Theirs is one env per worker, so "an opponent per episode" costs one net. Here
one batch of N worlds resets asynchronously, so a literal port would need up to
N distinct nets forward-passed every step. Instead there are `blocks` opponent
SLOTS per side; each slot holds one sampled checkpoint, and each world is
assigned to a slot. Two cadences, and they are different on purpose:

  * a world redraws its SLOT at its own episode reset -- that is their
    per-episode resample, and it is free (an index write);
  * a slot redraws its CHECKPOINT once per training iteration -- that is what
    bounds the cost at `blocks` forward passes per step instead of N.

The marginal distribution a learner faces is therefore still their distribution
(slots are i.i.d. draws from the window and the assignment is uniform over
slots). The deviation is that a world's opponent can change at an iteration
boundary in the MIDDLE of an episode, which theirs never does. With `blocks=4`
and a 64-step rollout against ~150-step episodes that is roughly one swap per
episode. It is recorded in PORT_STATUS as a deviation rather than hidden.

### And the slots are ONE forward pass, not `blocks` of them

The first version of this ran `blocks` separate `DevActorCritic.act` calls per
side per step and gathered the answers -- 640 tiny forward passes per 64-step
rollout against 64, which the stage-3 profile measured at 11.0 s of a 28.4 s
iteration. That cost is call COUNT, not arithmetic: these nets are 38k
parameters. All 2 x `blocks` opponents share an architecture and differ only in
weights, so `dev_ppo.StackedDevActors` holds the whole set as stacked weight
tensors and evaluates them in one broadcasting `matmul` per layer. The gather
then happens on the DISTRIBUTION rather than on a sample, so a row is drawn once
instead of `blocks` times and discarded `blocks - 1` times -- the same
distribution, fewer variates.

`batched_opponents=False` restores the per-slot path; it is the reference the
gate measures the batched one against, and the two agree to fp32 on the
assembled action (`tests/test_selfplay.py`, first two checks).
"""

import collections
import copy
import math

import numpy as np
import torch

from rower_soccer.competevo_port.dev_ppo import (DevActorCritic,
                                                 DevSelfPlayPPO,
                                                 StackedDevActors)

# Their `delta` for every `*-devants-*` config (`run-to-goal-devants-v0.yaml:48`,
# `robo-sumo-devants-v0.yaml:48`). Their fixed-morph ants use 0 (full history).
DEV_DELTA = 0.5

# Their history is UNBOUNDED: `save_model_interval: 1` writes one pickle per
# epoch and nothing ever deletes (measured in their tree: 110 files x 317 kB per
# agent for a 109-epoch run). An in-memory ring has to have a bound, so pick one
# that does not clip their actual schedule: at epoch E the delta=0.5 window is
# `[floor(E/2), E-1]`, i.e. ceil(E/2) entries, so `max_epoch_num: 1000` needs
# 500. 512 covers it with room, and a dev checkpoint is ~38k parameters
# (scale 6.7k + control 19k + critic 12k) = ~152 kB fp32, so the ring costs
# ~78 MB of HOST ram at full occupancy -- measured by the gate, not estimated.
# Past that bound the oldest entries are evicted and `sample_epoch` clamps into
# what is still stored; `n_clamped` counts every time that happens, so a run
# that is silently no longer playing their distribution says so.
RING_CAPACITY = 512

# `save_model_interval: 1`. Anything larger thins the ring and is a deviation.
CHECKPOINT_EVERY = 1

# How many distinct sampled checkpoints are live per side per iteration. Their
# effective number is (nthreads x episodes per worker), i.e. dozens; the cost
# here is one extra forward pass per block per step, so this is the knob that
# trades opponent diversity against throughput. 4 measured at ~7% of iteration
# wall time at 1024 worlds.
OPPONENT_BLOCKS = 4


class OpponentRing:
    """A bounded, in-memory stand-in for their `models/agent_i/epoch_%04d.p`.

    Entries are tagged by EPOCH in their numbering: the entry pushed after
    training epoch E is tagged E+1, matching `save_checkpoint`'s
    `epoch_%04d.p % (epoch + 1)`. Consequently at the start of epoch E the
    newest entry is tagged E and holds exactly the learner's current weights,
    which is what their `ckpt = end` branch loads.

    State dicts are stored detached on the CPU. The ring never holds a
    reference to a live parameter, so a checkpoint cannot be mutated by a later
    optimizer step -- the gate asserts this by round-tripping.
    """

    def __init__(self, capacity=RING_CAPACITY, delta=DEV_DELTA):
        assert capacity >= 1
        self.capacity, self.delta = capacity, delta
        self._epochs = collections.deque(maxlen=capacity)
        self._ckpts = collections.deque(maxlen=capacity)
        # Number of draws whose target epoch had already been evicted. Non-zero
        # means the run is no longer sampling their distribution.
        self.n_clamped = 0
        self.n_evicted = 0

    def __len__(self):
        return len(self._ckpts)

    @property
    def epochs(self):
        return list(self._epochs)

    def push(self, epoch, module):
        """Store a detached CPU copy of `module`'s parameters and buffers."""
        if len(self._ckpts) == self.capacity:
            self.n_evicted += 1
        self._epochs.append(int(epoch))
        self._ckpts.append({k: v.detach().to("cpu", copy=True)
                            for k, v in module.state_dict().items()})

    def sample_epoch(self, epoch, rng):
        """Their rule (`multi_evo_agent_runner.py:210-213`), transcribed.

        `rng` is a `numpy.random.Generator` or `RandomState`; both expose a
        HIGH-EXCLUSIVE `integers`/`randint`, which is the detail that makes the
        drawn checkpoint strictly past.
        """
        start = math.floor(epoch * self.delta)
        start = start if start > 1 else 1
        end = int(epoch)
        if start == end:
            return end
        draw = getattr(rng, "integers", None) or rng.randint
        return int(draw(start, end))

    def sample(self, epoch, rng):
        """(tagged epoch, state_dict) for this epoch's opponent, or None if the
        ring is empty (their epoch-0 branch: nothing to load)."""
        if not self._ckpts:
            return None
        target = self.sample_epoch(epoch, rng)
        return self.get(target)

    def get(self, target):
        """The stored entry for `target`, clamped into the surviving window.

        Exact when `checkpoint_every == 1` and the ring has not evicted the
        target. Otherwise: the newest stored entry at or before `target`, or the
        oldest stored entry if `target` predates the whole ring (which is the
        eviction case, counted).
        """
        if not self._ckpts:
            return None
        eps = self._epochs
        best = None
        for i, e in enumerate(eps):
            if e <= target and (best is None or e > eps[best]):
                best = i
        if best is None:                      # target older than everything left
            self.n_clamped += 1
            best = 0
        elif eps[best] != target:
            self.n_clamped += 1
        return eps[best], self._ckpts[best]

    def nbytes(self):
        """Host bytes held. The gate uses this to show the bound is real."""
        return sum(sum(v.numel() * v.element_size() for v in cp.values())
                   for cp in self._ckpts)


class _LaneEnv:
    """The five attributes `SelfPlayPPO.__init__` reads off an env.

    Each learner owns a `DevSelfPlayPPO` whose buffers are one lane wide
    (`n_agents = 1`) over its own half of the world batch. That reuses their
    GAE, their mask semantics, the globally standardized advantages, the two
    optimizers and the curriculum without reimplementing any of it; the only
    thing the sub-trainer must never do is drive the env itself, so `collect`
    is fenced off below.
    """

    def __init__(self, env, n_worlds, n_lanes=1):
        # `n_lanes` is the TEAM SIZE: 1 for their two-agent co-evolution, 2 for
        # 2v2, where one learner owns both of its team's lanes in each of its
        # worlds. Everything downstream (GAE, the mask, the globally
        # standardized advantages) already runs per (world, lane).
        self.n, self.n_agents = n_worlds, n_lanes
        self.obs_dim, self.act_dim = env.obs_dim, env.act_dim
        self.max_episode_steps = env.max_episode_steps
        self._device, self._dtype = env.device, env.dtype

    def reset(self):
        return torch.zeros(self.n, self.n_agents, self.obs_dim,
                           device=self._device, dtype=self._dtype)


class _LaneLearner(DevSelfPlayPPO):
    """A `DevSelfPlayPPO` that is filled from outside instead of collecting."""

    def collect(self):
        raise RuntimeError("a lane learner does not drive the env; "
                           "CoEvoPPO.collect fills its buffers")


class CoEvoPPO:
    """Their two-learner co-evolution loop over one batched env.

    World layout, fixed at construction:

        worlds [0, N/2)   ego = agent 0   opponent lane 1 = ring[1] sample
        worlds [N/2, N)   ego = agent 1   opponent lane 0 = ring[0] sample

    Learner e trains ONLY on lane e of its own half. The other lane of that half
    is opponent data and is thrown away, which is their
    `b = [ma_buffer_0[0], ma_buffer_1[1]]`.
    """

    def __init__(self, env, acs=None, delta=DEV_DELTA,
                 ring_capacity=RING_CAPACITY, checkpoint_every=CHECKPOINT_EVERY,
                 blocks=OPPONENT_BLOCKS, use_opponent_sample=True,
                 batched_opponents=True,
                 rollout_len=64, seed=0, device="cuda", **ppo_kw):
        assert env.n % 2 == 0, "the ego split needs an even world count"
        self.env, self.device = env, device
        self.T, self.N, self.A = rollout_len, env.n, env.n_agents
        self.n_ego = env.n // 2
        # Measured 2026-08-24, against the pre-generalisation file extracted
        # from commit 90c0bba: at L = 1 this code is BIT-IDENTICAL to the
        # two-agent version it replaced -- 0.000e+00 over every rollout buffer
        # (obs, act, logp, val, rew, mask) and both learners' GAE outputs,
        # given matched env state and a matched global RNG. So a 1v1 run under
        # this file is directly comparable to one under the old one, which is
        # what makes the seed-to-seed spread in M2E section 12 a statement
        # about the port rather than about this refactor.
        #
        # TWO SIDES, not two agents. Their loop is two-agent and this was an
        # assert on that; 2f needs two TEAMS of two, and the generalisation is
        # that "ego lane e" becomes "the lanes belonging to side e". At two
        # agents `team_lanes` is [[0], [1]] and every path below is what it
        # was, which is what `tests/test_selfplay.py` keeps honest.
        teams = (env.team.tolist() if hasattr(env, "team")
                 else list(range(self.A)))
        assert sorted(set(teams)) == [0, 1], (
            f"co-evolution needs exactly two sides, got teams {teams}")
        self.team_lanes = [
            torch.tensor([i for i, t in enumerate(teams) if t == e],
                         device=env.device, dtype=torch.long)
            for e in range(2)]
        self.L = len(self.team_lanes[0])
        assert self.L == len(self.team_lanes[1]), "sides must be the same size"
        self.blocks = max(int(blocks), 1)
        self.use_opponent_sample = use_opponent_sample
        self.checkpoint_every = max(int(checkpoint_every), 1)
        self.epoch = 0

        if acs is None:
            acs = [DevActorCritic(design_dim=env.design_dim,
                                  sim_obs_dim=env.sim_obs_dim,
                                  n_motor=env.n_motor) for _ in range(2)]
        assert len(acs) == 2
        self.acs = [ac.to(device) for ac in acs]
        # `RunningNorm` advances only in training mode and their sampling pass
        # must not move it (ppo.RunningNorm's docstring); `update()` toggles
        # train/eval around the optimizer pass, but a freshly constructed module
        # is in TRAIN mode, so without this the very first rollout would whiten
        # with statistics that moved underneath it.
        for ac in self.acs:
            ac.eval()

        self.learners = [
            _LaneLearner(_LaneEnv(env, self.n_ego, self.L), self.acs[e],
                         rollout_len=rollout_len, device=device, **ppo_kw)
            for e in range(2)]

        # Opponent slots. `opp_nets[e][k]` plays lane `1 - e` in ego-`e` worlds,
        # so it is loaded from `rings[1 - e]`. Weights are overwritten wholesale
        # every iteration; these modules are never optimized and never see a
        # gradient (all forward passes are under `torch.no_grad`, and their
        # RunningNorm is pinned by `eval()`).
        self.rings = [OpponentRing(ring_capacity, delta) for _ in range(2)]
        # deepcopy rather than a fresh DevActorCritic: at 2v2 the learners are
        # `TeamActorCritic`s and an opponent built from the env's dims would be
        # the wrong width. Copying the thing it must mirror is correct for any
        # architecture, and the weights are overwritten wholesale below anyway.
        self.opp_nets = [[copy.deepcopy(self.acs[1 - e]).to(device).eval()
                          for _ in range(self.blocks)] for e in range(2)]
        for e in range(2):
            for k in range(self.blocks):
                self.opp_nets[e][k].load_state_dict(self.acs[1 - e].state_dict())
                for p in self.opp_nets[e][k].parameters():
                    p.requires_grad_(False)
        # All 2 x `blocks` opponents share an architecture and differ only in
        # weights, so they are one batched forward, not 2 x `blocks` of them.
        # `opp_nets` stays the source of truth (it is what `resample_opponents`
        # loads into and what the gate inspects); the stack is re-synced from it
        # at the top of every rollout, so it can never be stale.
        # StackedDevActors is a hand-stacked mirror of DevActorCritic's action
        # path, and it slices the design head's input at `design_dim`. A
        # role-aware design head reads `design_dim + ROLE_DIM`, so the mirror
        # is silently the wrong shape -- it fails loudly here (20 vs 22) but
        # would be far worse if it broadcast. Fall back to the per-slot
        # reference path, which drives the REAL modules and therefore picks up
        # any policy override for free. Costs the ~1.57x the stacking bought.
        #
        # Teaching the mirror about the wider head is the faster fix and wants
        # its own equivalence gate before it is trusted; until then, correct
        # beats fast for an exploratory arm.
        if batched_opponents and getattr(self.acs[0], "role_in_design", False):
            print("[CoEvoPPO] role_in_design: using the per-slot opponent path "
                  "(StackedDevActors does not know the wider design head)",
                  flush=True)
            batched_opponents = False
        self.batched_opponents = bool(batched_opponents)
        self.opp_stack = StackedDevActors(self.opp_nets[0][0], 2,
                                          self.blocks).to(device)
        # Which sampled epoch each slot currently holds, and the epoch the draw
        # was made AT (the two differ by the lag, and `train_iter` increments
        # `self.epoch` after the draw, so the lag must not be measured against
        # the current value). Logged, and the gate reads both.
        self.opp_epoch = [[0] * self.blocks for _ in range(2)]
        self.opp_sample_epoch = 0

        d = env.device
        self.ego_worlds = [torch.arange(0, self.n_ego, device=d),
                           torch.arange(self.n_ego, self.N, device=d)]
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=d).manual_seed(seed + 1)
        # Per-world opponent slot, redrawn at that world's episode reset.
        self.slot = torch.randint(0, self.blocks, (self.N,), generator=self.gen,
                                  device=d)
        self._obs = env.reset()
        self.ep_fwd = [0.0, 0.0]

    # -- opponent bookkeeping ------------------------------------------------
    def resample_opponents(self):
        """Once per iteration: every slot draws a fresh checkpoint by their
        rule and the weights are loaded in. At epoch 0, or with opponent
        sampling off, the opponent is the other learner's CURRENT weights --
        their `mean_action or self.epoch == 0` branch, which loads `ckpt =
        self.epoch` for both agents."""
        self.opp_sample_epoch = self.epoch
        for e in range(2):
            for k in range(self.blocks):
                pick = None
                if self.use_opponent_sample and self.epoch > 0:
                    pick = self.rings[1 - e].sample(self.epoch, self.rng)
                if pick is None:
                    self.opp_nets[e][k].load_state_dict(
                        self.acs[1 - e].state_dict())
                    self.opp_epoch[e][k] = self.epoch
                else:
                    ep, sd = pick
                    self.opp_nets[e][k].load_state_dict(sd)
                    self.opp_epoch[e][k] = ep

    def push_checkpoints(self):
        """Their `save_checkpoint(epoch)`: after epoch E, tag E+1. Called by
        `train_iter` after the update, so a checkpoint is always post-update
        weights, as theirs are."""
        for e in range(2):
            self.rings[e].push(self.epoch, self.acs[e])

    # -- rollout -------------------------------------------------------------
    def _opponent_actions(self, e, obs_half, noise=None):
        """REFERENCE path: one forward pass per slot over the whole ego-`e`
        half, gathered by slot. This is what stage 3 shipped and what the
        equivalence gate measures the batched path against; it is still
        reachable with `batched_opponents=False`.

        Running every slot on every row wastes `blocks - 1` of the compute but
        keeps the shapes static (no boolean indexing, no host sync), which on
        this batch size is the cheaper of the two."""
        # `obs_half` is [M, obs] at one lane per side and [M, L, obs] at more;
        # a per-world slot applies to every lane of that world, because the
        # ring holds whole past TEAMS (design doc section 6), so the slot is
        # broadcast across L rather than drawn per agent.
        slots = self.slot[self.ego_worlds[e]]
        outs = torch.stack([net.act(obs_half, noise=noise)[0]
                            for net in self.opp_nets[e]])      # [K, M, (L,) act]
        idx = slots.view(-1, *([1] * (outs.dim() - 2)))
        idx = idx.unsqueeze(0).expand(1, *outs.shape[1:])
        return outs.gather(0, idx).squeeze(0)

    def _opponent_actions_batched(self, obs, noise=None):
        """Both sides' `blocks` opponents in ONE forward. `obs` is the full
        `[N, A, obs_dim]` observation; returns `[2, n_ego, act_dim]`, group 0
        being lane 1 of the ego-0 worlds and group 1 lane 0 of the ego-1
        worlds. The ego halves are contiguous by construction, so the two
        groups are slices."""
        M, L = self.n_ego, self.L
        # [2, M, L, obs] -> [2, M*L, obs]: the stacked actors take one row per
        # driven agent, and a world's L lanes all run on that world's slot.
        half = torch.stack([obs[:M][:, self.team_lanes[1]],
                            obs[M:][:, self.team_lanes[0]]])
        flat = half.reshape(2, M * L, -1)
        slot = self.slot.view(2, M, 1).expand(2, M, L).reshape(2, M * L)
        if noise is not None:
            # Callers supply lane-shaped noise [2, M, L, d]; the stacked actors
            # see one row per driven agent, so it flattens exactly as obs does.
            noise = tuple(z.reshape(2, M * L, z.shape[-1]) for z in noise)
        # Always [2, M, L, act], including at L = 1. One shape means the
        # caller's lane assignment is the same expression at both team sizes,
        # and the reference path below returns the same thing.
        return self.opp_stack.act(flat, slot, noise=noise).reshape(2, M, L, -1)

    def collect(self):
        env, T = self.env, self.T
        alphas = [lr.alpha() for lr in self.learners]
        self.ep_fwd = [0.0, 0.0]
        if self.batched_opponents:
            # `opp_nets` is the source of truth; re-syncing here (a few hundred
            # tiny copies, once per rollout) means a test or a caller that pokes
            # an opponent net directly cannot leave the stack behind.
            self.opp_stack.sync_from(self.opp_nets)
        for t in range(T):
            obs = self._obs.float()
            act = torch.zeros(self.N, self.A, env.act_dim, device=env.device,
                              dtype=obs.dtype)
            if self.batched_opponents:
                opp = self._opponent_actions_batched(obs)
                act[:self.n_ego][:, self.team_lanes[1]] = opp[0]
                act[self.n_ego:][:, self.team_lanes[0]] = opp[1]
            for e in range(2):
                w = self.ego_worlds[e]
                mine, theirs = self.team_lanes[e], self.team_lanes[1 - e]
                lr = self.learners[e]
                a, logp, v = lr.ac.act(obs[w][:, mine])
                act[w.unsqueeze(-1), mine.unsqueeze(0)] = a
                if not self.batched_opponents:
                    # The opponents occupy the OTHER side's lanes of the SAME
                    # worlds.
                    act[w.unsqueeze(-1), theirs.unsqueeze(0)] = (
                        self._opponent_actions(e, obs[w][:, theirs]))
                lr.obs_buf[t] = obs[w][:, mine]
                lr.act_buf[t] = a
                lr.logp_buf[t] = logp
                lr.val_buf[t] = v
            self._obs, rew, done, info = env.step(act.to(env.dtype))
            term = (~info["terminated"]).float()
            for e in range(2):
                w, mine = self.ego_worlds[e], self.team_lanes[e]
                lr = self.learners[e]
                r = rew[w][:, mine].float()
                if alphas[e] is not None:
                    r = (alphas[e] * info["dense"][w][:, mine].float()
                         + (1.0 - alphas[e]) * info["parse"][w][:, mine].float())
                lr.rew_buf[t] = r
                # Termination is per WORLD, so every lane of a world shares it.
                lr.mask_buf[t] = term[w].unsqueeze(-1).expand(-1, self.L)
                self.ep_fwd[e] += float(info["forward"][w][:, mine].mean()) / T
            # Their per-episode opponent resample, as an index write: a world
            # that just reset picks a new slot for its next episode.
            if bool(done.any()):
                di = done.nonzero(as_tuple=True)[0]
                self.slot[di] = torch.randint(0, self.blocks, (di.numel(),),
                                              generator=self.gen,
                                              device=env.device)
        for lr in self.learners:
            lr.total_steps += T * self.n_ego * self.L
        out = []
        with torch.no_grad():
            obs = self._obs.float()
            for e in range(2):
                w, mine = self.ego_worlds[e], self.team_lanes[e]
                last_v = self.learners[e].ac.value(obs[w][:, mine])
                out.append(self.learners[e]._gae(
                    last_v.view(self.n_ego, self.L)))
        return out

    def train_iter(self):
        """One of their epochs: sample against the ring, update BOTH learners
        (`optimize_policy:91-92` loops over `self.learners` in index order),
        then checkpoint."""
        self.resample_opponents()
        gae = self.collect()
        stats = {}
        for e in range(2):
            s = self.learners[e].update(*gae[e])
            stats.update({f"{k}_{e}": v for k, v in s.items()})
        self.epoch += 1
        if self.epoch % self.checkpoint_every == 0:
            self.push_checkpoints()
        return stats

    # -- diagnostics ---------------------------------------------------------
    @property
    def total_steps(self):
        """Ego transitions summed over both learners -- the comparable number to
        stage 2's `total_steps`, which also counted both lanes."""
        return sum(lr.total_steps for lr in self.learners)

    def opponent_lag(self):
        """Mean (epoch of the draw - sampled opponent epoch) over the live
        slots. 0 means every opponent is the current policy, i.e. the ring is
        doing nothing; delta=0.5 predicts it climbs to ~epoch/4."""
        eps = [ep for side in self.opp_epoch for ep in side]
        return float(self.opp_sample_epoch - np.mean(eps))


@torch.no_grad()
def evaluate_pair(env, acs, max_steps=None, alpha=None):
    """Their eval pass with two policies: BOTH agents on their current weights
    and mean actions (`mean_action=True` takes the non-sampling branch and loads
    `ckpt = self.epoch` for both), full episodes, win rate with truncated draws
    in the denominator.

    `alpha` is their curriculum weight at the epoch being evaluated. Pass it and
    the returned `ret_curriculum` is the quantity their runner prints as
    "Agent_i gets eval reward", which is NOT the env reward: their eval sampler
    goes through the same `custom_reward` wrapper as training
    (`multi_evo_agent_runner.py:147-164` is called from `sample_worker` whether
    or not `mean_action` is set), so what their logger accumulates is
    `alpha * dense + (1 - alpha) * parse` at the CURRENT epoch's alpha. `ret`
    (the raw env return `parse + dense`) is kept as well; the two coincide only
    while no goal is ever crossed, i.e. exactly the regime the early epochs are
    in, and diverge by up to +/-1000 per game once they are not."""
    for ac in acs:
        ac.eval()
    max_steps = max_steps or (env.max_episode_steps
                              + int(getattr(env, "has_design_step", False)))
    obs = env.reset()
    env.reset_win_stats()
    rets, lens, crets = [], [], []
    cur = torch.zeros(env.n, env.n_agents, device=env.device, dtype=env.dtype)
    for _ in range(max_steps):
        o = obs.float()
        a = torch.stack([acs[i].mean_action(o[:, i]) for i in range(env.n_agents)],
                        dim=1)
        obs, rew, done, info = env.step(a.to(env.dtype))
        if alpha is not None:
            cur += alpha * info["dense"] + (1.0 - alpha) * info["parse"]
        if bool(done.any()):
            idx = done.nonzero(as_tuple=True)[0]
            rets.append(env.last_return[idx].float().cpu().numpy())
            lens.append(env.last_len[idx].float().cpu().numpy())
            if alpha is not None:
                crets.append(cur[idx].float().cpu().numpy())
                cur[idx] = 0.0
    if not rets:
        return {"ret": np.zeros(env.n_agents), "win_rate": env.win_rate(),
                "ep_len": 0.0, "games": env.games,
                "ret_curriculum": np.zeros(env.n_agents)}
    out = {"ret": np.concatenate(rets, 0).mean(0), "win_rate": env.win_rate(),
           "ep_len": float(np.concatenate(lens).mean()), "games": env.games}
    out["ret_curriculum"] = (np.concatenate(crets, 0).mean(0) if crets
                             else out["ret"])
    return out
