"""D3 M3 E4R: the opponent ring for SHARED-WEIGHT self-play.

One agent -- one design head, one controller -- plays both sides of the 1v1.
Its opponent is a *past self* drawn from a ring of archived checkpoints.

Why a ring and not the current self
-----------------------------------
The user's success criterion has two halves that CONFLICT at equilibrium:
each iteration should beat all past iterations, and be roughly tied against
its current self. If the training opponent were the current self, then at
equilibrium both sides reach the line on the same step, `n_reached == 2`, and
`run_to_goal.py` scores `parse = 0`. **The sparse +/-1000 term vanishes exactly
when the agent is well matched** -- the training signal switches itself off at
the point we are trying to reach. Beating a weaker past self is where the
+/-1000 actually pays, so the ring is what carries the gradient.

Sampling rule, transcribed from CompetEvo via D2
------------------------------------------------
`competevo_port/selfplay.py` documents their rule from
`runner/multi_evo_agent_runner.py:190-225`, including a correction D2 had to
make to its own port map:

    start = max(1, floor(delta * epoch));  end = epoch
    ckpt  = randomstate.randint(start, end)      # HIGH-EXCLUSIVE

so the opponent is uniform on the integers `[max(1, floor(delta*epoch)),
epoch-1]` -- **strictly past**, never the current self. `delta` is a WINDOW,
not a mixing probability: `delta=0.5` means "uniform over the most recent
half", `delta=0` means the whole history (Bansal-style).

**We default to `delta = 0`.** The user's criterion is beating *all* past
iterations, not the recent ones, so the ring should span the whole history.
CompetEvo's own fixed-morph ants used `delta = 0` too; `0.5` was their dev
setting.

Two more details taken from the same source, both of which differ from what
the two-lineage E4 arm did:

  * **Resampled per EPISODE**, not per epoch. (D2's port map said "once per
    worker-batch"; D2 measured that and found it wrong -- the checkpoint is
    redrawn every episode.)
  * **The opponent acts STOCHASTICALLY** -- `noise_rate = 1.0` in
    `base_runner.py:27` makes `use_mean_action` False for the opponent. E4 used
    a mean action; this matches theirs.

Cost
----
Per-episode resampling is nearly free here, which is not obvious. `AntEnv.
reset_robot` already reparses the `Robot` and calls `reload_sim_model` on
EVERY episode, so a model recompile is happening regardless. Caching each ring
member's merged XML and its opponent `Robot` at archive time reduces the
per-episode swap to picking a cached triple. `_settle_opponent` is also skipped:
it exists only to fill `_opp_frozen` for the SCRIPTED opponent, and
`set_opponent` is a no-op under `opponent_mode: policy`.
"""
import math, os, pickle
import numpy as np


class OpponentRing:
    """Archived past selves, each cached as (merged scene XML, opponent Robot,
    policy). Sampling follows CompetEvo's rule above."""

    def __init__(self, cfg, base_src, build_scene, make_policy, robot_cls,
                 delta=0.0, seed=0, log=print):
        self.cfg = cfg
        self.base_src = base_src
        self.build_scene = build_scene
        self.make_policy = make_policy
        self.robot_cls = robot_cls
        self.delta = float(delta)
        self.rs = np.random.RandomState(seed + 7717)
        self.log = log
        self.members = {}          # epoch -> dict(xml_str, robot, policy, path)
        self.dir = os.path.join(cfg.cfg_dir, "ring")
        os.makedirs(self.dir, exist_ok=True)

    # ------------------------------------------------------------ archive --
    def add(self, epoch, policy_state, body_xml_str):
        """Archive one checkpoint: its weights and the body its design head
        produces at the mean action (the MODE of the design distribution, not
        the distribution -- a distinction that has changed conclusions three
        times on this project, and is recorded as such)."""
        from lxml import etree
        body_path = os.path.join(self.dir, f"body_{epoch:04d}.xml")
        with open(body_path, "w") as f:
            f.write(body_xml_str)
        tree = self.build_scene(self.base_src, body_path)
        merged = etree.tostring(tree, pretty_print=True).decode("utf-8")
        merged_path = os.path.join(self.dir, f"scene_{epoch:04d}.xml")
        with open(merged_path, "w") as f:
            f.write(merged)
        pol = self.make_policy()
        pol.load_state_dict(policy_state)
        pol.eval()
        for p in pol.parameters():
            p.requires_grad_(False)
        self.members[epoch] = dict(
            merged_path=merged_path, body_path=body_path,
            robot=self.robot_cls(self.cfg.robot_cfg, xml=body_path),
            policy=pol)
        pickle.dump({"policy_dict": policy_state, "epoch": epoch},
                    open(os.path.join(self.dir, f"policy_{epoch:04d}.p"), "wb"))
        return len(self.members)

    # ------------------------------------------------------------- sample --
    def sample_epoch(self, epoch):
        """CompetEvo's rule. Returns None when no strictly-past member exists."""
        avail = sorted(e for e in self.members if e < epoch)
        if not avail:
            return None
        start = math.floor(self.delta * epoch)
        start = start if start > 1 else 1
        cand = [e for e in avail if e >= start]
        if not cand:
            cand = avail
        return int(cand[self.rs.randint(len(cand))])

    def get(self, epoch):
        return self.members.get(epoch)

    def epochs(self):
        return sorted(self.members)


def ring_stats(chosen, epoch):
    """What the ring actually served this epoch -- logged so the sampling
    distribution is measured, not assumed. D2 gated theirs the same way."""
    if not chosen:
        return {}
    c = np.asarray(chosen, dtype=float)
    return {"ring/n_draws": len(c), "ring/mean_age": float(epoch - c.mean()),
            "ring/min_epoch": float(c.min()), "ring/max_epoch": float(c.max()),
            "ring/distinct": float(len(set(chosen)))}


# ---------------------------------------------------------------- eval --
def _install(env, member):
    """Point the env at one specific opponent, with ring sampling OFF.

    Ring sampling must be disabled or `reset_robot` would redraw a random past
    self on the next episode and the match would not be the match we asked
    for. Callers restore `ring_epoch` afterwards.
    """
    env.init_xml_str = open(member["merged_path"], "rb").read()
    env.opp_robot = member["robot"]
    env.opp_policy = member["policy"]
    env.opponent_body_xml = member["body_path"]
    env._opp_cache = None
    env._opp_name_cache = None


def _outcomes(eps, goal_x):
    """Split 'tied' into its two very different meanings.

    A 0-0 stalemate and a 1-1 race both read as 'tied' on any scalar, and the
    difference is the whole question: MUTUAL means the agent is well matched
    against itself, STALEMATE means it has stopped moving. `run_to_goal.py`
    scores `parse = 0` for both.
    """
    if not eps:
        return {}
    r = np.array([bool(e.get("reached")) for e in eps])
    o = np.array([bool(e.get("opp_reached")) for e in eps])
    fwd = np.array([float(e.get("max_fwd", 0.0)) for e in eps])
    n = len(eps)
    win = float((r & ~o).mean())
    loss = float((~r & o).mean())
    mutual = float((r & o).mean())
    stale = float((~r & ~o).mean())
    return {"n": n, "win_rate": win, "loss_rate": loss,
            "mutual_rate": mutual, "stalemate_rate": stale,
            "decisive_rate": win + loss,
            # tournament score, draws worth a half, for the matrix
            "score": win + 0.5 * mutual,
            "fwd_mean": float(fwd.mean()), "fwd_min": float(fwd.min()),
            "ep_len_mean": float(np.mean([e.get("n", 0) for e in eps]))}


def _play(env, agent, e2_eval, episodes, seed_base):
    act, wrap = e2_eval.gnn_actor(agent.policy_net, agent.running_state, True)
    ev = e2_eval.evaluate(env, act, wrap, episodes=episodes,
                          seed_base=seed_base, max_steps=env.max_nsteps + 5)
    return _outcomes(ev.pop("episodes", []), env.goal_x)


def make_current_member(ring, agent, env, e3_morph, sp, epoch):
    """A transient ring member built from the CURRENT weights and the body the
    current design head produces -- for the mirror match. Not archived."""
    body = sp.dump_mean_action_body(env, agent.policy_net, e3_morph)
    if not body:
        return None
    sd = {k: v.detach().cpu() for k, v in agent.policy_net.state_dict().items()}
    key = -1                                   # reserved slot for 'current'
    ring.members.pop(key, None)
    ring.add(key, sd, body)
    return ring.members.pop(key)               # built, but kept out of the ring


def mirror_match(env, agent, ring, e2_eval, episodes=20):
    """Current self vs current self. Reports the three-way outcome split."""
    from rower_soccer.t2a_port import e3_morph
    from rower_soccer.t2a_port import e4_selfplay as sp
    keep_epoch, keep_xml = env.ring_epoch, env.init_xml_str
    keep_rob, keep_pol = getattr(env, "opp_robot", None), env.opp_policy
    env.ring_epoch = None                      # freeze ring sampling
    try:
        cur = make_current_member(ring, agent, env, e3_morph, sp,
                                  keep_epoch or 0)
        if cur is None:
            return {}
        _install(env, cur)
        out = _play(env, agent, e2_eval, episodes, seed_base=50000)
    finally:
        env.ring_epoch, env.init_xml_str = keep_epoch, keep_xml
        if keep_rob is not None:
            env.opp_robot = keep_rob
        env.opp_policy = keep_pol
        env._opp_cache = None
        env._opp_name_cache = None
    return out


def ladder(env, agent, ring, e2_eval, episodes=10, k=5):
    """Current self against up to k past selves spread across the history."""
    eps_avail = [e for e in ring.epochs() if e >= 0]
    if not eps_avail:
        return {"n_opponents": 0, "mean_win": None, "spearman": None,
                "rows": []}
    idx = np.linspace(0, len(eps_avail) - 1, min(k, len(eps_avail)))
    picks = sorted({eps_avail[int(round(i))] for i in idx})
    keep_epoch, keep_xml = env.ring_epoch, env.init_xml_str
    keep_rob, keep_pol = getattr(env, "opp_robot", None), env.opp_policy
    env.ring_epoch = None
    rows = []
    try:
        for e in picks:
            _install(env, ring.get(e))
            o = _play(env, agent, e2_eval, episodes, seed_base=60000 + 100 * e)
            o["opponent_epoch"] = e
            o["age_gap"] = (keep_epoch or 0) - e
            rows.append(o)
    finally:
        env.ring_epoch, env.init_xml_str = keep_epoch, keep_xml
        if keep_rob is not None:
            env.opp_robot = keep_rob
        env.opp_policy = keep_pol
        env._opp_cache = None
        env._opp_name_cache = None
    w = [r["win_rate"] for r in rows]
    g = [r["age_gap"] for r in rows]
    rho = None
    if len(rows) >= 3 and len(set(w)) > 1 and len(set(g)) > 1:
        from scipy.stats import spearmanr
        rho = float(spearmanr(g, w).correlation)
    return {"n_opponents": len(rows), "mean_win": float(np.mean(w)),
            "spearman": rho, "rows": rows}
