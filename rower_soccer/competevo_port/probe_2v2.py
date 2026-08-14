"""Measurements behind `docs/DESIGN_2V2.md`. Four probes, one file.

    PYTHONPATH=. .venv/bin/python -m rower_soccer.competevo_port.probe_2v2 geometry
    ...                                                                    downed
    ...                                                                    credit
    ...                                                                    render --out /tmp/team.mp4

Every number the design doc quotes comes from one of these. The policy used
throughout is the 1v1 pair from `runs/competevo_port/m2e_fixed/policies.pt`
(83.9% goal rate, M2E_VALIDATION section 10), TRANSPLANTED into the 2v2 scene by
`Transplant` below -- so "trained" here means "trained at 1v1", never "trained
at 2v2", and nothing in this file trains anything.
"""

import argparse
import collections
import os

import numpy as np
import torch

FIXED_POLICIES = ("/workspace/utmist-vc2-phase2/runs/competevo_port/m2e_fixed/"
                  "policies.pt")


# ---------------------------------------------------------------------------
# Driving four ants with two 1v1 nets
# ---------------------------------------------------------------------------
class Transplant:
    """Map the 56-dim 2v2 observation onto the 52-dim one the 1v1 nets expect.

    Team obs is `[flag(1) | scale(20) | own qpos(15) | own qvel(14) |
    xy of (teammate, opp_a, opp_b) (6)]`. The 1v1 obs is the same thing with a
    single 2-dim "other" block, which for two agents is the opponent. So the
    only choice is which of the three others to show, and the only defensible
    one is the NEAREST OPPONENT: it is the entry that plays the same role in the
    1v1 obs the net was trained on. The teammate is simply invisible to a
    transplanted 1v1 policy -- which is a fact about the transplant, not about
    2v2, and the doc says so wherever it matters.

    `ac_0` drives every +x-attacking agent and `ac_1` every -x-attacking one,
    because the scene is not mirror-canonicalised: agent 0's obs is in world
    frame and a net trained to run toward +4 has no way to run toward -4.
    """

    def __init__(self, acs, env, mode="nearest_opp"):
        self.acs, self.mode = acs, mode
        self.env = env
        A, m = env.n_agents, env.meta
        self.n_own = 1 + env.design_dim + (m.agents[0].qpos[1] - m.agents[0].qpos[0]) \
            + (m.agents[0].qvel[1] - m.agents[0].qvel[0])
        self.opp_slots = torch.tensor(
            [[1, 2] for _ in range(A)], device=env.device, dtype=torch.long)
        # which net drives which agent: +x attackers -> ac_0, -x -> ac_1
        self.net_of = [0 if a.goal_x > 0 else 1 for a in m.agents]
        self.xy_at = self.n_own                       # first "other" xy slot

    def adapt(self, obs):
        """`[n, A, 56]` -> `[n, A, 52]`."""
        n, A, _ = obs.shape
        head = obs[..., :self.n_own]
        own_xy = obs[..., 1 + self.env.design_dim: 1 + self.env.design_dim + 2]
        others = obs[..., self.xy_at:].reshape(n, A, -1, 2)
        opp = others[:, :, 1:, :]                     # [n, A, 2, 2]
        d = (opp - own_xy.unsqueeze(-2)).norm(dim=-1)
        k = d.argmin(-1, keepdim=True).unsqueeze(-1).expand(n, A, 1, 2)
        near = opp.gather(2, k).squeeze(2)
        return torch.cat([head, near], dim=-1)

    @torch.no_grad()
    def act(self, obs, mean=True):
        o = self.adapt(obs.float())
        outs = []
        for i in range(self.env.n_agents):
            ac = self.acs[self.net_of[i]]
            outs.append(ac.mean_action(o[:, i]) if mean else ac.act(o[:, i])[0])
        return torch.stack(outs, dim=1)


def load_acs(path, device):
    from rower_soccer.competevo_port.dev_ppo import DevActorCritic
    blob = torch.load(path, map_location="cpu")
    acs = []
    for key in ("ac_0", "ac_1"):
        ac = DevActorCritic().to(device)
        ac.load_state_dict(blob[key])
        ac.eval()
        acs.append(ac)
    return acs


def fresh_acs(device, seed=0):
    from rower_soccer.competevo_port.dev_ppo import DevActorCritic
    torch.manual_seed(seed)
    return [DevActorCritic().to(device).eval() for _ in range(2)]


def make_env(worlds, device, seed=0, **kw):
    from rower_soccer.competevo_port.team_env import TeamRunToGoalDevEnv
    return TeamRunToGoalDevEnv(num_worlds=worlds, use_gpu=(device == "cuda"),
                               seed=seed, **kw)


# ---------------------------------------------------------------------------
# 1. geometry
# ---------------------------------------------------------------------------
def probe_geometry(args, device):
    import mujoco

    from rower_soccer.competevo_port.team_scene import (build_dev_team_scene,
                                                        spawn_table)
    model, meta = build_dev_team_scene(4, back_x=args.back_x)
    print(f"scene: nq={model.nq} nv={model.nv} nu={model.nu} "
          f"nbody={model.nbody} ngeom={model.ngeom}  obs={meta.obs_dim} "
          f"act={meta.act_dim}")
    print(f"other-xy ordering per agent (teammate first): "
          f"teammate={meta.teammate} opponents={meta.opponents}")
    hdr = ("ag tm  spawn(x,y)   goal   to-target  to-own   d0    d1    d2    d3")
    print(hdr)
    for r in spawn_table(meta, model):
        ds = "  ".join(f"{r.get(f'dist_to_{j}', 0.0):4.2f}" for j in range(4))
        print(f"{r['agent']}  {r['team']}  ({r['spawn_x']:+.2f},{r['spawn_y']:+.2f})"
              f"  {r['goal_x']:+.1f}   {r['dist_to_target_line']:5.2f}    "
              f"{r['dist_to_own_line']:5.2f}   {ds}")

    # Does standing on the goal-line cylinder disturb the ant? Compare the
    # spawn contact set for a back ant and a front ant.
    rods = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("leftgoal", "rightgoal")}
    def owner(g):
        nm = model.geom(g).name
        return int(nm.split("/")[0][5:]) if nm.startswith("agent") else -1
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for label, nsteps in (("at qpos0", 0), ("after 0.3 s free", 100), ("after 3.0 s free", 900)):
        mujoco.mj_forward(model, data)
        for _ in range(nsteps):
            mujoco.mj_step(model, data)
        per, with_rod = collections.Counter(), collections.Counter()
        for c in range(data.ncon):
            g1, g2 = data.contact[c].geom1, data.contact[c].geom2
            for o in {owner(g1), owner(g2)} - {-1}:
                per[o] += 1
                if g1 in rods or g2 in rods:
                    with_rod[o] += 1
        z = [round(float(data.qpos[a.qpos[0] + 2]), 3) for a in meta.agents]
        print(f"\ncontacts per agent, {label}: {dict(sorted(per.items()))}"
              f"   (vs a goal rod: {dict(sorted(with_rod.items())) or '{}'})")
        print(f"  torso z: {z}")

    # How much of the goal LINE can a body actually block?
    torso_r = float(model.geom_size[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "agent2/geom_0"), 0])
    span = 2 * abs(float(model.geom_size[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rightgoal"), 1]))
    reach = 2 * (0.4 + torso_r)
    print(f"\ngoal line spans {span:.1f} m in y; an ant's silhouette is "
          f"~{reach:.2f} m wide -> a body-blocking keeper covers "
          f"{100 * reach / span:.1f}% of it.")

    # Speed budget: at the measured 1v1 gait, how long is 5 m and how long 8 m?
    print("\n(gait speed is measured by the `downed` probe; see travel_rate)")


# ---------------------------------------------------------------------------
# 2. downed-player rules
# ---------------------------------------------------------------------------
END_NAMES = {0: "running", 1: "goal", 2: "wipeout", 3: "fall", 4: "timeout"}


def rollout_stats(env, driver, n_games, max_iters=20000, mean=True):
    """Run until `n_games` episodes have finished; aggregate over ALL worlds."""
    obs = env.reset()
    env.reset_win_stats()
    ends = collections.Counter()
    lens, n_down_at_end, first_down_step = [], [], []
    team_wins = np.zeros(env.n_teams)
    per_agent_reached = np.zeros(env.n_agents)
    down_ever = np.zeros(env.n_agents)
    ep_down_step = torch.full((env.n, env.n_agents), -1.0, device=env.device)
    start_x = env._agent_com_x().clone()
    travel = []
    games = 0
    it = 0
    while games < n_games and it < max_iters:
        it += 1
        a = driver.act(obs, mean=mean)
        obs, rew, done, info = env.step(a.to(env.dtype))
        if not bool(info["was_design"].all()):
            newly = info["newly_down"]
            ep_down_step = torch.where(newly & (ep_down_step < 0),
                                       env.ep_step.unsqueeze(-1).float(),
                                       ep_down_step)
        st = env.ep_step.unsqueeze(-1)
        if bool(info["was_design"].any()):
            w = info["was_design"].nonzero(as_tuple=True)[0]
            start_x[w] = env._agent_com_x()[w]
        if bool(done.any()):
            idx = done.nonzero(as_tuple=True)[0]
            for e in env.last_end[idx].tolist():
                ends[END_NAMES[e]] += 1
            lens.extend(env.last_len[idx].float().cpu().tolist())
            d = env.last_down[idx].cpu().numpy()
            n_down_at_end.extend(d.sum(-1).tolist())
            down_ever += d.sum(0)
            w = info["winner"][idx].float().cpu().numpy()
            team_wins += (w @ env.team_onehot.cpu().numpy()).sum(0) / 2.0
            per_agent_reached += info["reached"][idx].float().cpu().numpy().sum(0)
            mv = ((env._agent_com_x()[idx] - start_x[idx])
                  * env.move_sign).cpu().numpy()
            travel.append(mv)
            fd = ep_down_step[idx].cpu().numpy()
            first_down_step.extend(fd[fd >= 0].tolist())
            ep_down_step[idx] = -1.0
            games += len(idx)
    tot = max(games, 1)
    trav = np.concatenate(travel, 0) if travel else np.zeros((1, env.n_agents))
    mean_len = float(np.mean(lens)) if lens else 0.0
    return {
        "games": games, "iters": it,
        "ends": {k: ends[k] / tot for k in
                 ("goal", "wipeout", "fall", "timeout")},
        "ends_n": dict(ends),
        "mean_len": mean_len,
        "median_len": float(np.median(lens)) if lens else 0.0,
        "p_any_down": float(np.mean([x > 0 for x in n_down_at_end])
                            if n_down_at_end else 0.0),
        "mean_down_at_end": float(np.mean(n_down_at_end)) if n_down_at_end else 0.0,
        "down_rate_per_agent": (down_ever / tot).round(3).tolist(),
        "first_down_step": float(np.mean(first_down_step)) if first_down_step else float("nan"),
        "team_win": (team_wins / tot).round(3).tolist(),
        "reach_rate_per_agent": (per_agent_reached / tot).round(3).tolist(),
        "travel_per_agent": trav.mean(0).round(2).tolist(),
        "travel_rate_m_per_s": float(trav.mean() / max(mean_len, 1) / 0.015),
        "recoveries": getattr(env, "n_recoveries", 0),
    }


def probe_downed(args, device):
    from rower_soccer.competevo_port.team_env import DOWN_RULES
    pols = {"trained(1v1 transplant)": load_acs(args.policies, device),
            "untrained": fresh_acs(device, seed=args.seed)}
    rules = args.rules.split(",") if args.rules else list(DOWN_RULES)
    print(f"{args.worlds} worlds, >= {args.games} finished games per cell, "
          f"mean actions, back_x={args.back_x}\n")
    for pname, acs in pols.items():
        print(f"=== policy: {pname} ===")
        hdr = (f"{'rule':10s} {'games':>6s} {'len':>7s} {'med':>6s} "
               f"{'P(any down)':>11s} {'#down@end':>9s} {'1st down':>8s} "
               f"{'goal':>6s} {'wipe':>6s} {'fall':>6s} {'t/out':>6s} "
               f"{'travel m/s':>10s}")
        print(hdr)
        for rule in rules:
            env = make_env(args.worlds, device, seed=args.seed,
                           down_rule=rule, win_rule="team_first",
                           goal_credit="team",
                           recover_steps=args.recover_steps,
                           scene_kwargs={"back_x": args.back_x})
            d = Transplant(acs, env)
            s = rollout_stats(env, d, args.games)
            e = s["ends"]
            print(f"{rule:10s} {s['games']:6d} {s['mean_len']:7.1f} "
                  f"{s['median_len']:6.0f} {s['p_any_down']:11.3f} "
                  f"{s['mean_down_at_end']:9.2f} {s['first_down_step']:8.1f} "
                  f"{e['goal']:6.3f} {e['wipeout']:6.3f} {e['fall']:6.3f} "
                  f"{e['timeout']:6.3f} {s['travel_rate_m_per_s']:10.2f}")
            print(f"           per-agent down rate {s['down_rate_per_agent']}  "
                  f"reach rate {s['reach_rate_per_agent']}  "
                  f"travel {s['travel_per_agent']}  "
                  f"team win {s['team_win']}  recoveries {s['recoveries']}")
            del env
        print()


# ---------------------------------------------------------------------------
# 3. credit assignment
# ---------------------------------------------------------------------------
def probe_credit(args, device):
    """How much of the TEAM reward does an agent's own action control?

    The measurement is a one-step counterfactual on frozen states. For a batch
    of `W` states, held fixed:

      own[i]  = Var over K resamples of agent i's action, of agent i's dense r
      team[i] = Var over the same K resamples, of the TEAM dense r (i + mate)
      mate[i] = Var over K resamples of the TEAMMATE's action, of the team r
                -- the part of the shared signal agent i cannot influence

    `mate[i] / own[i]` is the credit-assignment noise ratio a shared team reward
    imposes on agent i, measured on this env rather than argued from theory.

    CONTROL: `cross[i]` = Var over resamples of agent i's action, of the
    TEAMMATE's own dense reward. Physically that can only be nonzero through
    contact, so on states where the two are metres apart it must be ~0. If it
    is not, the state restore is leaking and every other number here is junk.
    (Break `_restore` on purpose and this is the number that moves first.)
    """
    torch.manual_seed(args.seed)
    acs = (fresh_acs(device, seed=args.seed) if args.untrained
           else load_acs(args.policies, device))
    env = make_env(args.worlds, device, seed=args.seed, down_rule="frozen",
                   auto_reset=False, scene_kwargs={"back_x": args.back_x})
    drv = Transplant(acs, env)
    A, W = env.n_agents, env.n

    obs = env.reset()
    for _ in range(args.warmup):
        obs, *_ = env.step(drv.act(obs).to(env.dtype))

    snap = (env.qpos.clone(), env.qvel.clone(), env.scale.clone(),
            env.stage.clone(), env.down.clone(), env.ep_step.clone())

    def restore():
        env.qpos.copy_(snap[0]); env.qvel.copy_(snap[1])
        env.scale.copy_(snap[2]); env.stage.copy_(snap[3])
        env.down.copy_(snap[4]); env.ep_step.copy_(snap[5])
        if args.break_restore:      # the control's control
            pass                    # (deliberately skip forward())
        else:
            env.backend.forward()

    base = drv.act(obs)                                # [W, A, 28] mean actions
    std = args.action_std

    def sample_rewards(perturb_agent, k):
        """K draws with `perturb_agent`'s motor action resampled, everyone
        else's held at the deterministic mean. Returns `[K, W, A]` dense."""
        out = []
        for _ in range(k):
            restore()
            a = base.clone()
            noise = torch.randn(W, env.n_motor, device=device) * std
            a[:, perturb_agent, -env.n_motor:] = (
                a[:, perturb_agent, -env.n_motor:] + noise).clamp(-1, 1)
            _, _, _, info = env.step(a.to(env.dtype))
            out.append(info["dense"].float())
        return torch.stack(out)

    print(f"states W={W}, K={args.k} resamples, action noise sd={std}, "
          f"warmup {args.warmup} steps, policy="
          f"{'untrained' if args.untrained else args.policies}")
    print(f"{'agent':>5s} {'own var':>10s} {'team var':>10s} {'mate var':>10s} "
          f"{'mate/own':>9s} {'cross(ctl)':>11s} {'cross/own':>10s}")
    rows = []
    for i in range(A):
        mate = env.meta.teammate[i]
        d_i = sample_rewards(i, args.k)                 # [K, W, A]
        d_m = sample_rewards(mate, args.k)
        own = d_i[..., i].var(0).mean().item()
        team = (d_i[..., i] + d_i[..., mate]).var(0).mean().item()
        matev = (d_m[..., i] + d_m[..., mate]).var(0).mean().item()
        cross = d_i[..., mate].var(0).mean().item()
        rows.append((i, own, team, matev, cross))
        print(f"{i:5d} {own:10.4e} {team:10.4e} {matev:10.4e} "
              f"{matev / max(own, 1e-30):9.2f} {cross:11.4e} "
              f"{cross / max(own, 1e-30):10.2e}")
    xs = np.array([r[4] / max(r[1], 1e-30) for r in rows])
    print(f"\nCONTROL cross/own max = {xs.max():.2e} "
          f"({'PASS' if xs.max() < 1e-3 else 'FAIL'}; expected ~0 -- an agent's "
          f"action cannot move a distant teammate's own dense reward)")
    mr = np.array([r[3] / max(r[1], 1e-30) for r in rows])
    print(f"shared-reward noise ratio mate/own: mean {mr.mean():.2f} "
          f"range [{mr.min():.2f}, {mr.max():.2f}]")

    print(f"downed agents at the probe state: "
          f"{float(env.down.float().mean()):.3f} of agent-slots")

    # ---- episode level -----------------------------------------------------
    # The one-step number above is what a per-step advantage sees. What a PPO
    # update actually sees is the RETURN, so measure the same decomposition on
    # whole-episode dense returns: Var(R_i) against Var(R_i + R_mate).
    env2 = make_env(args.worlds, device, seed=args.seed + 1,
                    down_rule="frozen", scene_kwargs={"back_x": args.back_x})
    drv2 = Transplant(acs, env2)
    obs = env2.reset()
    acc = torch.zeros(env2.n, A, device=device)
    rets = []
    with torch.no_grad():
        for _ in range(args.ep_steps):
            obs, _, done, info = env2.step(drv2.act(obs).to(env2.dtype))
            acc += info["dense"].float()
            if bool(done.any()):
                idx = done.nonzero(as_tuple=True)[0]
                rets.append(acc[idx].cpu().numpy())
                acc[idx] = 0.0
    R = np.concatenate(rets, 0) if rets else np.zeros((1, A))
    print(f"\nepisode dense returns over {len(R)} episodes "
          f"({args.ep_steps} steps of {env2.n} worlds):")
    print(f"{'agent':>5s} {'mean':>9s} {'sd':>9s} {'sd(team)':>9s} "
          f"{'var ratio':>9s} {'corr(i,mate)':>13s}")
    for i in range(A):
        m = env2.meta.teammate[i]
        s_i, s_t = R[:, i].std(), (R[:, i] + R[:, m]).std()
        c = float(np.corrcoef(R[:, i], R[:, m])[0, 1])
        print(f"{i:5d} {R[:, i].mean():9.1f} {s_i:9.1f} {s_t:9.1f} "
              f"{(s_t / max(s_i, 1e-9)) ** 2:9.2f} {c:13.3f}")
    print(f"\nscale check: episode dense return sd ~ {R.std():.0f}, "
          f"GOAL_REWARD = 1000 (paid once, to every member of the scoring "
          f"team under goal_credit='team')")


# ---------------------------------------------------------------------------
# 3b. roles: what the observation and the design head can and cannot express
# ---------------------------------------------------------------------------
def probe_roles(args, device):
    """Three questions item 5 of the brief asks, each answered by a number.

    (a) Does a SHARED policy see anything that distinguishes the two roles?
        Only via `own root (x, y)` and the three other-agent `(x, y)` pairs.
        Measured: hold the random scale vector fixed and move only the spawn,
        and see how far the design head's output moves compared with the
        spread it produces across scale vectors.

    (b) Does the design head therefore emit DIFFERENT bodies for the two
        teammates? If yes, "one policy" already buys "two genomes" and a second
        policy per team is not needed to get morphological division of labour.

    (c) Counterfactual policy divergence (the dm_soccer role metric): how much
        does an agent's action change when only the OTHER agents' positions
        change? For a transplanted 1v1 net the teammate is not in the obs at
        all, so its teammate-CPD is identically zero -- which is the point.
    """
    from rower_soccer.competevo_port.team_scene import build_dev_team_scene
    acs = (fresh_acs(device, seed=args.seed) if args.untrained
           else load_acs(args.policies, device))
    env = make_env(args.worlds, device, seed=args.seed, down_rule="frozen",
                   scene_kwargs={"back_x": args.back_x})
    drv = Transplant(acs, env)
    obs = env.reset()                        # every world is in the design stage
    o = drv.adapt(obs.float())               # [W, A, 52]
    net = acs[0]

    with torch.no_grad():
        a0 = net.mean_action(o[:, 0])[:, :env.design_dim]
        a2 = net.mean_action(o[:, 2])[:, :env.design_dim]
        # (a)/(b): same random scale vector, only the spawn differs
        o_mix = o[:, 0].clone()
        o_mix[:, 1 + env.design_dim:] = o[:, 2, 1 + env.design_dim:]
        a_mix = net.mean_action(o_mix)[:, :env.design_dim]
    spread = a0.std(0).mean().item()          # across random scale vectors
    d_roles = (a0 - a2).abs().mean().item()   # both channels differ
    d_state = (a0 - a_mix).abs().mean().item()  # ONLY the sim state differs
    print(f"design head, {env.n} worlds, policy="
          f"{'untrained' if args.untrained else 'm2e_fixed ac_0'}")
    print(f"  spread of the design across random scale vectors  sd = {spread:.4f}")
    print(f"  |design(agent0) - design(agent2)|, mean abs         = {d_roles:.4f}")
    print(f"  ... holding the scale vector fixed, spawn only      = {d_state:.4f}"
          f"   ({100 * d_state / max(spread, 1e-9):.1f}% of the spread)")
    print(f"  design range is [-1, 1]; body scale is 1 + 0.3 * s, so "
          f"{d_state:.4f} of design is {100 * 0.3 * d_state:.2f}% of a link length")

    # (c) counterfactual policy divergence on the OTHER-AGENT channel
    with torch.no_grad():
        obs2 = obs.float()
        for _ in range(args.warmup):
            obs2, *_ = env.step(drv.act(obs2).to(env.dtype))
        base = drv.adapt(obs2.float())
        act_base = torch.stack([acs[drv.net_of[i]].mean_action(base[:, i])
                                for i in range(env.n_agents)], 1)
        # shuffle the "other" xy channel across worlds -- same own state, a
        # different opponent position
        perm = torch.randperm(env.n, device=device)
        pert = base.clone()
        pert[..., -2:] = base[perm][..., -2:]
        act_pert = torch.stack([acs[drv.net_of[i]].mean_action(pert[:, i])
                                for i in range(env.n_agents)], 1)
        cpd = (act_base - act_pert)[..., -env.n_motor:].abs().mean(0).mean(-1)
        scale = act_base[..., -env.n_motor:].abs().mean().item()
    print(f"\nCPD on the opponent-position channel, after {args.warmup} steps:")
    print(f"  per agent {[round(x, 4) for x in cpd.tolist()]}  "
          f"(mean |motor action| = {scale:.3f})")
    print(f"  CPD on the TEAMMATE channel is 0 by construction for a "
          f"transplanted 1v1 net: the teammate is not in its 52-dim input.")

    # role separability: over an episode, how distinguishable are the two
    # teammates by the only channel that carries their role -- root x?
    xs = [[], []]
    with torch.no_grad():
        obs3 = env.reset()
        for _ in range(args.ep_steps // 4):
            obs3, _, done, info = env.step(drv.act(obs3).to(env.dtype))
            cx = info["com_x"].float()
            xs[0].append(cx[:, 0].cpu().numpy())
            xs[1].append(cx[:, 2].cpu().numpy())
    f, b = np.concatenate(xs[0]), np.concatenate(xs[1])
    lo, hi = min(f.min(), b.min()), max(f.max(), b.max())
    hf, _ = np.histogram(f, 60, (lo, hi), density=True)
    hb, _ = np.histogram(b, 60, (lo, hi), density=True)
    w = (hi - lo) / 60
    ov = float(np.minimum(hf, hb).sum() * w)
    print(f"\nrole separability by com_x over {len(f)} agent-steps: "
          f"front mean {f.mean():+.2f}, back mean {b.mean():+.2f}, "
          f"distribution overlap {100 * ov:.1f}%")


# ---------------------------------------------------------------------------
# 4. render
# ---------------------------------------------------------------------------
def _team_render_model(n_agents=4, back_x=4.0):
    import mujoco

    from rower_soccer.competevo_port.design import (CONST_FIELDS,
                                                    WRITTEN_FIELDS,
                                                    DesignWriter,
                                                    build_design_spec)
    from rower_soccer.competevo_port.team_scene import build_dev_team_scene
    model, meta = build_dev_team_scene(n_agents=n_agents, back_x=back_x)
    spec = build_design_spec(model, meta, device="cpu", dtype=torch.float64)
    arrays = {}
    for name in tuple(WRITTEN_FIELDS) + tuple(CONST_FIELDS):
        arrays[name] = torch.from_numpy(np.asarray(getattr(model, name))).unsqueeze(0)
    writer = DesignWriter(spec, arrays, model=model, exact_constants=True)
    return model, meta, writer, arrays


def probe_render(args, device):
    import imageio.v2 as imageio
    import mujoco

    from rower_soccer.competevo_port.render_designs import apply_design
    acs = (fresh_acs(device, seed=args.seed) if args.untrained
           else load_acs(args.policies, device))
    env = make_env(args.worlds, device, seed=args.seed,
                   down_rule=args.down_rule, win_rule="team_first",
                   goal_credit="team", recover_steps=args.recover_steps,
                   scene_kwargs={"back_x": args.back_x})
    drv = Transplant(acs, env)
    rmodel, rmeta, rwriter, rarrays = _team_render_model(env.n_agents,
                                                         args.back_x)
    renderer = mujoco.Renderer(rmodel, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 14.0, -25.0, 90.0
    torsos = [a.torso_body for a in rmeta.agents]

    frames, shown, live = [], 0, None
    obs = env.reset()
    budget = args.episodes * (env.max_episode_steps + 2) + 8
    with torch.no_grad():
        for _ in range(budget):
            obs, rew, done, info = env.step(drv.act(obs).to(env.dtype))
            if shown < args.episodes and not bool(info["was_design"][0]):
                d0 = env.scale[0].detach().cpu().numpy()
                if live is None or not np.array_equal(d0, live):
                    apply_design(rmodel, rwriter, rarrays, d0)
                    live = d0.copy()
                rdata = mujoco.MjData(rmodel)
                rdata.qpos[:] = env.qpos[0].detach().double().cpu().numpy()
                mujoco.mj_forward(rmodel, rdata)
                cam.lookat[:] = rdata.xpos[torsos].mean(0)
                renderer.update_scene(rdata, camera=cam)
                frames.append(renderer.render())
            if bool(done[0]):
                shown += 1
            if shown >= args.episodes:
                break
    imageio.mimwrite(args.out, frames, fps=args.fps, macro_block_size=1,
                     quality=8)
    print(f"{args.out}: {len(frames)} frames "
          f"({len(frames) / args.fps:.1f}s), down_rule={args.down_rule}, "
          f"policy={'untrained' if args.untrained else 'm2e_fixed transplant'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("probe", choices=("geometry", "downed", "credit",
                                 "roles", "render"))
    p.add_argument("--policies", default=FIXED_POLICIES)
    p.add_argument("--worlds", type=int, default=128)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--back-x", type=float, default=4.0)
    p.add_argument("--rules", default=None)
    p.add_argument("--recover-steps", type=int, default=50)
    p.add_argument("--untrained", action="store_true")
    # credit
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--action-std", type=float, default=0.5)
    p.add_argument("--ep-steps", type=int, default=1500)
    p.add_argument("--break-restore", action="store_true",
                   help="negative control: skip the state restore's forward()")
    # render
    p.add_argument("--out", default="/tmp/team2v2.mp4")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=40)
    p.add_argument("--width", type=int, default=1120)
    p.add_argument("--height", type=int, default=560)
    p.add_argument("--down-rule", default="frozen")
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    {"geometry": probe_geometry, "downed": probe_downed,
     "credit": probe_credit, "roles": probe_roles,
     "render": probe_render}[args.probe](args, device)


if __name__ == "__main__":
    main()
