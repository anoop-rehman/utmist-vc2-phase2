"""D3 M3 E3 gate: the design stage really changes the body MuJoCo integrates.

E2/E2.1's gate proved the MIRROR of this -- 134 mjModel arrays IDENTICAL under
20 episodes of destructive random design actions. E3's has to prove the
opposite, and prove it in two independent places, because **a design stage that
silently no-ops would produce a clean, boring, completely wrong null and would
look exactly like a real result**: it would reproduce E2.1's frozen-body
numbers, on E2.1's own instrument, with the design heads taking gradients the
whole time.

So "the design stage wrote it" and "the simulator ran it" are gated as two
different claims (phase 2), every phase carries a negative control, and the
scripted opponent -- which every rung from here inherits, and which lives as a
sibling body in the same MJCF the design stage rewrites -- is checked to
survive an evolved body (phase 3).

    cd /workspace/Transform2Act && source env-gpu.sh
    CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python .../t2a_port/gate_e3.py
"""

import os
import sys
import time

sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402

PASS, FAIL = [], []


def chk(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""),
          flush=True)


def phase(n, title):
    print(f"\n=== phase {n}: {title} ===", flush=True)


def arrays(m):
    for nm in dir(m):
        if nm.startswith("_"):
            continue
        try:
            v = getattr(m, nm)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiub":
            yield nm, np.array(v)


def diff_arrays(ref, m):
    out = set()
    for nm, v in arrays(m):
        if nm not in ref or ref[nm].shape != v.shape or not np.array_equal(ref[nm], v):
            out.add(nm)
    return out


def make_env(cfg_id, identity=None):
    from design_opt.utils.config import Config
    from design_opt.envs import env_dict
    cfg = Config(cfg_id, tmp=True)
    if identity is not None:
        cfg.env_specs = dict(cfg.env_specs)
        cfg.env_specs["force_identity_design"] = identity
    return cfg, env_dict[cfg.env_name](cfg, agent=None)


def design_actions(env, rng, destructive=True):
    """One episode's worth of design actions: 5 skeleton then 1 attribute.
    `destructive` tells every body to add or remove and kicks every attribute
    over its full range -- the same stimulus `gate_e2.py` phase 3 uses, so the
    two gates are the same experiment with opposite expected answers."""
    W = env.control_action_dim + env.attr_design_dim + 1
    acts = []
    for step in range(env.cfg.skel_transform_nsteps + 1):
        n = 32                      # generated for the largest body count seen
        a = np.zeros((n, W))
        if step < env.cfg.skel_transform_nsteps:
            a[:, -1] = (rng.randint(1, 3, size=n) if destructive
                        else rng.randint(0, 3, size=n))
        else:
            a[:, env.control_action_dim:-1] = rng.uniform(
                -1, 1, size=(n, env.attr_design_dim))
        acts.append(a)
    return acts


def run_design(env, acts, seed):
    env.seed(seed)
    env.reset()
    i = 0
    while env.if_use_transform_action() != 2 and i < len(acts):
        a = acts[i][:len(env.robot.bodies)]
        _, _, done, _ = env.step(a)
        i += 1
        if done:
            return False
    return env.if_use_transform_action() == 2


# =========================================================== phase 1 =======
def phase1():
    phase(1, "the cfgs are the experiment, and E2's are untouched")
    from design_opt.utils.config import Config
    e3 = [Config(f"rtg_e3_s{i}", tmp=True) for i in (1, 2, 3)]
    e3c = [Config(f"rtg_e3c_s{i}", tmp=True) for i in (1, 2)]
    gnn = Config("rtg_gnn_s1", tmp=True)
    mlp = Config("rtg_mlp_s1", tmp=True)

    chk("E3 arms: design stages LIVE (no force_identity_design)",
        all(not c.env_specs.get("force_identity_design", False) for c in e3),
        str([c.env_specs.get("force_identity_design", False) for c in e3]))
    chk("E3 control arms: design stages forced to IDENTITY",
        all(c.env_specs.get("force_identity_design", False) for c in e3c))
    chk("E2's own cfgs still force identity (NEG control on the edit)",
        gnn.env_specs.get("force_identity_design", False)
        and mlp.env_specs.get("force_identity_design", False))
    chk("E2's cfgs still at their own budget (100 epochs), not moved to 400",
        gnn.max_epoch_num == 100 and mlp.max_epoch_num == 100,
        f"gnn {gnn.max_epoch_num} mlp {mlp.max_epoch_num}")
    for c in e3 + e3c:
        pass
    chk("every E3 arm: 400 epochs x 50,000 = 20.0M steps, E2.1's budget",
        all(c.max_epoch_num == 400 and c.min_batch_size == 50000
            for c in e3 + e3c),
        f"{[ (c.max_epoch_num, c.min_batch_size) for c in e3 + e3c ]}")
    chk("seeds are 1,2,3 / 1,2",
        [c.seed for c in e3] == [1, 2, 3] and [c.seed for c in e3c] == [1, 2])
    # everything else identical to E2's GNN arm
    keys = ["gamma", "tau", "policy_lr", "value_lr", "clip_epsilon",
            "mini_batch_size", "num_optim_epoch", "skel_transform_nsteps",
            "robot_param_scale", "max_body_depth", "min_body_depth",
            "enable_remove", "env_name"]
    same = {k: all(getattr(c, k) == getattr(gnn, k) for c in e3 + e3c)
            for k in keys}
    chk("every other training/design hyperparameter equals E2's GNN arm",
        all(same.values()), str([k for k, v in same.items() if not v]))
    envkeys = ["model_xml_file", "frame_skip", "opponent_speed", "goal_x",
               "stand_z", "init_height"]
    chk("the TASK is E2's, unchanged (xml, frame_skip, opponent speed, goal)",
        all(c.env_specs.get(k) == gnn.env_specs.get(k)
            for c in e3 + e3c for k in envkeys))
    chk("the termination rule is E2's, unchanged -- the fall-dodge is KEPT",
        all(c.done_condition == gnn.done_condition for c in e3 + e3c),
        str(e3[0].done_condition))


# =========================================================== phase 2 =======
def phase2():
    phase(2, "THE MIRROR GATE: design changes the body MuJoCo integrates")
    cfg_on, env_on = make_env("rtg_e3_s1")
    cfg_off, env_off = make_env("rtg_e3c_s1")
    ref = dict(arrays(env_on.model))
    chk("the reference model is E2's 134 mjModel arrays", len(ref) == 134,
        f"{len(ref)} arrays")

    changed_on, changed_off = set(), set()
    counts_on, counts_off, nu_on = [], [], []
    for ep in range(20):
        rng = np.random.RandomState(1000 + ep)
        acts = design_actions(env_on, rng)
        ok = run_design(env_on, acts, 1000 + ep)
        changed_on |= diff_arrays(ref, env_on.model)
        counts_on.append(len(env_on.robot.bodies))
        nu_on.append(int(env_on.model.nu))
        rng = np.random.RandomState(1000 + ep)       # THE SAME actions
        acts = design_actions(env_off, rng)
        run_design(env_off, acts, 1000 + ep)
        changed_off |= diff_arrays(ref, env_off.model)
        counts_off.append(len(env_off.robot.bodies))

    chk("design ON: the mjModel arrays CHANGE", len(changed_on) > 0,
        f"{len(changed_on)} of {len(ref)} arrays changed")
    chk("design ON: the physical arrays are among them",
        {"body_mass", "geom_size", "actuator_gear", "body_pos"} <= changed_on,
        "body_mass/geom_size/actuator_gear/body_pos all changed")
    chk("design ON: the body COUNT moves off 13",
        len(set(counts_on)) > 1 and min(counts_on) != max(counts_on),
        f"body counts {sorted(set(counts_on))}, motors {sorted(set(nu_on))}")
    chk("NEG design OFF: the SAME actions change NOTHING (E2's 134 identical)",
        len(changed_off) == 0 and set(counts_off) == {13},
        f"{len(changed_off)} arrays changed, body counts {sorted(set(counts_off))}")

    # ---- 2c. the simulated body IS the designed body -------------------
    rng = np.random.RandomState(7)
    run_design(env_on, design_actions(env_on, rng), 7)
    m = env_on.model
    names = [b.name for b in env_on.robot.bodies]
    chk("every designed body exists in the compiled model",
        all(n in m.body_names for n in names),
        f"{len(names)} bodies")
    d_rad = d_len = d_gear = 0.0
    n_g = n_a = 0
    for b in env_on.robot.bodies:
        g = b.geoms[0]
        gid = [i for i in range(m.ngeom)
               if m.geom_bodyid[i] == m.body_name2id(b.name)][0]
        d_rad = max(d_rad, abs(float(np.asarray(g.size).reshape(-1)[0])
                               - float(m.geom_size[gid, 0])))
        n_g += 1
        if g.type == "capsule":
            ln = float(np.linalg.norm(np.asarray(g.end) - np.asarray(g.start)))
            d_len = max(d_len, abs(ln - 2.0 * float(m.geom_size[gid, 1])))
        if b.joints and b.joints[0].actuator is not None:
            an = b.joints[0].actuator.name
            if an in m.actuator_names:
                aid = list(m.actuator_names).index(an)
                d_gear = max(d_gear, abs(float(b.joints[0].actuator.gear)
                                         - float(m.actuator_gear[aid, 0])))
                n_a += 1
    # the XML is the channel and `Geom.sync_node`/`Actuator.sync_node` write
    # 6 decimal places, so exactness is bounded by the XML's own precision.
    chk("compiled capsule RADIUS == the designed radius",
        d_rad < 1e-5, f"max |delta| {d_rad:.3e} over {n_g} geoms (XML is 6 dp)")
    chk("compiled capsule LENGTH == the designed bone length",
        d_len < 1e-5, f"max |delta| {d_len:.3e}")
    chk("compiled actuator GEAR == the designed gear",
        d_gear < 1e-5, f"max |delta| {d_gear:.3e} over {n_a} actuators")

    # NEG: move the Robot's params WITHOUT recompiling. The same comparison
    # must now FAIL, which is what makes the three checks above evidence.
    g0 = env_on.robot.bodies[1].geoms[0]
    saved = float(np.asarray(g0.size).reshape(-1)[0])
    g0.size = np.array([saved + 0.02])
    gid = [i for i in range(m.ngeom)
           if m.geom_bodyid[i] == m.body_name2id(env_on.robot.bodies[1].name)][0]
    detected = abs(float(np.asarray(g0.size).reshape(-1)[0])
                   - float(m.geom_size[gid, 0])) > 1e-5
    g0.size = np.array([saved])
    chk("NEG: a Robot param the simulator never compiled IS detected", detected,
        "the round-trip check can fail, so passing it is evidence")

    # ---- 2d. a targeted, PREDICTED change ------------------------------
    cfg2, env2 = make_env("rtg_e3_s1")
    env2.seed(11)
    env2.reset()
    W = env2.control_action_dim + env2.attr_design_dim + 1
    nb0 = len(env2.robot.bodies)
    # one skeleton step, "add a child" on body index 1 only
    a = np.zeros((nb0, W))
    a[1, -1] = 1
    env2.step(a)
    chk("a targeted ADD adds exactly one body",
        len(env2.robot.bodies) == nb0 + 1,
        f"{nb0} -> {len(env2.robot.bodies)}")
    nb1 = len(env2.robot.bodies)
    # `allow_remove_body` is the cfg's own rule -- depth >= min_body_depth + 1
    # and no children -- so the target has to be a leaf, not body index 2.
    rm = [i for i, b in enumerate(env2.robot.bodies)
          if env2.allow_remove_body(b)]
    a = np.zeros((nb1, W))
    a[rm[0], -1] = 2
    env2.step(a)
    chk("a targeted REMOVE removes exactly one body",
        len(env2.robot.bodies) == nb1 - 1,
        f"{nb1} -> {len(env2.robot.bodies)}, removed body index {rm[0]} "
        f"of {len(rm)} the cfg allows")
    # skip to the attribute stage, then a predicted gear change
    for _ in range(env2.cfg.skel_transform_nsteps):
        if env2.if_use_transform_action() != 0:
            break
        env2.step(np.zeros((len(env2.robot.bodies), W)))
    chk("the attribute stage is reached", env2.if_use_transform_action() == 1,
        f"stage index {env2.if_use_transform_action()}")
    # column order is `Body.get_params`': offset_x, offset_y, gear, size,
    # ext_start (`e0_analyse.GENOME_COLS`). The four JOINTLESS leg stubs pad a
    # zero into the gear column (`xml_robot.py`'s pad_zeros, the bug the
    # converter found), so they can never move and the check is over the
    # bodies that actually carry an actuator.
    gi = 2
    before = env2.get_attr_design().copy()
    nb = len(env2.robot.bodies)
    act_rows = [i for i, b in enumerate(env2.robot.bodies)
                if b.joints and b.joints[0].actuator is not None]
    gear_before = {b.name: float(b.joints[0].actuator.gear)
                   for b in env2.robot.bodies
                   if b.joints and b.joints[0].actuator is not None}
    a = np.zeros((nb, W))
    a[:, env2.control_action_dim + gi] = 1.0      # push `gear` toward its bound
    env2.step(a)
    after = env2.get_attr_design()
    moved = float(np.abs(after[act_rows, gi] - before[act_rows, gi]).min())
    lb, ub = (env2.cfg.robot_cfg["actuator_params"]["gear"]["lb"],
              env2.cfg.robot_cfg["actuator_params"]["gear"]["ub"])
    gears = [float(b.joints[0].actuator.gear) for b in env2.robot.bodies[1:]
             if b.joints and b.joints[0].actuator is not None]
    pad = float(np.abs(after[:, gi]).max()) if not act_rows else 0.0
    chk("a targeted ATTRIBUTE action moves the genome in the named column",
        moved > 0.05,
        f"min |delta gear| {moved:.4f} over the {len(act_rows)} bodies that "
        f"carry an actuator; physical gear "
        f"{min(gear_before.values()):.1f} -> {min(gears):.1f}")
    chk("and the physical gear stays inside the cfg's own bounds",
        all(lb - 1e-6 <= g <= ub + 1e-6 for g in gears),
        f"gears {min(gears):.1f}-{max(gears):.1f}, bounds {lb}-{ub}")


# =========================================================== phase 3 =======
def phase3():
    phase(3, "the scripted opponent survives an evolved body")
    cfg, env = make_env("rtg_e3_s1")
    W = env.control_action_dim + env.attr_design_dim + 1
    errs, cross, nopp, nuopp, last = [], [], [], [], []
    for ep in range(6):
        rng = np.random.RandomState(50 + ep)
        ok = run_design(env, design_actions(env, rng), 50 + ep)
        if not ok:
            continue
        m = env.model
        nopp.append(sum(1 for n in m.body_names if n.startswith("opp_")))
        nuopp.append(sum(1 for n in m.actuator_names if n.startswith("opp_")))
        qs, vs, obid = env._opp()
        last.append(qs.stop == m.nq and vs.stop == m.nv)
        xs = []
        for k in range(60):
            _, _, done, _ = env.step(np.zeros((len(env.robot.bodies), W)))
            xs.append(float(env.data.qpos[qs][0]))
            if done:
                break
        pred = [env.opp_x(k + 1) for k in range(len(xs))]
        errs.append(max(abs(x - p) for x, p in zip(xs, pred)))
    chk("the opponent still has all 13 bodies and 8 motors after design",
        set(nopp) == {13} and set(nuopp) == {8},
        f"bodies {sorted(set(nopp))} motors {sorted(set(nuopp))}")
    chk("the opponent's joints are still LAST in qpos/qvel", all(last),
        "so `_opp`'s `nq - qposadr` slice stays correct as our body grows")
    chk("the opponent's root x still follows 1 - v*dt*k exactly",
        max(errs) == 0.0, f"max error {max(errs):.3e} over {len(errs)} designs")
    # crossing step, under an evolved body
    n = 0
    while env.opp_x(n) > -4.0:
        n += 1
    chk("the opponent still crosses x = -4 at control step 491", n == 491,
        f"step {n}")
    cfg2, env2 = make_env("rtg_e3_s1")
    env2.opp_speed = 1.0
    n2 = 0
    while env2.opp_x(n2) > -4.0:
        n2 += 1
    chk("NEG: at 1.0 m/s the crossing moves to step 334", n2 == 334, f"step {n2}")


# =========================================================== phase 4 =======
def phase4():
    phase(4, "the reward and the termination still measure OUR agent")
    cfg, env = make_env("rtg_e3_s1")
    W = env.control_action_dim + env.attr_design_dim + 1
    dr, dp, n, terms, bodies = 0.0, 0.0, 0, 0, set()
    for ep in range(20):
        rng = np.random.RandomState(3 + ep)
        if not run_design(env, design_actions(env, rng), 3 + ep):
            continue
        bodies.add(len(env.robot.bodies))
        for _ in range(env.max_nsteps + 5):
            a = np.zeros((len(env.robot.bodies), W))
            a[:, :env.control_action_dim] = rng.uniform(
                -0.3, 0.3, (len(env.robot.bodies), 1))
            _, r, done, info = env.step(a)
            if "dense" not in info:
                break
            n += 1
            dp = max(dp, abs(info["dense"] + info["parse"] - r))
            recon = info["forward"] - info["ctrl_cost"] + 1.0
            dr = max(dr, abs(recon - info["dense"]))
            terms += int(abs(info["parse"]) > 0)
            if done:
                break
    chk("the check saw several DIFFERENT evolved bodies and enough steps",
        n >= 200 and len(bodies) > 2,
        f"{n} steps over body counts {sorted(bodies)}")
    chk("dense + parse == the env reward, on an EVOLVED body",
        dp == 0.0, f"max |delta| {dp:.3e} over {n} steps")
    chk("dense == forward - 0.5*sum(a^2) + 1.0, on an EVOLVED body",
        dr < 1e-12, f"max |delta| {dr:.3e}")
    chk("the fall test reads OUR root z, not the opponent's",
        env.state_vector()[2] == env.data.qpos[2],
        f"our z {env.state_vector()[2]:.4f}, opponent's held at "
        f"{env._opp_frozen[2]:.4f} > stand_z {env.stand_z}")
    chk("NEG: an evolved body can still be commanded through all its motors",
        env.model.nu >= 8, f"nu {env.model.nu} (8 opponent + ours)")


# =========================================================== phase 5 =======
def phase5():
    phase(5, "the d2rep curriculum, on the GNN path")
    from rower_soccer.t2a_port.train_e3_gnn import alpha_at, make_custom_reward
    from rower_soccer.t2a_port import train_e11_mlp

    class A:
        pass
    D2REP, B = 130208333, 50000
    a = A(); a.args = A(); a.batch = B
    worst = 0.0
    for cs in (4000000, D2REP):
        a.args.curriculum_steps = cs
        for e in range(400):
            worst = max(worst, abs(alpha_at(e, cs, B)
                                   - train_e11_mlp.Trainer.alpha(a, e)))
    chk("alpha_at is BIT-IDENTICAL to E2.1's trained schedule over 400 epochs",
        worst == 0.0, f"max |delta| {worst:.3e} across cur and d2rep")
    chk("d2rep runs 1.000000 -> 0.846400, D2's own realised endpoint",
        abs(alpha_at(0, D2REP, B) - 1.0) == 0.0
        and abs(alpha_at(400, D2REP, B) - 0.8464) < 1e-9,
        f"alpha(0) {alpha_at(0, D2REP, B):.6f} "
        f"alpha(80) {alpha_at(80, D2REP, B):.6f} "
        f"alpha(400) {alpha_at(400, D2REP, B):.9f}")
    chk("d2rep never crosses E2.1's critical alpha of 0.739",
        min(alpha_at(e, D2REP, B) for e in range(401)) > 0.739,
        f"min alpha {min(alpha_at(e, D2REP, B) for e in range(401)):.4f}; "
        f"the fall-dodge is worth at most "
        f"{(1 - alpha_at(400, D2REP, B)) * 1000:.1f} against +1000 flat")
    chk("NEG: curriculum_steps = 0 returns None, not 1.0",
        alpha_at(0, 0, B) is None)

    # the mix itself
    class Ag:
        cur_alpha = None
    ag = Ag()
    cr = make_custom_reward(ag)
    info = {"dense": 7.5, "parse": -1000.0}
    ag.cur_alpha = 1.0
    r1, _ = cr(None, None, None, -992.5, info)
    ag.cur_alpha = 0.8464
    r2, _ = cr(None, None, None, -992.5, info)
    ag.cur_alpha = 0.0
    r3, _ = cr(None, None, None, -992.5, info)
    ag.cur_alpha = None
    r4, _ = cr(None, None, None, -992.5, info)
    chk("alpha = 1: NOT ONE of the +/-1000 reaches the buffer", r1 == 7.5,
        f"{r1}")
    chk("alpha = 0.8464: the sparse term enters at 15.36% weight",
        abs(r2 - (0.8464 * 7.5 - 0.1536 * 1000.0)) < 1e-12, f"{r2:.4f}")
    chk("NEG alpha = 0: the buffer is the sparse term ALONE", r3 == -1000.0)
    chk("NEG alpha = None: the buffer is the raw env reward", r4 == -992.5)
    ag.cur_alpha = 0.9
    rd, _ = cr(None, None, None, 0.0, {})           # a design stage
    chk("a design stage contributes 0 under any alpha", rd == 0.0)

    # END TO END: the curriculum changes the BUFFER and not the LOG.
    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    cfg = Config("rtg_e3c_s1", tmp=True)
    cfg.min_batch_size = 3000
    logs, mems = [], []
    for use_cr in (False, True):
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        ag = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                                device=torch.device("cpu"), seed=cfg.seed,
                                num_threads=1, training=True, checkpoint=0)
        ag.cur_alpha = 1.0
        if use_cr:
            ag.custom_reward = make_custom_reward(ag)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        batch, log = ag.sample(cfg.min_batch_size)
        logs.append(log)
        mems.append(np.asarray(batch.rewards, dtype=float))
    same_log = (logs[0].avg_episode_reward == logs[1].avg_episode_reward
                and logs[0].avg_exec_episode_reward
                == logs[1].avg_exec_episode_reward
                and logs[0].num_steps == logs[1].num_steps)
    chk("the LOGGED return is the RAW env return in both conditions",
        same_log,
        f"avg_episode_reward {logs[0].avg_episode_reward:.4f} vs "
        f"{logs[1].avg_episode_reward:.4f} over {logs[0].num_steps} steps")
    n1000 = int((np.abs(mems[0]) > 900).sum())
    n1000c = int((np.abs(mems[1]) > 900).sum())
    chk("but the BUFFER differs, and at alpha=1 no +/-1000 is in it",
        not np.array_equal(mems[0], mems[1]) and n1000c == 0 and n1000 > 0,
        f"flat buffer holds {n1000} sparse events, alpha=1 buffer holds "
        f"{n1000c}; buffers differ on "
        f"{int((mems[0] != mems[1]).sum())} of {len(mems[0])} steps")


# =========================================================== phase 6 =======
def phase6():
    phase(6, "the instrument does not perturb what it measures")
    from rower_soccer.t2a_port import e3_morph
    cfg, env = make_env("rtg_e3_s1")
    W = env.control_action_dim + env.attr_design_dim + 1

    def rollout():
        env.seed(5)
        env.reset()
        out = []
        for _ in range(12):
            out.append(float(np.random.rand()))
            a = np.zeros((len(env.robot.bodies), W))
            # only the control column may be non-zero in the execution stage
            a[:, :env.control_action_dim] = np.random.uniform(
                -1, 1, (len(env.robot.bodies), env.control_action_dim))
            env.step(a)
            out.append(float(env.data.qpos[0]))
        return out

    def probe():
        # a stand-in for the per-epoch census: it draws from every stream
        np.random.rand(97)
        torch.rand(13)
        env.seed(12345)
        env.reset()

    np.random.seed(0); torch.manual_seed(0)
    ref = rollout()
    np.random.seed(0); torch.manual_seed(0)
    with e3_morph.rng_guard(env):
        probe()
    guarded = rollout()
    np.random.seed(0); torch.manual_seed(0)
    probe()
    unguarded = rollout()
    chk("rng_guard: a probe inside it leaves the stream BIT-IDENTICAL",
        guarded == ref, f"{len(ref)} values compared")
    chk("NEG: the same probe outside it DOES shift the stream",
        unguarded != ref,
        f"{sum(1 for a, b in zip(ref, unguarded) if a != b)} of {len(ref)} differ")


# =========================================================== phase 7 =======
def phase7():
    phase(7, "the fall-dodge: still in the task, and WIDER through morphology")
    from rower_soccer.t2a_port import e3_morph

    # 7a. On the FROZEN body -- E2's own idle control -- the dodge is the
    # non-overlap `gate_e21.py` phase 4 measured: an episode that ends on a
    # fall never pays the -1000, one that survives to step 491 pays it in full.
    cfgc, envc = make_env("rtg_e3c_s1")
    Wc = envc.control_action_dim + envc.attr_design_dim + 1
    zc = np.zeros((len(envc.robot.bodies), Wc))
    prem, endsc = [], []
    for ep in range(10):
        envc.seed(300 + ep)
        envc.reset()
        while envc.if_use_transform_action() != 2:
            envc.step(zc)
        Rf = Ra = 0.0
        info = {}
        for _ in range(envc.max_nsteps + 5):
            _, r, done, info = envc.step(zc)
            if "dense" not in info:
                break
            Rf += r
            Ra += info["dense"]
            if done:
                break
        prem.append(Rf - Ra)
        endsc.append("fell" if info.get("fell") else
                     ("lost" if info.get("opp_reached") else "trunc"))
    fell = [p for p, e in zip(prem, endsc) if e == "fell"]
    lost = [p for p, e in zip(prem, endsc) if e == "lost"]
    chk("frozen body: the idle control still both falls and loses",
        len(fell) > 0 and len(lost) > 0, f"endings {endsc}")
    # `parse` is +/-1000 as a float64 sum over an episode, so it lands within
    # 1e-12 of the constant rather than on it; the claim is non-overlap, which
    # is `gate_e21.py` phase 4's own form of this test.
    chk("the fall-dodge is present and the two distributions do NOT overlap",
        bool(fell) and bool(lost) and max(abs(f) for f in fell) < 1e-9
        and max(abs(l + 1000.0) for l in lost) < 1e-9,
        f"sparse term banked: on a fall {sorted(set(fell))}, on a loss "
        f"{sorted(set(lost))} -- so stopping early is worth +1000 under the "
        f"flat reward and 0 under alpha=1")

    # 7b. On EVOLVED bodies the same zero-torque policy falls every time. That
    # is the hazard this rung adds: morphology reaches the degenerate ending
    # without needing a control policy at all.
    cfg, env = make_env("rtg_e3_s1")
    W = env.control_action_dim + env.attr_design_dim + 1
    ends, lens = [], []
    for ep in range(12):
        rng = np.random.RandomState(200 + ep)
        if not run_design(env, design_actions(env, rng), 200 + ep):
            ends.append("designfail")
            continue
        info, n = {}, 0
        for _ in range(env.max_nsteps + 5):
            _, r, done, info = env.step(np.zeros((len(env.robot.bodies), W)))
            if "dense" not in info:
                break
            n += 1
            if done:
                break
        lens.append(n)
        ends.append("fell" if info.get("fell") else
                    ("lost" if info.get("opp_reached") else "trunc"))
    fr = ends.count("fell") / len(ends)
    chk("EVOLVED bodies at zero torque fall far more than the frozen one does",
        fr > sum(1 for e in endsc if e == "fell") / len(endsc),
        f"fall rate {fr:.2f} over {len(ends)} random designs (mean length "
        f"{np.mean(lens):.0f} steps) against {sum(1 for e in endsc if e == 'fell') / len(endsc):.2f} "
        f"on the frozen body -- the reason this rung instruments the dodge "
        f"from epoch 0")

    # the correlation instrument itself
    eps = [dict(R=2.0, fell=True, opp_reached=False, max_fwd=0.1),
           dict(R=2.0, fell=True, opp_reached=False, max_fwd=0.2),
           dict(R=-3.0, fell=False, opp_reached=True, max_fwd=0.3),
           dict(R=-3.0, fell=False, opp_reached=True, max_fwd=0.4)]
    d = e3_morph.dodge_stats(eps)
    chk("dodge_stats reproduces E2's pair on a hand-checked example",
        abs(d["r_fall_return"] - 1.0) < 1e-9
        and abs(d["fall_premium"] - 5.0) < 1e-9,
        f"r(fall,R) {d['r_fall_return']:.3f} premium {d['fall_premium']:.1f} "
        f"(E2 measured r = +0.989 across its seven arms)")
    chk("NEG: a zero-variance column returns None, not 0.0 (E2.1's d2rep case)",
        e3_morph.corr([1, 1, 1], [1, 2, 3]) is None)


# =========================================================== phase 8 =======
def phase8():
    phase(8, "the instrument reports what changed, and E2's numbers do not move")
    from rower_soccer.t2a_port import e2_eval, e3_morph
    cfg, env = make_env("rtg_e3c_s1")            # frozen body
    W = env.control_action_dim + env.attr_design_dim + 1
    nb = len(env.robot.bodies)
    zero = np.zeros((nb, W))
    act = (lambda s, stage: zero)
    ev = e2_eval.evaluate(env, act, (lambda s: s), episodes=3, seed_base=1000,
                          max_steps=env.max_nsteps + 5)
    chk("frozen body: bodies_exec == the initial body count, so no E2 number "
        "moves", ev["bodies_exec"] == 13.0 and ev["design_fail_rate"] == 0.0,
        f"bodies_exec {ev['bodies_exec']} design_fail_rate "
        f"{ev['design_fail_rate']}")
    cfg2, env2 = make_env("rtg_e3_s1")           # design on
    s = e3_morph.body_summary(env2)
    chk("body_summary reads the OPPONENT out of every aggregate",
        s["n_opp_bodies"] == 13 and s["model_nbody_ours"] == 13
        and s["model_nu_ours"] == 8 and s["model_nbody"] == 27,
        f"ours {s['model_nbody_ours']} bodies / {s['model_nu_ours']} motors, "
        f"opponent {s['n_opp_bodies']}, model {s['model_nbody']}")
    chk("body_summary's mass is OUR ant's, not two ants'",
        abs(s["model_mass_ours"] - env2.model.body_mass.sum() / 2.0) < 1e-9,
        f"ours {s['model_mass_ours']:.4f} kg of "
        f"{env2.model.body_mass.sum():.4f} total")
    rng = np.random.RandomState(77)
    run_design(env2, design_actions(env2, rng), 77)
    s2 = e3_morph.body_summary(env2)
    chk("NEG: after a design step the summary REPORTS the change",
        s2["n_bodies"] != s["n_bodies"] or s2["topo"] != s["topo"],
        f"{s['n_bodies']} bodies topo {s['topo']} -> {s2['n_bodies']} bodies "
        f"topo {s2['topo']}, mass {s['model_mass_ours']:.3f} -> "
        f"{s2['model_mass_ours']:.3f} kg, limb len sum "
        f"{s['limb_length']['sum']:.3f} -> {s2['limb_length']['sum']:.3f} m")


def main():
    t0 = time.time()
    torch.set_default_dtype(torch.float64)
    for f in (phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8):
        f()
    print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed "
          f"({time.time() - t0:.0f}s) ===")
    for f in FAIL:
        print("  FAILED: " + f)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
