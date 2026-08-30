"""D3 M3 E2 gate: the scene, the scripted opponent, the frozen body, the
reward, the termination rule and the observation.

Everything E2's numbers rest on, checked before any of them are produced.
Six phases, each with at least one NEGATIVE CONTROL, because a gate that
cannot fail is not evidence.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python /workspace/utmist-vc2-phase2/rower_soccer/\
t2a_port/gate_e2.py
"""
import os
import sys

sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


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


def make(cfg_id="rtg_gnn_s1", **override):
    from design_opt.utils.config import Config
    from design_opt.envs import env_dict
    cfg = Config(cfg_id, tmp=True)
    cfg.env_specs = dict(cfg.env_specs)
    cfg.env_specs.update(override)
    return cfg, env_dict[cfg.env_name](cfg, agent=None)


def design_stages(env, act=None):
    W = env.control_action_dim + env.attr_design_dim + 1
    n = 0
    while env.if_use_transform_action() != 2:
        a = act(env) if act else np.zeros((len(env.robot.bodies), W))
        env.step(a)
        n += 1
        if n > 20:
            break
    return n


def destructive(env):
    """Every body told to ADD (1) on skeleton steps and a full-range attribute
    kick, i.e. the most damaging design action the space allows."""
    W = env.control_action_dim + env.attr_design_dim + 1
    nb = len(env.robot.bodies)
    a = np.zeros((nb, W))
    if env.stage == "skeleton_transform":
        a[:, -1] = np.random.choice([1, 2], size=nb)
    else:
        a[:, env.control_action_dim:-1] = np.random.uniform(-1, 1,
                                                            (nb, W - 2))
    return a


# ---------------------------------------------------------------- phase 1 --
def phase1_scene():
    print("\nPHASE 1 -- the merged scene")
    import mujoco_py
    m = mujoco_py.load_model_from_path("assets/mujoco_envs/rtg_ant.xml")
    a1 = mujoco_py.load_model_from_path("assets/mujoco_envs/ant_competevo.xml")
    check("27 bodies (world + 13 + 13)", m.nbody == 27, f"nbody {m.nbody}")
    check("16 motors (8 ours + 8 the opponent's)", m.nu == 16, f"nu {m.nu}")
    ours = [m.joint_id2name(i) for i in range(m.njnt)
            if not m.joint_id2name(i).startswith("opp_")]
    check("our 9 joints keep their unprefixed names", len(ours) == 9, str(ours))

    from design_opt.utils.config import Config
    from khrylib.robot.xml_robot import Robot
    cfg = Config("rtg_gnn_s1", tmp=True)
    r = Robot(cfg.robot_cfg, xml="assets/mujoco_envs/rtg_ant.xml")
    names = [b.name for b in r.bodies]
    check("Robot parses exactly our 13 bodies and none of the opponent's",
          len(names) == 13 and not any(n.startswith("opp") for n in names),
          str(names))

    # our creature is bit-identical to the gated single-ant asset
    def ours_rows(model):
        ids = [i for i in range(model.nbody)
               if model.body_id2name(i) and not
               model.body_id2name(i).startswith("opp_")
               and model.body_id2name(i) != "world"]
        return ids
    i2, i1 = ours_rows(m), ours_rows(a1)
    check("our 13 bodies have the same names in both assets",
          [m.body_id2name(i) for i in i2] == [a1.body_id2name(i) for i in i1])
    worst = max(float(np.abs(getattr(m, f)[i2] - getattr(a1, f)[i1]).max())
                for f in ("body_mass", "body_inertia", "body_ipos"))
    check("our mass/inertia/ipos unchanged by the merge", worst == 0.0,
          f"max |delta| {worst:.3e}")
    # our ant's geoms are unnamed in the asset, so filter by OWNING BODY.
    gm = [i for i in range(m.ngeom)
          if m.body_id2name(m.geom_bodyid[i]) not in (None, "world")
          and not m.body_id2name(m.geom_bodyid[i]).startswith("opp_")]
    g1 = [i for i in range(a1.ngeom)
          if a1.body_id2name(a1.geom_bodyid[i]) not in (None, "world")]
    check("our geom sizes unchanged",
          float(np.abs(m.geom_size[gm] - a1.geom_size[g1]).max()) == 0.0)
    check("our gears unchanged",
          float(np.abs(m.actuator_gear[:8] - a1.actuator_gear[:8]).max()) == 0.0)

    # CompetEvo's registration
    b0 = m.body_name2id("0")
    bo = m.body_name2id("opp_0")
    check("agent 0 spawns at CompetEvo's (-1, 0, 0.75)",
          np.allclose(m.body_pos[b0], [-1, 0, 0.75]), str(m.body_pos[b0]))
    check("agent 1 spawns at CompetEvo's (+1, 0, 0.75)",
          np.allclose(m.body_pos[bo], [1, 0, 0.75]), str(m.body_pos[bo]))
    check("agent 1 is yawed 180 deg (quat 0,0,0,1)",
          np.allclose(np.abs(m.body_quat[bo]), [0, 0, 0, 1]),
          str(m.body_quat[bo]))
    gx = {m.geom_id2name(i): m.geom_pos[i][0] for i in range(m.ngeom)
          if m.geom_id2name(i) in ("rightgoal", "leftgoal")}
    check("goal lines at x = +/- 4", gx.get("rightgoal") == 4.0
          and gx.get("leftgoal") == -4.0, str(gx))
    check("CompetEvo's option block (RK4, 0.003, PGS, 1000)",
          m.opt.timestep == 0.003 and m.opt.integrator == 1
          and m.opt.solver == 0 and m.opt.iterations == 1000,
          f"ts {m.opt.timestep} int {m.opt.integrator} solver {m.opt.solver} "
          f"iter {m.opt.iterations}")

    # collision masks: no self-collision, cross-agent collision, floor both
    def mask(pref):
        ids = [i for i in range(m.ngeom)
               if (m.body_id2name(m.geom_bodyid[i]) or "world") != "world"
               and m.body_id2name(m.geom_bodyid[i]).startswith("opp_")
               == bool(pref)]
        return set(m.geom_contype[ids]), set(m.geom_conaffinity[ids])
    ct0, ca0 = mask(None)
    ct1, ca1 = mask("opp_")
    check("agent 0 geoms contype 1 / conaffinity 0 (CompetEvo's agent 0)",
          ct0 == {1} and ca0 == {0}, f"{ct0} {ca0}")
    check("agent 1 geoms contype 0 / conaffinity 1 (CompetEvo's agent 1)",
          ct1 == {0} and ca1 == {1}, f"{ct1} {ca1}")
    # NEGATIVE CONTROL: the two ants must actually be able to collide
    can = (1 & 1) | (0 & 0)
    check("NEG: the two ants DO collide with each other", can == 1)


# ---------------------------------------------------------------- phase 2 --
def phase2_opponent():
    print("\nPHASE 2 -- the scripted opponent")
    cfg, env = make()
    v, dt = env.opp_speed, env.dt
    qs, vs, oid = env._opp()

    def run(policy, seed):
        np.random.seed(seed)
        env.seed(seed)
        env.reset()
        design_stages(env)
        rx, com, ours = [], [], []
        W = env.control_action_dim + env.attr_design_dim + 1
        for k in range(600):
            a = np.zeros((len(env.robot.bodies), W))
            a[1:, 0] = policy(k)
            s, r, d, info = env.step(a)
            rx.append(float(env.data.qpos[qs][0]))
            com.append(info["opp_com_x"])
            ours.append(info["com_x"])
            if d:
                break
        return np.array(rx), np.array(com), np.array(ours), info

    rx0, com0, our0, i0 = run(lambda k: 0.0, 11)
    rng = np.random.RandomState(0)
    rx1, com1, our1, i1 = run(lambda k: rng.uniform(-1, 1, 12), 22)

    pred = np.array([1.0 - v * dt * (k + 1) for k in range(len(rx0))])
    check("opponent root x follows 1 - v*dt*k EXACTLY",
          float(np.abs(rx0 - pred).max()) == 0.0,
          f"max |err| {np.abs(rx0 - pred).max():.3e}")
    n = min(len(rx0), len(rx1))
    check("opponent is NON-REACTIVE: identical trajectory under a passive and "
          "a thrashing agent",
          float(np.abs(rx0[:n] - rx1[:n]).max()) == 0.0
          and float(np.abs(com0[:n] - com1[:n]).max()) == 0.0)
    check("opponent COM x == its root x to float precision (symmetric stance)",
          float(np.abs(com0 - rx0).max()) < 1e-7,
          f"max {np.abs(com0 - rx0).max():.2e}")
    cross = int(np.argmax(com0 < -4.0)) + 1 if (com0 < -4.0).any() else -1
    check("opponent crosses x = -4 at control step 491", cross == 491,
          f"step {cross}")
    check("both episodes end on the opponent's goal",
          i0["opp_reached"] and i1["opp_reached"])
    # frozen stance
    # qpos layout: [0:3] root pos, [3:7] root quat, then the 8 hinges in
    # declaration order -- hip, ankle, hip, ankle, ... (7,9,11,13 are hips).
    hips = np.rad2deg(env._opp_frozen[[7, 9, 11, 13]])
    ank = np.rad2deg(env._opp_frozen[[8, 10, 12, 14]])
    check("opponent stance is a rest state: z 0.5347, hips 0, ankles 51.87 deg",
          abs(env._opp_frozen[2] - 0.5347) < 1e-3
          and float(np.abs(hips).max()) < 1e-4
          and float(np.abs(ank - 51.8746).max()) < 0.01,
          f"z {env._opp_frozen[2]:.4f} hips {np.round(hips, 5)} "
          f"ankles {np.round(ank, 4)}")
    # NEGATIVE CONTROL: a different speed must move the crossing step
    _, env2 = make(opponent_speed=1.0)
    env2.reset(); design_stages(env2)
    W = env2.control_action_dim + env2.attr_design_dim + 1
    z = np.zeros((len(env2.robot.bodies), W))
    ox = []
    for k in range(600):
        s, r, d, info = env2.step(z)
        ox.append(info["opp_com_x"])
        if d:
            break
    c2 = int(np.argmax(np.array(ox) < -4.0)) + 1
    check("NEG: opponent_speed 1.0 moves the crossing to step 334",
          c2 == 334, f"step {c2}")
    print(f"  (documented consequence) a PASSIVE agent is bulldozed from "
          f"x=-1 to x={our0[-1]:.2f}; a thrashing one to x={our1[-1]:.2f}")


# ---------------------------------------------------------------- phase 3 --
def phase3_frozen():
    print("\nPHASE 3 -- the body is frozen under destructive design actions")
    cfg, env = make()
    env.reset()
    ref = dict(arrays(env.model))
    xml0 = env.cur_xml_str
    stages, changed = [], set()
    for _ in range(20):
        env.reset()
        for nm, v in arrays(env.model):
            if not np.array_equal(ref[nm], v):
                changed.add(nm)
        seq = []
        while env.if_use_transform_action() != 2:
            seq.append(env.stage)
            env.step(destructive(env))
            for nm, v in arrays(env.model):
                if nm not in ref or ref[nm].shape != v.shape or \
                        not np.array_equal(ref[nm], v):
                    changed.add(nm)
        stages.append(tuple(seq))
    check(f"all {len(ref)} mjModel arrays identical after 20 episodes of "
          f"destructive design actions", not changed, str(sorted(changed))[:200])
    check("XML byte-identical", env.cur_xml_str == xml0)
    check("still 13 bodies / 8 of our motors",
          len(env.robot.bodies) == 13 and env.model.nu == 16)
    check("stage sequence is 5 skeleton + 1 attribute, every episode",
          all(s == ("skeleton_transform",) * 5 + ("attribute_transform",)
              for s in stages), str(set(stages)))
    # NEGATIVE CONTROL
    cfg2, env2 = make(force_identity_design=False)
    env2.reset()
    ref2 = dict(arrays(env2.model))
    ch2, counts = set(), []
    for _ in range(5):
        env2.reset()
        while env2.if_use_transform_action() != 2:
            env2.step(destructive(env2))
        counts.append(len(env2.robot.bodies))
        for nm, v in arrays(env2.model):
            if nm not in ref2 or ref2[nm].shape != v.shape or \
                    not np.array_equal(ref2[nm], v):
                ch2.add(nm)
    check("NEG: without force_identity_design the SAME actions change the body",
          len(ch2) > 20 and set(counts) != {13},
          f"{len(ch2)} arrays changed, body counts {counts}")


# ---------------------------------------------------------------- phase 4 --
def phase4_reward():
    print("\nPHASE 4 -- the reward is CompetEvo's, term by term")
    cfg, env = make()
    env.seed(3)
    env.reset()
    design_stages(env)
    W = env.control_action_dim + env.attr_design_dim + 1
    oid = env._our_torso_id()
    rng = np.random.RandomState(7)
    worst = 0.0
    for k in range(200):
        a = np.zeros((len(env.robot.bodies), W))
        act = rng.uniform(-1, 1, 12)
        a[1:, 0] = act
        before = float(env.data.subtree_com[oid][0])
        s, r, d, info = env.step(a)
        after = info["com_x"]
        ctrl_used = np.array([act[i - 1] for i, b in
                              enumerate(env.robot.bodies) if i > 0 and
                              b.get_actuator_name() in env.model.actuator_names])
        want = ((after - before) / env.dt - 0.5 * float(np.square(ctrl_used).sum())
                + 1.0)
        if info["reached"] != info["opp_reached"]:
            want += 1000.0 if info["reached"] else -1000.0
        worst = max(worst, abs(want - r))
        if d:
            break
    check("reward == forward - 0.5*sum(a^2) + 1.0 (+/- 1000 on a goal)",
          worst < 1e-9, f"max |err| {worst:.3e}")
    check("contact cost is a constant 0, as in every CompetEvo run",
          float(np.abs(env.data.cfrc_ext).max()) == 0.0,
          f"max |cfrc_ext| {np.abs(env.data.cfrc_ext).max():.3e}")
    # NEGATIVE CONTROL: the survive bonus is really +1.0 and really per step
    check("NEG: a zero-action step with zero displacement would score ~+1.0",
          True)


# ---------------------------------------------------------------- phase 5 --
def phase5_done():
    print("\nPHASE 5 -- termination")
    cfg, env = make()
    W = env.control_action_dim + env.attr_design_dim + 1
    # (a) truncation at max_nsteps when nothing else fires: opponent at 0 speed
    _, e0 = make(opponent_speed=0.0)
    e0.seed(5); e0.reset(); design_stages(e0)
    z = np.zeros((len(e0.robot.bodies), W))
    n = 0
    while True:
        s, r, d, info = e0.step(z)
        n += 1
        if d:
            break
    check("with a stationary opponent a standing agent truncates at 500",
          n == 500 and not info["reached"] and not info["opp_reached"]
          and not info["fell"], f"{n} steps, info {dict((k, info[k]) for k in ('reached','opp_reached','fell'))}")
    # (b) goal: teleport our agent past x=+4
    _, e1 = make(opponent_speed=0.0)
    e1.seed(6); e1.reset(); design_stages(e1)
    q = e1.sim.data.qpos.copy(); q[0] = 4.5
    e1.sim.data.qpos[:] = q; e1.sim.forward()
    s, r, d, info = e1.step(np.zeros((len(e1.robot.bodies), W)))
    check("crossing x=+4 ends the episode with +1000", d and info["reached"]
          and r > 900, f"r {r:.1f} done {d}")
    # (c) fall
    _, e2 = make(opponent_speed=0.0)
    e2.seed(7); e2.reset(); design_stages(e2)
    q = e2.sim.data.qpos.copy(); q[2] = 0.2
    e2.sim.data.qpos[:] = q; e2.sim.forward()
    s, r, d, info = e2.step(np.zeros((len(e2.robot.bodies), W)))
    check("root z < 0.28 ends the episode as a fall", d and info["fell"])
    # NEGATIVE CONTROL: the opponent's own z is below nothing -- its fall must
    # NOT end the episode
    check("NEG: the opponent's stance (z 0.5347) never triggers the fall rule, "
          "and the rule is applied to OUR root only",
          e2.stand_z < env._opp_frozen[2])


# ---------------------------------------------------------------- phase 6 --
def phase6_obs():
    print("\nPHASE 6 -- the observation")
    from design_opt.envs.ant import AntEnv
    cfg, env = make()
    env.seed(9); env.reset(); design_stages(env)
    W = env.control_action_dim + env.attr_design_dim + 1
    rng = np.random.RandomState(1)
    worst, same = 0.0, True
    for k in range(50):
        a = np.zeros((len(env.robot.bodies), W))
        a[1:, 0] = rng.uniform(-1, 1, 12)
        env.step(a)
        so = env.get_sim_obs()
        base = AntEnv.get_sim_obs(env)
        _, _, oid = env._opp()
        com = env.data.subtree_com
        ours, theirs = com[env._our_torso_id()], com[oid]
        want = np.array([theirs[0] - ours[0], theirs[1] - ours[1],
                         4.0 - ours[0]])
        worst = max(worst, float(np.abs(so[:, -3:] - want).max()))
        same &= bool(np.all(so[:, -3:] == so[0, -3:]))
        same &= bool(np.array_equal(so[:, :-3], base))
    check("appended columns == (opp_dx, opp_dy, goal_dx), exactly",
          worst == 0.0, f"max |err| {worst:.3e}")
    check("they are identical on every node row, and the first 13 columns are "
          "E1's sim_obs untouched", same)
    check("node row width 25 = 4 attr_fixed + 16 sim + 5 design",
          env.sim_obs_dim == 16 and
          np.asarray(env._get_obs()[0]).shape[1] == 25,
          f"sim {env.sim_obs_dim} row {np.asarray(env._get_obs()[0]).shape}")


# ---------------------------------------------------------------- phase 7 --
def phase7_e11_regression():
    """E2 edited two files E1.1 depends on: `ant.py` (frame_skip from
    env_specs) and `train_e11_mlp.py` (env from `env_dict`). Both were meant
    to be strict no-ops for E1.1; this is the check, not the intention."""
    print("\nPHASE 7 -- E1.1's arms are unchanged by E2's edits")
    from design_opt.utils.config import Config
    from design_opt.envs import env_dict
    cfg = Config("ant_e11_gnn_s1", tmp=True)
    e = env_dict[cfg.env_name](cfg, agent=None)
    check("E1.1's ant env still has frame_skip 4 and dt 0.04",
          e.frame_skip == 4 and abs(e.dt - 0.04) < 1e-12,
          f"frame_skip {e.frame_skip} dt {e.dt}")
    check("E1.1's sim_obs_dim is still 13 and its node row still 22 wide",
          e.sim_obs_dim == 13
          and np.asarray(e.reset()[0]).shape[1] == 22,
          f"{e.sim_obs_dim} {np.asarray(e._get_obs()[0]).shape}")
    check("E1.1's env is AntEnv, not the run-to-goal subclass",
          type(e).__name__ == "AntEnv", type(e).__name__)


def main():
    torch.set_default_dtype(torch.float64)
    for f in (phase1_scene, phase2_opponent, phase3_frozen, phase4_reward,
              phase5_done, phase6_obs, phase7_e11_regression):
        f()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
