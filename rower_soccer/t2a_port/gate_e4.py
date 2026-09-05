"""D3 M3 E4 GATE. Six assertions that must all pass before any E4 arm launches.

The briefing's rule for E3 was that a design stage which silently no-ops would
produce "a clean, boring, completely wrong null that looks exactly like a real
result". E4's equivalent failure is an opponent that is present in the XML but
inert, or one that acts in the wrong frame: either would give a tidy
convergence number that means nothing. These gates are the mirror of E2.1's.

  1  REGRESSION      no opponent_src -> byte-identical to the checked-in scene
  2  HETEROGENEOUS   an evolved opponent's bodies/motors match its source
  3  ROTATION        the pi-z transform is exact, in observation AND in physics
  4  COST ISOLATION  the learner is NOT charged for the opponent's torques
  5  SCRIPTED PARITY opponent_mode=scripted reproduces run_to_goal exactly
  6  OPPONENT BITES  a real opponent changes the outcome vs an inert one
                     -- the negative control; without it the coupled channel
                     could be silently dead.
"""
import argparse, os, pickle, sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/Transform2Act")
sys.path.insert(0, "/workspace/utmist-vc2-phase2")

torch.set_default_dtype(torch.float64)

FAILS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def build_agent(cfg_id, seed=0):
    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    cfg = Config(cfg_id, tmp=True)
    return Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                              device=torch.device("cpu"), seed=seed,
                              num_threads=1, training=False, checkpoint=0)


# ------------------------------------------------------------------ 1, 2 --
def gate_scene():
    from rower_soccer.t2a_port import rtg_scene
    from lxml import etree
    from lxml.etree import parse, XMLParser
    import mujoco_py, tempfile

    base = "/workspace/Transform2Act/assets/mujoco_envs/ant_competevo.xml"
    ref = "/workspace/Transform2Act/assets/mujoco_envs/rtg_ant.xml"
    new = etree.tostring(rtg_scene.build(base), pretty_print=True)
    check("1 REGRESSION: build(src) byte-identical to rtg_ant.xml",
          new == open(ref, "rb").read(), f"{len(new)} bytes")

    for body in ("rtg_evolved_s2body.xml", "rtg_evolved_s3body.xml"):
        src = f"/workspace/Transform2Act/assets/mujoco_envs/{body}"
        t = rtg_scene.build(base, src)
        p = tempfile.mktemp(suffix=".xml")
        open(p, "w").write(etree.tostring(t, pretty_print=True).decode())
        m = mujoco_py.load_model_from_path(p)
        os.unlink(p)
        nopp = sum(1 for n in m.body_names if n.startswith("opp_"))
        mopp = sum(1 for n in m.actuator_names if n.startswith("opp_"))
        r = parse(src, XMLParser(remove_blank_text=True)).getroot()
        b = r.find("worldbody").find("body")
        want_b = len(list(b.iter("body")))
        want_m = len([e for e in r.find("actuator").findall("motor")
                      if not e.attrib["joint"].startswith("opp_")])
        check(f"2 HETEROGENEOUS: {body} -> opponent {nopp}b/{mopp}m",
              nopp == want_b and mopp == want_m,
              f"source says {want_b}b/{want_m}m")


# --------------------------------------------------------------------- 3 --
def rotated_equivalent(env, qpos, qvel):
    """Write our agent's root state and the pi-z image of it into the
    opponent's slot; joint angles and joint velocities are intrinsic and are
    copied across unchanged."""
    from design_opt.envs.run_to_goal_sp import qmul_zpi
    qs, vs, _ = env._opp()
    d = env.sim.data
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    # ours: root at [0:7], joints after
    d.qpos[0:7] = qpos[:7]
    d.qvel[0:6] = qvel[:6]
    njoint_q = qs.start - 7
    d.qpos[7:7 + njoint_q] = qpos[7:7 + njoint_q]
    d.qvel[6:6 + njoint_q] = qvel[6:6 + njoint_q]
    # theirs: rotated root, identical joints
    d.qpos[qs.start + 0] = -qpos[0]
    d.qpos[qs.start + 1] = -qpos[1]
    d.qpos[qs.start + 2] = qpos[2]
    d.qpos[qs.start + 3:qs.start + 7] = qmul_zpi(qpos[3:7])
    d.qvel[vs.start + 0] = -qvel[0]
    d.qvel[vs.start + 1] = -qvel[1]
    d.qvel[vs.start + 2] = qvel[2]
    d.qvel[vs.start + 3:vs.start + 6] = qvel[3:6]
    d.qpos[qs.start + 7:qs.start + 7 + njoint_q] = qpos[7:7 + njoint_q]
    d.qvel[vs.start + 6:vs.start + 6 + njoint_q] = qvel[6:6 + njoint_q]
    env.sim.forward()


def gate_rotation(agent):
    env = agent.env
    env.stage = "execution"
    rng = np.random.RandomState(0)
    worst_obs = 0.0
    for trial in range(5):
        q = np.zeros(env.model.nq)
        v = np.zeros(env.model.nv)
        q[0], q[1], q[2] = rng.uniform(-1, 1), rng.uniform(-1, 1), 0.75
        ax = rng.randn(3); ax /= np.linalg.norm(ax)
        th = rng.uniform(-1.5, 1.5)
        q[3:7] = [np.cos(th / 2), *(np.sin(th / 2) * ax)]
        nj = env._opp()[0].start - 7
        q[7:7 + nj] = rng.uniform(-0.4, 0.4, nj)
        v[0:6] = rng.uniform(-1, 1, 6)
        v[6:6 + nj] = rng.uniform(-1, 1, nj)
        rotated_equivalent(env, q, v)
        ours = env.get_sim_obs()
        theirs = env._opp_sim_obs()
        worst_obs = max(worst_obs, float(np.abs(ours - theirs).max()))
    check("3a ROTATION (observation): our sim_obs == opponent's rotated "
          "sim_obs in mirror-equivalent states",
          worst_obs < 1e-9, f"max|delta| = {worst_obs:.3e} over 5 random states")

    # physics: step both with mirror-equivalent torques, stay equivalent
    from design_opt.envs.run_to_goal_sp import qmul_zpi
    rotated_equivalent(env, q, v)
    qs, vs, _ = env._opp()
    nu_ours = sum(1 for n in env.model.actuator_names
                  if not n.startswith("opp_"))
    ctrl = np.zeros(env.model.nu)
    t = rng.uniform(-1, 1, nu_ours)
    ctrl[:nu_ours] = t
    ctrl[nu_ours:nu_ours + nu_ours] = t          # same body-local torques
    env.sim.data.ctrl[:] = ctrl
    for _ in range(20):
        env.sim.step()
    d = env.sim.data
    err = max(abs(-d.qpos[qs.start + 0] - d.qpos[0]),
              abs(-d.qpos[qs.start + 1] - d.qpos[1]),
              abs(d.qpos[qs.start + 2] - d.qpos[2]),
              float(np.abs(d.qpos[qs.start + 3:qs.start + 7]
                           - qmul_zpi(d.qpos[3:7])).max()))
    check("3b ROTATION (physics): the two bodies stay mirror-equivalent "
          "after 20 steps of equal body-local torque",
          err < 1e-6, f"max|delta| = {err:.3e}")


# --------------------------------------------------------------------- 4 --
def gate_cost(agent):
    """The learner must be charged 0.5*sum(a_ours^2) and not one unit more.

    Tested differentially: the SAME learner action, once against an inert
    opponent and once against an active one. If the opponent's torques leaked
    into the cost the two would differ. The absolute value is checked too.
    """
    from rower_soccer.t2a_port.e3_morph import tensorfy
    env = agent.env

    class Const:
        def __init__(self, v): self.v = v
        def select_action(self, obs, mean):
            # obs is a BATCH: obs[0] is the sample, obs[0][0] the node matrix
            return torch.full((obs[0][0].shape[0], env.control_action_dim),
                              self.v, dtype=torch.float64)

    def one_step(opp_torque):
        np.random.seed(3); torch.manual_seed(3)
        state = env.reset()
        env.set_opponent_policy(Const(opp_torque))
        while env.stage != "execution":
            with torch.no_grad():
                a = agent.policy_net.select_action(
                    tensorfy([state]), True).numpy().astype(np.float64)
            state, _, _, _ = env.step(a)
        adim = np.asarray(a).shape[-1]
        act = np.zeros((len(env.robot.bodies), adim))
        act[1:, :env.control_action_dim] = 0.3
        _, _, _, info = env.step(act)
        return info, env.action_to_control(act[:, :env.control_action_dim])

    i0, ours = one_step(0.0)
    i1, _ = one_step(0.25)
    if "ctrl_cost" not in i0 or "ctrl_cost" not in i1:
        check("4 COST ISOLATION: ctrl_cost excludes the opponent's torques",
              False, "step returned no ctrl_cost (simulation error)")
        return
    want = 0.5 * float(np.square(ours).sum())
    check("4a COST ISOLATION: ctrl_cost identical with opponent inert vs "
          "active", abs(i0["ctrl_cost"] - i1["ctrl_cost"]) < 1e-12,
          f"inert {i0['ctrl_cost']:.9f} vs active {i1['ctrl_cost']:.9f}")
    check("4b COST ISOLATION: ctrl_cost == 0.5*sum(our action^2)",
          abs(i1["ctrl_cost"] - want) < 1e-12,
          f"charged {i1['ctrl_cost']:.9f}, ours alone {want:.9f}")


# ------------------------------------------------------------------ 5, 6 --
def rollout(env, policy, nsteps=500, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    from rower_soccer.t2a_port.e3_morph import tensorfy
    state = env.reset()
    total, k, info = 0.0, 0, {}
    for _ in range(nsteps + 10):
        with torch.no_grad():
            a = policy.select_action(tensorfy([state]), True).numpy().astype(np.float64)
        state, r, done, info = env.step(a)
        total += r
        if env.stage == "execution":
            k += 1
        if done:
            break
    return dict(ret=total, steps=k, reached=info.get("reached", False),
                opp_reached=info.get("opp_reached", False),
                com_x=info.get("com_x", np.nan))


def gate_scripted_parity(cfg_a, cfg_b):
    """opponent_mode=scripted on the SP env must reproduce run_to_goal."""
    a = build_agent(cfg_a); b = build_agent(cfg_b)
    b.env.opp_mode = "scripted"
    b.policy_net.load_state_dict(a.policy_net.state_dict())
    ra = rollout(a.env, a.policy_net, seed=7)
    rb = rollout(b.env, b.policy_net, seed=7)
    same = (abs(ra["ret"] - rb["ret"]) < 1e-9 and ra["steps"] == rb["steps"]
            and abs(ra["com_x"] - rb["com_x"]) < 1e-9)
    check("5 SCRIPTED PARITY: run_to_goal_sp(scripted) == run_to_goal",
          same, f"ret {ra['ret']:.6f} vs {rb['ret']:.6f}, "
                f"steps {ra['steps']} vs {rb['steps']}")


def gate_opponent_bites(agent, snap_policy, train_speed, tol=0.25):
    """NEGATIVE CONTROL, and the strongest gate here.

    A snapshot must not merely perturb the episode -- it must RACE. The first
    version of this gate only asserted "the return changed", and it passed
    while the opponent was standing still and collapsing: the stage flag was
    wrong, the policy ran its skeleton head, every control column came back
    exactly 0.0, and the small return difference was the inert body's weight
    shifting the contacts. A gate that can pass on a dead opponent is not a
    gate, so it now checks the thing that matters: the snapshot reproduces the
    speed it was trained at, in the OTHER slot, through the pi-z rotation.
    """
    env = agent.env

    class Inert:
        def select_action(self, obs, mean):
            return torch.zeros(obs[0][0].shape[0], env.control_action_dim,
                               dtype=torch.float64)

    def race(pol):
        from rower_soccer.t2a_port.e3_morph import tensorfy
        np.random.seed(11); torch.manual_seed(11)
        env.set_opponent_policy(pol)
        state = env.reset()
        while env.stage != "execution":
            with torch.no_grad():
                a = agent.policy_net.select_action(
                    tensorfy([state]), True).numpy().astype(np.float64)
            state, _, _, _ = env.step(a)
        # resolve the opponent's body id AFTER the design stages: they
        # recompile the model and every body id moves
        _, _, obid = env._opp()
        x0 = float(env.data.subtree_com[obid][0])
        tq, n, ret, info = 0.0, 0, 0.0, {}
        for _ in range(env.max_nsteps + 1):
            if env.opp_policy is not None:
                tq = max(tq, float(np.abs(env.opp_control(env.opp_action())).max()))
            with torch.no_grad():
                a = agent.policy_net.select_action(
                    tensorfy([state]), True).numpy().astype(np.float64)
            state, r, done, info = env.step(a)
            ret += r; n += 1
            if done:
                break
        x1 = float(env.data.subtree_com[obid][0])
        return dict(ret=ret, steps=n, dx=x0 - x1, torque=tq,
                    speed=(x0 - x1) / (n * env.dt),
                    reached=bool(info.get("opp_reached", False)))

    r0 = race(Inert())
    r1 = race(snap_policy)
    check("6a OPPONENT ACTS: the snapshot applies non-zero torque",
          r1["torque"] > 1e-3, f"max|torque| = {r1['torque']:.4f} "
                               f"(inert {r0['torque']:.4f})")
    check("6b OPPONENT RACES: the snapshot reaches its own goal line",
          r1["reached"] and not r0["reached"],
          f"trained travelled {r1['dx']:.2f} m in {r1['steps']} steps "
          f"(reached {r1['reached']}); inert {r0['dx']:.2f} m "
          f"(reached {r0['reached']})")
    rel = abs(r1["speed"] - train_speed) / train_speed
    check("6c ROTATION END-TO-END: the snapshot's speed in slot 1 matches the "
          "speed it trained at in slot 0",
          rel < tol, f"{r1['speed']:.3f} m/s vs trained {train_speed:.3f} m/s "
                     f"({100 * rel:.1f}% off, tol {100 * tol:.0f}%)")
    check("6d OPPONENT BITES: outcome differs from an inert opponent",
          abs(r0["ret"] - r1["ret"]) > 1e-6 or r0["steps"] != r1["steps"],
          f"inert ret {r0['ret']:.1f}/{r0['steps']} steps | "
          f"trained ret {r1['ret']:.1f}/{r1['steps']} steps")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="rtg_e4_s1a")
    p.add_argument("--ref-cfg", default="rtg_e31_s1")
    p.add_argument("--bite-cfg", default="rtg_e4_gate")
    p.add_argument("--snapshot",
                   default="/workspace/Transform2Act/results/rtg_e31_s2/models/epoch_0400.p")
    a = p.parse_args()
    print("=== D3 M3 E4 GATE ===", flush=True)
    gate_scene()
    agent = build_agent(a.cfg)
    gate_rotation(agent)
    gate_cost(agent)
    gate_scripted_parity(a.ref_cfg, a.cfg)
    snap = pickle.load(open(a.snapshot, "rb"))["policy_dict"]
    # a real pairing: s2's trained controller driving s2's own evolved body,
    # which is exactly the (body, policy) pair a refresh installs
    bite_agent = build_agent(a.bite_cfg)
    snap_agent = build_agent(a.bite_cfg)
    snap_agent.policy_net.load_state_dict(snap)
    snap_agent.policy_net.eval()
    import json
    cs = json.load(open("/workspace/utmist-vc2-phase2/rower_soccer/docs/"
                        "t2a/e4_null/e31_comparison_set.json"))
    spd = cs["bodies"]["rtg_e31_s2"]["own_eval"]["speed"]
    gate_opponent_bites(bite_agent, snap_agent.policy_net, spd)
    print()
    if FAILS:
        print("GATE FAILED:", ", ".join(FAILS)); sys.exit(1)
    print("GATE PASSED: all 6")


if __name__ == "__main__":
    main()
