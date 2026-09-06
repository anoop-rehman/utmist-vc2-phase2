"""D3 E4B warm-start gate. Two-sided by construction.

A freshly-initialised policy and a failed weight load are indistinguishable
from a startup log -- both print "loaded" and both stand still. So this gate
measures the SAME quantities on a cold agent and a warm one and requires them
to differ in the predicted direction and magnitude. If the load silently did
nothing, the warm arm reads like the cold arm and the gate fails.

  W1  the loaded weights hash-match the checkpoint file
  W2  warm mean|action| is orders above cold (cold collapsed to ~0.0006)
  W3  warm speed is within tolerance of the checkpoint's PUBLISHED speed
  W4  NEGATIVE CONTROL: cold speed is ~0, so W3 is a real discriminator and
      not something any agent would pass
"""
import argparse, hashlib, os, pickle, sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/Transform2Act")
sys.path.insert(0, "/workspace/utmist-vc2-phase2")
torch.set_default_dtype(torch.float64)
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def sha_sd(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def run(agent, env, episodes=6, seed_base=900):
    from rower_soccer.t2a_port import e2_eval
    from khrylib.utils.torch import to_cpu, to_test
    with to_cpu(agent.policy_net), to_test(agent.policy_net):
        act, wrap = e2_eval.gnn_actor(agent.policy_net, agent.running_state, True)
        ev = e2_eval.evaluate(env, act, wrap, episodes=episodes,
                              seed_base=seed_base, max_steps=env.max_nsteps + 5)
    eps = ev.pop("episodes", [])
    return ev, eps


def mean_action_mag(agent, env):
    from rower_soccer.t2a_port.e3_morph import tensorfy
    from khrylib.utils.torch import to_cpu, to_test
    with to_cpu(agent.policy_net), to_test(agent.policy_net):
        np.random.seed(3); torch.manual_seed(3)
        state = env.reset()
        while env.stage != "execution":
            with torch.no_grad():
                a = agent.policy_net.select_action(tensorfy([state]), True).numpy()
            state, _, _, _ = env.step(a.astype(np.float64))
        mags = []
        for _ in range(40):
            with torch.no_grad():
                a = agent.policy_net.select_action(tensorfy([state]), True).numpy()
            mags.append(float(np.abs(a[:, :env.control_action_dim]).max()))
            state, _, done, _ = env.step(a.astype(np.float64))
            if done:
                break
    return float(np.mean(mags)), float(np.max(mags))


def build(cfg_id, ckpt=None):
    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    cfg = Config(cfg_id, tmp=True)
    ag = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                            device=torch.device("cpu"), seed=0,
                            num_threads=1, training=False, checkpoint=0)
    if ckpt:
        blob = pickle.load(open(ckpt, "rb"))
        ag.policy_net.load_state_dict(blob["policy_dict"], strict=True)
        ag.value_net.load_state_dict(blob["value_dict"], strict=True)
    return ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="rtg_e4r_s2")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--published-speed", type=float, required=True)
    ap.add_argument("--tol", type=float, default=0.30)
    a = ap.parse_args()
    print("=== D3 E4B WARM-START GATE ===", flush=True)
    print(f"  checkpoint {a.ckpt}, published speed {a.published_speed:.3f} m/s\n")

    blob = pickle.load(open(a.ckpt, "rb"))
    file_sha = sha_sd(blob["policy_dict"])

    cold = build(a.cfg)
    warm = build(a.cfg, a.ckpt)
    check("W1 LOAD TOOK: in-memory policy hash == checkpoint hash",
          sha_sd(warm.policy_net.state_dict()) == file_sha,
          f"{sha_sd(warm.policy_net.state_dict())} vs {file_sha}; "
          f"cold is {sha_sd(cold.policy_net.state_dict())}")

    cm, cx = mean_action_mag(cold, cold.env)
    wm, wx = mean_action_mag(warm, warm.env)
    check("W2 MEAN ACTION: warm is far above cold",
          wm > 20 * max(cm, 1e-9),
          f"warm mean {wm:.4f} max {wx:.4f} | cold mean {cm:.6f} max {cx:.6f}")

    cev, _ = run(cold, cold.env)
    wev, _ = run(warm, warm.env)
    rel = abs(wev["speed"] - a.published_speed) / a.published_speed
    check("W3 SPEED: warm matches the checkpoint's published speed",
          rel < a.tol,
          f"warm {wev['speed']:.3f} m/s vs published {a.published_speed:.3f} "
          f"({100*rel:.0f}% off, tol {100*a.tol:.0f}%); goal {wev['goal_rate']:.2f} "
          f"fwd {wev['max_fwd']:.2f} m")
    check("W4 NEGATIVE CONTROL: cold is near-stationary, so W3 discriminates",
          abs(cev["speed"]) < 0.5,
          f"cold {cev['speed']:.3f} m/s, goal {cev['goal_rate']:.2f}, "
          f"fwd {cev['max_fwd']:.2f} m")
    print()
    if FAILS:
        print("WARM-START GATE FAILED:", ", ".join(FAILS)); sys.exit(1)
    print("WARM-START GATE PASSED")


if __name__ == "__main__":
    main()
