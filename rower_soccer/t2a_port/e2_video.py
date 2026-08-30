"""D3 M3 E2: render a best/median/worst clip from a checkpoint.

Run as a SUBPROCESS by both trainers (`--json` prints the scalars back), never
in-process: the GNN arm holds a CUDA context and mujoco-py's offscreen GL
context in the same process as a forking sampler is a way to lose a run. The
trainer then logs the mp4 into its OWN wandb run in the same `wandb.log` call
as that epoch's metrics, so metrics and media share one run and one step.

With morphology frozen every panel is the SAME creature, so the clip shows
gait and tactics -- the start, whether it dodges the scripted opponent, whether
it crosses the line -- and NOT design variation.

    cd /workspace/Transform2Act && source env-gpu.sh
    MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python \\
        .../t2a_port/e2_video.py --arm gnn --cfg rtg_gnn_s1 --epoch 100 \\
        --out /workspace/utmist-vc2-phase2/runs/d3_e2_rtg/renders/x.mp4
"""
import argparse
import json
import os
import sys

sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def load_arm(arm, cfg_id, epoch, tag=None, device="cpu"):
    """Return (cfg, env, make_actor, action_std). `make_actor(mean_action)`
    gives the (act, wrap) pair `e2_eval` drives, so one load serves both the
    mean-action and the stochastic protocol.

    `arm="idle"` is the task's negative control: zero torque on every motor,
    no checkpoint. It is what "not won by standing still" is measured with,
    through exactly the same instrument as the trained arms."""
    from design_opt.utils.config import Config
    from design_opt.envs import env_dict
    from rower_soccer.t2a_port import e2_eval
    cfg = Config(cfg_id, tmp=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if arm == "idle":
        env = env_dict[cfg.env_name](cfg, agent=None)
        W = env.control_action_dim + env.attr_design_dim + 1
        nb = len(env.robot.bodies)
        zero = np.zeros((nb, W))

        def make(mean=True):
            return (lambda state, stage: zero), (lambda s: s)
        return cfg, env, make, 0.0

    if arm == "gnn":
        from design_opt.agents.transform2act_agent import Transform2ActAgent
        from khrylib.utils.torch import to_test
        ag = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                                device=torch.device(device), seed=cfg.seed,
                                num_threads=1, training=False,
                                checkpoint=int(epoch))
        to_test(ag.policy_net)
        std = ag.policy_net.state_dict().get("control_action_log_std")
        std = float(std.exp().mean().item()) if std is not None else float("nan")
        return (cfg, ag.env,
                lambda mean=True: e2_eval.gnn_actor(ag.policy_net,
                                                    ag.running_state, mean),
                std)

    from rower_soccer.t2a_port.train_e11_mlp import Actor, RunningNorm, flat_obs
    env = env_dict[cfg.env_name](cfg, agent=None)
    d = f"/workspace/Transform2Act/results/{cfg_id}" + (f"_{tag}" if tag else "")
    blob = torch.load(os.path.join(d, f"epoch_{int(epoch):04d}.p"),
                      map_location="cpu")
    names = list(env.model.actuator_names)
    rows = [i for i, b in enumerate(env.robot.bodies)
            if i > 0 and b.get_actuator_name() in names]
    od = flat_obs(env.reset()).shape[0]
    hd = [int(x) for x in blob["args"].get("hdims", "64,64").split(",")]
    actor = Actor(od, len(rows), hd, 0.0)
    actor.load_state_dict(blob["actor"])
    actor.eval()
    norm = RunningNorm(od)
    norm.load(blob["norm"])
    W = env.control_action_dim + env.attr_design_dim + 1
    return (cfg, env,
            lambda mean=True: e2_eval.mlp_actor(actor, norm, rows,
                                                len(env.robot.bodies), W, mean),
            float(actor.log_std.exp().mean().item()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["gnn", "mlp", "idle"], required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--epoch", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=9)
    p.add_argument("--seed-base", type=int, default=777)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--camera", default="pitch")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    from rower_soccer.t2a_port import e2_eval
    cfg, env, make, std = load_arm(a.arm, a.cfg, a.epoch, a.tag)
    act, wrap = make(True)
    path, sc = e2_eval.best_median_worst(
        env, act, wrap, a.out, episodes=a.episodes, seed_base=a.seed_base,
        fps=a.fps, camera=a.camera, max_frames=a.max_frames, stride=a.stride,
        max_steps=cfg.done_condition.get("max_nsteps", 500) + 5)
    sc["video/action_std"] = std
    if a.json:
        print("E2VIDEO " + json.dumps({"mp4": path, "scalars": sc}))
    else:
        print(path, sc)


if __name__ == "__main__":
    main()
