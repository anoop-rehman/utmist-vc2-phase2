"""D3 M3 E3: render a best/median/worst clip from a checkpoint NAME.

Two differences from `e2_video.py`, which is left untouched so E2's clips stay
reproducible from the file that made them:

  * it loads a checkpoint by NAME, not by epoch number, so the trainer can
    render off a transient `_video_tmp.p` it deletes afterwards. A GNN
    checkpoint here is 157 MB; a ~15-minute video cadence off archival
    checkpoints would cost 6 GB per seed on a disk with 13 GB free;
  * with the design stages live the three panels are three DIFFERENT
    creatures, so the clip shows design variation as well as gait. Each panel
    is labelled with its own body count (`e2_eval.best_median_worst` appends
    `nb=` only when the design stages actually changed the body, so E2's
    frozen-body clips are byte-identical to the ones already logged).

Run as a SUBPROCESS by the trainer, never in-process: the trainer holds a CUDA
context and mujoco-py's offscreen GL context in the same process as a forking
sampler is a way to lose a run.

    cd /workspace/Transform2Act && source env-gpu.sh
    MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python \\
        .../t2a_port/e3_video.py --cfg rtg_e3_s1 --ckpt epoch_0100 --out x.mp4
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


def load_gnn(cfg_id, ckpt):
    """(cfg, env, make_actor, action_std). `ckpt` is a checkpoint BASENAME
    without `.p` -- `epoch_0100`, `best`, `_video_tmp`. Transform2Act's own
    `load_checkpoint` already takes a str and builds `model_dir/<name>.p`."""
    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    from khrylib.utils.torch import to_test
    from rower_soccer.t2a_port import e2_eval
    cfg = Config(cfg_id, tmp=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    ag = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                            device=torch.device("cpu"), seed=cfg.seed,
                            num_threads=1, training=False, checkpoint=str(ckpt))
    to_test(ag.policy_net)
    std = ag.policy_net.state_dict().get("control_action_log_std")
    std = float(std.exp().mean().item()) if std is not None else float("nan")
    return (cfg, ag.env,
            lambda mean=True: e2_eval.gnn_actor(ag.policy_net,
                                                ag.running_state, mean), std)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--ckpt", required=True)
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
    cfg, env, make, std = load_gnn(a.cfg, a.ckpt)
    act, wrap = make(True)
    path, sc = e2_eval.best_median_worst(
        env, act, wrap, a.out, episodes=a.episodes, seed_base=a.seed_base,
        fps=a.fps, camera=a.camera, max_frames=a.max_frames, stride=a.stride,
        max_steps=cfg.done_condition.get("max_nsteps", 500) + 5)
    sc["video/action_std"] = std
    if a.json:
        print("E3VIDEO " + json.dumps({"mp4": path, "scalars": sc}))
    else:
        print(path, sc)


if __name__ == "__main__":
    main()
