"""Render a Transform2Act checkpoint offscreen, without touching a live run.

Their `eval.py` calls `visualize_agent`, which calls `_get_viewer('human')` --
a GLFW window. On a headless pod that crashes, which is why nobody has looked
at a Transform2Act rollout here yet.

This does the same thing through mujoco-py's offscreen path instead, and is
deliberately built so it CANNOT disturb training:

  * **CPU only.** `device=torch.device('cpu')`, no CUDA context. After the
    2026-08-25 incident where tearing down several MPS clients at once killed
    two unrelated training runs with an illegal memory access, nothing that
    only needs a 200k-parameter forward pass should be opening a CUDA context
    next to a live job.
  * **Read-only.** It loads a checkpoint file and never writes into the run
    directory.

    cd /workspace/Transform2Act && source env-gpu.sh
    MUJOCO_GL=osmesa .venv-gpu/bin/python \
        .../t2a_port/render_checkpoint.py --cfg hopper_gpu --epoch 400 \
        --out /tmp/hopper_400.mp4

The design stages come first and involve no physics: the skeleton is edited,
then the attributes, and only then does the body move. `--skip-design` starts
the video at the first execution step, which is what you want if the question
is "what does it look like when it runs" rather than "how was it built".
"""

import argparse
import os
import sys

sys.path.append("/workspace/Transform2Act")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="hopper_gpu")
    p.add_argument("--epoch", default="best")
    p.add_argument("--out", default="/tmp/t2a_rollout.mp4")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--skip-design", action="store_true",
                   help="start at the first execution step")
    p.add_argument("--camera", default="track")
    args = p.parse_args()

    import imageio
    from design_opt.agents.transform2act_agent import (Transform2ActAgent,
                                                       tensorfy)
    from design_opt.utils.config import Config

    torch.set_default_dtype(torch.float64)
    cfg = Config(args.cfg, tmp=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    epoch = int(args.epoch) if args.epoch.isnumeric() else args.epoch
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=cfg.seed,
                               num_threads=1, training=False, checkpoint=epoch)
    env, policy = agent.env, agent.policy_net
    policy.eval()

    frames, stage_of = [], []
    state = env.reset()
    if agent.running_state is not None:
        state = agent.running_state(state)

    with torch.no_grad():
        for t in range(args.max_steps):
            action = policy.select_action(
                tensorfy([state]), True).numpy().astype(np.float64)
            state, _, done, info = env.step(action)
            if agent.running_state is not None:
                state = agent.running_state(state)
            stage = info.get("stage", "?")
            if args.skip_design and stage != "execution":
                continue
            # The design stages REPLACE the MjModel, so the offscreen context
            # has to be taken from whatever sim exists right now rather than
            # cached once at the top.
            img = env.sim.render(args.width, args.height,
                                 camera_name=args.camera)
            frames.append(np.flipud(img))       # mujoco-py returns bottom-up
            stage_of.append(stage)
            if done:
                break

    if not frames:
        raise SystemExit("no frames rendered")
    imageio.mimwrite(args.out, frames, fps=args.fps, macro_block_size=1,
                     quality=8)
    from collections import Counter
    print(f"cfg {args.cfg}  epoch {epoch}  -> {args.out}")
    print(f"  {len(frames)} frames, {len(frames)/args.fps:.1f}s at {args.fps} fps")
    print(f"  stages: {dict(Counter(stage_of))}")
    print(f"  bodies in the final design: {env.sim.model.nbody - 1}, "
          f"actuators: {env.sim.model.nu}")


if __name__ == "__main__":
    main()
