"""D3 M3 E1.1: best / median / worst rollout video from an MLP-arm checkpoint.

The GNN arm's clips come from `e0_video.py` unchanged, because that arm is an
ordinary Transform2Act cfg. The MLP arm needs its own roller because its policy
is `train_e11_mlp.Actor`, not `Transform2ActPolicy` -- but it writes **the same
mp4 + JSON sidecar contract**, so `e0_wandb_media.py` uploads it by exactly the
path E0 fixed: a separate `<name>_media` run, `wandb.log` with NO explicit step,
and `epoch` declared as the step metric. That is the whole point of reusing the
contract rather than adding a second uploader.

Simpler than `e0_video.py` in one respect and it matters: the morphology is
frozen, so every episode is the SAME body and the three panels differ only in
control and reset noise. There is no design to assert equal between passes, so
the two-pass rank-then-render structure is kept only to bound memory.

    cd /workspace/Transform2Act && source env-gpu.sh
    MUJOCO_GL=osmesa .venv-gpu/bin/python .../t2a_port/e11_mlp_video.py \
        --cfg ant_e11_mlp_s1 --tag pub --epoch best \
        --out .../renders/e11_mlp_s1_best.mp4 --wandb-run d3_e11_mlp_s1_media

CPU only -- no CUDA context.
"""

import argparse
import json
import os
import sys

sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rower_soccer.t2a_port.train_e11_mlp import (Actor, RunningNorm,  # noqa: E402
                                                 flat_obs)


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 12], fill=(0, 0, 0))
    d.text((2, 1), text, fill=(255, 255, 0))
    return np.asarray(im)


def roll(env, actor, norm, act_rows, args, render):
    nbody = len(env.robot.bodies)
    width = env.control_action_dim + env.attr_design_dim + 1
    zero = np.zeros((nbody, width))
    state = env.reset()
    frames = []
    while env.if_use_transform_action() != 2:
        state, _, done, _ = env.step(zero)
        if done:
            return 0.0, 0, frames
    ret, n = 0.0, 0
    with torch.no_grad():
        while n < args.max_steps:
            o = norm(flat_obs(state))
            a, _ = actor.select_action(
                torch.as_tensor(o, dtype=torch.float64).unsqueeze(0), True)
            full = np.zeros((nbody, width))
            full[act_rows, 0] = a.numpy()[0]
            state, r, done, _ = env.step(full)
            ret += r
            n += 1
            if render and len(frames) < args.max_frames:
                frames.append(np.flipud(env.sim.render(args.width, args.height,
                                                       camera_name=args.camera)))
            if done:
                break
    return ret, n, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--tag", default=None, help="results-dir suffix, as in train_e11_mlp")
    p.add_argument("--epoch", default="best")
    p.add_argument("--episodes", type=int, default=9)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--camera", default="track")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--hdims", default="64,64")
    p.add_argument("--out", required=True)
    p.add_argument("--wandb-run", default=None)
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--video-key", default="video/best_median_worst")
    p.add_argument("--step", type=int, default=0)
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    from design_opt.envs.ant import AntEnv
    from design_opt.utils.config import Config

    cfg = Config(args.cfg, tmp=False)
    assert cfg.env_specs.get("force_identity_design", False), (
        "the MLP arm's cfg must force the design stages to the identity")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = AntEnv(cfg, agent=None)
    names = list(env.model.actuator_names)
    act_rows = [i for i, b in enumerate(env.robot.bodies)
                if i > 0 and b.get_actuator_name() in names]

    d = f"/workspace/Transform2Act/results/{args.cfg}" + (
        f"_{args.tag}" if args.tag else "")
    ck = os.path.join(d, f"{args.epoch}.p" if not args.epoch.isnumeric()
                      else f"epoch_{int(args.epoch):04d}.p")
    blob = torch.load(ck, map_location="cpu")
    obs_dim = flat_obs(env.reset()).shape[0]
    actor = Actor(obs_dim, env.model.nu, [int(x) for x in args.hdims.split(",")],
                  0.0)
    actor.load_state_dict(blob["actor"])
    actor.eval()
    norm = RunningNorm(obs_dim)
    norm.load(blob["norm"])
    print(f"loaded {ck}  norm n={norm.n}")

    stats = []
    for i in range(args.episodes):
        np.random.seed(args.seed + i)
        torch.manual_seed(args.seed + i)
        env.np_random.seed(args.seed + i)
        r, n, _ = roll(env, actor, norm, act_rows, args, False)
        stats.append((r, n))
    order = np.argsort([s[0] for s in stats])
    pick = {int(order[-1]): "best", int(order[len(order) // 2]): "median",
            int(order[0]): "worst"}
    print(f"{args.cfg}{'_' + args.tag if args.tag else ''} {args.epoch}: "
          "returns " + " ".join(f"{s[0]:.1f}" for s in stats))

    panels = {}
    for i in range(args.episodes):
        np.random.seed(args.seed + i)
        torch.manual_seed(args.seed + i)
        env.np_random.seed(args.seed + i)
        want = i in pick
        r, n, frames = roll(env, actor, norm, act_rows, args, want)
        if want and frames:
            panels[pick[i]] = ([label(f, f"{pick[i]}  R={r:.1f}  {n} steps")
                                for f in frames], r, n)

    tiles = [panels[k][0] for k in ("best", "median", "worst") if k in panels]
    if not tiles:
        raise SystemExit("no frames rendered")
    T = max(len(t) for t in tiles)
    tiles = [t + [t[-1]] * (T - len(t)) for t in tiles]
    video = np.stack([np.concatenate([t[j] for t in tiles], axis=1)
                      for j in range(T)])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import imageio
    imageio.mimwrite(args.out, video, fps=args.fps, macro_block_size=1,
                     quality=8)
    print(f"  -> {args.out}  {T} frames, {video.shape[2]}x{video.shape[1]}")
    for k in ("best", "median", "worst"):
        if k in panels:
            print(f"  {k:>6}: R {panels[k][1]:8.1f}  {panels[k][2]:4d} steps")

    if args.wandb_run:
        side = {"mp4": args.out, "run": args.wandb_run,
                "project": args.wandb_project, "key": args.video_key,
                "step": int(args.step), "fps": args.fps,
                "scalars": {"video/best_R": panels["best"][1],
                            "video/median_R": panels["median"][1],
                            "video/worst_R": panels["worst"][1],
                            "video/best_steps": panels["best"][2],
                            "video/worst_steps": panels["worst"][2],
                            "epoch": int(args.step)}}
        json.dump(side, open(args.out + ".json", "w"), indent=1)
        print(f"  sidecar -> {args.out}.json  (upload with e0_wandb_media.py)")


if __name__ == "__main__":
    main()
