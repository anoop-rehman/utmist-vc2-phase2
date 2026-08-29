"""D3 M3 E0: best / median / worst rollout video from a checkpoint, to wandb.

The same idea as D1's `train_soccer2v2_warp.render_best_median_worst` -- roll a
batch, rank by episode reward, tile three panels -- but post-hoc from a saved
checkpoint rather than inside the trainer, because the E0 seeds were already
mid-flight when video logging was asked for and restarting them to add it would
have thrown away the run.

On Transform2Act the three panels carry more than three control policies: every
sampled episode draws its OWN design (`ant.py:310, 318` reset the robot each
reset), so best/median/worst is also a sample of three morphologies out of the
distribution the skeleton stage is producing at that epoch. For E0 -- whose
question is whether that distribution is still wide -- that is the point of the
video, not a side effect.

**Two passes, not one.** Rendering every episode to pick three would hold
~1,000 frames x N episodes in memory. Instead pass 1 rolls N episodes with no
rendering and records their returns; pass 2 re-seeds identically and renders
only the three chosen.

**Rendering perturbs the rollout.** Measured, not assumed: two no-render passes
from the same seed are bit-identical, and a pass that calls `env.sim.render`
gives the same episode length and the same design but a return that differs in
the 4th decimal (9.3468136 -> 9.3453570 on one episode; three episodes checked,
all identical in length and body count). mujoco-py's render path touches the
sim, and the solver's warm start carries that into the next step. So pass 2
cannot be asserted equal to pass 1 on the return.

That perturbation also breaks the replay in a way a tolerance cannot fix, and
this is the reason for the per-episode seeding below. One episode terminating a
step later (measured: 236 -> 237 at `ant_e0_s2` epoch 30) means one extra action
is drawn from the shared torch generator, so EVERY LATER EPISODE draws from a
shifted stream and gets a genuinely different design. Widening the tolerance
would have hidden that rather than fixed it.

So `seed_all` runs at the start of every episode with `seed + i`, not once per
pass. Each episode's design and start state then depend only on its own index,
and rendering episode 4 cannot touch episode 5. With that, the design is
identical between passes by construction.

What is asserted is therefore the DESIGN, and ONLY the design: body count,
exact. Episode length and return are reported, not asserted, because the
perturbation is chaotic once an episode is long -- measured at `ant_e0_s2`
epoch 90, the same design ran 597 steps unrendered and 427 rendered, a 28%
difference from a 4th-decimal nudge, because the tilt/height test trips at a
different point after the trajectories separate. A tolerance that passed that
would be vacuous, and one that failed it would reject a correct replay. The
design is the thing per-episode seeding actually guarantees, so it is the thing
worth asserting.

The consequence, stated rather than hidden: the best/median/worst RANKING comes
from pass 1, and pass 2's returns differ from it by up to ~7% on long episodes,
so on a close call the "best" panel may not be the true best of the nine. Every
panel is labelled with its own pass-2 return, so the numbers on screen are
always right even when the rank ordering is marginal.

Every panel is LABELLED with the return pass 2 measured for the frames it is
showing, so the label always describes the video, and the only thing pass 1 is
trusted for is the ranking.

    cd /workspace/Transform2Act && source env-gpu.sh
    MUJOCO_GL=osmesa .venv-gpu/bin/python .../t2a_port/e0_video.py \
        --cfg ant_e0_s1 --epoch 10 --wandb-run d3_e0_ant_s1

CPU only -- no CUDA context, safe beside live MPS clients.
"""

import argparse
import os
import sys

sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def tensorfy(np_list):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x) for x in y] for y in np_list]
    return [torch.tensor(y) for y in np_list]


def seed_all(env, seed):
    """Everything a rollout draws from, reset to a known point, so pass 2 can
    replay pass 1 exactly. `env.seed` reseeds the gym `np_random` that
    `ant.reset_state(True)` uses for the execution-stage start noise; the torch
    generator is what the policy samples actions from."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)


def roll(env, policy, args, render, initial_body):
    """One episode. Returns (exec_return, n_exec_steps, frames, n_bodies)."""
    state = env.reset()
    total, steps, frames = 0.0, 0, []
    with torch.no_grad():
        for _ in range(args.max_steps + env.cfg.skel_transform_nsteps + 1):
            if initial_body and env.if_use_transform_action() != 2:
                a = np.zeros((len(env.robot.bodies), env.attr_design_dim + 2))
            else:
                a = policy.select_action(
                    tensorfy([state]),
                    args.mean_action).numpy().astype(np.float64)
            state, reward, done, info = env.step(a)
            if info.get("stage") == "execution":
                total += float(reward)
                steps += 1
                if render and len(frames) < args.max_frames:
                    # The design stages REPLACE the MjModel, so the offscreen
                    # context must come from whatever sim exists right now.
                    frames.append(np.flipud(env.sim.render(
                        args.width, args.height, camera_name=args.camera)))
            if done:
                break
    return total, steps, frames, len(env.robot.bodies)


def label(img, text):
    """Burn a caption into the top-left of a panel. Falls back to the bare
    image if Pillow is not importable, because a missing font must not cost a
    video."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--epoch", default="best")
    p.add_argument("--episodes", type=int, default=9,
                   help="how many to rank before picking three")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--max-frames", type=int, default=300,
                   help="cap per panel; a 1,000-step episode at 40 fps is 25 s "
                        "of video and minutes of osmesa")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--camera", default="track")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--mean-action", action="store_true")
    p.add_argument("--untrained", action="store_true")
    p.add_argument("--initial-body", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--wandb-run", default=None,
                   help="wandb run NAME (= id); omit to only write the mp4")
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--video-key", default="video/best_median_worst",
                   help="wandb media key. Use a distinct one for the "
                        "initial-body clip so it does not overwrite the "
                        "rollout video at the same step.")
    p.add_argument("--step", type=int, default=None,
                   help="wandb step; defaults to the checkpoint epoch")
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config

    cfg = Config(args.cfg, tmp=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    epoch = 0 if args.untrained else (int(args.epoch) if args.epoch.isnumeric()
                                      else args.epoch)
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=cfg.seed,
                               num_threads=1, training=False, checkpoint=epoch)
    env, policy = agent.env, agent.policy_net
    policy.eval()          # their sampler runs under to_test; see topology_census
    env.cfg = cfg

    # ---- pass 1: rank, no rendering ------------------------------------
    stats = []
    for i in range(args.episodes):
        seed_all(env, args.seed + i)
        r, n, _, nb = roll(env, policy, args, False, args.initial_body)
        stats.append((r, n, nb))
    order = np.argsort([s[0] for s in stats])
    pick = {int(order[-1]): "best", int(order[len(order) // 2]): "median",
            int(order[0]): "worst"}
    print(f"{args.cfg} epoch {epoch}: returns " +
          " ".join(f"{s[0]:.2f}" for s in stats))

    # ---- pass 2: replay the same episodes, render the three chosen -----
    panels, drift, step_drift = {}, 0.0, 0
    for i in range(args.episodes):
        seed_all(env, args.seed + i)
        want = i in pick
        r, n, frames, nb = roll(env, policy, args, want, args.initial_body)
        rel = abs(r - stats[i][0]) / max(abs(stats[i][0]), 1e-9)
        assert nb == stats[i][2], (
            f"replay built a DIFFERENT DESIGN at index {i}: pass 1 gave "
            f"{stats[i][2]} bodies, pass 2 gave {nb}. Per-episode seeding makes "
            f"the design independent of rendering, so this is a real bug, not "
            f"drift. Do not loosen it.")
        drift = max(drift, rel)
        step_drift = max(step_drift, abs(n - stats[i][1]))
        if want:
            panels[pick[i]] = (
                [label(f, f"{pick[i]}  R={r:.1f}  {n} steps  {nb} bodies")
                 for f in frames], r, n, nb)

    print(f"  render drift: return {drift:.2e} relative, "
          f"episode length {step_drift} step(s)")
    order3 = ["best", "median", "worst"]
    tiles = [panels[k][0] for k in order3 if panels.get(k) and panels[k][0]]
    if not tiles:
        raise SystemExit("no frames rendered")
    T = max(len(t) for t in tiles)
    tiles = [t + [t[-1]] * (T - len(t)) for t in tiles]      # pad with the last
    video = np.stack([np.concatenate([t[j] for t in tiles], axis=1)
                      for j in range(T)])

    out = args.out or (f"/workspace/utmist-vc2-phase2/runs/d3_e0_ant/renders/"
                       f"{args.cfg}_e{epoch if isinstance(epoch, int) else 0:04d}"
                       f"_bmw.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import imageio
    imageio.mimwrite(out, video, fps=args.fps, macro_block_size=1, quality=8)
    print(f"  -> {out}  {T} frames, {video.shape[2]}x{video.shape[1]}")
    for k in order3:
        if k in panels:
            _, r, n, nb = panels[k]
            print(f"  {k:>6}: R {r:8.2f}  {n:4d} steps  {nb:2d} bodies")

    # Rendering and uploading run in DIFFERENT venvs and this script does not
    # upload. `.venv-gpu` has mujoco-py and no wandb; the repo's `.venv` has
    # wandb and no mujoco-py. Worse, this file puts `/workspace/utmist-vc2-phase2`
    # on `sys.path`, where the repo's `wandb/` RUN-ARTIFACT directory imports as
    # a namespace package and shadows the real one -- `import wandb` succeeds and
    # `wandb.init` then raises AttributeError. So the mp4 gets a JSON sidecar and
    # `e0_wandb_media.py`, run under `.venv`, uploads both.
    if args.wandb_run:
        import json
        step = args.step if args.step is not None else (
            epoch if isinstance(epoch, int) else 0)
        side = {"mp4": out, "run": args.wandb_run,
                "project": args.wandb_project, "key": args.video_key,
                "step": int(step), "fps": args.fps,
                "scalars": {"video/best_R": panels["best"][1],
                            "video/median_R": panels["median"][1],
                            "video/worst_R": panels["worst"][1],
                            "video/best_bodies": panels["best"][3],
                            "video/median_bodies": panels["median"][3],
                            "video/worst_bodies": panels["worst"][3],
                            "video/best_steps": panels["best"][2],
                            "video/worst_steps": panels["worst"][2],
                            "epoch": int(step)}}
        json.dump(side, open(out + ".json", "w"), indent=1)
        print(f"  sidecar -> {out}.json  (upload with e0_wandb_media.py)")


if __name__ == "__main__":
    main()
