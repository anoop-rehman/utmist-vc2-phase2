"""D3 M3 E2: ONE instrument for every arm, and the video that goes with it.

E1.1's measurement trap, restated because E2 inherits it verbatim:
Transform2Act's `exec_R_eps` is a separate **mean-action evaluation** pass
(`transform2act_agent.py:214`), while `train_e11_mlp.py`'s is the **stochastic
training** return. Reading the two training logs side by side compares a
deterministic evaluation against a noisy average and flatters the GNN by ~1.3x.

So neither training log is the result. Both arms are measured by
`evaluate()` below -- same code, same protocol, same episode seeds -- and it is
called from three places: inside each trainer every `--eval-every` epochs (so
the wandb curve is already the right statistic), and post-hoc from
`e2_posthoc.py` for the headline table.

`--stochastic` reports the same episodes with sampled actions, because each
architecture learns its own action noise and the two protocols can disagree in
opposite directions. Both columns get reported; the mean-action one is the
headline.
"""

import numpy as np
import torch


# ---------------------------------------------------------------- actors --
def gnn_actor(policy, running_state, mean_action=True):
    """Transform2Act's GNN. It acts in the design stages too -- with
    `force_identity_design` the env throws those actions away, but they are
    still drawn, so the rollout is exactly the one training produced."""
    def tf(l):
        if isinstance(l[0], list):
            return [[torch.tensor(x) for x in y] for y in l]
        return [torch.tensor(y) for y in l]

    def act(state, stage):
        with torch.no_grad():
            return policy.select_action(tf([state]),
                                        mean_action).numpy().astype(np.float64)
    def wrap(s):
        return running_state(s) if running_state is not None else s
    return act, wrap


def mlp_actor(actor, norm, rows, nbody, width, mean_action=True):
    from rower_soccer.t2a_port.train_e11_mlp import flat_obs

    def act(state, stage):
        if stage != 2:
            return np.zeros((nbody, width))
        o = norm(flat_obs(state))
        with torch.no_grad():
            mu, _ = actor.select_action(torch.as_tensor(o).unsqueeze(0),
                                        mean_action)
        full = np.zeros((nbody, width))
        full[rows, 0] = mu.numpy()[0]
        return full
    return act, (lambda s: s)


# -------------------------------------------------------------- rollouts --
def roll(env, act, wrap, seed, max_steps=1000, render=False,
         camera="pitch", width=400, height=224, max_frames=200, stride=3):
    """One episode from a named seed. Everything the rollout draws from is
    reset here, per episode, so rendering episode i cannot shift the random
    stream episode i+1 sees (E0's `e0_video.py` measured that it does)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    state = wrap(env.reset())
    nb = len(env.robot.bodies)
    while env.if_use_transform_action() != 2:
        state, _, done, _ = env.step(act(state, env.if_use_transform_action()))
        state = wrap(state)
        if done:
            return None
    R, n, xs, ys, frames = 0.0, 0, [], [], []
    info = {}
    x0 = float(env.data.subtree_com[env._our_torso_id()][0])
    for _ in range(max_steps):
        state, r, done, info = env.step(act(state, 2))
        state = wrap(state)
        R += float(r)
        n += 1
        xs.append(info["com_x"])
        ys.append(float(env.data.subtree_com[env._our_torso_id()][1]))
        # every `stride`-th control step: a 500-step episode at 0.015 s is
        # 7.5 s of wall time, and osmesa is the cost here, not the physics.
        if render and len(frames) < max_frames and (n - 1) % stride == 0:
            frames.append(np.flipud(env.sim.render(width, height,
                                                   camera_name=camera)))
        if done:
            break
    xs = np.asarray(xs)
    path = float(np.abs(np.diff(np.concatenate([[x0], xs]))).sum())
    return dict(R=R, n=n, reached=bool(info.get("reached", False)),
                opp_reached=bool(info.get("opp_reached", False)),
                fell=bool(info.get("fell", False)),
                net_dx=float(xs[-1] - x0), max_x=float(xs.max()),
                path=path, net_over_path=float((xs[-1] - x0) / path)
                if path else 0.0,
                max_abs_y=float(np.abs(ys).max()), bodies=nb, frames=frames)


def evaluate(env, act, wrap, episodes=20, seed_base=1000, max_steps=1000):
    eps = []
    for i in range(episodes):
        e = roll(env, act, wrap, seed_base + i, max_steps=max_steps)
        if e is not None:
            eps.append(e)
    if not eps:
        return {}
    g = lambda k: np.array([e[k] for e in eps], dtype=float)
    R = g("R")
    return dict(n_eps=len(eps), R_mean=float(R.mean()), R_sd=float(R.std(ddof=1))
                if len(R) > 1 else 0.0,
                R_min=float(R.min()), R_max=float(R.max()),
                ep_len=float(g("n").mean()),
                goal_rate=float(g("reached").mean()),
                loss_rate=float(g("opp_reached").mean()),
                fall_rate=float(g("fell").mean()),
                net_dx=float(g("net_dx").mean()),
                max_x=float(g("max_x").mean()),
                net_over_path=float(g("net_over_path").mean()),
                max_abs_y=float(g("max_abs_y").mean()),
                speed=float((g("net_dx") / (g("n") * env.dt)).mean()),
                episodes=[{k: v for k, v in e.items() if k != "frames"}
                          for e in eps])


# ----------------------------------------------------------------- video --
def label(img, text):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def best_median_worst(env, act, wrap, path, episodes=9, seed_base=777,
                      fps=12, camera="pitch", width=400, height=224,
                      max_frames=200, stride=3, max_steps=1000):
    """Two passes: rank without rendering, then replay and render three.

    Rendering perturbs mujoco-py's rollout in the 4th decimal
    (`e0_video.py` measured it), so the ranking comes from pass 1 and every
    panel is LABELLED with its own pass-2 numbers. With morphology frozen all
    three panels are the SAME creature, so what the clip shows is gait and
    tactics -- how it starts, whether it dodges the opponent, whether it
    reaches the line -- not design variation."""
    stats = [roll(env, act, wrap, seed_base + i, max_steps=max_steps)
             for i in range(episodes)]
    stats = [s for s in stats if s is not None]
    if not stats:
        return None, {}
    order = np.argsort([s["R"] for s in stats])
    pick = {int(order[-1]): "best", int(order[len(order) // 2]): "median",
            int(order[0]): "worst"}
    panels = {}
    for i in range(len(stats)):
        if i not in pick:
            continue
        e = roll(env, act, wrap, seed_base + i, max_steps=max_steps,
                 render=True, camera=camera, width=width, height=height,
                 max_frames=max_frames, stride=stride)
        tag = ("GOAL" if e["reached"] else
               ("lost" if e["opp_reached"] else ("fell" if e["fell"] else "--")))
        panels[pick[i]] = ([label(f, f"{pick[i]}  R={e['R']:.0f}  "
                                  f"dx={e['net_dx']:.2f}m  {e['n']}st  {tag}")
                            for f in e["frames"]], e)
    tiles = [panels[k][0] for k in ("best", "median", "worst")
             if k in panels and panels[k][0]]
    if not tiles:
        return None, {}
    T = max(len(t) for t in tiles)
    tiles = [t + [t[-1]] * (T - len(t)) for t in tiles]
    video = np.stack([np.concatenate([t[j] for t in tiles], axis=1)
                      for j in range(T)])
    import imageio
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, video, fps=fps, macro_block_size=1, quality=8)
    sc = {}
    for k in ("best", "median", "worst"):
        if k in panels:
            e = panels[k][1]
            sc[f"video/{k}_R"] = e["R"]
            sc[f"video/{k}_dx"] = e["net_dx"]
            sc[f"video/{k}_steps"] = e["n"]
            sc[f"video/{k}_goal"] = float(e["reached"])
    return path, sc
