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
    # D3 M3 E3: `nb` above is the body count BEFORE the design stages ran --
    # 13, always, because `reset_model` rebuilds the initial ant every episode.
    # With the design stages live the body that actually runs the episode is
    # this one. Under E2/E2.1's `force_identity_design` the two are equal by
    # construction, which is why no earlier number moves.
    nb_exec = len(env.robot.bodies)
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
    return dict(R=R, n=n, x0=x0, max_fwd=float(xs.max() - x0),
                bodies_exec=nb_exec,
                reached=bool(info.get("reached", False)),
                opp_reached=bool(info.get("opp_reached", False)),
                # D3 M3 E4: both torso positions at episode end, so the race
                # margin (how much further the loser still had to run) can be
                # computed. Additive keys -- every existing consumer looks up
                # by name, so E2/E3 aggregation is unaffected. Without these
                # `e4_selfplay.race_stats` silently returns no margin and one
                # of the two pre-registered degeneracy guards is dead.
                com_x=float(info["com_x"]) if "com_x" in info else None,
                opp_com_x=float(info["opp_com_x"])
                if "opp_com_x" in info else None,
                fell=bool(info.get("fell", False)),
                net_dx=float(xs[-1] - x0), max_x=float(xs.max()),
                needs=float(env.goal_x - x0),
                frac_of_goal=float((xs.max() - x0) / (env.goal_x - x0)),
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
    # D3 M3 E3: `roll` returns None when the DESIGN stages end the episode --
    # an evolved body that fails to compile or to reset. With the body frozen
    # that cannot happen and `n_eps == episodes` always; with the design stages
    # live it can, and silently dropping those episodes would bias every rate
    # in this dict toward the designs that survive. Counted, not dropped.
    g = lambda k: np.array([e[k] for e in eps], dtype=float)
    R = g("R")
    return dict(n_eps=len(eps), n_requested=int(episodes),
                design_fail_rate=float((episodes - len(eps)) / episodes),
                bodies_exec=float(g("bodies_exec").mean()),
                R_mean=float(R.mean()), R_sd=float(R.std(ddof=1))
                if len(R) > 1 else 0.0,
                R_min=float(R.min()), R_max=float(R.max()),
                ep_len=float(g("n").mean()),
                goal_rate=float(g("reached").mean()),
                loss_rate=float(g("opp_reached").mean()),
                fall_rate=float(g("fell").mean()),
                net_dx=float(g("net_dx").mean()),
                max_x=float(g("max_x").mean()),
                max_fwd=float(g("max_fwd").mean()),
                max_fwd_best=float(g("max_fwd").max()),
                frac_of_goal=float(g("frac_of_goal").mean()),
                frac_of_goal_best=float(g("frac_of_goal").max()),
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
    panel is LABELLED with its own pass-2 numbers.

    What the three panels are depends on the rung, and the prose here used to
    claim otherwise:

      * With morphology FROZEN (E2, E2.1) all three are the same creature, so
        the clip shows gait and tactics only.
      * With the design stages LIVE (E3, E3.1, E4B) they are three DIFFERENT
        creatures, which is why `nb=` is appended to each label.

    "Whether it dodges the opponent" is also rung-specific and is FALSE for
    E4B: E2/E3's opponent was a scripted mover that could be dodged, whereas
    E4B's is a past self that races and collides head-on. Read an E4B clip as
    a race, not a dodge."""
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
        # D3 M3 E3: with the design stages live the three panels are three
        # DIFFERENT creatures, so the body count belongs on the label. Appended
        # only when the design stages actually changed the body, so every E2
        # and E2.1 clip renders byte-identically to the ones already logged.
        if e["bodies_exec"] != e["bodies"]:
            tag += f"  nb={e['bodies_exec']}"
        # How far the OPPONENT travelled. Without this the panel cannot
        # distinguish a real match from one rendered against an inert body --
        # which is exactly the failure that made every clip before 2026-09-06
        # misleading. `opp_dx` near 0 means the purple side never moved.
        odx = None
        if e.get("opp_com_x") is not None:
            odx = abs(float(e["opp_com_x"]) - 1.0)   # it starts at x = +1
            tag += f"  oppdx={odx:.2f}m"
        e["opp_dx"] = odx
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
            if e.get("opp_dx") is not None:
                sc[f"video/{k}_opp_dx"] = float(e["opp_dx"])
    return path, sc
