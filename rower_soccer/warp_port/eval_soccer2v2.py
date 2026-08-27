"""D1 unit 1f -- evaluation rollout + video from a TRAINED soccer2v2 checkpoint.

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.eval_soccer2v2 \
        --ckpt runs_v2/soccer2v2_1f_base/final.pt \
        --out  runs_v2/soccer2v2_1f_base/final_eval.mp4 \
        --worlds 64 --matches 4

WHY THIS EXISTS AND NOT `train_soccer2v2_warp.render_clip`
-----------------------------------------------------------
The trainer's in-loop `render_clip` is a monitor: ONE world, world 0, 15 s, no
aggregate behind it. D1_UNIT1F 4.2 records the failure mode that gives -- the
first warm-start render was a single world in which nobody reached the ball,
and "the warm start does not move" was nearly written down; over 64 worlds the
opposite was true. So this module AGGREGATES FIRST over `worlds x matches`
complete 45 s matches and only then picks what to film, by rank within that
population.

Everything else is reused rather than rewritten: `make_env` (so the eval env is
the trained env, from the run's own `config.json`), `Soccer2v2Renderer` from the
1e probe (the same render-only MjModel with the identical qpos layout), and the
trainer's DETERMINISTIC mean-action convention -- at the entropy ceiling the
action std is a large fraction of the +/-1 range, so a sampled clip shows the
exploration noise rather than the policy. `--stochastic` runs the sampled
policy instead, which is the one training's own `match` metrics were measured
under; the tool reports both when asked, so the video's numbers and the
training log's numbers are comparable on the same axis.

WHAT IS FILMED
--------------
Four panels, one per QUARTILE of the match population ranked by ball path
length (how far the ball actually travelled during the match) -- the quartile
midpoints, i.e. ranks at 12.5 / 37.5 / 62.5 / 87.5 %. That brackets the
distribution instead of asserting a typical world exists. Panels are labelled
with their quartile and their final score, so a viewer can see which end of the
distribution they are looking at.

The filmed frames are REPLAYED from qpos recorded during the aggregate rollout,
not re-simulated: the picture is the state that produced the numbers.

UPRIGHT DIAGNOSTIC
------------------
D1_UNIT1F 5.10 leaves `upright ~ 0.6` unexplained and asks whether it is four
ants knocking each other over or the shoot gait degrading off its training
distribution. This module separates them the cheap way: mean uprightness
conditioned on the distance to the nearest OTHER creature. If contact is the
cause, uprightness should be much lower when someone is close than when the
creature is alone. It is a correlation, not a controlled experiment -- players
are close precisely when they are all doing the same thing near the ball -- and
is reported as such.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from rower_soccer.warp_port.ppo import ActorCritic, _flatten_checkpoint
from rower_soccer.warp_port.train_soccer2v2_warp import build_parser, make_env


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
def env_args_from_run(ckpt_path, overrides=None):
    """The trainer's own parser defaults, overwritten by the run's config.json.

    An eval env that differs from the trained env (match length, pitch scale,
    ball mass, spawn) silently measures a different task, so the env spec is
    taken from the run that produced the checkpoint rather than re-typed.
    """
    args = build_parser().parse_args(["--run-name", "_eval"])
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                            "config.json")
    used = None
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and k not in ("worlds", "seed", "resume"):
                setattr(args, k, v)
        used = cfg_path
    for k, v in (overrides or {}).items():
        setattr(args, k, v)
    return args, used


def load_policy(ac, path, device):
    """Load an `export_sb3_compatible` export (final.pt / latest.pt) or a
    resume checkpoint, STRICTLY -- eval silently running a partly-random policy
    is exactly the failure this refuses."""
    flat = _flatten_checkpoint(torch.load(path, map_location=device,
                                          weights_only=True))
    own = ac.state_dict()
    missing = [k for k in own if k not in flat]
    # p_idx/t_idx are this env's observation layout; they are saved inside
    # mlp_extractor.state_dict() but must not be taken from the file if they
    # disagree -- that would slice the wrong columns out of every observation.
    for k in ("mlp_extractor.p_idx", "mlp_extractor.t_idx"):
        if k in flat:
            assert torch.equal(flat[k].to(own[k].device).long(), own[k].long()), \
                f"{k} in checkpoint disagrees with this env's obs layout"
            flat[k] = own[k]
    assert not missing, f"checkpoint is missing {len(missing)} tensors: {missing[:6]}"
    ac.load_state_dict(flat, strict=True)
    return len(flat)


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_matches(env, ac, n_matches, deterministic=True, record=True, seed=0):
    """`n_matches` complete matches over every world; returns per-match stats.

    Returns (rows, qpos) where `rows` is a list of dicts, one per (match,
    world), and `qpos` is a list of [T, n, nq] float16 CPU tensors (one per
    match) or None. Recording qpos rather than frames keeps the memory flat:
    64 worlds x 1800 steps x nq is ~15 MB per match, where the frames would be
    tens of GB.
    """
    rows, qpos_all = [], []
    gen = torch.Generator(device=env.device).manual_seed(seed + 4242)
    for m in range(n_matches):
        obs = env.reset()
        T = env.episode_steps
        up_sum = torch.zeros(env.n, env.n_agents, device=env.device)
        up_near = torch.zeros(env.n, env.n_agents, device=env.device)
        n_near = torch.zeros(env.n, env.n_agents, device=env.device)
        up_far = torch.zeros(env.n, env.n_agents, device=env.device)
        n_far = torch.zeros(env.n, env.n_agents, device=env.device)
        down = torch.zeros(env.n, env.n_agents, device=env.device)
        falls = torch.zeros(env.n, env.n_agents, device=env.device)
        recov = torch.zeros(env.n, env.n_agents, device=env.device)
        was_down = torch.zeros(env.n, env.n_agents, dtype=torch.bool,
                               device=env.device)
        up_early = torch.zeros(env.n, env.n_agents, device=env.device)
        up_late = torch.zeros(env.n, env.n_agents, device=env.device)
        edge = int(5.0 / 0.025)              # 5 s at each end of the match
        ball_path = torch.zeros(env.n, device=env.device)
        near_ball = torch.zeros(env.n, device=env.device)
        prev_ball = env.ball_xy().clone()
        qbuf = torch.empty(T, env.n, env.model.nq, dtype=torch.float16,
                           device="cpu") if record else None

        for t in range(T):
            d = ac.dist(obs.float())
            a = (d.mean if deterministic else d.sample()).clamp(-1, 1)
            obs, _, done = env.step(a)

            if record:
                qbuf[t] = env.qpos.detach().to("cpu", torch.float16)
            up = env.upright()                                   # [n, A]
            up_sum += up
            is_down = up < 0.5
            down += is_down.float()
            # A fall is an up->down CROSSING and a recovery is the reverse, so
            # `recoveries / falls` says whether a creature that goes over ever
            # gets back up -- which `mean upright` alone cannot distinguish
            # from everyone being permanently half-tilted.
            falls += (is_down & ~was_down).float()
            recov += (~is_down & was_down).float()
            was_down = is_down
            if t < edge:
                up_early += up
            if t >= T - edge:
                up_late += up
            xy = torch.stack([env.root_xy(k) for k in range(env.n_agents)], 1)
            dmat = torch.cdist(xy, xy)                           # [n, A, A]
            dmat += torch.eye(env.n_agents, device=env.device) * 1e6
            nearest = dmat.min(-1).values                        # [n, A]
            close = (nearest < 1.5).float()
            up_near += up * close
            n_near += close
            up_far += up * (1 - close)
            n_far += (1 - close)
            ball = env.ball_xy()
            ball_path += torch.linalg.norm(ball - prev_ball, dim=-1)
            prev_ball = ball.clone()
            near_ball += torch.linalg.norm(
                ball[:, None, :] - xy, dim=-1).min(-1).values
            if done:
                break

        sc = env.score.cpu().numpy()
        for w in range(env.n):
            rows.append(dict(
                match=m, world=w,
                home=float(sc[w, 0]), away=float(sc[w, 1]),
                goals=float(sc[w, 0] + sc[w, 1]),
                throw_ins=float(env.throw_ins[w]),
                ball_dist=float(torch.linalg.norm(env.ball_xy()[w])),
                ball_path=float(ball_path[w]),
                nearest_ant_to_ball=float(near_ball[w] / T),
                upright=float(up_sum[w].mean() / T),
                down_frac=float(down[w].mean() / T),
                falls=float(falls[w].sum()),
                recoveries=float(recov[w].sum()),
                down_at_end=float(was_down[w].float().mean()),
                upright_first5s=float(up_early[w].mean() / edge),
                upright_last5s=float(up_late[w].mean() / edge),
                up_near=float(up_near[w].sum() / n_near[w].sum().clamp(min=1)),
                n_near=float(n_near[w].sum()),
                up_far=float(up_far[w].sum() / n_far[w].sum().clamp(min=1)),
                n_far=float(n_far[w].sum()),
            ))
        if record:
            qpos_all.append(qbuf)
    return rows, (qpos_all if record else None)


def summarise(rows, env):
    """Aggregate over every completed match. Every number here is a mean over
    `len(rows)` matches, never a single world."""
    g = np.array([r["goals"] for r in rows])
    h = np.array([r["home"] for r in rows])
    a = np.array([r["away"] for r in rows])
    w = lambda k: np.array([r[k] for r in rows])
    nn, nf = w("n_near").sum(), w("n_far").sum()
    return dict(
        matches=len(rows),
        goals_per_match=float(g.mean()),
        home_goals=float(h.mean()), away_goals=float(a.mean()),
        p_0_goals=float((g == 0).mean()),
        p_1_goal=float((g == 1).mean()),
        p_2plus_goals=float((g >= 2).mean()),
        max_goals=float(g.max()),
        p_home_win=float((h > a).mean()), p_away_win=float((a > h).mean()),
        p_draw=float((h == a).mean()),
        throw_ins=float(w("throw_ins").mean()),
        ball_dist=float(w("ball_dist").mean()),
        ball_path_m=float(w("ball_path").mean()),
        nearest_ant_to_ball_m=float(w("nearest_ant_to_ball").mean()),
        p_ball_path_lt_30m=float((w("ball_path") < 30.0).mean()),
        upright=float(w("upright").mean()),
        down_frac=float(w("down_frac").mean()),
        falls_per_match=float(w("falls").mean()),
        recoveries_per_match=float(w("recoveries").mean()),
        recovery_ratio=float(w("recoveries").sum() / max(w("falls").sum(), 1)),
        p_player_down_at_end=float(w("down_at_end").mean()),
        upright_first5s=float(w("upright_first5s").mean()),
        upright_last5s=float(w("upright_last5s").mean()),
        upright_when_crowded=float((w("up_near") * w("n_near")).sum() / max(nn, 1)),
        upright_when_alone=float((w("up_far") * w("n_far")).sum() / max(nf, 1)),
        frac_time_crowded=float(nn / max(nn + nf, 1)),
        diverged=int(env.n_diverged),
    )


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------
def _label(img, text, sub=""):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                               max(14, img.shape[0] // 26))
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                max(11, img.shape[0] // 34))
    except OSError:                                    # pragma: no cover
        f = fs = ImageFont.load_default()
    d.rectangle([0, 0, im.width, int(im.height * 0.115)], fill=(16, 16, 16))
    d.text((8, 4), text, fill=(255, 255, 255), font=f)
    if sub:
        d.text((8, 4 + int(im.height * 0.055)), sub, fill=(190, 190, 190), font=fs)
    return np.asarray(im)


def render_grid(env, qpos_all, picks, path, fps=40, panel=(800, 600),
                cols=2, ren=None, pip=True, pip_frac=0.34, pip_dist=6.0,
                pip_elev=-42.0):
    """Replay the recorded qpos of the picked (match, world) pairs into one
    tiled clip. Streams frame by frame -- holding 4 x 1800 panels in RAM would
    be ~10 GB for a 45 s clip.

    Each panel is a top-down view of the whole pitch with a ball-tracking
    close-up inset (`pip`). The top-down alone cannot answer "are they falling
    over": on a 30 m pitch an ant is ~20 px and a fallen one looks much like a
    standing one from directly above. The inset is the same frame from the
    broadcast camera at `pip_dist` m, looking at the ball, which is where the
    creatures are -- so posture and gait are visible next to the tactics.
    """
    import imageio
    from rower_soccer.warp_port.probe_soccer2v2 import Soccer2v2Renderer

    class _Q:            # the renderer only ever reads `.qpos[w]`
        __slots__ = ("qpos",)

    ren = ren or Soccer2v2Renderer(env, width=panel[0], height=panel[1])
    pw, ph = panel
    iw, ih = int(pw * pip_frac) // 2 * 2, int(ph * pip_frac) // 2 * 2
    zoom = Soccer2v2Renderer(env, width=iw, height=ih) if pip else None
    if zoom is not None:
        # steeper than the broadcast camera: at 6 m a -28 deg elevation puts a
        # third of the inset above the horizon, which shows nothing.
        zoom.free.elevation = pip_elev
    shim = _Q()
    T = min(qpos_all[p["match"]].shape[0] for p in picks)
    rows = int(np.ceil(len(picks) / cols))
    pad = 8
    with imageio.get_writer(path, fps=fps, quality=8,
                            macro_block_size=8) as wr:
        for t in range(T):
            grid = np.full((rows * ph + (rows + 1) * pad,
                            cols * pw + (cols + 1) * pad, 3), 20, np.uint8)
            for i, p in enumerate(picks):
                q = qpos_all[p["match"]][t].float()
                shim.qpos = q
                f = _label(ren.frame(shim, p["world"], "topdown"),
                           p["title"], p["sub"])
                if pip:
                    b = q[p["world"], env.bq:env.bq + 3].numpy()
                    z = zoom.frame(shim, p["world"], "free",
                                   lookat=[float(b[0]), float(b[1]), 0.35],
                                   distance=pip_dist)
                    y0, x0 = ph - ih - 6, pw - iw - 6
                    f = f.copy()
                    f[y0 - 2:y0 + ih + 2, x0 - 2:x0 + iw + 2] = 235
                    f[y0:y0 + ih, x0:x0 + iw] = z
                r, c = divmod(i, cols)
                y, x = pad + r * (ph + pad), pad + c * (pw + pad)
                grid[y:y + ph, x:x + pw] = f
            wr.append_data(grid)
    return T


def pick_quartiles(rows, n=4, key="ball_path"):
    """One match per quartile of `key`, at the quartile MIDPOINTS. Returns the
    picks with the rank they were chosen at, so the selection rule travels with
    the picture instead of living only in a report."""
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    picks = []
    for q in range(n):
        frac = (q + 0.5) / n
        idx = order[min(len(order) - 1, int(frac * len(order)))]
        r = rows[idx]
        picks.append(dict(
            match=r["match"], world=r["world"], rank=int(frac * 100), row=r,
            title=f"Q{q + 1}  ({int(frac * 100)}th pct by ball travel)",
            sub=(f"blue {r['home']:.0f} - {r['away']:.0f} red   "
                 f"ball travelled {r['ball_path']:.0f} m   "
                 f"upright {r['upright']:.2f}")))
    return picks


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs_v2/soccer2v2_1f_base/final.pt")
    p.add_argument("--out", default="runs_v2/soccer2v2_1f_base/final_eval.mp4")
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--matches", type=int, default=4)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--stochastic", action="store_true",
                   help="sample instead of the mean action -- the convention "
                        "training's own match metrics were measured under")
    p.add_argument("--also-stochastic", action="store_true",
                   help="run a second aggregate pass with sampling, for "
                        "comparison against the training log (no video)")
    p.add_argument("--panel-w", type=int, default=800)
    p.add_argument("--panel-h", type=int, default=600)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-pip", action="store_true",
                   help="drop the ball-tracking close-up inset")
    p.add_argument("--pip-dist", type=float, default=6.0)
    p.add_argument("--match-secs", type=float, default=0.0,
                   help="override the run's match length (smoke tests only; "
                        "an eval at a different match length is a different "
                        "task from the one that was trained)")
    p.add_argument("--json-out", default="")
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    over = {"match_secs": args.match_secs} if args.match_secs > 0 else None
    eargs, cfg_used = env_args_from_run(args.ckpt, over)
    env = make_env(eargs, num_worlds=args.worlds, seed=args.seed)
    dev = str(env.device)
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(),
                     z_dim=eargs.z_dim,
                     state_dependent_std=eargs.state_dependent_std).to(dev)
    n = load_policy(ac, args.ckpt, dev)
    ac.eval()
    print(f"[eval] {args.ckpt}: {n} tensors loaded strictly; env from "
          f"{cfg_used or 'trainer defaults'}", flush=True)
    print(f"[eval] worlds={env.n} match={eargs.match_secs}s "
          f"({env.episode_steps} steps) matches={args.matches} "
          f"-> {env.n * args.matches} matches, "
          f"std={float(ac.log_std.exp().mean()):.3f}, "
          f"actions={'sampled' if args.stochastic else 'mean (deterministic)'}",
          flush=True)

    t0 = time.time()
    rows, qpos = run_matches(env, ac, args.matches,
                             deterministic=not args.stochastic,
                             record=not args.no_video, seed=args.seed)
    stats = summarise(rows, env)
    stats["mode"] = "sampled" if args.stochastic else "deterministic"
    stats["rollout_secs"] = round(time.time() - t0, 1)
    if torch.cuda.is_available():
        stats["gpu_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    print("[eval] " + json.dumps(stats, indent=1), flush=True)

    out = {"ckpt": args.ckpt, "primary": stats}
    if args.also_stochastic:
        r2, _ = run_matches(env, ac, args.matches, deterministic=args.stochastic,
                            record=False, seed=args.seed + 1)
        s2 = summarise(r2, env)
        s2["mode"] = "deterministic" if args.stochastic else "sampled"
        print("[eval] second pass " + json.dumps(s2, indent=1), flush=True)
        out["second"] = s2

    if not args.no_video:
        picks = pick_quartiles(rows, 4)
        out["picks"] = [{k: v for k, v in p.items() if k != "row"} for p in picks]
        for p_ in picks:
            print(f"[film] {p_['title']}: match {p_['match']} world "
                  f"{p_['world']} | {p_['sub']}", flush=True)
        t1 = time.time()
        T = render_grid(env, qpos, picks, args.out,
                        panel=(args.panel_w, args.panel_h),
                        pip=not args.no_pip, pip_dist=args.pip_dist)
        print(f"[film] {args.out}: {T} frames, {T / 40.0:.1f}s at 40 fps, "
              f"{os.path.getsize(args.out) / 1e6:.1f} MB, "
              f"{time.time() - t1:.0f}s to render", flush=True)
        out["video"] = args.out

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
