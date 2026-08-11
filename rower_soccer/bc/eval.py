"""Is this BC policy any good? Two answers, because one of them lies.

    # held-out action agreement only (fast, no simulator)
    PYTHONPATH=. .venv/bin/python -m rower_soccer.bc.eval \
        runs_v2/bc/ant_action_v1/best.pt --data runs_v2/bc/ant_2v2_v1.npz

    # + behaviour in the CPU soccer env, against the scripted baseline, with video
    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.bc.eval \
        runs_v2/bc/ant_action_v1/best.pt --data runs_v2/bc/ant_2v2_v1.npz \
        --rollout --episodes 3 --seconds 30 --video runs_v2/bc/ant_action_v1/roll.mp4

**Agreement** is the cheap answer: mean squared error against the held-out
matches' recorded actions, overall and per actuator, sliced by controller
(human vs the scripted chase), by skill, and by mirrored/original. It is a real
measurement and it is also the one that lies, in the specific way this project
has been burned by before: an action MSE of 0.2 on a corpus whose actions
saturate 52% of the time can be produced by a policy that walks and by a policy
that vibrates. Nothing about "how close is each torque" says whether the ant
gets to the ball.

**Rollout** is the answer that does not lie. Four ants, one r=0.15 drill ball
(the ball every current checkpoint trained on — the stock 0.35 m ball is a
different task), 2v2, the drill's own integration step. Two arms on the same
seeds so the comparison is paired:

    bc        home = the BC policy,  away = the scripted chase
    baseline  home = the scripted chase, away = the scripted chase

and the numbers are behavioural: ball touches, possession (nearest player, and
nearest-within-1.5 m), how far the team moved the ball, goals, time upright,
distance walked. The scripted chase is a strong baseline at exactly one thing —
being near the ball — so "possession vs baseline" is the honest headline, and
being *beaten* by it is the expected first result on a corpus this size.

And then `--video`: watch it. Metrics lie, videos do not. The rule has been
learned five times over in this repo; the harness makes it one flag.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["agreement", "rollout", "compare", "make_eval_env", "BASELINE_SKILL"]

#: What an unclaimed seat plays in the game, and therefore the baseline every BC
#: policy has to beat before it is worth putting in a video.
BASELINE_SKILL = "scripted"

#: `world_zaxis` z-component; +1 upright, 0 on its side. Same threshold as
#: `skills/demo_follow_soccer.py`.
UPRIGHT_MIN = 0.5

#: Metres. A player nearer than this to the ball is "on the ball" — roughly two
#: ant body lengths, and about where the dribble expert's ball_ego inputs live.
CLOSE_RADIUS = 1.5


# --- held-out agreement ----------------------------------------------------

def agreement(runner, ds, *, batch: int = 8192, device: str = "cpu") -> dict:
    """Action agreement of `runner` on every sample of `ds`.

    `runner` is a `model.BCRunner` (or anything with `.ac` and `.obs_keys`).
    Reported on the CLAMPED prediction, because that is what a player applies.

    Beyond MSE it reports:
      explained      1 - MSE/Var, i.e. how much better than predicting the
                     corpus mean action. Negative means worse than a constant.
      sign_agree     fraction of actuator commands pushing the same way. A
                     torque's sign is most of its behaviour.
      sat_agree      fraction of |target| >= 0.999 entries the policy also
                     saturates the same way — 52% of this corpus is saturated,
                     so a policy that never reaches the rail is a different
                     animal however good its MSE.
      latent_mse     z error, where the demo recorded a z (never on mirrored
                     rows). Only meaningful for the latent architecture.
    """
    import torch

    ac = runner.ac
    cols = _dataset_columns(runner, ds)
    obs_all = np.asarray(ds.arrays["obs"])
    act_all = np.asarray(ds.arrays["action"], np.float32)
    z_all = np.asarray(ds.arrays["z"], np.float32)
    n = act_all.shape[0]
    if n == 0:
        raise ValueError("nothing to score: the dataset is empty")

    pred = np.empty_like(act_all)
    zpred = np.full_like(z_all, np.nan)
    has_z = hasattr(ac, "z")
    with torch.no_grad():
        for i in range(0, n, batch):
            o = torch.as_tensor(obs_all[i:i + batch][:, cols], dtype=torch.float32,
                                device=device)
            pred[i:i + batch] = ac.dist(o).mean.clamp(-1.0, 1.0).cpu().numpy()
            if has_z:
                zpred[i:i + batch] = ac.z(o).cpu().numpy()

    err = pred - act_all
    se = err ** 2
    per_row = se.mean(1)
    var = float(act_all.var(0).mean())
    sat = np.abs(act_all) >= 0.999
    out = dict(
        n=int(n),
        action_mse=float(per_row.mean()),
        action_rmse=float(math.sqrt(per_row.mean())),
        action_mae=float(np.abs(err).mean()),
        explained=float(1.0 - per_row.mean() / var) if var > 0 else float("nan"),
        per_actuator_mse=[round(float(x), 5) for x in se.mean(0)],
        per_actuator_mae=[round(float(x), 5) for x in np.abs(err).mean(0)],
        sign_agree=float((np.sign(pred) == np.sign(act_all)).mean()),
        sat_agree=(float(((np.abs(pred) >= 0.999) & (np.sign(pred) == np.sign(act_all)))[sat].mean())
                   if sat.any() else None),
        pred_saturated=float((np.abs(pred) >= 0.999).mean()),
        target_saturated=float(sat.mean()),
        pred_std=[round(float(x), 4) for x in pred.std(0)],
        target_std=[round(float(x), 4) for x in act_all.std(0)],
    )
    zm = np.isfinite(z_all).all(1) & np.isfinite(zpred).all(1)
    out["latent_mse"] = float(((zpred[zm] - z_all[zm]) ** 2).mean()) if zm.any() else None
    out["latent_n"] = int(zm.sum())

    a = ds.arrays
    for key, names in (("controller", ds.controller_vocab),
                       ("skill", ds.skill_vocab), ("split", ("train", "val"))):
        col = np.asarray(a[key])
        d = {}
        for i, name in enumerate(names):
            m = col == i
            if m.any():
                d[name] = dict(n=int(m.sum()), action_mse=round(float(per_row[m].mean()), 5),
                               explained=round(float(1 - per_row[m].mean()
                                                     / max(act_all[m].var(0).mean(), 1e-9)), 4))
        out[f"by_{key}"] = d
    mm = np.asarray(a["mirrored"])
    if mm.max(initial=0):
        out["by_mirrored"] = {
            ("original" if v == 0 else "mirrored"):
                dict(n=int((mm == v).sum()),
                     action_mse=round(float(per_row[mm == v].mean()), 5))
            for v in (0, 1)}
    return out


def _dataset_columns(runner, ds) -> np.ndarray:
    """Columns of the dataset's obs vector that make up the policy's input."""
    keys = list(ds.meta["obs_keys"])
    sizes = [int(s) for s in ds.meta["obs_sizes"]]
    off, i = {}, 0
    for k, n in zip(keys, sizes):
        off[k] = (i, i + n)
        i += n
    missing = [k for k in runner.obs_keys if k not in off]
    if missing:
        raise ValueError(f"the dataset has no observation keys {missing}, which "
                         f"the policy was trained on")
    bad = [(k, off[k][1] - off[k][0], w) for k, w in zip(runner.obs_keys, runner.obs_sizes)
           if off[k][1] - off[k][0] != w]
    if bad:
        raise ValueError(f"observation key widths differ between the policy and "
                         f"this dataset: {bad}")
    return np.concatenate([np.arange(*off[k]) for k in runner.obs_keys]).astype(np.int64)


# --- the rollout env -------------------------------------------------------

def make_eval_env(creature: str = "ant", *, seed: int = 0,
                  pitch_half=(15.0, 11.0), video: bool = False,
                  aspect: float = 4 / 3):
    """A 2v2 CPU soccer env matching the one the demos were recorded in.

    Same choices as `game/match.py:MatchSim`: the r=0.15 drill ball, the drill's
    physics step, a pinned pitch (`RandomizedPitch` otherwise resamples its size
    every episode, which would make paired seeds mean nothing), and no
    terminate-on-goal so a 30 s rollout is 30 s of play rather than 4 s and a
    reset.
    """
    from rower_soccer.skills.soccer import make_skill_soccer_env

    env = make_skill_soccer_env(home=(creature, creature), away=(creature, creature),
                                time_limit=1e6, random_state=seed, match_dt=True,
                                terminate_on_goal=False)
    arena = env.task.arena
    if hasattr(arena, "_min_size"):
        arena._min_size = arena._max_size = tuple(pitch_half)
    if video:
        _add_topdown_camera(env, pitch_half, aspect=aspect)
    return env


def _add_topdown_camera(env, pitch_half, aspect=4 / 3, height_mult=3.0,
                        margin=1.18):
    """A straight-down camera that frames the whole pitch, plus shadow removal.

    `fovy` is the VERTICAL field of view, so the half-height has to satisfy both
    axes: the pitch's own half-y, and its half-x divided by the frame aspect.
    Get that wrong and the ants are four pixels across in the middle of a lot of
    grass — which is exactly how a video stops being watchable and the "watch
    the eval video" rule quietly stops being followed.

    The pitch also ships four 8192 px shadowmaps that cost ~90 ms/frame and show
    nothing useful from above; turning them off is the difference between a
    video you wait for and one you don't.
    """
    half = margin * max(float(pitch_half[1]), float(pitch_half[0]) / max(aspect, 1e-6))
    h = half * height_mult
    wb = env.task.arena.mjcf_model.worldbody
    wb.add("camera", name="bc_topdown", pos=[0, 0, h], xyaxes=[1, 0, 0, 0, 1, 0],
           fovy=2.0 * math.degrees(math.atan(half / h)))
    for light in env.task.arena.mjcf_model.find_all("light"):
        light.castshadow = "false"
    env.task.arena.mjcf_model.visual.quality.offsamples = 0


def _camera_id(env, suffix="bc_topdown"):
    model = env.physics.model
    for i in range(model.ncam):
        name = model.camera(i).name
        if name and name.endswith(suffix):
            return i
    return 0


# --- rollout ---------------------------------------------------------------

def _make_agents(env, spec: Sequence[str], ckpt: Optional[str], creature: str,
                 seed: int, device: str, action_mode: str):
    """One callable per player slot. `spec[i]` is 'bc' or a registry skill id."""
    from rower_soccer.skills import SkillController
    from rower_soccer.bc.model import BCRunner

    agents, kinds = [], []
    bc = None
    for i, what in enumerate(spec):
        if what == "bc":
            if ckpt is None:
                raise ValueError("a 'bc' slot needs --checkpoint")
            if bc is None:
                bc = BCRunner(ckpt, device=device)
            agents.append(("bc", bc))
        else:
            ctrl = SkillController(creature, action_mode=action_mode, seed=seed,
                                   player_index=i, name=f"p{i}", quiet=True,
                                   preload=(what,) if what != "idle" else ())
            ctrl.set_command(what)
            agents.append(("skill", ctrl))
        kinds.append(what)
    return agents, kinds


def rollout(*, checkpoint: Optional[str] = None, home=("bc", "bc"),
            away=(BASELINE_SKILL, BASELINE_SKILL), creature: str = "ant",
            seconds: float = 30.0, seed: int = 0, device: str = "cpu",
            action_mode: str = "mean", video: Optional[str] = None,
            fps: int = 20, render_size=(640, 480), verbose: bool = False) -> dict:
    """Play one episode and measure behaviour. Returns per-team metrics.

    Home is slots 0-1, away 2-3 — the same order `env.step` wants and the same
    order the demos recorded.
    """
    from rower_soccer.skills.soccer import SoccerFrameSource

    spec = list(home) + list(away)
    env = make_eval_env(creature, seed=seed, video=bool(video),
                        aspect=render_size[0] / max(render_size[1], 1))
    src = SoccerFrameSource(env)
    if len(spec) != src.n_players:
        raise ValueError(f"{len(spec)} agents for {src.n_players} players")
    agents, kinds = _make_agents(env, spec, checkpoint, creature, seed, device,
                                 action_mode)

    ts = env.reset()
    hz = int(round(1.0 / env.task.control_timestep))
    n_steps = int(round(seconds * hz))
    stride = max(1, int(round(hz / max(fps, 1))))
    cam = _camera_id(env) if video else None
    frames: List[np.ndarray] = []

    P = src.n_players
    teams = np.array([0, 0, 1, 1])
    touches = np.zeros(P, int)
    goals = [0, 0]
    near_ticks = np.zeros(P, int)
    close_ticks = np.zeros(P, int)
    upright = np.zeros(P, int)
    walked = np.zeros(P)
    dist_sum = np.zeros(P)
    ball_path = 0.0
    prev_xy = np.stack([src.root_xy(i) for i in range(P)])
    prev_ball = src.ball_xy().copy()
    goal_latch = False
    t0 = time.time()

    for step in range(n_steps):
        fr = src.frames(ts)
        actions = []
        for i, (kind, agent) in enumerate(agents):
            if kind == "bc":
                actions.append(agent.action(fr[i].obs))
            else:
                actions.append(agent.action(fr[i]))
        actions = [np.asarray(a, np.float64) for a in actions]
        if not all(np.isfinite(a).all() for a in actions):
            raise RuntimeError(f"non-finite action at step {step}")

        # touches: rising edge on the arena's own last-hit record, exactly as
        # `game/match.py:_detect_touch` does it.
        ts = env.step(actions)
        ball = env.task.ball
        if getattr(ball, "hit", False) and ball.last_hit is not None:
            try:
                touches[env.task.players.index(ball.last_hit)] += 1
            except ValueError:
                pass
        g = env.task.arena.detected_goal()
        if g is None:
            goal_latch = False
        elif not goal_latch:
            goal_latch = True
            from dm_control.locomotion.soccer.team import Team
            goals[0 if g == Team.HOME else 1] += 1

        bxy = src.ball_xy()
        ball_path += float(np.linalg.norm(bxy - prev_ball))
        prev_ball = bxy.copy()
        xy = np.stack([src.root_xy(i) for i in range(P)])
        walked += np.linalg.norm(xy - prev_xy, axis=1)
        prev_xy = xy
        d = np.linalg.norm(xy - bxy[None, :], axis=1)
        dist_sum += d
        near_ticks[int(np.argmin(d))] += 1
        if d.min() < CLOSE_RADIUS:
            close_ticks[int(np.argmin(d))] += 1
        for i in range(P):
            upright[i] += int(np.asarray(fr[i].obs["world_zaxis"]).ravel()[2]
                              > UPRIGHT_MIN)

        if cam is not None and step % stride == 0:
            frames.append(env.physics.render(camera_id=cam, width=render_size[0],
                                             height=render_size[1]))
        if verbose and step % hz == 0:
            print(f"    t={step // hz:3d}s ball={np.round(bxy, 1)} "
                  f"nearest=p{int(np.argmin(d))} d={d.min():.2f}", flush=True)

    wall = time.time() - t0
    if video and frames:
        import imageio
        os.makedirs(os.path.dirname(os.path.abspath(video)) or ".", exist_ok=True)
        imageio.mimsave(video, frames, fps=max(1, int(round(hz / stride))))

    def team_block(t):
        m = teams == t
        return dict(
            touches=int(touches[m].sum()),
            possession=float(near_ticks[m].sum() / n_steps),
            close_possession=float(close_ticks[m].sum() / n_steps),
            mean_ball_distance=float(dist_sum[m].sum() / (n_steps * m.sum())),
            min_mean_ball_distance=float((dist_sum[m] / n_steps).min()),
            walked_m=float(walked[m].sum()),
            upright_frac=float(upright[m].sum() / (n_steps * m.sum())),
            goals_for=goals[t], goals_against=goals[1 - t],
            agents=[kinds[i] for i in range(P) if m[i]])

    return dict(seed=seed, seconds=seconds, steps=n_steps, hz=hz,
                wall_seconds=round(wall, 1),
                home=team_block(0), away=team_block(1),
                ball_path_m=round(ball_path, 2),
                per_player=dict(touches=touches.tolist(),
                                possession=[round(float(x / n_steps), 4)
                                            for x in near_ticks],
                                upright=[round(float(x / n_steps), 4) for x in upright]),
                video=video if (video and frames) else None)


def compare(checkpoint: str, *, episodes: int = 3, seconds: float = 30.0,
            creature: str = "ant", device: str = "cpu", video: Optional[str] = None,
            baseline: bool = True, verbose: bool = False) -> dict:
    """Paired arms on the same seeds: BC-vs-scripted and scripted-vs-scripted.

    The baseline arm is not decoration. The scripted chase is the follow expert
    retargeted at the ball every tick, i.e. the best "just go to the ball"
    policy this repo has; a BC prior that gets less of the ball than it does has
    learned to imitate the *look* of play without the substance, which is
    exactly the failure a low action MSE cannot see.
    """
    arms = {"bc": (("bc", "bc"), (BASELINE_SKILL, BASELINE_SKILL))}
    if baseline:
        arms["baseline"] = ((BASELINE_SKILL, BASELINE_SKILL),
                            (BASELINE_SKILL, BASELINE_SKILL))
    out: Dict[str, dict] = {}
    for name, (home, away) in arms.items():
        eps = []
        for s in range(episodes):
            vid = None
            if video and name == "bc" and s == 0:
                vid = video
            elif video and name == "baseline" and s == 0:
                root, ext = os.path.splitext(video)
                vid = f"{root}_baseline{ext}"
            print(f"[eval] arm {name} seed {s} ...", flush=True)
            eps.append(rollout(checkpoint=checkpoint, home=home, away=away,
                               creature=creature, seconds=seconds, seed=s,
                               device=device, video=vid, verbose=verbose))
        out[name] = dict(episodes=eps, mean=_mean_of(eps))
    return out


def _mean_of(eps: Sequence[dict]) -> dict:
    keys = ("touches", "possession", "close_possession", "mean_ball_distance",
            "walked_m", "upright_frac", "goals_for")
    m = {}
    for side in ("home", "away"):
        m[side] = {k: round(float(np.mean([e[side][k] for e in eps])), 4) for k in keys}
    m["ball_path_m"] = round(float(np.mean([e["ball_path_m"] for e in eps])), 2)
    return m


# --- reporting -------------------------------------------------------------

def format_agreement(a: dict, title: str = "held-out agreement") -> str:
    L = [f"{title}: n={a['n']}",
         f"  action MSE {a['action_mse']:.5f}  RMSE {a['action_rmse']:.4f}  "
         f"MAE {a['action_mae']:.4f}  explained {a['explained']:+.3f}",
         f"  sign agreement {a['sign_agree']:.1%}   "
         f"saturation: target {a['target_saturated']:.1%} / policy "
         f"{a['pred_saturated']:.1%}"
         + (f" / agreeing {a['sat_agree']:.1%}" if a["sat_agree"] is not None else "")]
    if a.get("latent_mse") is not None:
        L.append(f"  latent (z) MSE {a['latent_mse']:.4f} on {a['latent_n']} rows")
    L.append("  per actuator MSE " +
             " ".join(f"{x:.3f}" for x in a["per_actuator_mse"]))
    L.append("  policy std       " + " ".join(f"{x:.3f}" for x in a["pred_std"]))
    L.append("  demo std         " + " ".join(f"{x:.3f}" for x in a["target_std"]))
    for key in ("by_split", "by_controller", "by_skill", "by_mirrored"):
        if not a.get(key):
            continue
        L.append(f"  {key[3:]:11s} " + "   ".join(
            f"{k}: {v['action_mse']:.4f} (n={v['n']})" for k, v in a[key].items()))
    return "\n".join(L)


def format_rollout(cmp: dict) -> str:
    L = ["rollout (CPU soccer, r=0.15 drill ball, 2v2)",
         "  poss = fraction of ticks this team holds the nearest player; "
         "close = same, and within 1.5 m"]
    hdr = ("arm", "team", "agents", "touch", "poss", "close", "d_ball", "walk",
           "upright", "goals")
    L.append("  " + "  ".join(f"{h:>8s}" for h in hdr))
    for arm, block in cmp.items():
        for side in ("home", "away"):
            m = block["mean"][side]
            ag = block["episodes"][0][side]["agents"]
            tag = f"{len(ag)}x{ag[0][:6]}" if len(set(ag)) == 1 else "+".join(
                a[:3] for a in ag)
            L.append("  " + "  ".join(f"{v:>8}" for v in (
                arm, side, tag, f"{m['touches']:.1f}", f"{m['possession']:.2f}",
                f"{m['close_possession']:.2f}", f"{m['mean_ball_distance']:.2f}",
                f"{m['walked_m']:.0f}", f"{m['upright_frac']:.2f}",
                f"{m['goals_for']:.1f}")))
        L.append(f"    ball travelled {block['mean']['ball_path_m']:.1f} m")
    if "baseline" in cmp and "bc" in cmp:
        b = cmp["bc"]["mean"]["home"]
        s = cmp["baseline"]["mean"]["home"]
        L.append(f"  BC home vs scripted home: possession "
                 f"{b['possession']:.2f} vs {s['possession']:.2f}, touches "
                 f"{b['touches']:.1f} vs {s['touches']:.1f}, upright "
                 f"{b['upright_frac']:.2f} vs {s['upright_frac']:.2f}")
    return "\n".join(L)


# --- CLI -------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a best.pt / final.pt from bc.train")
    p.add_argument("--data", default=None,
                   help="dataset .npz for the agreement half (omit to skip it)")
    p.add_argument("--split", default="val", choices=["val", "train", "all"])
    p.add_argument("--contract", default="registry", choices=["registry", "all"])
    p.add_argument("--mirror", action="store_true",
                   help="also score the mirrored corpus (never train-honest, but "
                        "it says whether the policy learned the pitch symmetry)")
    p.add_argument("--rollout", action="store_true")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--creature", default="ant")
    p.add_argument("--video", default=None, help="mp4/gif path for seed 0")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", default=None, help="write the full report here")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    import torch
    torch.set_num_threads(1)
    from rower_soccer.bc.model import BCRunner

    report: Dict[str, object] = dict(checkpoint=os.path.abspath(args.checkpoint))
    runner = BCRunner(args.checkpoint, device=args.device)
    print(f"[eval] {runner}", flush=True)
    cfg = runner.meta["config"]
    print(f"[eval] arch={cfg['arch']} loss={cfg['loss']} "
          f"frozen_decoder={cfg['freeze_decoder']} "
          f"trained_epoch={runner.meta.get('epoch')}", flush=True)
    report["meta"] = {k: v for k, v in runner.meta.items()
                      if k not in ("src_cols", "proprio_indices", "task_indices",
                                   "frozen")}

    if args.data:
        from rower_soccer.bc.dataset import BCDataset
        from rower_soccer.bc.train import select_corpus
        ds = BCDataset.load(args.data)
        ds = select_corpus(ds, contract=args.contract, verbose=True)
        if args.split != "all":
            from rower_soccer.bc.dataset import SPLIT_TRAIN, SPLIT_VAL
            want = SPLIT_VAL if args.split == "val" else SPLIT_TRAIN
            ds = ds.select(ds.arrays["split"] == want)
        if args.mirror:
            from rower_soccer.bc.augment import mirror_dataset
            ds = mirror_dataset(ds, append=True)
        a = agreement(runner, ds, device=args.device)
        report["agreement"] = a
        print(format_agreement(a, f"held-out agreement ({args.split} split)"),
              flush=True)

    if args.rollout:
        cmp = compare(args.checkpoint, episodes=args.episodes, seconds=args.seconds,
                      creature=args.creature, device=args.device, video=args.video,
                      baseline=not args.no_baseline, verbose=args.verbose)
        report["rollout"] = cmp
        print(format_rollout(cmp), flush=True)
        if args.video:
            print(f"[eval] wrote {args.video} — WATCH IT. The agreement number "
                  "cannot tell a walk from a twitch.", flush=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1, default=float)
        print(f"[eval] wrote {args.json}", flush=True)
    return report


if __name__ == "__main__":
    main()
