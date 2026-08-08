"""Replay a demo file. Three modes, because "deterministic" means three things.

    state       write the recorded qpos back into a fresh env and render it.
                Exact by construction; this is the video you show people.
    action      rebuild the env from the demo's seed, force it to the recorded
                tick-0 state, then re-step it with the RECORDED ACTIONS and compare
                qpos tick by tick.  This is the real determinism check: if it
                diverges, the demo does not describe the match it claims to.
    controller  re-run the SkillController over the recorded observations, root
                poses, skills and targets and compare the actions bit-for-bit.
                This is the check that matters for the BC dataset -- it proves the
                demo carries everything needed to reproduce a policy's output
                without a simulator at all.

    MUJOCO_GL=egl .venv/bin/python -m rower_soccer.game.replay demos/x.demo.npz \
        --mode state --video out.mp4
    .venv/bin/python -m rower_soccer.game.replay demos/x.demo.npz --mode controller
"""

from __future__ import annotations

import argparse

import numpy as np

from rower_soccer.game.recording import read_demo


def _env_from(demo, render_size=None):
    """A MatchSim rebuilt from the demo's metadata -- same creature, same pitch, same
    seed, same dt.  No controller: replay never runs a policy unless asked to."""
    from rower_soccer.game.match import MatchSim
    m = demo.meta
    cam = m.camera or {}
    size = render_size or (int(cam.get("px_w", 960)), int(cam.get("px_h", 640)))
    creature = m.players[0].creature if m.players else "ant"
    sim = MatchSim(creature=creature, pitch_half=m.pitch_half,
                   match_seconds=m.time_limit, seed=m.seed,
                   physics_dt=m.physics_dt, render_size=size, countdown=0.0)
    return sim


def _set_state(sim, qpos, qvel=None):
    d = sim.physics.data
    d.qpos[:] = qpos
    if qvel is not None:
        d.qvel[:] = qvel
    else:
        d.qvel[:] = 0.0
    sim.physics.forward()


def replay_state(demo, video=None, fps=None, stride=1, render_size=None):
    """Render the recorded trajectory. Returns the frame list (or [] if streamed)."""
    sim = _env_from(demo, render_size)
    qpos, qvel = demo.arrays["qpos"], demo.arrays.get("qvel")
    tgt = demo.arrays["target"]
    frames = []
    writer = None
    if video:
        import imageio
        writer = imageio.get_writer(video, fps=fps or int(round(1 / demo.meta.control_dt / stride)))
    try:
        for t in range(0, len(qpos), stride):
            _set_state(sim, qpos[t], None if qvel is None else qvel[t])
            for p in range(min(len(sim._markers), tgt.shape[1])):
                sim._markers[p].pos = np.array([tgt[t, p, 0], tgt[t, p, 1], 0.35])
            frame = sim.render()
            if writer is not None:
                writer.append_data(frame)
            else:
                frames.append(frame)
    finally:
        if writer is not None:
            writer.close()
    return frames


def replay_actions(demo, video=None, stride=1, render_size=None, tol=1e-4):
    """Re-simulate from the recorded actions and measure the drift.

    Returns a report dict.  `max_qpos_err` is the worst per-tick deviation between
    the re-simulated state and the recorded one.  On CPU MuJoCo with the same model
    and the same inputs this should be ~0 (float32 storage rounding only, ~1e-7).
    """
    from rower_soccer.game.match import restore_rng
    sim = _env_from(demo, render_size)
    sim.start_match(demo_path=None)           # same reset the recorder did
    restore_rng(sim.env, demo.meta.rng_state)  # ...and the same RNG position
    q, a = demo.arrays["qpos"], demo.arrays["action"]
    qv = demo.arrays.get("qvel")
    _set_state(sim, q[0], None if qv is None else qv[0])
    writer = None
    if video:
        import imageio
        writer = imageio.get_writer(video, fps=int(round(1 / demo.meta.control_dt)))
    errs = np.zeros(len(q), np.float64)
    try:
        for t in range(len(q) - 1):
            # env.step, not a hand-rolled substep loop: the task's before/after
            # substep hooks drive the goal and off-court detectors and the ball's
            # possession trackers, and MultiturnTask re-spawns on a goal. Replaying
            # only the integrator would silently drift the moment anyone scores.
            sim.timestep = sim.env.step([a[t, p] for p in range(a.shape[1])])
            errs[t + 1] = float(np.max(np.abs(sim.physics.data.qpos - q[t + 1])))
            if writer is not None and t % stride == 0:
                writer.append_data(sim.render())
    finally:
        if writer is not None:
            writer.close()
    return dict(n=len(q), max_qpos_err=float(errs.max()),
                mean_qpos_err=float(errs.mean()),
                err_at_1s=float(errs[min(len(errs) - 1, int(1 / demo.meta.control_dt))]),
                deterministic=bool(errs.max() <= tol), tol=tol)


def replay_controller(demo, tol=1e-5, players=None, max_ticks=None):
    """Re-run the skill layer over the recorded rows; compare actions, z and the
    reconstructed expert input.

    Needs no simulator: `obs` + `player_pos` + `player_mat` is exactly a
    `skills.PlayerFrame`, which is why the schema stores the root pose separately
    (dm_soccer's observation deliberately omits it).

    `tol` is 1e-5, not 0: dm_soccer emits float64 observations and the demo stores
    them as float32 (halving the largest array in the file), so rebuilding the
    expert's input from the stored obs lands ~1e-7 away from the vector the live
    match fed it, and the network turns that into ~1e-6 on the action. The check
    that IS exact is `max_obs_err`, which compares against the stored `skill_obs` --
    the actual vector that went into the policy.
    """
    from rower_soccer.skills import PlayerFrame, SkillController
    m = demo.meta
    P = demo.n_players
    players = range(P) if players is None else players
    n = demo.n_ticks if max_ticks is None else min(demo.n_ticks, max_ticks)
    ck = {s: v["path"] for s, v in (m.checkpoints or {}).items() if v.get("path")}
    vocab = m.skill_vocab
    out = {}
    for p in players:
        ctrl = SkillController(m.players[p].creature, checkpoints=ck,
                               action_mode=m.action_mode, seed=m.skill_seed,
                               player_index=p, quiet=True, name=m.players[p].slot)
        da = dz = do = 0.0
        cmp_n = 0
        for t in range(n):
            skill = vocab[int(demo.arrays["skill"][t, p])]
            if skill == "idle":
                continue
            obs = _obs_dict(demo, t, p)
            frame = PlayerFrame(obs=obs, root_pos=demo.arrays["player_pos"][t, p],
                                root_mat=demo.arrays["player_mat"][t, p])
            ctrl.set_command(skill, tuple(demo.arrays["target"][t, p]))
            # In MODE_NOISE the action depends on the controller's OWN tick, which
            # resets on a skill switch -- so restore it rather than assume t.
            ctrl.tick = int(demo.arrays["ctrl_tick"][t, p])
            r = ctrl.act(frame)
            da = max(da, float(np.max(np.abs(r.action - demo.arrays["action"][t, p]))))
            if r.z is not None:
                zrec = demo.arrays["z"][t, p][:len(r.z)]
                dz = max(dz, float(np.max(np.abs(r.z - zrec))))
            if "skill_obs" in demo.arrays:
                n_o = int(demo.arrays["skill_obs_n"][t, p])
                if n_o:
                    rec_o = demo.arrays["skill_obs"][t, p, :n_o]
                    do = max(do, float(np.max(np.abs(r.obs_vector[:n_o] - rec_o))))
            cmp_n += 1
        out[m.players[p].slot] = dict(ticks=cmp_n, max_action_err=da, max_z_err=dz,
                                      max_obs_err=do,
                                      ok=bool(cmp_n == 0 or (da <= tol and dz <= tol)))
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    out["tol"] = tol
    return out


def _obs_dict(demo, t, p):
    """Row (t, p) of `obs` split back into the dm_soccer observation dict."""
    vec = demo.arrays["obs"][t, p]
    d, i = {}, 0
    for k, n in zip(demo.meta.obs_keys, demo.meta.obs_sizes):
        d[k] = vec[i:i + n]
        i += n
    return d


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("demo")
    p.add_argument("--mode", default="state",
                   choices=["state", "action", "controller", "all"])
    p.add_argument("--video", default=None, help="mp4 output (state/action modes)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-ticks", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    a = p.parse_args(argv)

    demo = read_demo(a.demo)
    size = (a.width, a.height) if a.width and a.height else None
    from rower_soccer.game.recording import summarize
    print(summarize(a.demo))

    if a.mode in ("state", "all"):
        frames = replay_state(demo, video=a.video if a.mode != "all" else None,
                              stride=a.stride, render_size=size)
        print(f"[replay:state] rendered {len(frames) or 'streamed'} frames"
              + (f" -> {a.video}" if a.video and a.mode != 'all' else ""))
    if a.mode in ("action", "all"):
        r = replay_actions(demo, video=a.video if a.mode == "action" else None,
                           stride=a.stride, render_size=size)
        print(f"[replay:action] {'DETERMINISTIC' if r['deterministic'] else 'DIVERGED'} "
              f"max|dqpos|={r['max_qpos_err']:.3e} mean={r['mean_qpos_err']:.3e} "
              f"@1s={r['err_at_1s']:.3e} over {r['n']} ticks")
    if a.mode in ("controller", "all"):
        r = replay_controller(demo, max_ticks=a.max_ticks)
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"[replay:controller] {k:8s} ticks={v['ticks']:5d} "
                      f"max|da|={v['max_action_err']:.3e} max|dz|={v['max_z_err']:.3e} "
                      f"max|dobs|={v['max_obs_err']:.3e} "
                      f"{'OK' if v['ok'] else 'MISMATCH'}")
        print(f"[replay:controller] {'OK' if r['ok'] else 'MISMATCH'}")


if __name__ == "__main__":
    main()
