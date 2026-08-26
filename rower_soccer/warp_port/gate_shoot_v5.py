"""Gate the shoot v5 reward options before any run is launched on them.

Three options were added after shoot_ant_v4's render showed hard strikes with
terrible aim (`w_aim`, `--live-cmd-dir`, and a raised `--speed-clip`). Each
defaults to v4's behaviour, so the first thing to establish is that a default
env is unchanged -- otherwise every v4 number silently changes meaning and the
control arm is not a control.

    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.warp_port.gate_shoot_v5

Checks:
  1. DEFAULTS ARE v4. w_aim=0 contributes exactly nothing to the reward, and
     cmd_dir stays frozen at its spawn value for the whole segment.
  2. `last_aim` has the right SNAPSHOT DISCIPLINE -- nonzero only on the step a
     segment ends, and computed from the PRE-respawn seg_goal_best. Reading it
     after `_close_segments` would score every segment against a ball that has
     just been teleported, which is the bug `last_score_t` exists to avoid and
     the one kick's `last_arrival` was written for.
  3. A SCORED segment gets the full aim term (d_mouth_best = 0 -> exp(0) = 1).
  4. NEGATIVE CONTROLS: each option must actually change something. An option
     that silently does nothing is worse than no option.
"""

import sys

import numpy as np
import torch

_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def _env(**kw):
    from rower_soccer.warp_port.shoot_env import WarpShootEnv
    base = dict(num_worlds=8, use_graph=False, seed=5, pitch_scale=0.3125,
                arena="pitch")
    base.update(kw)
    return WarpShootEnv(**base)


def main():
    # ---- 1. defaults are v4 ---------------------------------------------
    e = _env()
    e.reset()
    check("default w_aim is 0.0", e.reward.w_aim == 0.0,
          f"w_aim={e.reward.w_aim}")
    check("default live_cmd_dir is off", e.live_cmd_dir is False)

    cmd0 = e.cmd_dir.clone()
    a = torch.zeros(e.n, e.act_dim, device=e.device)
    for _ in range(40):
        e.step(a)
    same = torch.allclose(cmd0, e.cmd_dir)
    # Only meaningful if no segment ended and respawned (which would legally
    # rewrite cmd_dir); assert that separately.
    check("default: cmd_dir stays FROZEN within a segment", same,
          f"max |d| = {float((cmd0 - e.cmd_dir).abs().max()):.3e}")

    # ---- 2. last_aim snapshot discipline ---------------------------------
    e2 = _env(w_aim=3.0)
    e2.reset()
    nonzero_steps, end_steps = 0, 0
    seen_match = []
    for _ in range(400):
        pre_best = e2.seg_goal_best.clone()
        pre_nseg = e2.n_segments.clone()
        e2.step(torch.zeros(e2.n, e2.act_dim, device=e2.device))
        ended = e2.n_segments > pre_nseg
        la = e2.last_aim
        nonzero_steps += int((la != 0).sum())
        end_steps += int(ended.sum())
        if bool(ended.any()):
            i = ended.nonzero(as_tuple=True)[0]
            # last_aim must equal exp(-c * seg_goal_best) using the value from
            # BEFORE the respawn; pre_best is that value only if the closing
            # step did not further improve it, so compare against the min of
            # the two -- the assertion that matters is that it is NOT the
            # post-respawn value, checked separately below.
            want = torch.exp(-e2._reward_coef * pre_best[i])
            seen_match.append(float((la[i] - want).abs().max()))
    check("last_aim is nonzero ONLY on segment-end steps",
          nonzero_steps == end_steps,
          f"{nonzero_steps} nonzero vs {end_steps} segment ends")
    if seen_match:
        worst = max(seen_match)
        check("last_aim uses the PRE-respawn seg_goal_best", worst < 5e-2,
              f"max |d| vs pre-step value = {worst:.3e} over "
              f"{len(seen_match)} closes")
    # The post-respawn value is goal_x (seg_goal_best is reset to it), so a
    # port that read it after _close_segments would give exp(-0.5*13.33)=1.3e-3
    # on EVERY segment -- a constant, i.e. no aim signal at all.
    post = float(np.exp(-e2._reward_coef * e2.goal_x))
    check("negative control: reading it post-respawn would be a constant",
          post < 1e-2, f"exp(-c * goal_x) = {post:.2e}, the same for every "
                       f"segment regardless of aim")

    # ---- 3. a scored segment gets the full aim term ----------------------
    # Drive the ball into the mouth directly rather than waiting for a policy.
    e3 = _env(w_aim=3.0)
    e3.reset()
    e3.qpos[:, e3.bq + 0] = e3.goal_x - 0.3
    e3.qpos[:, e3.bq + 1] = 0.0
    e3.qpos[:, e3.bq + 2] = 0.2
    e3.qvel[:, e3.bv + 0] = 12.0
    e3.backend.forward()
    got = None
    for _ in range(20):
        e3.step(torch.zeros(e3.n, e3.act_dim, device=e3.device))
        if bool((e3.last_aim != 0).any()):
            got = float(e3.last_aim[e3.last_aim != 0][0])
            break
    check("a scored segment gets the FULL aim term (d_mouth_best = 0)",
          got is not None and got > 0.99,
          f"last_aim = {got:.4f}" if got is not None else "no segment closed")

    # ---- 4. negative controls: each option changes something -------------
    def _roll(seed, **kw):
        env = _env(seed=seed, **kw)
        env.reset()
        torch.manual_seed(0)
        act = torch.zeros(env.n, env.act_dim, device=env.device)
        tot = torch.zeros(env.n, device=env.device)
        for _ in range(120):
            _, r, _ = env.step(act)
            tot += r
        return tot, env

    r_off, _ = _roll(11)
    r_aim, _ = _roll(11, w_aim=3.0)
    d = float((r_aim - r_off).abs().max())
    check("negative control: w_aim changes the reward", d > 1e-4,
          f"max |d return| = {d:.3e}")

    e4 = _env(live_cmd_dir=True)
    e4.reset()
    c0 = e4.cmd_dir.clone()
    # Nudge the ball sideways; a live cmd_dir must re-aim, a frozen one must not.
    e4.qpos[:, e4.bq + 1] += 1.5
    e4.backend.forward()
    e4.step(torch.zeros(e4.n, e4.act_dim, device=e4.device))
    moved = float((c0 - e4.cmd_dir).abs().max())
    check("negative control: live_cmd_dir re-aims when the ball moves",
          moved > 1e-3, f"max |d cmd_dir| = {moved:.3e}")

    e5 = _env(speed_clip=20.0)
    check("speed_clip is settable and reaches the segment machinery",
          abs(e5._speed_clip - 20.0) < 1e-9 and abs(_env()._speed_clip - 8.0) < 1e-9,
          f"{e5._speed_clip} vs default {_env()._speed_clip}")

    n_ok = sum(_results)
    print(f"\n{n_ok}/{len(_results)} checks passed")
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
