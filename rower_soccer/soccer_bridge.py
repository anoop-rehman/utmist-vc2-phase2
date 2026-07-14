"""Run a drill-trained FOLLOW policy inside the soccer env (obs-adapter bridge).

The soccer env (`custom_soccer_env.create_soccer_env`, via
`rower_soccer.envs.build.make_soccer_env`) and the follow drill
(`rower_soccer.drills.follow`) use the **same `Creature` walker**, so a creature's
proprioception is byte-identical in construction across both — the soccer per-player
obs just leaves the keys unprefixed (`joints_pos`) where the drill flattener prefixes
them `creature/joints_pos`. The drill's *task* observables (`target_ego`,
`target_ego_future`) don't exist in soccer, but are recomputable from the root pose
(`absolute_root_pos` + `absolute_root_mat`, both present in the soccer obs) with the
exact egocentric transform used in `drills/follow.py:_to_ego`.

So we can drive the soccer env with an unmodified drill follow policy by reconstructing
its exact input vector each step. Per the intended scope (worm, first milestone):

  * target is HARDCODED (a static world point; a real high-level source comes later),
  * other players / walls / goalposts / the ball are ignored,
  * the only environmental change that matters is the pitch dimensions (40x30),
  * proprioception + kinematic sensors are reused verbatim.

Caveats (out-of-distribution vs training):
  * The soccer Task defaults to physics_timestep=0.005 (5 substeps) vs the drill's
    0.0025 (10 substeps); the control rate is identical (40 Hz). We match the drill's
    0.0025 by default (the dt the policy trained on); --no-match-physics-dt keeps the
    soccer-native 0.005.
  * The follow policy trained on a flat 30x30 floor; the pitch is 40x30 with walls and
    goalposts. follow is egocentric, so arena size barely matters; wall/goal collisions
    are unhandled by design.

Heavy deps (torch/SB3) are imported lazily inside the loader so the adapter stays
importable for unit tests.

See docs/SOCCER_BRIDGE.md for the full field-by-field comparison of the drill model
inputs vs. the soccer observations and exactly what this bridge converts, synthesizes,
and drops.
"""

import argparse

import numpy as np

# Reused verbatim (no policy/env changes): the .pt loader + guard.
from rower_soccer.play_interactive import build_obs, load_latent_policy, skill_layout


def reference_follow_layout(creature_kind):
    """Build a throwaway drill FOLLOW env for `creature_kind` and read its exact obs
    contract: the sorted key list, the proprio base names (soccer's unprefixed keys),
    and the task keys we must synthesize. Self-configuring, so it stays correct for
    rower as well as worm."""
    from rower_soccer.drills.follow import make_follow_env
    env = make_follow_env(random_state=0, creature_kind=creature_kind)
    keys = sorted(env.reset().observation.keys())
    proprio_bases = [k.split("/", 1)[1] for k in keys if k.startswith("creature/")]
    task_keys = [k for k in keys if not k.startswith("creature/")]
    return keys, proprio_bases, task_keys


def _to_ego(root_xy, root_mat, world_xy):
    """Identical to drills/follow.py:_to_ego — project (world_xy - root_xy) onto the
    root body's forward (xmat col 0) and left (xmat col 1) axes."""
    fwd, left = root_mat[:2, 0], root_mat[:2, 1]
    d = np.asarray(world_xy, dtype=np.float64) - root_xy
    return np.array([np.dot(d, fwd), np.dot(d, left)], dtype=np.float32)


def soccer_to_drill_follow_dict(soccer_obs0, target_xy, proprio_bases, task_keys):
    """Reconstruct the drill FOLLOW observation dict from one soccer player's obs.

    Proprio keys are re-prefixed `creature/` and copied verbatim; the task keys
    (`target_ego`, `target_ego_future`) are synthesized from a STATIC target, so both
    collapse to the same value (matching a stopped drill target, `_target_vel = 0`).
    """
    # The soccer env keeps a leading singleton buffer dim (no
    # strip_singleton_obs_buffer_dim), so ravel before reshaping.
    root_xy = np.asarray(soccer_obs0["absolute_root_pos"], dtype=np.float64).ravel()[:2]
    root_mat = np.asarray(soccer_obs0["absolute_root_mat"], dtype=np.float64).ravel().reshape(3, 3)
    tgt_ego = _to_ego(root_xy, root_mat, target_xy)

    out = {f"creature/{b}": np.asarray(soccer_obs0[b], dtype=np.float32) for b in proprio_bases}
    for k in task_keys:
        # target_ego and target_ego_future are identical for a static target.
        out[k] = tgt_ego.copy()
    return out


def drill_follow_obs(soccer_obs0, target_xy, follow_keys, proprio_bases, task_keys):
    """Flat 41-dim (worm) / 77-dim (rower) drill FOLLOW vector in trained sorted order."""
    d = soccer_to_drill_follow_dict(soccer_obs0, target_xy, proprio_bases, task_keys)
    return build_obs(d, follow_keys)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--creature", default="worm")
    p.add_argument("--model", default="runs_v2/follow_drill_model.pt",
                   help=".pt state dict (warp_port.ppo.export_sb3_compatible layout)")
    p.add_argument("--target", type=float, nargs=2, default=(8.0, 4.0),
                   metavar=("X", "Y"), help="hardcoded static world target (m)")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    p.add_argument("--match-physics-dt", action=argparse.BooleanOptionalAction, default=True,
                   help="match the drill physics dt (0.0025) the policy trained on "
                        "(default on; --no-match-physics-dt keeps soccer's native 0.005)")
    p.add_argument("--video", default=None, help="optional top_camera mp4 output path")
    args = p.parse_args()

    from rower_soccer.envs.build import make_soccer_env

    target = np.array(args.target, dtype=np.float64)
    follow_keys, proprio_bases, task_keys = reference_follow_layout(args.creature)

    # Single creature on the pitch; no away team -> no teammate_/opponent_ keys.
    env = make_soccer_env(home_team=(args.creature,), n_away=0, time_limit=1e6)
    if args.match_physics_dt:
        env.task.set_timesteps(control_timestep=0.025, physics_timestep=0.0025)
    ts = env.reset()

    obs0 = ts.observation[0]
    obs_dim = int(drill_follow_obs(obs0, target, follow_keys, proprio_bases, task_keys).shape[0])
    _, prop_idx, task_idx = skill_layout(
        soccer_to_drill_follow_dict(obs0, target, proprio_bases, task_keys), follow_keys)
    act_dim = int(env.action_spec()[0].shape[0])
    print(f"[bridge] creature={args.creature} obs={obs_dim} "
          f"(proprio={len(prop_idx)} task={len(task_idx)}) act={act_dim} target={tuple(target)}",
          flush=True)

    policy = load_latent_policy(args.model, obs_dim, act_dim, prop_idx, task_idx, args.device)

    frames = []
    root_xy0 = np.asarray(obs0["absolute_root_pos"], dtype=np.float64).ravel()[:2]
    d0 = float(np.linalg.norm(root_xy0 - target))
    for _ in range(args.steps):
        vec = drill_follow_obs(ts.observation[0], target, follow_keys, proprio_bases, task_keys)
        action, _ = policy.predict(vec, deterministic=True)
        ts = env.step([action])  # soccer expects a per-player list
        if args.video is not None:
            frames.append(env.physics.render(camera_id="top_camera", width=640, height=480))

    root_xyf = np.asarray(ts.observation[0]["absolute_root_pos"], dtype=np.float64).ravel()[:2]
    df = float(np.linalg.norm(root_xyf - target))
    print(f"[bridge] distance to target: start={d0:.2f}m  end={df:.2f}m  "
          f"({'closer' if df < d0 else 'no closer'})", flush=True)

    if args.video is not None and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=40)
        print(f"[bridge] wrote {args.video} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
