"""WS5 integration smoke tests for the ant sprint.

Plain-python (no pytest in this venv):

    MUJOCO_GL=egl .venv/bin/python -m tests.integration_smoke          # CPU only
    MUJOCO_GL=egl .venv/bin/python -m tests.integration_smoke --warp   # + GPU parity

Covers the checks in docs/ANT_SPRINT_WORKSTREAMS.md's integration checklist that
do not need a trained policy or a browser:

  1. ant is registered in envs/build.CREATURE_XMLS and the file exists
  2. the ant CPU follow drill has the exact 65+4 observation contract, in the
     sorted-key order the warp envs assume
  3. the accelerometer observable carries the warp obs-contract scaling
     (/100, clip +/-50) -- the bug the sim2sim probe caught
  4. a 4-ant 2v2 CPU soccer env builds, resets and steps with finite obs
  5. soccer_bridge can rebuild a drill policy's input vector inside soccer
  6. that env steps at >= realtime (40 Hz control) on this host
  7. (--warp) the warp follow env and the CPU follow drill produce the SAME
     observation vector from the same physical state
"""

import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Expected ant follow observation, in dm_control sorted-key order. This IS the
# contract: warp_port/follow_env.py builds the same 69 numbers in the same order
# and any silent reordering here would feed a trained decoder permuted inputs.
ANT_FOLLOW_LAYOUT = [
    ("creature/bodies_pos", 27),
    ("creature/body_height", 1),
    ("creature/joints_pos", 8),
    ("creature/joints_vel", 8),
    ("creature/sensors_accelerometer", 3),
    ("creature/sensors_gyro", 3),
    ("creature/sensors_velocimeter", 3),
    ("creature/touch_sensors", 9),
    ("creature/world_zaxis", 3),
    ("target_ego", 2),
    ("target_ego_future", 2),
]
ANT_PROPRIO_DIM = 65
ANT_OBS_DIM = 69
ANT_ACT_DIM = 8

_results = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        ok, err = True, ""
    except Exception as e:                                          # noqa: BLE001
        import traceback
        ok, detail, err = False, "", traceback.format_exc()
    dt = time.perf_counter() - t0
    _results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt:.1f}s) {detail}", flush=True)
    if err:
        print(err, flush=True)


# ---------------------------------------------------------------------------
def t_ant_registered():
    from rower_soccer.envs.build import CREATURE_XMLS
    assert "ant" in CREATURE_XMLS, sorted(CREATURE_XMLS)
    path = CREATURE_XMLS["ant"]
    assert os.path.exists(path), path
    return path


def t_follow_obs_contract():
    from rower_soccer.drills.follow import make_follow_env
    env = make_follow_env(random_state=0, creature_kind="ant")
    ts = env.reset()
    got = [(k, int(np.asarray(ts.observation[k]).size))
           for k in sorted(ts.observation.keys())]
    assert got == ANT_FOLLOW_LAYOUT, f"\n got {got}\nwant {ANT_FOLLOW_LAYOUT}"
    total = sum(n for _, n in got)
    proprio = sum(n for k, n in got if "/" in k)
    assert (total, proprio) == (ANT_OBS_DIM, ANT_PROPRIO_DIM), (total, proprio)
    spec = env.action_spec()
    assert spec.shape == (ANT_ACT_DIM,), spec.shape
    return f"obs={total} proprio={proprio} act={spec.shape[0]}"


def t_accel_scaling():
    """creature.py must apply the warp obs contract's /100 + clip to the
    accelerometer. Without it the SAME policy scores 0.41 on CPU vs 0.92 in warp
    (WS5 sim2sim probe, 2026-08-08) -- an obs bug that reads as a physics gap."""
    from rower_soccer.drills.follow import make_follow_env
    env = make_follow_env(random_state=0, creature_kind="ant")
    env.reset()
    walker = env.task._walker
    physics = env.physics
    sensors = walker.mjcf_model.sensor.accelerometer

    peak_raw = 0.0
    rng = np.random.RandomState(0)
    for _ in range(120):
        ts = env.step(np.clip(rng.randn(ANT_ACT_DIM), -1, 1))
        raw = np.reshape(physics.bind(sensors).sensordata, -1)
        obs = np.asarray(ts.observation["creature/sensors_accelerometer"]).ravel()
        peak_raw = max(peak_raw, float(np.abs(raw).max()))
        assert np.allclose(obs, np.clip(raw / 100.0, -50.0, 50.0), atol=1e-6), \
            f"accelerometer not scaled: obs={obs} raw={raw}"
    assert peak_raw > 10.0, ("accelerometer never left the noise floor, so this "
                            f"test proved nothing (peak {peak_raw})")
    return f"peak raw |a| = {peak_raw:.0f} m/s^2 -> obs bounded by 50"


def t_soccer_2v2_builds():
    from rower_soccer.envs.build import make_soccer_env
    env = make_soccer_env(home_team=("ant", "ant"), away_team=("ant", "ant"),
                          time_limit=45.0, random_state=0)
    ts = env.reset()
    specs = env.action_spec()
    assert len(specs) == 4, len(specs)
    for s in specs:
        assert s.shape == (ANT_ACT_DIM,), s.shape
    rng = np.random.RandomState(0)
    for _ in range(40):
        a = [np.clip(0.3 * rng.randn(*s.shape), s.minimum, s.maximum) for s in specs]
        ts = env.step(a)
        for p, obs in enumerate(ts.observation):
            for k, v in obs.items():
                v = np.asarray(v, dtype=np.float64)
                assert np.isfinite(v).all(), f"player {p} obs[{k}] not finite"
    keys = sorted(ts.observation[0].keys())
    return f"4 players, {len(keys)} obs keys/player"


def t_soccer_bridge_obs():
    """A drill policy must be drivable inside the soccer env: the bridge has to
    rebuild the exact 69-vector from one player's soccer observation. This needs
    `absolute_root_pos` / `absolute_root_mat` to be enabled on the soccer
    walkers (envs/build.make_creature(expose_root_pose=True)) -- without them
    the bridge raised KeyError and nothing could be driven in soccer at all."""
    from rower_soccer import soccer_bridge as SB
    from rower_soccer.envs.build import make_soccer_env
    keys, prop_bases, task_keys = SB.reference_follow_layout("ant")
    assert keys == [k for k, _ in ANT_FOLLOW_LAYOUT], keys
    env = make_soccer_env(home_team=("ant", "ant"), away_team=("ant", "ant"),
                          time_limit=1e6, random_state=0)
    ts = env.reset()
    for p in range(4):
        vec = SB.drill_follow_obs(ts.observation[p], np.array([3.0, 1.0]),
                                  keys, prop_bases, task_keys)
        assert vec.shape == (ANT_OBS_DIM,), vec.shape
        assert np.isfinite(vec).all()
    return f"bridge rebuilt {ANT_OBS_DIM}-dim follow obs for all 4 players"


def t_soccer_2v2_realtime():
    from rower_soccer.envs.build import make_soccer_env
    env = make_soccer_env(home_team=("ant", "ant"), away_team=("ant", "ant"),
                          time_limit=1e6, random_state=0)
    # Match the drill physics dt the policies trained at (0.0025 -> 10 substeps);
    # the soccer default 0.005 is cheaper, so this is the pessimistic case.
    env.task.set_timesteps(control_timestep=0.025, physics_timestep=0.0025)
    env.reset()
    specs = env.action_spec()
    rng = np.random.RandomState(0)
    acts = [np.clip(0.3 * rng.randn(*s.shape), s.minimum, s.maximum) for s in specs]
    for _ in range(20):
        env.step(acts)
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        env.step(acts)
    hz = n / (time.perf_counter() - t0)
    assert hz >= 40.0, f"2v2 ant soccer runs at {hz:.1f} Hz, below the 40 Hz control rate"
    return f"{hz:.0f} Hz control = {hz / 40:.2f}x realtime (need 1.0x)"


def t_warp_cpu_obs_parity():
    """The warp follow env and the CPU follow drill must emit the SAME vector
    for the same physical state. Any mismatch is an obs bug masquerading as a
    sim2sim physics gap."""
    import mujoco
    import torch
    from dm_control import composer
    from rower_soccer.drills.follow import FollowTask
    from rower_soccer.warp_port.follow_env import WarpFollowEnv

    wenv = WarpFollowEnv(num_worlds=1, use_graph=False, seed=0,
                         creature_xml="creature_configs/ant.xml",
                         target_speed_range=(0.07, 0.6),
                         spawn_dist_range=(1.07, 3.22))
    assert wenv.obs_dim == ANT_OBS_DIM and wenv.act_dim == ANT_ACT_DIM
    wobs = wenv.reset()
    # advance a little so the state is not the trivial spawn pose
    for _ in range(20):
        wobs, _, _ = wenv.step(torch.zeros(1, wenv.act_dim, device="cuda"))
    meta = wenv.meta
    qpos = wenv.qpos[0].cpu().numpy()
    qvel = wenv.qvel[0].cpu().numpy()

    class Pinned(FollowTask):
        def initialize_episode(self, physics, random_state):
            super().initialize_episode(physics, random_state)
            w = self._walker
            # `set_pose` positions the ATTACHMENT FRAME, not the root body.
            # dm_control's add_free_entity puts the freejoint on a wrapper body
            # and hangs seg0 off it at the XML's `pos` (0.75 m up for the ant);
            # warp's build_creature_scene puts the freejoint on seg0 itself, so
            # warp's qpos_root IS seg0's world pose. Feeding warp's qpos
            # straight to set_pose spawns the CPU creature 0.75 m in the air.
            # Anyone placing a creature in the soccer env hits this.
            w.set_pose(physics, position=np.zeros(3),
                       quaternion=np.array([1.0, 0.0, 0.0, 0.0]))
            physics.forward()
            off = np.array(physics.bind(w.root_body).xpos, dtype=np.float64)
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(
                rot, np.asarray(qpos[meta.qpos_root + 3:meta.qpos_root + 7],
                                dtype=np.float64))
            w.set_pose(physics,
                       position=(qpos[meta.qpos_root:meta.qpos_root + 3]
                                 - rot.reshape(3, 3) @ off),
                       quaternion=qpos[meta.qpos_root + 3:meta.qpos_root + 7])
            w.set_velocity(physics, velocity=qvel[meta.qvel_root:meta.qvel_root + 3],
                           angular_velocity=qvel[meta.qvel_root + 3:meta.qvel_root + 6])
            physics.bind(w.observable_joints).qpos = qpos[meta.joint_qpos]
            physics.bind(w.observable_joints).qvel = qvel[meta.joint_qvel]
            self._target_xy = wenv.target_xy[0].cpu().numpy().astype(np.float64)
            self._target_vel = wenv.target_vel[0].cpu().numpy().astype(np.float64)
            self._target.set_pose_xy(physics, self._target_xy, self._target_height)
            physics.forward()

    cenv = composer.Environment(task=Pinned(creature_kind="ant"),
                                time_limit=1e6, random_state=0,
                                strip_singleton_obs_buffer_dim=True)
    ts = cenv.reset()
    cvec = np.concatenate([np.asarray(ts.observation[k], np.float64).ravel()
                           for k, _ in ANT_FOLLOW_LAYOUT])
    wvec = wobs[0].cpu().numpy().astype(np.float64)
    # Blocks that are pure kinematics agree to float precision; the sensor
    # blocks (velocimeter/gyro/accel/touch) are solver outputs of two different
    # backends at the same state and only agree loosely.
    worst = ""
    off = 0
    tols = {"creature/sensors_accelerometer": 5e-1, "creature/touch_sensors": 5e-1,
            "creature/sensors_gyro": 5e-2, "creature/sensors_velocimeter": 5e-2}
    for k, n in ANT_FOLLOW_LAYOUT:
        d = float(np.abs(wvec[off:off + n] - cvec[off:off + n]).max())
        tol = tols.get(k, 1e-4)
        assert d <= tol, f"{k}: max|warp-cpu| = {d:.4g} > {tol}"
        worst += f"{k.split('/')[-1]}={d:.2g} "
        off += n
    return worst.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warp", action="store_true",
                   help="also run the GPU warp<->CPU observation parity check")
    args = p.parse_args()

    check("ant registered in CREATURE_XMLS", t_ant_registered)
    check("ant follow drill obs contract (69 = 65 proprio + 4 task)", t_follow_obs_contract)
    check("accelerometer carries the warp /100 clip scaling", t_accel_scaling)
    check("4-ant 2v2 CPU soccer env builds and steps", t_soccer_2v2_builds)
    check("soccer_bridge rebuilds the follow obs inside soccer", t_soccer_bridge_obs)
    check("4-ant 2v2 CPU soccer env >= realtime", t_soccer_2v2_realtime)
    if args.warp:
        check("warp <-> CPU observation parity (ant)", t_warp_cpu_obs_parity)

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
