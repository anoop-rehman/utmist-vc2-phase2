"""The mirror, checked against the simulator instead of against an argument.

This module is why `augment.py` is allowed to exist. A wrong actuator
permutation or a wrong observation sign produces training data that looks
perfect in every histogram and teaches the policy a body it does not have, so
the transforms are verified where they cannot lie:

  * `test_mirrored_rollout_matches` — mirror a contact-rich 4-ant state AND the
    applied actions, step MuJoCo, and require the result to be the mirror of the
    unmirrored rollout to machine precision.
  * `test_game_obs_mirror_matches_env` — mirror the state and compare
    `mirror_game_obs` against the observation dm_soccer computes for the
    mirrored state, key by key, for all four players.

Run with:

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python \
        -m rower_soccer.bc.tests.run_tests --slow
"""

import os

import numpy as np

from rower_soccer.bc import augment as A
from rower_soccer.bc import dataset as D
from rower_soccer.game import recording as rec

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ANT_XML = os.path.join(REPO, "creature_configs", "ant.xml")

_ENV = None


def _env():
    """One 2v2 ant env for the whole module (building it costs ~15 s)."""
    global _ENV
    if _ENV is None:
        from rower_soccer.skills.soccer import make_skill_soccer_env
        env = make_skill_soccer_env(home=("ant", "ant"), away=("ant", "ant"),
                                    time_limit=45.0,
                                    random_state=np.random.RandomState(0))
        env.reset()
        _ENV = env
    return _ENV


def _stir(env, n=25, seed=3):
    """Reset, then drive random actions until the ants are in real contact.

    Resets first so the probe state does not depend on which test ran before,
    and re-reads `env.physics` afterwards: `RandomizedPitch` recompiles the
    model on reset, which invalidates any handle taken earlier.
    """
    env.reset()
    rng = np.random.RandomState(seed)
    for _ in range(n):
        env.step([rng.uniform(-1, 1, 8) for _ in range(4)])
    ph = env.physics
    return (ph, np.array(ph.data.qpos), np.array(ph.data.qvel),
            np.array(ph.data.ctrl))


def _flat64(obs, keys):
    """`recording.flatten_obs` without its float32 cast.

    The demo stores float32 and that is fine for training, but a mirror check at
    1e-15 must not be measuring `np.float32(34.666668)`.
    """
    return np.concatenate([np.asarray(obs[k], np.float64).ravel() for k in keys])


def _observation(env):
    """The per-player observation dm_soccer would report for the CURRENT state.

    `Updater.update()` is a no-op when `physics.time()` has not advanced, which
    it has not when the state was written directly — so the buffers are reset
    instead, which fills them from the live physics. (Getting this wrong makes
    every key look wrong at once; that is what it looked like the first time.)
    """
    env._observation_updater.reset(physics=env.physics,
                                   random_state=env.random_state)
    return env._observation_updater.get_observation()


# --- the inertial reflection constant ---------------------------------------

def test_inertial_reflection_matches_the_compiled_model():
    """`INERTIAL_REFLECTION['ant']` is a compiler output, so re-derive it."""
    N = A.derive_inertial_reflection(ANT_XML)
    np.testing.assert_allclose(N, A.INERTIAL_REFLECTION["ant"], atol=1e-12)
    # and the same body inside the soccer scene must agree
    import mujoco
    ph = _env().physics
    bid = mujoco.mj_name2id(ph.model.ptr, mujoco.mjtObj.mjOBJ_BODY, "creature/seg0")
    Ri = np.zeros(9)
    mujoco.mju_quat2Mat(Ri, ph.model.body_iquat[bid])
    Ri = Ri.reshape(3, 3)
    np.testing.assert_allclose(Ri.T @ np.diag(A.M3) @ Ri,
                               A.INERTIAL_REFLECTION["ant"], atol=1e-12)


# --- the actuator map, against the integrator -------------------------------

def test_mirrored_rollout_matches():
    """Mirror state + action, step, and compare with the mirrored rollout.

    Ten control ticks of ten 2.5 ms substeps each, i.e. exactly what the game
    integrates between two recorded rows, on four ants in mutual contact with
    the ball. If the actuator permutation or any sign were wrong the two
    trajectories would separate immediately and visibly.
    """
    env = _env()
    bm = A.body_mirror("ant")
    ph, q0, v0, _ = _stir(env)
    model = ph.model.ptr

    def rollout(q, v, ctrl, n_ticks=10, n_sub=10):
        ph.data.qpos[:] = q
        ph.data.qvel[:] = v
        # The constraint solver warm-starts from the previous call's qacc. Left
        # alone, the second rollout would start from the FIRST rollout's
        # leftovers and the two would differ by solver noise that grows; zeroing
        # it makes the two runs start from identical, and mirror-symmetric,
        # conditions. (Found by watching tick 0 diverge at 1e-8.)
        ph.data.qacc_warmstart[:] = 0
        ph.forward()
        traj = []
        for _ in range(n_ticks):
            ph.data.ctrl[:] = ctrl
            for _ in range(n_sub):
                ph.step()
            traj.append((np.array(ph.data.qpos), np.array(ph.data.qvel)))
        return traj

    rng = np.random.RandomState(11)
    acts = rng.uniform(-1, 1, (4, 8))
    ctrl = acts.reshape(-1)
    ctrl_m = np.concatenate([A.mirror_action(acts[i], bm) for i in range(4)])

    plain = rollout(q0, v0, ctrl)
    qm, vm = A.mirror_mj_state(model, q0, v0, bm)
    mirrored = rollout(qm, vm, ctrl_m)

    moved = float(np.abs(plain[-1][0] - q0).max())
    assert moved > 0.1, "the probe state did not move; the test proves nothing"
    for k, ((q, v), (q2, v2)) in enumerate(zip(plain, mirrored)):
        eq, ev = A.mirror_mj_state(model, q, v, bm)
        assert np.abs(eq - q2).max() < 1e-9, f"qpos diverged at tick {k}"
        assert np.abs(ev - v2).max() < 1e-8, f"qvel diverged at tick {k}"


def test_mirror_mj_state_is_an_involution():
    env = _env()
    bm = A.body_mirror("ant")
    ph, q0, v0, _ = _stir(env, n=5, seed=7)
    model = ph.model.ptr
    q1, v1 = A.mirror_mj_state(model, *A.mirror_mj_state(model, q0, v0, bm), bm)
    np.testing.assert_allclose(q1, q0, atol=1e-15)
    np.testing.assert_allclose(v1, v0, atol=1e-15)


# --- the game observation, against dm_soccer --------------------------------

def _arena_landmarks(env):
    """Ground-truth world xy of each landmark key, per team, from the arena.

    dm_soccer's `CoreObservablesAdder._add_player_arena_observables` walks a
    clockwise list of eight corners and ROTATES it by four for the away team,
    which is what this table spells out.
    """
    ar = env.task.arena
    home, away, field = ar.home_goal, ar.away_goal, ar.field
    return {
        "home": {
            "team_goal_back_right": np.array(home.lower[:2]),
            "team_goal_front_left": np.array(home.upper[:2]),
            "field_front_left": np.array(field.upper),
            "field_back_right": np.array(field.lower),
            "opponent_goal_back_left": np.array(away.upper[:2]),
            "opponent_goal_front_right": np.array(away.lower[:2]),
        },
        "away": {
            "team_goal_back_right": np.array(away.upper[:2]),
            "team_goal_front_left": np.array(away.lower[:2]),
            "field_front_left": np.array(field.lower),
            "field_back_right": np.array(field.upper),
            "opponent_goal_back_left": np.array(home.lower[:2]),
            "opponent_goal_front_right": np.array(home.upper[:2]),
        },
    }


def test_game_obs_mirror_matches_env():
    env = _env()
    bm = A.body_mirror("ant")
    ph, q0, v0, c0 = _stir(env, n=25, seed=3)
    model = ph.model.ptr

    ph.data.qpos[:] = q0
    ph.data.qvel[:] = v0
    ph.data.ctrl[:] = c0
    ph.data.qacc_warmstart[:] = 0
    ph.forward()
    plain = _observation(env)
    keys, sizes = rec.obs_layout(plain[0])

    # ...and the same physics, mirrored. The CONTROL has to be mirrored too:
    # the accelerometer and the touch sensors are functions of qacc, which is a
    # function of the applied torque, so leaving `ctrl` alone makes exactly two
    # keys look wrong (which is how this line came to be here).
    qm, vm = A.mirror_mj_state(model, q0, v0, bm)
    ph.data.qpos[:] = qm
    ph.data.qvel[:] = vm
    ph.data.ctrl[:] = np.concatenate(
        [A.mirror_action(c0[8 * i:8 * i + 8], bm) for i in range(4)])
    for i, p in enumerate(env.task.players):
        p.walker._prev_action[:] = A.mirror_action(
            np.array(plain[i]["prev_action"]).ravel(), bm)
    ph.data.qacc_warmstart[:] = 0
    ph.forward()
    mirrored = _observation(env)

    lm = _arena_landmarks(env)
    teams = ["home", "home", "away", "away"]
    worst = {}
    for p in range(4):
        vec = _flat64(plain[p], keys)
        want = _flat64(mirrored[p], keys)
        got = A.mirror_game_obs(vec, keys, sizes, bm, lm[teams[p]])
        off = D.key_offsets(keys, sizes)
        for k, sl in off.items():
            e = float(np.abs(got[sl] - want[sl]).max())
            worst[k] = max(worst.get(k, 0.0), e)
    bad = {k: v for k, v in worst.items() if v > 1e-11}
    assert not bad, f"keys whose mirror disagrees with dm_soccer: {bad}"
    assert len(worst) == len(keys) == 47


def test_landmark_recovery_matches_the_arena():
    """`dataset.recover_landmarks` reads the pitch off the demos; prove it."""
    env = _env()
    env.reset()
    rng = np.random.RandomState(5)
    rows = {"home": [], "away": []}
    teams = ["home", "home", "away", "away"]
    for _ in range(30):
        ts = env.step([rng.uniform(-1, 1, 8) for _ in range(4)])
        keys, sizes = rec.obs_layout(ts.observation[0])
        for p in range(4):
            rows[teams[p]].append(_flat64(ts.observation[p], keys))
    off = D.key_offsets(keys, sizes)
    truth = _arena_landmarks(env)
    for team, block in rows.items():
        world, resid = D.recover_landmarks(np.stack(block), off)
        for k, W in world.items():
            np.testing.assert_allclose(W, truth[team][k], atol=1e-9,
                                       err_msg=f"{team}/{k}")
            assert resid[k] < 1e-9, (team, k, resid[k])
