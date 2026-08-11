"""Synthetic demos: the format, in miniature, with no simulator.

The real corpus is 7 MB per match and takes a second to read; these fakes carry
the same schema and the same *structure* (a real egocentric view of real pitch
landmarks, computed from a real root pose) so the loader, the split, the filters
and the landmark recovery are all exercised for free.
"""

import numpy as np

from rower_soccer.game import recording as rec
from rower_soccer.bc.dataset import LANDMARK_KEYS

#: A cut-down game observation: everything `bc` reads by name, nothing else.
OBS_KEYS = ["absolute_root_mat", "absolute_root_pos"] + list(LANDMARK_KEYS)
OBS_SIZES = [9, 3] + [2] * len(LANDMARK_KEYS)

#: The real pitch_half=(15, 11) geometry, read off the recorded corpus
#: (`dataset.recover_landmarks` on demos/20260810-*). The away team sees the
#: same points negated in BOTH axes, which is what dm_soccer's per-team rotation
#: of its clockwise corner list comes out as.
HOME_LANDMARKS = {
    "team_goal_back_right": (-15.0, -3.63),
    "team_goal_front_left": (-9.6666667, 3.63),
    "field_front_left": (9.6666667, 5.6666667),
    "field_back_right": (-9.6666667, -5.6666667),
    "opponent_goal_back_left": (15.0, 3.63),
    "opponent_goal_front_right": (9.6666667, -3.63),
}
AWAY_LANDMARKS = {k: (-v[0], -v[1]) for k, v in HOME_LANDMARKS.items()}

FOLLOW_V3 = ("bodies_pos", "body_height", "joints_pos", "joints_vel",
             "sensors_accelerometer", "sensors_gyro", "sensors_velocimeter",
             "touch_sensors", "world_zaxis", "target_ego3", "target_ego3_future")
FOLLOW_V1 = FOLLOW_V3[:-2] + ("target_ego", "target_ego_future")


def rot(yaw, tilt=0.0):
    cy, sy, ct, st = np.cos(yaw), np.sin(yaw), np.cos(tilt), np.sin(tilt)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Rx = np.array([[1.0, 0, 0], [0, ct, -st], [0, st, ct]])
    return Rz @ Rx


def make_demo(path, *, match_id="m0", n_ticks=24, seed=0, follow_fields=FOLLOW_V3,
              controllers=("human", "scripted", "scripted", "idle"),
              skills=("follow", "scripted", "scripted", "idle"),
              end_tick=None, obs_keys=None, obs_sizes=None, tilt=0.15):
    """Write one synthetic demo and return its path."""
    rng = np.random.default_rng(seed)
    keys = list(obs_keys or OBS_KEYS)
    sizes = list(obs_sizes or OBS_SIZES)
    P = len(controllers)
    teams = ["home", "home", "away", "away"][:P]
    width = 71 if tuple(follow_fields) == FOLLOW_V3 else 69
    meta = rec.DemoMeta(
        match_id=match_id, created_utc="2026-08-10T00:00:00Z", seed=seed,
        n_players=P, obs_keys=keys, obs_sizes=sizes, z_dim=16, act_dim=8,
        pitch_half=(15.0, 11.0),
        players=[rec.PlayerMeta(i, ["home_1", "home_2", "away_1", "away_2"][i],
                                teams[i], "ant", controllers[i], f"p{i}", 8)
                 for i in range(P)],
        available_skills=["follow", "idle", "scripted"],
        skill_obs={
            "idle": dict(fields=[], obs_dim=0, kind="zero", target_source="none"),
            "follow": dict(fields=list(follow_fields), obs_dim=width,
                           kind="latent", target_source="command"),
            "scripted": dict(fields=list(follow_fields), obs_dim=width,
                             kind="latent", target_source="ball"),
        })
    w = rec.DemoWriter(str(path), meta)
    w.add_event("match_start", 0, 0.0, seed=seed)
    total = int(np.sum(sizes))
    for t in range(n_ticks):
        obs = np.zeros((P, total), np.float32)
        ppos = np.zeros((P, 3), np.float32)
        pmat = np.zeros((P, 9), np.float32)
        for p in range(P):
            R = rot(rng.uniform(-np.pi, np.pi), rng.uniform(-tilt, tilt))
            x = np.array([rng.uniform(-9, 9), rng.uniform(-5, 5), 0.75])
            lm = HOME_LANDMARKS if teams[p] == "home" else AWAY_LANDMARKS
            row, i = obs[p], 0
            for k, n in zip(keys, sizes):
                if k == "absolute_root_mat":
                    row[i:i + 9] = R.ravel()
                elif k == "absolute_root_pos":
                    row[i:i + 3] = x
                elif k in lm:
                    row[i:i + 2] = R[:2, :2].T @ (np.array(lm[k]) - x[:2])
                else:
                    row[i:i + n] = rng.normal(size=n)
                i += n
            ppos[p], pmat[p] = x, R.ravel()
        sk = np.array([rec.SKILL_INDEX[s] for s in skills[:P]], np.int8)
        sobs = np.full((P, width), np.nan, np.float32)
        son = np.zeros(P, np.int16)
        for p in range(P):
            if skills[p] != "idle":
                sobs[p] = rng.normal(size=width).astype(np.float32)
                son[p] = width
        w.record_tick(
            tick=t, t=t * 0.025, obs=obs, skill=sk, skill_req=sk,
            target=rng.normal(size=(P, 2)).astype(np.float32),
            aim=np.zeros((P, 2), np.float32),
            z=rng.normal(size=(P, 16)).astype(np.float32),
            skill_obs=sobs, skill_obs_n=son,
            ctrl_tick=np.arange(P, dtype=np.int32) + t,
            action=rng.uniform(-1, 1, size=(P, 8)).astype(np.float32),
            score=np.zeros(2, np.int16), player_pos=ppos, player_mat=pmat,
            ball_pos=rng.normal(size=3).astype(np.float32),
            ball_vel=rng.normal(size=3).astype(np.float32),
            qpos=np.zeros(7 + 8 * P, np.float64),
            qvel=np.zeros(6 + 8 * P, np.float64))
    w.add_event("match_end", n_ticks - 1 if end_tick is None else end_tick,
                0.0, reason="time")
    return w.close()
