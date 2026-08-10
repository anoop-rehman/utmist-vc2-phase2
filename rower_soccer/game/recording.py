"""Demo recording -- schema v1. THIS IS THE BC DATASET, not a replay gimmick.

A demo is one `.npz` (``np.savez_compressed``) holding a fixed-length record per
control tick per player plus a sparse event log.  Numpy-only: no extra deps, memory
-mappable, and `obs[t, p]` is already the tensor a BC dataloader wants.

Design rules (deliberate, do not "simplify" away):

* **Record what actually ran, never what was requested.**  If a human presses
  "shoot" and the shoot expert is not loaded, the tick records the skill that was
  actually executed.  A BC dataset that lies about its labels is worse than a
  smaller one.
* **Record the *game* obs, not the per-skill obs.**  Each skill's policy input is a
  deterministic function of (game obs, target) -- exactly what `skills.py` computes
  -- so storing the game obs keeps every skill's input reconstructible from one
  array, and keeps the file rectangular.  `meta["obs_keys"]/["obs_sizes"]` let a
  reader split `obs[t, p]` back into the dm_soccer observation dict.
* **Record full physics state too.**  `qpos`/`qvel` make replay exact by
  construction and make the action-resimulation determinism test possible.
* Nothing in the file references the running server: a demo is self-describing.

::

    meta_json   JSON, see `DemoMeta` below
    tick        int64   [T]        control tick index, 0-based, contiguous
    t           float32 [T]        sim time (s) = tick * control_dt
    obs         float32 [T, P, O]  per-player dm_soccer obs, flat, meta.obs_keys order
    skill       int8    [T, P]     index into meta.skill_vocab (skill that RAN)
    skill_req   int8    [T, P]     index into meta.skill_vocab (skill REQUESTED)
    target      float32 [T, P, 2]  commanded world target xy (m)
    aim         float32 [T, P, 2]  commanded aim direction (unit, or 0 if none)
    z           float32 [T, P, Z]  latent emitted by the expert (NaN when no policy ran)
    action      float32 [T, P, A]  actuator command actually applied, in [-1, 1]
    skill_obs   float32 [T, P, Om] the exact vector fed to the expert, NaN-padded to
                                   the widest skill's obs_dim
    skill_obs_n int16   [T, P]     how much of `skill_obs` is real
    ctrl_tick   int32   [T, P]     the SkillController's own tick counter. NOT the
                                   match tick: it resets on every skill switch, and
                                   with `skill_seed` + the player index it is what
                                   makes a MODE_NOISE action reproducible (see
                                   game/skills.py)
    player_pos  float32 [T, P, 3]  root world position (NOT in obs: obs is egocentric)
    player_mat  float32 [T, P, 9]  root world rotation, row-major.  `player_pos`+
                                   `player_mat` are exactly what a `skills.PlayerFrame`
                                   needs on top of `obs`, so a demo can be re-run
                                   through a SkillController without a simulator.
    ball_pos    float32 [T, 3]
    ball_vel    float32 [T, 3]
    score       int16   [T, 2]     cumulative (home, away)
    qpos        float64 [T, Q]     full MuJoCo qpos   (replay ground truth)
    qvel        float64 [T, V]     full MuJoCo qvel   (omitted if store_qvel=False)
    events_json JSON, list of dicts; every event carries {"tick", "t", "type"}

Event types (`EVENT_TYPES`): match_start, match_end, goal, ball_touch,
skill_change, target_set, slot_claim, slot_release.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np

SCHEMA_NAME = "rower_soccer.demo"
SCHEMA_VERSION = 1

#: Fixed vocabulary; these are `rower_soccer.skills`' skill_ids (WS3 owns the names).
#: APPEND ONLY -- the indices are baked into every recorded file.
#: "idle" = zero torque, "scripted" = the follow expert retargeted at the ball.
SKILL_VOCAB = ("idle", "follow", "dribble", "kick", "shoot", "scripted")
SKILL_INDEX = {s: i for i, s in enumerate(SKILL_VOCAB)}

EVENT_TYPES = (
    "match_start", "match_end", "goal", "ball_touch",
    "skill_change", "target_set", "slot_claim", "slot_release",
    # A state write, not an input: MatchSim.unflip stood this player upright.
    # replay_actions re-applies these at their recorded ticks, or the
    # resimulation would diverge from the recorded qpos the moment anyone flips.
    "unflip",
)

DEMO_SUFFIX = ".demo.npz"


@dataclass
class PlayerMeta:
    """One of the four seats. `index` is the dm_soccer per-player list index."""
    index: int
    slot: str            # "home_1" | "home_2" | "away_1" | "away_2"
    team: str            # "home" | "away"
    creature: str        # "ant" | "rower" | "worm"
    controller: str      # "human" | "scripted" | "idle"
    display_name: str = ""
    act_dim: int = 0


@dataclass
class DemoMeta:
    """Everything a consumer needs to interpret the arrays without the server."""
    schema: str = SCHEMA_NAME
    version: int = SCHEMA_VERSION
    match_id: str = ""
    created_utc: str = ""
    git_sha: str = ""
    # --- sim reproduction -------------------------------------------------
    seed: int = 0
    control_dt: float = 0.025
    physics_dt: float = 0.0025
    time_limit: float = 45.0
    pitch_half: tuple = (15.0, 11.0)      # RandomizedPitch min==max size (half-extents, m)
    terminate_on_goal: bool = False
    #: `env.random_state.get_state()` captured immediately after the match's reset,
    #: as [name, keys, pos, has_gauss, cached_gaussian].
    #:
    #: The seed alone is NOT enough. dm_soccer's `Task.before_step` calls
    #: `_throw_in`, which draws from this RNG whenever the ball leaves the field,
    #: and `MultiturnTask.after_step` re-spawns everyone from it on a goal. The
    #: server steps the sim during the lobby too, so by kickoff the stream sits at
    #: an offset that depends on what happened while people were still joining.
    #: Restoring the state makes a replay exact no matter what came before.
    rng_state: list = field(default_factory=list)
    # --- layout -----------------------------------------------------------
    n_players: int = 4
    players: list = field(default_factory=list)     # list[PlayerMeta]
    obs_keys: list = field(default_factory=list)    # sorted dm_soccer obs keys
    obs_sizes: list = field(default_factory=list)   # flat width of each key
    skill_vocab: list = field(default_factory=lambda: list(SKILL_VOCAB))
    available_skills: list = field(default_factory=list)  # skills with a real policy
    z_dim: int = 16
    act_dim: int = 8
    # --- the obs contract the policies were trained on --------------------
    # warp_port/follow_env.py scales the accelerometer /100 and clips to +/-50 and
    # says "any future body must apply the same scaling at deployment -- it is part
    # of the obs contract". The CPU drill env does NOT, so the game applies it in
    # skills.py. Recorded so a BC consumer knows whether raw or scaled accel is in
    # `obs` (it is RAW there; the scaling happens inside the skill adapter).
    accel_scale: float = 100.0
    accel_clip: float = 50.0
    skill_obs: dict = field(default_factory=dict)   # skill -> {"fields":[...], "obs_dim":n}
    checkpoints: dict = field(default_factory=dict)  # skill -> {"path", "sha256", "bytes"}
    # --- how the expert turned an obs into an action ----------------------
    # `follow_ant_v1` trained with ent_ceil=0: action std ~1.0 = the whole action
    # range, so PPO scored the SAMPLED policy and the distribution mean does not
    # locomote. WS3's MODE_AUTO therefore runs it in "noise" mode, where the noise
    # is a pure function of (skill_seed, player index, ctrl_tick) -- all three are
    # recorded, so the run is still bit-exact reproducible.
    skill_backend: str = ""                         # "ws3"
    action_mode: str = "auto"                       # requested: auto | mean | noise
    resolved_modes: dict = field(default_factory=dict)   # skill -> mean | noise
    skill_seed: int = 0
    # --- rendering / input affine ----------------------------------------
    camera: dict = field(default_factory=dict)      # cam_height, half_x, half_y, px_w, px_h
    store_qvel: bool = True
    notes: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        d["players"] = [asdict(p) if isinstance(p, PlayerMeta) else dict(p)
                        for p in self.players]
        d["pitch_half"] = list(self.pitch_half)
        return json.dumps(d, indent=1, sort_keys=True, default=_json_default)

    @staticmethod
    def from_json(s: str) -> "DemoMeta":
        d = json.loads(s)
        players = [PlayerMeta(**p) for p in d.pop("players", [])]
        d.pop("schema", None)
        m = DemoMeta(**{k: v for k, v in d.items() if k in DemoMeta.__annotations__})
        m.players = players
        m.pitch_half = tuple(m.pitch_half)
        return m


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def flatten_obs(obs: dict, keys) -> np.ndarray:
    """dm_soccer per-player obs dict -> flat float32 vector in `keys` order.

    The soccer env keeps a leading singleton buffer dim (it does not pass
    `strip_singleton_obs_buffer_dim`), hence the ravel.
    """
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in keys])


def obs_layout(obs: dict):
    """(sorted keys, per-key flat sizes) for one player's dm_soccer obs dict."""
    keys = sorted(obs.keys())
    sizes = [int(np.asarray(obs[k]).size) for k in keys]
    return keys, sizes


def split_obs(vec: np.ndarray, keys, sizes) -> dict:
    """Inverse of `flatten_obs`: flat vector (or [T, O] batch) -> dict of arrays."""
    out, i = {}, 0
    for k, n in zip(keys, sizes):
        out[k] = vec[..., i:i + n]
        i += n
    return out


def sha256_file(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class DemoWriter:
    """Buffers ticks in RAM and writes one npz at `close()`.

    A 45 s 2v2 match is ~1800 ticks x ~2 kB = a few MB, so buffering is simpler and
    faster than streaming, and it keeps the file atomic: a crashed match leaves no
    half-written demo (call `close()` from a finally: block to keep what you have).
    """

    def __init__(self, path, meta: DemoMeta):
        self.path = str(path)
        if not self.path.endswith(".npz"):
            self.path += DEMO_SUFFIX
        self.meta = meta
        self._cols: dict[str, list] = {}
        self.events: list[dict] = []
        self._n = 0
        self.closed = False

    # -- writing -----------------------------------------------------------
    def record_tick(self, **cols):
        """One control tick. Keys must match the schema's array names (minus meta)."""
        for k, v in cols.items():
            self._cols.setdefault(k, []).append(np.asarray(v))
        self._n += 1

    def add_event(self, type_: str, tick: int, t: float, **payload):
        if type_ not in EVENT_TYPES:
            raise ValueError(f"unknown event type {type_!r}; extend EVENT_TYPES")
        ev = {"tick": int(tick), "t": float(t), "type": type_}
        ev.update(payload)
        self.events.append(ev)
        return ev

    @property
    def n_ticks(self):
        return self._n

    def close(self):
        if self.closed:
            return self.path
        self.closed = True
        arrays = {k: np.stack(v) if v else np.zeros((0,), np.float32)
                  for k, v in self._cols.items()}
        # int64 tick / int8 skills / int16 score: keep the dtypes the schema promises
        # rather than whatever np.stack inferred from the python ints.
        for k, dt in (("tick", np.int64), ("skill", np.int8), ("skill_req", np.int8),
                      ("skill_obs_n", np.int16), ("ctrl_tick", np.int32),
                      ("score", np.int16)):
            if k in arrays:
                arrays[k] = arrays[k].astype(dt, copy=False)
        for k, a in list(arrays.items()):
            # qpos/qvel stay float64. They are the replay's initial condition, and a
            # legged creature in contact is chaotic: rounding the start state to
            # float32 (~1e-7 relative) grows to ~0.5 m of divergence inside a second
            # of re-simulation. Everything else is observations and network I/O,
            # where float32 is already the working precision.
            if a.dtype == np.float64 and k not in ("qpos", "qvel"):
                arrays[k] = a.astype(np.float32)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        # A file OBJECT, not a path: np.savez_compressed appends ".npz" to a path
        # that lacks it, which would silently write next to the intended name.
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                meta_json=np.array(self.meta.to_json()),
                events_json=np.array(json.dumps(self.events, default=_json_default)),
                **arrays)
        os.replace(tmp, self.path)
        return self.path


@dataclass
class Demo:
    """A loaded demo. `arrays` holds every recorded array by schema name."""
    meta: DemoMeta
    events: list
    arrays: dict
    path: str = ""

    def __getattr__(self, name):
        # demo.obs / demo.action / demo.qpos ...
        try:
            return self.__dict__["arrays"][name]
        except KeyError:
            raise AttributeError(name) from None

    def __contains__(self, name):
        return name in self.arrays

    @property
    def n_ticks(self):
        return int(self.arrays["tick"].shape[0])

    @property
    def n_players(self):
        return int(self.meta.n_players)

    def skill_names(self, p: int):
        v = self.meta.skill_vocab
        return [v[i] for i in self.arrays["skill"][:, p]]

    def obs_dict(self, p: int) -> dict:
        """[T, O] player-p obs split back into named [T, n] arrays."""
        return split_obs(self.arrays["obs"][:, p, :], self.meta.obs_keys, self.meta.obs_sizes)

    def events_of(self, *types):
        return [e for e in self.events if e["type"] in types]

    def final_score(self):
        s = self.arrays.get("score")
        return (0, 0) if s is None or len(s) == 0 else (int(s[-1, 0]), int(s[-1, 1]))

    def bc_pairs(self, players=None, skills=None):
        """Flatten to supervised pairs for next sprint's BC.

        Returns dict with obs [N, O], target [N, 2], skill [N], z [N, Z],
        action [N, A], player [N]. `skills` filters by name.
        """
        P = self.n_players
        players = range(P) if players is None else players
        keep_idx = None
        if skills is not None:
            keep_idx = {SKILL_INDEX[s] if isinstance(s, str) else int(s) for s in skills}
        o, tg, sk, z, a, pi = [], [], [], [], [], []
        for p in players:
            m = np.ones(self.n_ticks, bool)
            if keep_idx is not None:
                m = np.isin(self.arrays["skill"][:, p], list(keep_idx))
            o.append(self.arrays["obs"][m, p]); tg.append(self.arrays["target"][m, p])
            sk.append(self.arrays["skill"][m, p]); z.append(self.arrays["z"][m, p])
            a.append(self.arrays["action"][m, p])
            pi.append(np.full(int(m.sum()), p, np.int8))
        cat = np.concatenate
        return dict(obs=cat(o), target=cat(tg), skill=cat(sk), z=cat(z),
                    action=cat(a), player=cat(pi))


def read_demo(path) -> Demo:
    path = str(path)
    with np.load(path, allow_pickle=False) as f:
        meta = DemoMeta.from_json(str(f["meta_json"]))
        events = json.loads(str(f["events_json"]))
        arrays = {k: f[k] for k in f.files if k not in ("meta_json", "events_json")}
    if meta.version != SCHEMA_VERSION:
        # Forward-compatible reads are fine for v1 -> v1; anything else is a bug
        # until a migration exists, and a silent misread is the worst outcome.
        raise ValueError(f"{path}: demo schema v{meta.version}, this reader is "
                         f"v{SCHEMA_VERSION}")
    return Demo(meta=meta, events=events, arrays=arrays, path=path)


def summarize(path) -> str:
    d = read_demo(path)
    goals = d.events_of("goal")
    touches = d.events_of("ball_touch")
    per = []
    for p in range(d.n_players):
        pm = d.meta.players[p]
        names = np.array(d.meta.skill_vocab)[d.arrays["skill"][:, p]]
        u, c = np.unique(names, return_counts=True)
        mix = " ".join(f"{n}:{k}" for n, k in zip(u, c))
        per.append(f"    {pm.slot:7s} {pm.controller:8s} {pm.display_name or '-':12s} {mix}")
    hs, as_ = d.final_score()
    return "\n".join([
        f"{os.path.basename(path)}  schema v{d.meta.version}  match {d.meta.match_id}",
        f"  {d.n_ticks} ticks @ {1/d.meta.control_dt:.0f} Hz "
        f"= {d.n_ticks * d.meta.control_dt:.1f} s   seed={d.meta.seed}",
        f"  score home {hs} - {as_} away   goals={len(goals)}  touches={len(touches)}",
        f"  obs {d.arrays['obs'].shape}  z {d.arrays['z'].shape}  "
        f"action {d.arrays['action'].shape}  qpos {d.arrays['qpos'].shape}",
        *per,
    ])


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Inspect a recorded demo file.")
    p.add_argument("demo", nargs="+")
    p.add_argument("--events", action="store_true", help="print the full event log")
    a = p.parse_args(argv)
    for path in a.demo:
        print(summarize(path))
        if a.events:
            for e in read_demo(path).events:
                print("   ", json.dumps(e, default=_json_default))


if __name__ == "__main__":
    main()
