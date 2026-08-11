"""Demos -> one consolidated BC dataset.

    from rower_soccer.bc import build_dataset
    ds = build_dataset(sorted(glob.glob("demos/*.demo.npz")))
    ds.save("runs_v2/bc/ant_2v2_v1.npz")

One SAMPLE is one (tick, player) pair of one match. It carries everything a
behaviour-cloning trainer can want from that instant and nothing it has to go
back to the demo file for:

    obs          [O]   the dm_soccer game observation (high-level policy input)
    expert_obs   [Om]  the EXACT vector the drill expert consumed, NaN-padded
    expert_obs_n       how much of `expert_obs` is real
    action       [A]   the actuator command that was applied, in [-1, 1]
    z            [Z]   the latent the expert emitted (NaN when none ran)
    skill              stable int id + the string, via `meta["skill_vocab"]`
    target       [2]   the world xy target actually in force
    provenance         demo file, match id, tick, player slot, team, controller

Design decisions worth knowing about
------------------------------------

**Nothing is thrown away silently.** Every filter increments a named counter in
`meta["dropped"]`, and `stats.py` prints them. A dataset that quietly halved
itself is the kind of thing that costs a week.

**Scripted bots are KEPT and TAGGED.** Three of the four seats in the current
corpus are the scripted chase, and a BC prior trained on chase-the-ball is not
worthless — but it is not human play either. `controller` (human / scripted /
idle) is a per-sample column so a trainer can weight or exclude them; nothing
here decides for it.

**The split is by MATCH, never by tick.** Consecutive ticks are 25 ms apart and
almost identical; splitting them across train/val measures memorisation. The
assignment is a pure function of the match id (sha256), so it is stable across
runs, machines, and the order the demo files happen to be listed in.

**Per-demo observation layouts.** `meta.skill_obs[skill]["fields"]` is recorded
in every demo. Two demos in the current corpus predate the v3 skill contract
(`follow` was 69 wide with a 2-D `target_ego`, now 71 with `target_ego3`), so
the loader keys each sample to a LAYOUT id — a (skill, field tuple, width)
triple — instead of assuming the live registry. Selecting one layout is then a
one-line mask, and mixing two of them by accident is impossible.

**Pitch landmarks.** Six of the game observation's keys are egocentric views of
fixed pitch corners, and the pitch mirror needs their WORLD positions (see
`augment.mirror_game_obs`). They are recovered here, per demo per team, by
least squares over the whole match from the recorded root poses — no arena, no
dm_control. The residual is stored; `augment` refuses to mirror a demo whose
recovery did not converge.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from rower_soccer.game import recording as rec

__all__ = ["BCDataset", "build_dataset", "load_dataset", "split_of_match",
           "key_offsets", "recover_landmarks", "playing_mask",
           "SCHEMA_NAME", "SCHEMA_VERSION", "SPLIT_TRAIN", "SPLIT_VAL",
           "CONTROLLER_VOCAB", "TEAM_VOCAB", "LANDMARK_KEYS"]

SCHEMA_NAME = "rower_soccer.bc.dataset"
SCHEMA_VERSION = 1

SPLIT_TRAIN, SPLIT_VAL = 0, 1
SPLIT_NAMES = ("train", "val")

#: Stable int ids for `PlayerMeta.controller`. APPEND ONLY.
CONTROLLER_VOCAB = ("human", "scripted", "idle")
TEAM_VOCAB = ("home", "away")

#: The game-observation keys that are an egocentric view of a FIXED pitch point.
#: Their mirror images are not themselves (the goal/field corners recorded are a
#: diagonal pair, and a y-mirror maps that pair onto the other diagonal), so the
#: mirror needs the world point. See `augment.mirror_game_obs`.
LANDMARK_KEYS = ("team_goal_back_right", "team_goal_front_left",
                 "field_front_left", "field_back_right",
                 "opponent_goal_back_left", "opponent_goal_front_right")

_IDLE = rec.SKILL_INDEX["idle"]


# --- split -----------------------------------------------------------------

def _match_hash(match_id: str, salt: str = "") -> int:
    return int(hashlib.sha256((salt + "|" + str(match_id)).encode()).hexdigest()[:16], 16)


def split_of_match(match_id: str, val_fraction: float = 0.25, salt: str = "",
                   mode: str = "quota", all_match_ids: Optional[Sequence[str]] = None) -> int:
    """`SPLIT_TRAIN` / `SPLIT_VAL` for one match, deterministically.

    Two modes, and the difference matters once the corpus grows:

      quota (default) — sort the matches by their hash and take the first
        ``round(val_fraction * n)`` (at least one whenever ``val_fraction > 0``
        and there is more than one match) as validation. Guarantees a usable
        val set on a corpus of eight matches, which a threshold cannot; the
        price is that ADDING a match can move an existing one across the split.
      hash — threshold the hash. A match's side never changes when the corpus
        grows, but with few matches the realised fraction is luck.

    Either way the answer depends only on the match ids, never on the order the
    files were listed in.
    """
    if mode == "hash":
        return SPLIT_VAL if (_match_hash(match_id, salt) % 10**9) / 10**9 < val_fraction \
            else SPLIT_TRAIN
    if mode != "quota":
        raise ValueError(f"unknown split mode {mode!r}; use 'quota' or 'hash'")
    if all_match_ids is None:
        raise ValueError("quota split needs the full list of match ids")
    ids = sorted(set(str(m) for m in all_match_ids))
    if len(ids) < 2 or val_fraction <= 0:
        return SPLIT_TRAIN
    k = max(1, int(round(val_fraction * len(ids))))
    k = min(k, len(ids) - 1)                      # never leave train empty
    order = sorted(ids, key=lambda m: _match_hash(m, salt))
    return SPLIT_VAL if str(match_id) in set(order[:k]) else SPLIT_TRAIN


# --- helpers ---------------------------------------------------------------

def key_offsets(keys, sizes) -> Dict[str, slice]:
    """Observation key -> slice into the flat vector, in `keys` order."""
    out, i = {}, 0
    for k, n in zip(keys, sizes):
        out[k] = slice(i, i + n)
        i += n
    return out


def playing_mask(demo) -> np.ndarray:
    """Ticks inside the PLAYING phase, from the event log.

    `MatchSim.step` only records while `phase == PHASE_PLAYING` (countdown ticks
    are stepped but not written), so on a well-formed demo this is all-True and
    the filter is a belt-and-braces check rather than a real cut. It is still
    computed from the events instead of assumed, because a demo produced by some
    future recorder that DOES write the countdown must not poison the corpus
    with 3 s of zero-torque ticks per match.
    """
    t = np.asarray(demo.arrays["tick"]).astype(np.int64)
    m = np.ones(t.shape[0], bool)
    starts = [int(e["tick"]) for e in demo.events if e.get("type") == "match_start"]
    ends = [int(e["tick"]) for e in demo.events if e.get("type") == "match_end"]
    if starts:
        m &= t >= min(starts)
    if ends:
        m &= t <= max(ends)
    return m


def recover_landmarks(obs: np.ndarray, offsets: Dict[str, slice],
                      keys: Sequence[str] = LANDMARK_KEYS):
    """World xy of each fixed pitch landmark, from the recorded egocentric views.

    dm_soccer emits these as ``ego = R2.T @ (W - x)`` where ``R2`` is the top-left
    2x2 block of the root's world rotation and ``x`` its world xy (see
    `dm_control.composer.Entity.global_vector_to_local_frame`: the 2-D branch
    uses `xmat[:2, :2]`, so a tilted body's transform is not orthonormal and
    inverting it per tick is ill-conditioned near a 90-degree roll). Stacking
    every tick of the match instead turns it into an overdetermined 2-unknown
    least-squares problem that the ant's constant yawing conditions beautifully:
    on the current corpus the residual is ~1e-6, i.e. float32 storage noise.

    Args:
      obs: [T, O] one team's rows of the game observation (any number of players
        of that team, stacked along T).
      offsets: key -> slice into O.
    Returns:
      (dict key -> (2,) float64 world xy, dict key -> float max residual)
    """
    R = obs[:, offsets["absolute_root_mat"]].astype(np.float64).reshape(-1, 3, 3)
    x = obs[:, offsets["absolute_root_pos"]].astype(np.float64)[:, :2]
    A = np.transpose(R[:, :2, :2], (0, 2, 1))            # [T, 2, 2] = R2.T
    world, resid = {}, {}
    for k in keys:
        if k not in offsets:
            continue
        ob = obs[:, offsets[k]].astype(np.float64)
        b = ob + np.einsum("tij,tj->ti", A, x)           # R2.T @ W
        W, *_ = np.linalg.lstsq(A.reshape(-1, 2), b.reshape(-1), rcond=None)
        pred = np.einsum("tij,tj->ti", A, W[None, :] - x)
        world[k] = W
        resid[k] = float(np.abs(pred - ob).max())
    return world, resid


def _layout_of(meta, skill_name: str):
    """(fields, obs_dim) the demo says this skill consumed, or None."""
    entry = (meta.skill_obs or {}).get(skill_name)
    if not entry:
        return None
    fields = tuple(entry.get("fields") or ())
    dim = int(entry.get("obs_dim", 0))
    if not fields or dim <= 0:
        return None
    return fields, dim


# --- the dataset -----------------------------------------------------------

class BCDataset:
    """Arrays + metadata. Dumb on purpose: no torch, no shuffling, no batching.

    `arrays` holds every per-sample column; `meta` is JSON-able provenance.
    """

    def __init__(self, arrays: Dict[str, np.ndarray], meta: dict):
        self.arrays = arrays
        self.meta = meta

    # -- basics ------------------------------------------------------------
    def __len__(self):
        return int(self.arrays["action"].shape[0]) if "action" in self.arrays else 0

    def __getattr__(self, name):
        try:
            return self.__dict__["arrays"][name]
        except KeyError:
            raise AttributeError(name) from None

    def __contains__(self, name):
        return name in self.arrays

    def __repr__(self):
        return (f"<BCDataset {len(self)} samples from {len(self.meta.get('demos', []))} "
                f"demos, obs {self.arrays['obs'].shape[-1] if len(self) else '?'}>")

    @property
    def skill_vocab(self):
        return tuple(self.meta["skill_vocab"])

    @property
    def controller_vocab(self):
        return tuple(self.meta["controller_vocab"])

    @property
    def layouts(self):
        return self.meta["layouts"]

    def skill_names(self) -> np.ndarray:
        return np.array(self.skill_vocab)[self.arrays["skill"]]

    def controller_names(self) -> np.ndarray:
        return np.array(self.controller_vocab)[self.arrays["controller"]]

    def obs_offsets(self) -> Dict[str, slice]:
        return key_offsets(self.meta["obs_keys"], self.meta["obs_sizes"])

    def landmarks_for(self, demo_idx: int, team: int) -> Dict[str, np.ndarray]:
        """World xy of each landmark key for one demo and one team."""
        lm = np.asarray(self.arrays["landmarks"])[int(demo_idx), int(team)]
        return {k: lm[i] for i, k in enumerate(self.meta["landmark_keys"])}

    # -- selection ---------------------------------------------------------
    def select(self, mask) -> "BCDataset":
        """A new dataset holding the masked samples (metadata carried over)."""
        mask = np.asarray(mask)
        out = {}
        n = len(self)
        for k, v in self.arrays.items():
            out[k] = v[mask] if (isinstance(v, np.ndarray) and v.ndim >= 1
                                 and v.shape[0] == n and k != "landmarks") else v
        meta = dict(self.meta)
        meta["n_samples"] = int(out["action"].shape[0])
        return BCDataset(out, meta)

    def train(self) -> "BCDataset":
        return self.select(self.arrays["split"] == SPLIT_TRAIN)

    def val(self) -> "BCDataset":
        return self.select(self.arrays["split"] == SPLIT_VAL)

    def with_skill(self, *names) -> "BCDataset":
        want = {self.skill_vocab.index(n) if isinstance(n, str) else int(n) for n in names}
        return self.select(np.isin(self.arrays["skill"], list(want)))

    def with_controller(self, *names) -> "BCDataset":
        want = {self.controller_vocab.index(n) if isinstance(n, str) else int(n)
                for n in names}
        return self.select(np.isin(self.arrays["controller"], list(want)))

    def with_layout(self, layout_id: int) -> "BCDataset":
        return self.select(self.arrays["layout"] == int(layout_id))

    def expert_obs_of(self, i: int) -> np.ndarray:
        """Sample `i`'s expert vector, trimmed to its real width."""
        n = int(self.arrays["expert_obs_n"][i])
        return self.arrays["expert_obs"][i, :n]

    # -- io ----------------------------------------------------------------
    def save(self, path) -> str:
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        meta = dict(self.meta)
        meta["n_samples"] = len(self)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, meta_json=np.array(json.dumps(meta, sort_keys=True,
                                                                  default=_json_default)),
                                **self.arrays)
        os.replace(tmp, path)
        return path

    @staticmethod
    def load(path) -> "BCDataset":
        with np.load(str(path), allow_pickle=False) as f:
            meta = json.loads(str(f["meta_json"]))
            arrays = {k: f[k] for k in f.files if k != "meta_json"}
        if meta.get("schema") != SCHEMA_NAME:
            raise ValueError(f"{path}: not a {SCHEMA_NAME} file (got {meta.get('schema')!r})")
        if int(meta.get("version", 0)) != SCHEMA_VERSION:
            raise ValueError(f"{path}: dataset schema v{meta.get('version')}, this "
                             f"reader is v{SCHEMA_VERSION}")
        return BCDataset(arrays, meta)


load_dataset = BCDataset.load


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --- the builder -----------------------------------------------------------

def build_dataset(paths: Iterable[str],
                  *,
                  drop_idle: bool = True,
                  playing_only: bool = True,
                  keep_controllers: Optional[Sequence[str]] = None,
                  require_expert_obs: bool = False,
                  val_fraction: float = 0.25,
                  split_mode: str = "quota",
                  split_salt: str = "",
                  creature: Optional[str] = None,
                  verbose: bool = False) -> BCDataset:
    """Load demos into one dataset.

    Args:
      paths: demo `.npz` files. Files whose game-observation layout disagrees
        with the first accepted demo's are SKIPPED (with a reason in
        `meta["skipped"]`) rather than silently concatenated into a corpus whose
        columns mean different things per row.
      drop_idle: drop ticks whose EXECUTED skill was `idle` (zero torque). These
        are 40% of the current corpus — an unclaimed or waiting seat — and they
        teach a policy to do nothing.
      playing_only: keep only ticks inside the match's PLAYING phase.
      keep_controllers: e.g. `("human",)`. None keeps every controller, tagged.
      require_expert_obs: drop samples with no recorded expert input vector.
      val_fraction / split_mode / split_salt: see `split_of_match`.
      creature: require every player to be this creature kind (the mirror
        augmentation is per-body, so a mixed corpus must be built per kind).
    """
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("no demo paths given")

    accepted, skipped = [], []
    obs_keys = obs_sizes = None
    act_dim = z_dim = None
    layouts: List[dict] = []
    layout_id: Dict[tuple, int] = {}
    dropped = dict(non_playing=0, idle=0, controller=0, no_expert_obs=0, nonfinite=0)

    cols: Dict[str, list] = {}
    demo_records: List[dict] = []
    landmark_rows: List[np.ndarray] = []
    keep_controllers = None if keep_controllers is None else set(keep_controllers)

    # -- pass 1: read, validate, collect rows ------------------------------
    for path in paths:
        try:
            demo = rec.read_demo(path)
        except Exception as exc:                            # noqa: BLE001
            skipped.append(dict(path=path, reason=f"unreadable: {exc}"))
            continue
        m = demo.meta
        if obs_keys is None:
            obs_keys, obs_sizes = list(m.obs_keys), [int(s) for s in m.obs_sizes]
            act_dim, z_dim = int(m.act_dim), int(m.z_dim)
        elif list(m.obs_keys) != obs_keys or [int(s) for s in m.obs_sizes] != obs_sizes:
            skipped.append(dict(path=path, reason="game obs layout differs from the "
                                                  "first accepted demo"))
            continue
        elif int(m.act_dim) != act_dim or int(m.z_dim) != z_dim:
            skipped.append(dict(path=path,
                                reason=f"act/z dims {m.act_dim}/{m.z_dim} != "
                                       f"{act_dim}/{z_dim}"))
            continue
        kinds = {p.creature for p in m.players}
        if creature is not None and kinds != {creature}:
            skipped.append(dict(path=path, reason=f"creatures {sorted(kinds)} != {creature}"))
            continue
        if list(m.skill_vocab) != list(rec.SKILL_VOCAB):
            skipped.append(dict(path=path, reason="skill vocab differs from recording.SKILL_VOCAB"))
            continue

        offsets = key_offsets(obs_keys, obs_sizes)
        di = len(demo_records)
        a = demo.arrays
        T, P = a["obs"].shape[0], demo.n_players
        play = playing_mask(demo) if playing_only else np.ones(T, bool)
        dropped["non_playing"] += int(P * (~play).sum())

        # landmarks, per team, from every tick of every player of that team
        lm_row = np.full((len(TEAM_VOCAB), len(LANDMARK_KEYS), 2), np.nan)
        lm_res = {}
        for ti, team in enumerate(TEAM_VOCAB):
            idx = [p for p in range(P) if m.players[p].team == team]
            if not idx:
                continue
            block = np.concatenate([a["obs"][:, p, :] for p in idx], axis=0)
            world, resid = recover_landmarks(block, offsets)
            for ki, k in enumerate(LANDMARK_KEYS):
                if k in world:
                    lm_row[ti, ki] = world[k]
            lm_res[team] = max(resid.values()) if resid else float("nan")
        landmark_rows.append(lm_row)

        n_kept_demo = 0
        for p in range(P):
            pm = m.players[p]
            ctrl = pm.controller if pm.controller in CONTROLLER_VOCAB else "idle"
            keep = play.copy()
            if drop_idle:
                idle = a["skill"][:, p] == _IDLE
                dropped["idle"] += int((keep & idle).sum())
                keep &= ~idle
            if keep_controllers is not None and ctrl not in keep_controllers:
                dropped["controller"] += int(keep.sum())
                keep[:] = False
            son = a["skill_obs_n"][:, p].astype(np.int64) if "skill_obs_n" in a \
                else np.zeros(T, np.int64)
            if require_expert_obs:
                bad = son <= 0
                dropped["no_expert_obs"] += int((keep & bad).sum())
                keep &= ~bad
            finite = np.isfinite(a["obs"][:, p, :]).all(1) & np.isfinite(a["action"][:, p, :]).all(1)
            dropped["nonfinite"] += int((keep & ~finite).sum())
            keep &= finite
            n = int(keep.sum())
            if n == 0:
                continue
            n_kept_demo += n

            # per-sample layout id: (skill, fields, width) as the DEMO recorded it
            lay = np.zeros(n, np.int16)
            sk = a["skill"][keep, p].astype(np.int64)
            for s in np.unique(sk):
                name = m.skill_vocab[int(s)]
                spec = _layout_of(m, name)
                key = (name, spec[0], spec[1]) if spec else (name, (), 0)
                if key not in layout_id:
                    layout_id[key] = len(layouts)
                    layouts.append(dict(id=len(layouts), skill=name,
                                        fields=list(key[1]), obs_dim=int(key[2])))
                lay[sk == s] = layout_id[key]

            _append(cols, "obs", a["obs"][keep, p, :])
            _append(cols, "action", a["action"][keep, p, :])
            _append(cols, "z", a["z"][keep, p, :])
            _append(cols, "expert_obs", a["skill_obs"][keep, p, :] if "skill_obs" in a
                    else np.zeros((n, 0), np.float32))
            _append(cols, "expert_obs_n", son[keep].astype(np.int16))
            _append(cols, "skill", sk.astype(np.int8))
            _append(cols, "skill_req", a["skill_req"][keep, p].astype(np.int8))
            _append(cols, "layout", lay)
            _append(cols, "target", a["target"][keep, p, :])
            _append(cols, "aim", a["aim"][keep, p, :] if "aim" in a
                    else np.zeros((n, 2), np.float32))
            _append(cols, "root_pos", a["player_pos"][keep, p, :])
            _append(cols, "root_mat", a["player_mat"][keep, p, :])
            _append(cols, "ball_pos", a["ball_pos"][keep, :])
            _append(cols, "ball_vel", a["ball_vel"][keep, :])
            _append(cols, "ctrl_tick", a["ctrl_tick"][keep, p].astype(np.int32))
            _append(cols, "tick", a["tick"][keep].astype(np.int32))
            _append(cols, "t", a["t"][keep].astype(np.float32))
            _append(cols, "score", a["score"][keep, :].astype(np.int16))
            _append(cols, "demo", np.full(n, di, np.int16))
            _append(cols, "player", np.full(n, p, np.int8))
            _append(cols, "team", np.full(n, TEAM_VOCAB.index(pm.team), np.int8))
            _append(cols, "controller", np.full(n, CONTROLLER_VOCAB.index(ctrl), np.int8))
            _append(cols, "mirrored", np.zeros(n, np.int8))

        demo_records.append(dict(
            index=di, path=os.path.abspath(path), file=os.path.basename(path),
            match_id=m.match_id, created_utc=m.created_utc, git_sha=m.git_sha,
            seed=int(m.seed), control_dt=float(m.control_dt),
            pitch_half=list(m.pitch_half), n_ticks=int(T), n_players=int(P),
            n_samples=int(n_kept_demo), action_mode=m.action_mode,
            skill_seed=int(m.skill_seed),
            creatures=sorted(kinds), landmark_residual=lm_res,
            final_score=list(demo.final_score()),
            n_goals=len(demo.events_of("goal")),
            players=[dict(slot=q.slot, team=q.team, controller=q.controller,
                          creature=q.creature, display_name=q.display_name)
                     for q in m.players]))
        accepted.append(path)
        if verbose:
            print(f"[bc] {os.path.basename(path)}: {n_kept_demo} samples "
                  f"({T} ticks x {P} players)")

    if not accepted:
        raise ValueError("no demo was accepted; skipped: "
                         + json.dumps(skipped, indent=1))
    kinds = sorted({c for d in demo_records for c in d["creatures"]})
    if len(kinds) != 1:
        # The mirror augmentation and every action-space model are per-body, so
        # a corpus that mixes creatures has no single act_dim meaning. Build one
        # dataset per kind (`creature=`) instead of discovering this at train time.
        raise ValueError(f"demos mix creature kinds {kinds}; pass creature=<kind>")

    # `skill_obs` is NaN-padded per demo to THAT demo's widest skill, so two
    # demos recorded either side of a contract change bring different widths.
    # Pad every block out to the corpus width before concatenating; the padding
    # is NaN and `expert_obs_n` says where the real numbers stop, exactly as in
    # the demo schema.
    blocks = cols["expert_obs"]
    w = max([int(b.shape[1]) for b in blocks]
            + [int(d["obs_dim"]) for d in layouts] + [0])
    cols["expert_obs"] = [
        b if b.shape[1] == w else
        np.concatenate([b, np.full((b.shape[0], w - b.shape[1]), np.nan, np.float32)],
                       axis=1)
        for b in blocks]

    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    n = int(arrays["action"].shape[0])
    if n == 0:
        raise ValueError("every tick was filtered out; loosen drop_idle / "
                         "keep_controllers")

    # split by match
    match_of_demo = [d["match_id"] for d in demo_records]
    split_of_demo = np.array([split_of_match(mid, val_fraction, split_salt,
                                             split_mode, match_of_demo)
                              for mid in match_of_demo], np.int8)
    for d in demo_records:
        d["split"] = SPLIT_NAMES[int(split_of_demo[d["index"]])]
    arrays["split"] = split_of_demo[arrays["demo"]]
    arrays["landmarks"] = np.stack(landmark_rows).astype(np.float64)

    meta = dict(
        schema=SCHEMA_NAME, version=SCHEMA_VERSION,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n_samples=n, obs_keys=obs_keys, obs_sizes=obs_sizes,
        act_dim=act_dim, z_dim=z_dim,
        skill_vocab=list(rec.SKILL_VOCAB), controller_vocab=list(CONTROLLER_VOCAB),
        team_vocab=list(TEAM_VOCAB), split_names=list(SPLIT_NAMES),
        landmark_keys=list(LANDMARK_KEYS),
        layouts=layouts, demos=demo_records, skipped=skipped, dropped=dropped,
        creature=kinds[0],
        filters=dict(drop_idle=drop_idle, playing_only=playing_only,
                     keep_controllers=None if keep_controllers is None
                     else sorted(keep_controllers),
                     require_expert_obs=require_expert_obs),
        split=dict(mode=split_mode, val_fraction=val_fraction, salt=split_salt),
    )
    return BCDataset(arrays, meta)


def _append(cols, key, value):
    cols.setdefault(key, []).append(np.asarray(value))


# --- CLI -------------------------------------------------------------------

def main(argv=None):
    import argparse
    import glob as _glob
    p = argparse.ArgumentParser(description="Build a BC dataset from demo files.")
    p.add_argument("demos", nargs="+", help="demo .npz paths or globs")
    p.add_argument("-o", "--out", default=None, help="write the dataset here")
    p.add_argument("--keep-idle", action="store_true")
    p.add_argument("--controllers", default=None,
                   help="comma-separated subset of " + ",".join(CONTROLLER_VOCAB))
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--split-mode", default="quota", choices=["quota", "hash"])
    p.add_argument("--require-expert-obs", action="store_true")
    a = p.parse_args(argv)
    paths = []
    for pat in a.demos:
        paths.extend(sorted(_glob.glob(pat)) or [pat])
    ds = build_dataset(paths, drop_idle=not a.keep_idle,
                       keep_controllers=a.controllers.split(",") if a.controllers else None,
                       require_expert_obs=a.require_expert_obs,
                       val_fraction=a.val_fraction, split_mode=a.split_mode,
                       verbose=True)
    from rower_soccer.bc import stats as _stats
    print(_stats.summary(ds))
    if a.out:
        print("wrote", ds.save(a.out))


if __name__ == "__main__":
    main()
