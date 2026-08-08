"""Thin game-side adapter over WS3's `rower_soccer.skills`.

WS3 owns everything that turns (skill_id, target_xy) + a game observation into
torques: obs reconstruction, checkpoint loading/validation, the frozen decoder, the
`scripted` chase baseline.  This module does NOT reimplement any of it.  It exists
because the game's needs are slightly different in shape from a per-player
controller's, and mediating that here keeps `match.py` and `server.py` free of skill
plumbing:

  * one call per player per tick, `act(p, obs, skill, target)`, with the *requested*
    skill supplied fresh each tick (a human can retarget or switch at any moment)
    rather than a `set_command` / `act` two-step;
  * `set_command`'s loud failures (`UnknownSkill`, `SkillUnavailable`) turned into a
    recorded downgrade to `idle`.  A play server must not die because someone
    pressed 3 before WS1 trained kick -- but it must not silently run `follow` and
    label the tick `kick` either, because that label goes straight into the BC
    dataset.  So: run nothing, record `idle`, tell the client the skill is
    unavailable;
  * a single place that reports which skills are live, for the client's key hints
    and for the demo metadata.

`SkillController` is per-player and holds the active command; the game holds the
same command in `MatchSim.commands` because that is what the UI displays and what
the recorder writes.  We push ours into WS3's on change, rather than keeping two
sources of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IDLE = "idle"


@dataclass
class SkillTick:
    """What the game records for one player for one tick."""
    action: np.ndarray            # (act_dim,) in [-1, 1]
    z: np.ndarray | None          # (z_dim,) latent actually emitted, None for idle
    skill: str                    # the skill that RAN (never the one merely asked for)
    target: np.ndarray            # the world xy actually used (scripted: the ball)
    obs_vector: np.ndarray        # the exact expert input, for BC + bit-exact replay
    ctrl_tick: int                # the controller's own tick counter (see below)


class GameSkillLayer:
    """Drives all four players through one `SkillControllerPool`.

    `action_mode` is load-bearing, not a tuning knob.  `follow_ant_v1` trained with
    `ent_ceil = 0`, so its action std finished near 1.0 -- the full action range --
    and PPO scored the SAMPLED policy.  Its distribution mean therefore does not
    locomote; WS3's `MODE_AUTO` detects that and runs `MODE_NOISE`, where the noise
    is a pure function of `(seed, player_index, controller tick)`.  That keeps
    replay bit-exact, but only if the demo records all three -- hence `skill_seed`
    and `player index` in the header and `ctrl_tick` per row.  The controller resets
    its tick on a skill switch, so it cannot be derived from the match tick.
    """

    backend = "ws3"

    def __init__(self, env, creature="ant", checkpoints=None, device="cpu",
                 action_mode="auto", seed=0, target_clip=None):
        from rower_soccer.skills import SkillControllerPool
        from rower_soccer.skills.registry import DEFAULT_TARGET_CLIP

        self.creature = creature
        self.seed = int(seed)
        n = len(env.task.players)
        base_kw = dict(device=device, seed=self.seed,
                       checkpoints=dict(checkpoints or {}),
                       target_clip=DEFAULT_TARGET_CLIP if target_clip is None
                       else target_clip)

        def _pool(mode):
            kw = dict(base_kw)
            if mode is not None:
                kw["action_mode"] = mode
            return SkillControllerPool([creature] * n, names=list(SLOT_NAMES[:n]), **kw)

        # `auto` is resolved HERE rather than assumed of WS3, whose action-mode API
        # is still moving. If WS3 knows "auto", use it; otherwise take its default
        # and upgrade to "noise" when any loaded expert says its mean does not
        # locomote. Getting this wrong is not subtle -- the ants simply stand there.
        if action_mode in (None, "auto"):
            try:
                self.pool = _pool("auto")
            except (ValueError, TypeError):
                self.pool = _pool(None)
                if _any_noise_driven(self.pool):
                    self.pool = _pool("noise")
        else:
            self.pool = _pool(action_mode)
        self.action_mode = str(getattr(self.pool[0], "action_mode", action_mode))
        self.act_dim = self.pool[0].act_dim
        self.skills = tuple(self.pool[0].available_skills())
        self.checkpoints = {}
        self.z_dim = 0
        self.obs_dim = {}
        for s in self.skills:
            try:
                self.checkpoints[s] = self._resolve(s)
            except Exception:                       # noqa: BLE001 - zero-kind skills
                pass
            try:
                self.obs_dim[s] = int(self.pool[0].layout(s)[0])
            except Exception:                       # noqa: BLE001
                self.obs_dim[s] = 0
        # z width: ask a real expert rather than assuming 16.
        for s in self.skills:
            try:
                self.z_dim = max(self.z_dim, int(self.pool[0]._expert(s).info.z_dim))
            except Exception:                       # noqa: BLE001 - idle has no expert
                continue
        self.z_dim = self.z_dim or 16
        self.max_obs_dim = max(self.obs_dim.values() or [0])
        self.resolved_modes = {}
        for s in self.skills:
            try:
                self.resolved_modes[s] = self.pool[0].resolved_mode(s)
            except Exception:                       # noqa: BLE001 - optional API
                self.resolved_modes[s] = self.action_mode
        self._unavailable_warned = set()
        self.bind(env)

    def _resolve(self, skill):
        from rower_soccer.skills import get_spec, resolve_checkpoint
        c = self.pool[0]
        path = c._overrides.get(skill) or get_spec(skill).checkpoint_for(self.creature)
        return resolve_checkpoint(path)

    # -- lifecycle ---------------------------------------------------------
    def bind(self, env):
        """(Re)bind to the env.  dm_control recompiles the model when the arena
        resizes on reset, replacing `env.physics` -- every handle derived from it
        must be re-fetched, never cached across resets."""
        self._env = env
        self._walkers = [p.walker for p in env.task.players]

    def reset(self):
        self.pool.reset()

    def skill_obs_meta(self):
        """`{skill: {"fields": [...], "obs_dim": n}}` for the demo file, so a BC
        consumer can rebuild any expert's input from the recorded game obs."""
        from rower_soccer.skills import get_spec
        out = {}
        for s in self.skills:
            spec = get_spec(s)
            out[s] = {"fields": list(spec.fields), "obs_dim": self.obs_dim.get(s, 0),
                      "target_source": spec.target_source, "kind": spec.kind}
        return out

    # -- the tick ----------------------------------------------------------
    def act(self, p, obs, skill, target, aim=None, ball=None) -> SkillTick:
        """`ball` is the ball's WORLD (pos, vel).

        Not optional in practice, and not derivable from `obs`: dm_soccer builds
        `ball_ego_position` with `objtype='body', reftype='body'`, which in MuJoCo
        means the INERTIAL frames, while the drills compute `ball_ego` in the BODY
        frame. For the ant those differ by a whole axis permutation (WS3 measured
        |ximat - xmat| = 1.09), so feeding the game's own ball observation to a
        drill-trained expert hands it a permuted vector. The world position plus
        the drill's own transform sidesteps it -- and is why the demo schema stores
        `ball_pos`/`ball_vel` rather than trusting the observation.
        """
        from rower_soccer.skills import PlayerFrame, SkillError

        ctrl = self.pool[p]
        target = np.asarray(target, np.float64).reshape(2)
        try:
            if ctrl.skill_id != skill:
                ctrl.set_command(skill, target)
            elif ctrl.command is not None and ctrl.command.target_xy is not None:
                ctrl.set_target(target)
        except SkillError as exc:
            # Asked for a skill WS1 has not trained yet (or a typo in a key binding).
            # Stand still and say so once; never fake the label.
            if (p, skill) not in self._unavailable_warned:
                self._unavailable_warned.add((p, skill))
                print(f"[skills] player {p}: {exc}", flush=True)
            ctrl.clear_command()

        ph = self._env.physics
        b = ph.bind(self._walkers[p].root_body)
        bp, bv = ball if ball is not None else (None, None)
        frame = PlayerFrame(obs=obs, root_pos=np.array(b.xpos),
                            root_mat=np.array(b.xmat), ball_pos=bp, ball_vel=bv)
        ctrl_tick = int(ctrl.tick)          # read BEFORE act(): act() increments it
        out = ctrl.act(frame)
        used = target if out.target_xy is None else np.asarray(out.target_xy, np.float64)
        return SkillTick(action=out.action, z=out.z, skill=out.skill_id,
                         target=used, obs_vector=out.obs_vector, ctrl_tick=ctrl_tick)


SLOT_NAMES = ("home_1", "home_2", "away_1", "away_2")


def _any_noise_driven(pool):
    """True if some loaded expert was scored as a SAMPLED policy, so its
    distribution mean is not the behaviour the run achieved. `follow_ant_v1` is
    exactly that case (`ent_ceil = 0` left its action std at ~1.0)."""
    c = pool[0]
    for s in c.available_skills():
        try:
            if getattr(c._expert(s), "noise_driven", False):
                return True
        except Exception:                           # noqa: BLE001 - zero-kind skills
            continue
    return False


def build_controller(env, creature="ant", checkpoints=None, device="cpu", **kw):
    """The game's one entry point into the skill layer."""
    layer = GameSkillLayer(env, creature=creature, checkpoints=checkpoints,
                           device=device, **kw)
    print(f"[skills] backend={layer.backend} skills={layer.skills} "
          f"act_dim={layer.act_dim} z_dim={layer.z_dim} obs_dim={layer.obs_dim} "
          f"modes={layer.resolved_modes} seed={layer.seed}", flush=True)
    return layer
