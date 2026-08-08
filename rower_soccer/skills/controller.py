"""`SkillController` — the piece between "drill experts" and "game".

One controller drives ONE player. Give it a skill and a world target; feed it a
`PlayerFrame` each tick; it returns the torques.

    from rower_soccer.skills import SkillController, PlayerFrame

    ctrl = SkillController("ant")                 # loads follow_ant_v1 lazily
    ctrl.set_command("follow", target_xy=(12.0, -4.0))
    out = ctrl.act(frame)                         # SkillOutput
    timestep = env.step([out.action])

Design notes that matter to the callers
---------------------------------------
* **Creature-agnostic.** Nothing here knows what an ant is. Widths come from the
  creature's MJCF (`contract.py`), the obs layout from the skill's field list
  (`registry.py`). Swapping in rower/worm is a `SkillController("rower")` plus a
  checkpoint entry.

* **Reproducible by default, in one of two ways.** Gameplay and replay both need
  identical output for identical input, so nothing here ever touches a global
  RNG. `action_mode="mean"` emits the action distribution's mean. `"noise"` adds
  `std * eps` where `eps` is drawn from a generator seeded by
  `(seed, player, tick)` — so it is still a pure function of the inputs and a
  demo still replays bit-for-bit, but the policy gets the exploration noise it
  was scored with. `"auto"` (the default) picks between them by inspecting the
  checkpoint, and says which it picked and why.

  The reason `"mean"` is not simply the answer: `follow_ant_v1` trained with
  `ent_ceil = 0`, so its `log_std` finished pinned at the ceiling (std ~= 1.0 on
  every joint, against actions clamped to [-1, 1]). PPO scores the *sampled*
  policy, so its 0.997 fitness belongs to `clamp(mean + N(0, 1))`, not to the
  mean. Measured here on CPU MuJoCo, 15 s toward a point 3 m away: sampled walks
  there (fitness 0.944 at the pitch's solref, 0.996 at the drill's), the mean
  crouches at the spawn point and never moves (0.23). See MODE_AUTO below.

* **Clean skill switching.** `set_command` with a different `skill_id` clears
  every scrap of per-skill state before the next `act()`. Today the experts are
  feed-forward and the only per-tick state is the target, so "clean" is easy —
  but the switch path is written as if it were not, because the paper's expert
  carries an LSTM and dribble may well arrive with one. The env-side
  `prev_action` buffer that `creature.py` keeps is deliberately NOT part of any
  skill's observation (`PROPRIO_V1` has no `prev_action` field), so it cannot
  leak across a switch either.

* **Weights are shared, state is not.** Experts are cached globally by
  checkpoint, so four players on one checkpoint hold one copy of the weights.

* **Ready for BC/self-play.** `SkillOutput` carries `z` and the exact
  `obs_vector`, which is what PIPELINE_V2's stage-5 BC trains on. A future
  high-level policy replaces the human calling `set_command`, and nothing else
  in this class changes.
"""

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from rower_soccer.skills.api import (PlayerFrame, SkillCommand, SkillOutput,
                                     SkillUnavailable, _as_xy)
from rower_soccer.skills.contract import (CreatureContract, check_soccer_obs_widths,
                                          contract_for)
from rower_soccer.skills.fields import FieldContext, ball_world_xy, get_field
from rower_soccer.skills import registry as R
from rower_soccer.skills.policy import load_policy

__all__ = ["SkillController", "SkillControllerPool",
           "MODE_AUTO", "MODE_MEAN", "MODE_NOISE"]

#: Emit the action distribution's mean. Bit-exact, no seed needed. Correct for
#: any checkpoint whose mean policy is the behaviour it was scored on.
MODE_MEAN = "mean"
#: Emit `mean + std * eps` with `eps` a pure function of `(seed, player, tick)`.
#: Still fully reproducible — a replay that re-runs the same ticks with the same
#: seed gets the same torques — but it restores the exploration noise a
#: noise-driven checkpoint needs in order to locomote at all.
MODE_NOISE = "noise"
#: Inspect the checkpoint and choose. `MODE_NOISE` when the expert's action std is
#: at/near the [-1, 1] action range (`policy.NOISE_DRIVEN_STD`), else `MODE_MEAN`.
#: The choice is announced once per expert, with the std that drove it, because a
#: silent choice here is exactly the kind of thing that costs this project runs.
MODE_AUTO = "auto"

_ANNOUNCED = set()


class SkillController:
    """Runs one player's active skill against soccer observations."""

    def __init__(self,
                 creature: str = "ant",
                 *,
                 device: str = "cpu",
                 action_mode: str = MODE_AUTO,
                 checkpoints: Optional[Mapping[str, str]] = None,
                 target_clip: float = R.DEFAULT_TARGET_CLIP,
                 preload: Sequence[str] = (),
                 seed: int = 0,
                 player_index: int = 0,
                 quiet: bool = False,
                 name: str = ""):
        """
        Args:
          creature: creature kind (`"ant"`, `"rower"`, `"worm"`) or a path to a
            creature MJCF.
          device: torch device for the experts. `"cpu"` is right for the game
            server — one 1.4 MB MLP per tick is microseconds, and it keeps the
            play loop off the training GPU.
          action_mode: `MODE_AUTO` (default), `MODE_MEAN`, or `MODE_NOISE`. All
            three are reproducible; see the module docstring for which to use.
          checkpoints: per-skill override, `{"follow": "/path/best.pt"}`. Takes
            precedence over the registry, so WS4 can point at a fresh run without
            editing this package.
          target_clip: metres; see `fields._target_ego`. 0 disables.
          preload: skills to load immediately rather than on first use — pass the
            slot's expected skills to keep the first tick off the disk.
          seed, player_index: seed the per-tick noise stream in `MODE_NOISE`.
            Distinct `player_index` values keep four ants from receiving the
            identical noise sequence. **WS4: record both in the demo header** —
            with them plus the tick index a replay is bit-exact.
          quiet: suppress the one-time `MODE_AUTO` announcement.
          name: label for error messages (e.g. `"home_0"`).
        """
        if action_mode not in (MODE_AUTO, MODE_MEAN, MODE_NOISE):
            raise ValueError(
                f"action_mode must be one of {MODE_AUTO!r}, {MODE_MEAN!r}, "
                f"{MODE_NOISE!r}; got {action_mode!r}")
        self.creature = creature
        self.name = name or creature
        self.device = device
        self.action_mode = action_mode
        self.target_clip = float(target_clip)
        self.seed = int(seed)
        self.player_index = int(player_index)
        self.quiet = bool(quiet)
        self.tick = 0
        self._overrides: Dict[str, str] = dict(checkpoints or {})
        self._experts: Dict[str, object] = {}
        self._contract: CreatureContract = contract_for(creature)
        self._layouts: Dict[str, Tuple[int, list, list]] = {}
        self._command: Optional[SkillCommand] = None
        self._obs_checked = False
        for s in preload:
            self._expert(s)

    @property
    def deterministic(self) -> bool:
        """True for every mode this class offers: output is a pure function of
        (observation, seed, player_index, tick), so a demo always replays."""
        return True

    def resolved_mode(self, skill_id: Optional[str] = None) -> str:
        """The mode `act()` will actually use for a skill (resolves MODE_AUTO)."""
        sid = skill_id or self.skill_id
        if self.action_mode != MODE_AUTO or sid is None:
            return self.action_mode if self.action_mode != MODE_AUTO else MODE_MEAN
        spec = R.get_spec(sid)
        if spec.kind == R.KIND_ZERO:
            return MODE_MEAN
        return MODE_NOISE if self._expert(sid).noise_driven else MODE_MEAN

    # -- introspection -----------------------------------------------------
    @property
    def contract(self) -> CreatureContract:
        return self._contract

    @property
    def act_dim(self) -> int:
        return self._contract.act_dim

    @property
    def command(self) -> Optional[SkillCommand]:
        return self._command

    @property
    def skill_id(self) -> Optional[str]:
        return None if self._command is None else self._command.skill_id

    def available_skills(self) -> Tuple[str, ...]:
        """Skills runnable right now for this creature, including overrides."""
        out = []
        for sid, spec in R.SKILLS.items():
            if sid in self._overrides or spec.is_available(self.creature):
                out.append(sid)
        return tuple(sorted(out))

    def zero_action(self) -> np.ndarray:
        return np.zeros(self.act_dim, dtype=np.float32)

    # -- commanding --------------------------------------------------------
    def set_command(self, skill_id: str, target_xy=None) -> SkillCommand:
        """Set the active skill and target. Clears per-skill state on a change.

        Safe to call every tick with the same skill (retargeting is not a switch
        and does not reset anything). Raises `UnknownSkill` / `SkillUnavailable`
        rather than silently falling back, so a bad key binding surfaces at once.
        """
        spec = R.get_spec(skill_id)
        target = _as_xy(target_xy)
        if spec.needs_target() and target is None:
            if self._command is not None and self._command.skill_id == skill_id:
                target = self._command.target_xy          # retain on re-arm
            else:
                raise SkillUnavailable(
                    f"[{self.name}] skill '{skill_id}' needs a target_xy "
                    "(the world point the human clicked)")
        switching = self._command is None or self._command.skill_id != skill_id
        if switching:
            self._reset_skill_state()
            if spec.kind != R.KIND_ZERO:
                self._expert(skill_id)                    # fail now, not mid-tick
        self._command = SkillCommand(skill_id, target)
        return self._command

    def set_target(self, target_xy) -> SkillCommand:
        """Retarget the active skill without touching its state."""
        if self._command is None:
            raise SkillUnavailable(
                f"[{self.name}] no active skill to retarget; call set_command()")
        self._command = self._command.with_target(target_xy)
        return self._command

    def clear_command(self):
        """Drop the active skill; `act()` then emits zero torque."""
        self._reset_skill_state()
        self._command = None

    def reset(self):
        """Call on episode reset. Clears state; keeps the command and the cache."""
        self._reset_skill_state()
        self._obs_checked = False
        self.tick = 0

    def _reset_skill_state(self):
        """Everything that could carry meaning from the previous skill into the
        next one. Today: the experts' (empty) internal state and the noise
        stream's phase. The noise phase matters — a skill switch that left the
        tick counter mid-stream would make a replay's actions depend on the
        history of switches, not just on the tick."""
        for e in self._experts.values():
            e.reset()
        self.tick = 0

    # -- the tick ----------------------------------------------------------
    def act(self, frame: PlayerFrame) -> SkillOutput:
        """Torques for this tick. Never raises on a missing command — an
        uncommanded slot stands still rather than stopping the match."""
        if self._command is None:
            return SkillOutput(self.zero_action(), None, "idle", None,
                               np.zeros(0, dtype=np.float32))
        spec = R.get_spec(self._command.skill_id)
        if not self._obs_checked:
            check_soccer_obs_widths(self._contract, frame.obs, spec.fields)
            self._obs_checked = True

        if spec.kind == R.KIND_ZERO:
            self.tick += 1
            return SkillOutput(self.zero_action(), None, spec.skill_id,
                               self._command.target_xy, np.zeros(0, dtype=np.float32))

        target = self._resolve_target(spec, frame)
        vec = self.build_obs(spec, frame, target)
        expert = self._expert(spec.skill_id)
        mode = self.resolved_mode(spec.skill_id)
        noise = self._noise(expert.info.act_dim) if mode == MODE_NOISE else None
        action, z = expert.act(vec, mode=mode, noise=noise)
        self.tick += 1
        return SkillOutput(action=action, z=z, skill_id=spec.skill_id,
                           target_xy=None if target is None else (float(target[0]),
                                                                  float(target[1])),
                           obs_vector=vec)

    def _noise(self, act_dim: int) -> np.ndarray:
        """Exploration noise for this tick, as a pure function of
        (seed, player_index, tick).

        Re-seeded every tick rather than advanced, so the stream does not depend
        on how many ticks a replay chose to skip, or on how many players share the
        process. numpy's SeedSequence does the mixing, which is exactly what it is
        for; torch's Generator is not used here so the values are identical on CPU
        and CUDA.
        """
        ss = np.random.SeedSequence([self.seed, self.player_index, self.tick])
        return np.random.default_rng(ss).standard_normal(act_dim).astype(np.float32)

    def action(self, frame: PlayerFrame) -> np.ndarray:
        """`act(frame).action`, for callers that do not record demos."""
        return self.act(frame).action

    # -- obs assembly ------------------------------------------------------
    def build_obs(self, spec, frame: PlayerFrame, target_xy) -> np.ndarray:
        """The drill observation vector for `spec`, from a soccer observation.

        Exposed (rather than private) because it is the piece a replay or a BC
        dataset builder wants on its own, without running the network.
        """
        ctx = FieldContext(frame=frame, target_xy=target_xy,
                           target_clip=self.target_clip)
        parts, expect = [], self.layout(spec.skill_id)[0]
        for name in spec.fields:
            parts.append(np.asarray(get_field(name).build(ctx), dtype=np.float32).ravel())
        vec = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        if vec.size != expect:
            # Widths were checked against the contract on the first tick; this can
            # only fire if the observation changed shape mid-episode.
            raise ValueError(
                f"[{self.name}] assembled obs is {vec.size} wide, expected "
                f"{expect} for skill '{spec.skill_id}' on {self.creature}")
        return vec

    def layout(self, skill_id: str):
        """(obs_dim, proprio_indices, task_indices) for a skill on this creature."""
        hit = self._layouts.get(skill_id)
        if hit is None:
            hit = R.get_spec(skill_id).layout(self._contract)
            self._layouts[skill_id] = hit
        return hit

    def _resolve_target(self, spec, frame) -> Optional[np.ndarray]:
        if spec.target_source == R.TARGET_BALL:
            # The scripted chase: aim the locomotion expert at the ball, every
            # tick. Recovered from the player's own egocentric ball observation,
            # so it needs nothing from the simulator.
            return ball_world_xy(frame)
        if spec.target_source == R.TARGET_NONE:
            return None
        if self._command.target_xy is None:
            raise SkillUnavailable(
                f"[{self.name}] skill '{spec.skill_id}' has no target")
        return np.asarray(self._command.target_xy, dtype=np.float64)

    # -- experts -----------------------------------------------------------
    def _expert(self, skill_id: str):
        hit = self._experts.get(skill_id)
        if hit is not None:
            return hit
        spec = R.get_spec(skill_id)
        path = self._overrides.get(skill_id) or spec.checkpoint_for(self.creature)
        _, p_idx, t_idx = self.layout(skill_id)
        expert = load_policy(path, proprio_indices=p_idx, task_indices=t_idx,
                             act_dim=self._contract.act_dim, device=self.device,
                             label=f"[{self.name}] skill '{skill_id}'")
        self._experts[skill_id] = expert
        self._announce(skill_id, expert)
        return expert

    def _announce(self, skill_id, expert):
        """Say once, per checkpoint, what mode this expert will run in and why."""
        if self.quiet or self.action_mode != MODE_AUTO:
            return
        key = (expert.info.path, self.action_mode)
        if key in _ANNOUNCED:
            return
        _ANNOUNCED.add(key)
        if expert.noise_driven:
            print(f"[skills] '{skill_id}' <- {expert.info.path}\n"
                  f"[skills]   {expert.info.describe()}\n"
                  f"[skills]   action std {expert.info.action_std:.2f} fills the "
                  f"[-1, 1] action range, so this checkpoint was SCORED as a "
                  f"sampled policy and its mean action does not locomote.\n"
                  f"[skills]   -> running MODE_NOISE (seed={self.seed}, "
                  f"player={self.player_index}); still bit-exact for replay. "
                  f"Pass action_mode='mean' to override.", flush=True)
        else:
            print(f"[skills] '{skill_id}' <- {expert.info.path} "
                  f"({expert.info.describe()}) -> MODE_MEAN", flush=True)

    def __repr__(self):
        return (f"<SkillController {self.name} {self._contract.describe()} "
                f"skill={self.skill_id} target={self._command.target_xy if self._command else None}>")


class SkillControllerPool:
    """One controller per player slot, for the 2v2 game.

    Thin on purpose: expert weights are already shared through the module-level
    checkpoint cache, so this only keeps the per-slot controllers together and
    steps them in the order `env.step()` expects (home players first).
    """

    def __init__(self, creatures: Sequence[str], **kwargs):
        names = kwargs.pop("names", None)
        kwargs.pop("player_index", None)   # assigned per slot, below
        self.controllers = [
            SkillController(c, player_index=i,
                            name=(names[i] if names else f"player_{i}"), **kwargs)
            for i, c in enumerate(creatures)
        ]

    def __len__(self):
        return len(self.controllers)

    def __getitem__(self, i) -> SkillController:
        return self.controllers[i]

    def __iter__(self):
        return iter(self.controllers)

    def set_command(self, i: int, skill_id: str, target_xy=None):
        return self.controllers[i].set_command(skill_id, target_xy)

    def act(self, frames: Sequence[PlayerFrame]):
        """One `SkillOutput` per player, in env order."""
        if len(frames) != len(self.controllers):
            raise ValueError(
                f"{len(frames)} frames for {len(self.controllers)} controllers")
        return [c.act(f) for c, f in zip(self.controllers, frames)]

    def actions(self, frames: Sequence[PlayerFrame]):
        """The list `env.step()` wants."""
        return [o.action for o in self.act(frames)]

    def reset(self):
        for c in self.controllers:
            c.reset()
