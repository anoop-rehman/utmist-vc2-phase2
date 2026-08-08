"""The authoritative match: one CPU dm_control soccer env, stepped at control rate.

One env, one thread, 4 players.  Throughput is irrelevant here (a 2v2 ant match is
~9 ms/step at physics_dt 0.0025, ~2.8x realtime before rendering) -- what matters is
that this is the single source of truth: every client sees a render of THIS state,
and every input is applied to THIS state, at a tick boundary, routed by slot.

Why CPU dm_control and not warp: the game needs the real soccer pitch, goals, ball
and 2v2 observations, all of which exist here and not in the warp drill scenes.  The
known GPU->CPU contact gap (warp resolves contacts ~6.7x softer; see
docs/ANT_SPRINT_WORKSTREAMS.md) is WS5's probe; the ant is statically stable on four
legs, so it should be far less gap-sensitive than the worm this was measured on.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from rower_soccer.game import recording as rec
from rower_soccer.game.recording import DemoMeta, DemoWriter, PlayerMeta

SLOTS = ("home_1", "home_2", "away_1", "away_2")   # index == dm_soccer player index
CONTROL_DT = 0.025          # 40 Hz, the rate every drill trained at
PHYSICS_DT = 0.0025         # 10 substeps, matching the drills (soccer's native is 0.005)

PHASE_LOBBY, PHASE_COUNTDOWN, PHASE_PLAYING, PHASE_ENDED = (
    "lobby", "countdown", "playing", "ended")


@dataclass
class PlayerCommand:
    """The high level, and ONLY the high level. Humans never emit torques."""
    skill: str = "idle"
    target: np.ndarray = field(default_factory=lambda: np.zeros(2))
    aim: np.ndarray = field(default_factory=lambda: np.zeros(2))
    controller: str = "idle"        # human | scripted | idle
    name: str = ""


def register_ant():
    """Put the ant in CREATURE_XMLS without editing envs/build.py (WS5 owns that
    file and lands the same entry).  `setdefault` makes this a no-op afterwards."""
    import os
    from rower_soccer.envs import build as B
    B.CREATURE_XMLS.setdefault(
        "ant", os.path.join(B._REPO_ROOT, "creature_configs", "ant.xml"))
    return B


class MatchSim:
    def __init__(self, creature="ant", pitch_half=(15.0, 11.0), match_seconds=45.0,
                 seed=0, physics_dt=PHYSICS_DT, render_size=(960, 640),
                 controller=None, shadows=False, countdown=3.0):
        B = register_ant()
        self.creature = creature
        self.pitch_half = (float(pitch_half[0]), float(pitch_half[1]))
        self.match_seconds = float(match_seconds)
        self.seed = int(seed)
        self.physics_dt = float(physics_dt)
        self.control_dt = CONTROL_DT
        self.countdown = float(countdown)
        self.render_w, self.render_h = int(render_size[0]), int(render_size[1])

        team = (creature, creature)
        # time_limit is huge on purpose: the MATCH clock is ours, so a goal (which
        # MultiturnTask absorbs by re-spawning) and the 45 s whistle are decisions
        # this class makes, not side effects of composer truncation.
        self.env = B.make_soccer_env(home_team=team, away_team=team,
                                     time_limit=1e6, random_state=self.seed,
                                     terminate_on_goal=False)
        self.task = self.env.task
        self.arena = self.task.arena
        # RandomizedPitch samples its size from [_min_size, _max_size] every episode;
        # pinning both makes the pitch fixed AND reproducible for replay.
        self.arena._min_size = self.arena._max_size = self.pitch_half
        self.task.set_timesteps(control_timestep=self.control_dt,
                                physics_timestep=self.physics_dt)

        self._add_scene_furniture(shadows)
        self.n_players = len(self.task.players)
        self.act_dim = int(self.env.action_spec()[0].shape[0])

        self.controller = controller
        self.commands = [PlayerCommand() for _ in range(self.n_players)]

        self.phase = PHASE_LOBBY
        self.tick = 0
        self.score = [0, 0]
        self.match_id = ""
        self.writer: DemoWriter | None = None
        self.demo_path = None
        self.last_demo = None
        self._goal_latch = False
        self._pending_events = []
        self._skill_obs_w = 0

        self._reset_env()

    # -- scene -------------------------------------------------------------
    def _add_scene_furniture(self, shadows):
        """A straight-down camera with a documented screen->world affine, plus one
        target marker per player.  Same trick as warp_port/scene.py's `topdown_cam`
        and play_interactive.py's SoccerScene: a fixed orthographic-ish camera makes
        pixel->world a two-line affine instead of version-fragile camera-matrix
        internals."""
        px, py = self.pitch_half
        aspect = self.render_w / self.render_h
        # Fit the pitch with margin in BOTH axes, then let the wider axis win.
        self.half_y = max(py * 1.10, (px * 1.06) / aspect)
        self.half_x = self.half_y * aspect
        # High camera => the z=0 affine is accurate for creatures standing at z~0.75
        # (edge error = half * body_height / cam_height, ~0.2 m at 4x).
        self.cam_height = 4.0 * self.half_y
        fovy = 2.0 * math.degrees(math.atan(self.half_y / self.cam_height))

        wb = self.arena.mjcf_model.worldbody
        wb.add("camera", name="topdown", pos=[0.0, 0.0, self.cam_height],
               xyaxes=[1, 0, 0, 0, 1, 0], fovy=fovy)
        if not shadows:
            # The pitch ships 4 lights each rendering an 8192x8192 shadowmap: ~90 ms
            # a frame, a fixed cost that dwarfs the physics step and would cap the
            # stream near 9 FPS. Purely cosmetic from straight overhead.
            for light in self.arena.mjcf_model.find_all("light"):
                light.castshadow = "false"
            self.arena.mjcf_model.visual.quality.offsamples = 0

        self._marker_specs = []
        for i, slot in enumerate(SLOTS[:len(self.task.players)]):
            rgba = [0.35, 0.6, 1.0, 0.85] if slot.startswith("home") else [1.0, 0.4, 0.35, 0.85]
            self._marker_specs.append(wb.add(
                "geom", name=f"target_{slot}", type="sphere", size=[0.35],
                rgba=rgba, contype=0, conaffinity=0, mass=1e-6))

    def _rebind(self):
        """dm_control recompiles the model on reset, replacing `physics`.  Every
        handle derived from it (camera id, marker bindings, walker bindings) must be
        re-fetched or the picture and the root poses silently refer to a dead model."""
        ph = self.env.physics
        self.physics = ph
        self.cam_id = next(i for i in range(ph.model.ncam)
                           if (ph.model.camera(i).name or "").endswith("topdown"))
        self._markers = [ph.bind(g) for g in self._marker_specs]
        self._roots = [ph.bind(p.walker.root_body) for p in self.task.players]
        if self.controller is not None:
            self.controller.bind(self.env)

    def _reset_env(self):
        self.timestep = self.env.reset()
        self._rebind()
        self._goal_latch = False

    # -- geometry ----------------------------------------------------------
    def uv_to_world(self, u, v):
        """Normalized click (u, v in [0, 1], origin top-left of the frame) -> world xy.

        Straight-down camera: image-right -> world +x, image-UP -> world +y.
        """
        return (float((u * 2.0 - 1.0) * self.half_x),
                float((1.0 - v * 2.0) * self.half_y))

    def world_to_uv(self, x, y):
        return (float((x / self.half_x + 1.0) * 0.5), float((1.0 - y / self.half_y) * 0.5))

    # -- match lifecycle ---------------------------------------------------
    def start_match(self, demo_path=None, meta_extra=None):
        self.match_id = uuid.uuid4().hex[:12]
        self.score = [0, 0]
        self.tick = 0
        self._reset_env()
        for c in self.commands:
            c.skill = "scripted" if c.controller == "scripted" else "idle"
        self.writer = None
        self.demo_path = None
        if demo_path:
            self.writer = DemoWriter(demo_path, self._build_meta(meta_extra))
            self.demo_path = self.writer.path
        self.phase = PHASE_COUNTDOWN if self.countdown > 0 else PHASE_PLAYING
        self._countdown_left = self.countdown
        self._emit("match_start", seed=self.seed, pitch_half=list(self.pitch_half),
                   players=[{"slot": SLOTS[i], "controller": c.controller,
                             "name": c.name} for i, c in enumerate(self.commands)])
        return self.match_id

    def end_match(self, reason="time"):
        if self.phase == PHASE_ENDED:
            return self.demo_path
        self._emit("match_end", reason=reason, score=list(self.score))
        self.phase = PHASE_ENDED
        if self.writer is not None:
            self.last_demo = self.writer.close()
            self.writer = None
        return self.last_demo

    def abort(self):
        """Stop and keep whatever was recorded (server shutdown, everyone left)."""
        return self.end_match(reason="aborted")

    @property
    def match_time(self):
        return self.tick * self.control_dt

    @property
    def time_left(self):
        if self.phase == PHASE_COUNTDOWN:
            return self.match_seconds
        return max(0.0, self.match_seconds - self.match_time)

    def _emit(self, type_, **payload):
        ev = {"tick": int(self.tick), "t": float(self.match_time), "type": type_}
        ev.update(payload)
        self._pending_events.append(ev)
        if self.writer is not None:
            self.writer.add_event(type_, self.tick, self.match_time, **payload)
        return ev

    def drain_events(self, keep=40):
        """Take the events since the last call (the UI's news feed)."""
        out, self._pending_events = self._pending_events[-keep:], []
        return out

    # -- the tick ----------------------------------------------------------
    def step(self):
        """Advance one control tick.  Returns the per-player SkillOutputs."""
        if self.phase == PHASE_COUNTDOWN:
            self._countdown_left -= self.control_dt
            if self._countdown_left <= 0:
                self.phase = PHASE_PLAYING
        obs_list = self.timestep.observation
        # The recorded row must be the state the actions were CHOSEN from, so
        # capture it before stepping. Getting this backwards silently shifts every
        # (obs, action) pair by one tick -- which a BC run would train on happily and
        # which would make the action-replay determinism check fail by a hair.
        pre = self._capture_state() if self.writer is not None else None

        outs, actions = [], []
        for p in range(self.n_players):
            cmd = self.commands[p]
            skill = cmd.skill if self.phase == PHASE_PLAYING else "idle"
            if self.controller is None:
                out = None
                actions.append(np.zeros(self.act_dim, np.float32))
            else:
                out = self.controller.act(p, obs_list[p], skill, cmd.target, cmd.aim)
                actions.append(out.action)
            outs.append(out)
            # Show the human where their creature is actually being sent (for
            # `scripted` that is the ball, which is the point of showing it).
            tgt = out.target if out is not None else cmd.target
            self._markers[p].pos = np.array([tgt[0], tgt[1], 0.35])

        self.timestep = self.env.step(actions)
        touched = self._detect_touch()
        scored = self._detect_goal()

        if self.phase == PHASE_PLAYING:
            if self.writer is not None:
                self._record(obs_list, pre, outs, actions)
            self.tick += 1
            if self.match_time >= self.match_seconds:
                self.end_match("time")
        return outs, touched, scored

    def _capture_state(self):
        d = self.physics.data
        bpos, _ = self.task.ball.get_pose(self.physics)
        bvel, _ = self.task.ball.get_velocity(self.physics)
        return dict(
            rng=_rng_state(self.env),
            # float64: see DemoWriter.close -- this is the replay's initial condition.
            qpos=np.asarray(d.qpos, np.float64).copy(),
            qvel=np.asarray(d.qvel, np.float64).copy(),
            player_pos=np.stack([np.asarray(r.xpos, np.float32) for r in self._roots]),
            player_mat=np.stack([np.asarray(r.xmat, np.float32).reshape(9)
                                 for r in self._roots]),
            ball_pos=np.asarray(bpos, np.float32).copy(),
            ball_vel=np.asarray(bvel, np.float32).copy())

    def _detect_touch(self):
        if not self.task.ball.hit:
            return None
        last = self.task.ball.last_hit
        if last is None:
            return None
        try:
            p = self.task.players.index(last)
        except ValueError:
            return None
        self._emit("ball_touch", player=p, slot=SLOTS[p],
                   team="home" if p < 2 else "away",
                   repossessed=bool(self.task.ball.repossessed),
                   intercepted=bool(self.task.ball.intercepted))
        return p

    def _detect_goal(self):
        """Rising edge on the arena's goal detector.

        MultiturnTask re-spawns everyone the instant a goal is detected but leaves
        the detector latched until the next step's substeps clear it, so an edge
        detector (not a level read) is what counts a goal exactly once.
        """
        from dm_control.locomotion.soccer.team import Team
        g = self.arena.detected_goal()
        if g is None:
            self._goal_latch = False
            return None
        if self._goal_latch:
            return None
        self._goal_latch = True
        team = "home" if g == Team.HOME else "away"
        self.score[0 if team == "home" else 1] += 1
        last = self.task.ball.last_hit
        scorer = self.task.players.index(last) if last in self.task.players else None
        self._emit("goal", team=team, scorer=scorer,
                   scorer_slot=None if scorer is None else SLOTS[scorer],
                   score=list(self.score))
        return team

    # -- recording ---------------------------------------------------------
    def _build_meta(self, extra=None):
        obs0 = self.timestep.observation[0]
        keys, sizes = rec.obs_layout(obs0)
        ctl = self.controller
        players = []
        for i, c in enumerate(self.commands):
            players.append(PlayerMeta(index=i, slot=SLOTS[i],
                                      team="home" if i < 2 else "away",
                                      creature=self.creature, controller=c.controller,
                                      display_name=c.name, act_dim=self.act_dim))
        ck = {}
        for s, path in (getattr(ctl, "checkpoints", {}) or {}).items():
            entry = {"path": path}
            try:
                entry["sha256"] = rec.sha256_file(path)
                import os
                entry["bytes"] = os.path.getsize(path)
            except OSError:
                pass
            ck[s] = entry
        meta = DemoMeta(
            match_id=self.match_id,
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            git_sha=_git_sha(),
            seed=self.seed, control_dt=self.control_dt, physics_dt=self.physics_dt,
            time_limit=self.match_seconds, pitch_half=self.pitch_half,
            terminate_on_goal=False, rng_state=_rng_state(self.env),
            n_players=self.n_players, players=players,
            obs_keys=keys, obs_sizes=sizes,
            available_skills=list(getattr(ctl, "skills", ())),
            z_dim=int(getattr(ctl, "z_dim", 16) or 16), act_dim=self.act_dim,
            skill_obs=(ctl.skill_obs_meta() if hasattr(ctl, "skill_obs_meta") else {}),
            checkpoints=ck,
            skill_backend=str(getattr(ctl, "backend", "")),
            action_mode=str(getattr(ctl, "action_mode", "auto")),
            resolved_modes=dict(getattr(ctl, "resolved_modes", {})),
            skill_seed=int(getattr(ctl, "seed", 0)),
            camera=dict(cam_height=self.cam_height, half_x=self.half_x,
                        half_y=self.half_y, px_w=self.render_w, px_h=self.render_h),
            notes=f"skill backend={getattr(ctl, 'backend', 'none')}",
        )
        if extra:
            meta.notes = (meta.notes + " | " + str(extra)).strip()
        self._skill_obs_w = int(getattr(ctl, "max_obs_dim", 0) or 0)
        return meta

    def _record(self, obs_list, pre, outs, actions):
        w = self.writer
        m = w.meta
        if self.tick == 0:
            # The RNG must be pinned to the FIRST RECORDED tick, not to the reset:
            # the countdown ticks in between are stepped but not recorded, and a
            # throw-in during them advances the stream (see DemoMeta.rng_state).
            m.rng_state = pre["rng"]
        P, Z, O = self.n_players, m.z_dim, self._skill_obs_w
        obs = np.stack([rec.flatten_obs(obs_list[p], m.obs_keys) for p in range(P)])
        skill = np.array([rec.SKILL_INDEX[o.skill if o else "idle"] for o in outs], np.int8)
        skill_req = np.array([rec.SKILL_INDEX.get(self.commands[p].skill, 0)
                              for p in range(P)], np.int8)
        target = np.stack([np.asarray(o.target if o is not None else self.commands[p].target,
                                      np.float32).reshape(2) for p, o in enumerate(outs)])
        aim = np.stack([np.asarray(self.commands[p].aim, np.float32).reshape(2)
                        for p in range(P)])
        z = np.full((P, Z), np.nan, np.float32)
        sobs = np.full((P, O), np.nan, np.float32)
        sobs_n = np.zeros(P, np.int16)
        ctrl_tick = np.zeros(P, np.int32)
        for p, o in enumerate(outs):
            if o is None:
                continue
            if o.z is not None:
                z[p, :len(o.z)] = o.z
            v = np.asarray(o.obs_vector, np.float32).ravel()
            sobs[p, :min(len(v), O)] = v[:O]
            sobs_n[p] = min(len(v), O)
            ctrl_tick[p] = o.ctrl_tick
        row = dict(tick=self.tick, t=self.match_time, obs=obs, skill=skill,
                   skill_req=skill_req, target=target, aim=aim, z=z,
                   skill_obs=sobs, skill_obs_n=sobs_n, ctrl_tick=ctrl_tick,
                   action=np.stack(actions).astype(np.float32),
                   score=np.array(self.score, np.int16),
                   player_pos=pre["player_pos"], player_mat=pre["player_mat"],
                   ball_pos=pre["ball_pos"], ball_vel=pre["ball_vel"],
                   qpos=pre["qpos"])
        if m.store_qvel:
            row["qvel"] = pre["qvel"]
        w.record_tick(**row)

    # -- rendering ---------------------------------------------------------
    def render(self):
        return self.physics.render(camera_id=self.cam_id,
                                   width=self.render_w, height=self.render_h)

    def snapshot(self):
        """Small JSON-able state for the clients (positions in normalized uv, so the
        browser can draw an overlay on top of the stream without knowing the affine)."""
        players = []
        for p in range(self.n_players):
            xy = self._roots[p].xpos[:2]
            u, v = self.world_to_uv(xy[0], xy[1])
            c = self.commands[p]
            tu, tv = self.world_to_uv(c.target[0], c.target[1])
            players.append(dict(slot=SLOTS[p], u=u, v=v, tu=tu, tv=tv,
                                skill=c.skill, controller=c.controller, name=c.name))
        bpos, _ = self.task.ball.get_pose(self.physics)
        bu, bv = self.world_to_uv(bpos[0], bpos[1])
        return dict(phase=self.phase, tick=self.tick, t=round(self.match_time, 2),
                    time_left=round(self.time_left, 1), score=list(self.score),
                    match_id=self.match_id, players=players,
                    ball=dict(u=bu, v=bv),
                    countdown=round(max(0.0, getattr(self, "_countdown_left", 0.0)), 1))


def _rng_state(env):
    """`env.random_state.get_state()` as JSON-able data (see DemoMeta.rng_state)."""
    try:
        name, keys, pos, has_gauss, cached = env.random_state.get_state()
    except (AttributeError, ValueError):
        return []
    return [name, [int(k) for k in keys], int(pos), int(has_gauss), float(cached)]


def restore_rng(env, state):
    """Put `env.random_state` back where the demo says it was. Returns True on
    success; a demo recorded before this field existed simply has none."""
    if not state:
        return False
    name, keys, pos, has_gauss, cached = state
    env.random_state.set_state(
        (name, np.array(keys, dtype=np.uint32), int(pos), int(has_gauss),
         float(cached)))
    return True


def _git_sha():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:      # noqa: BLE001
        return ""
