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

import mujoco
import numpy as np

from rower_soccer.game import recording as rec
from rower_soccer.game.recording import DemoMeta, DemoWriter, PlayerMeta

SLOTS = ("home_1", "home_2", "away_1", "away_2")   # index == dm_soccer player index

# Seconds between unflips of the same player. Long enough that righting yourself
# is a recovery, not a physics exploit (an uncooled unflip is a free "stand
# perfectly still" button, since it also zeroes your velocity).
UNFLIP_COOLDOWN = 5.0
# Same drop height the warp drills spawn at (SceneMeta.spawn_z): high enough
# that the righted creature settles onto its feet instead of clipping the floor.
UNFLIP_DROP_Z = 0.75
CONTROL_DT = 0.025          # 40 Hz, the rate every drill trained at
PHYSICS_DT = 0.0025         # 10 substeps, matching the drills (soccer's native is 0.005)

PHASE_LOBBY, PHASE_COUNTDOWN, PHASE_PLAYING, PHASE_ENDED = (
    "lobby", "countdown", "playing", "ended")


# -- chase camera + marker constants ---------------------------------------
CHASE_BACK = 6.0        # metres behind the player, along its attacking axis
CHASE_UP = 4.0          # metres above it -> ~34 deg downtilt
CHASE_FOVY = 55.0       # degrees; a game-like field of view, not a pitch-fitting one
MARKER_RADIUS = 0.45    # a touch wider than the old sphere: flat discs read smaller
MARKER_HALF_H = 0.012   # flush with the turf -- it must never occlude the ball
BALLCAM_BACK = 9.0      # metres behind the ball, along -y
BALLCAM_UP = 5.5        # metres above it -> ~31 deg downtilt
BALLCAM_FOVY = 50.0     # close enough that the 0.35 m ball is clearly visible
# Dashes are a FIXED world size with a FIXED gap, and a longer line simply uses
# more of them. Stretching a fixed COUNT across the span instead makes a long
# line read as solid and a short one as a few stubs -- the dash pattern has to
# be a property of the pitch, not of the distance.
DASH_N = 26             # the most a line can use; surplus are parked
DASH_LEN = 0.16         # half-length of one dash (0.32 m long)
DASH_PITCH = 0.62       # centre-to-centre spacing -> a 0.30 m gap
DASH_W = 0.05           # half-width


# Per-player colours, hardcoded. These were derived from HSV before, which was
# a mistake: hue is not a uniform axis, and the same +/-12 degree step that gave
# two clearly different blues gave one red and one ORANGE, because the
# red-to-orange transition is compressed right where the red team sits. Two hex
# codes chosen by eye beat a formula that has to be corrected per team.
# Chosen against the actual renderer (scripts/color_picker.py), not from a
# swatch: the four lights, the material's specular, and the saturated green turf
# all shift what a hex code looks like on a creature. #faaca5 is a pale pink in
# a picker and reads as YELLOW on an ant, which is how these ended up being
# eyeballed in-engine rather than derived.
PLAYER_HEX = ("#598eff", "#59d0ff",     # home
              "#e65b2d", "#b03636")     # away


def marker_rgba(i, alpha=0.9):
    """Player i's colour as rgba floats.

    Used for its target disc, its aim-line dashes, AND the creature itself, so
    your ant and your marks all match and your teammate's are visibly different.

    Note for anyone changing creature colour: each walker carries its own
    material (`creature_N/self`) as well as per-geom rgba, and where a geom has
    a material MuJoCo renders the MATERIAL. Painting only `geom.rgba` leaves
    every creature washed toward the material's tan -- measured: two players
    whose colours differ by 60 rgb points both rendered as the same orange.
    `_build_scene` sets both.
    """
    h = PLAYER_HEX[i % len(PLAYER_HEX)].lstrip("#")
    r, g, b = (int(h[k:k + 2], 16) / 255.0 for k in (0, 2, 4))
    return [r, g, b, alpha]


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
        # Per-player views are rendered at HALF linear resolution. Render cost is
        # ~pixel-linear, so four of these cost about what one full-size frame
        # does -- which is what keeps 20 Hz on the CPU rasteriser. Measured:
        # 31 ms for one 960x640 frame, budget 50 ms at 20 Hz.
        self.chase_w, self.chase_h = self.render_w // 2, self.render_h // 2

        team = (creature, creature)
        # time_limit is huge on purpose: the MATCH clock is ours, so a goal (which
        # MultiturnTask absorbs by re-spawning) and the 45 s whistle are decisions
        # this class makes, not side effects of composer truncation.
        self.env = B.make_soccer_env(home_team=team, away_team=team,
                                     time_limit=1e6, random_state=self.seed,
                                     terminate_on_goal=False,
                                     # The ball the skills trained on (r=0.15) —
                                     # the stock 0.35 ball is a different task.
                                     ball=B.drill_ball())
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
        # Wall-clock stamps for the unflip cooldown; -inf so the first one is free.
        self._unflip_at = [float("-inf")] * self.n_players

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

        # Broadcast camera: the TV main-camera position — halfway line, behind
        # the -y touchline, elevated, looking slightly above the centre spot
        # (~28 deg downtilt). Fixed, not ball-tracking: a static camera keeps
        # click->world a pure function of (u, v), so drag input keeps working in
        # this view via the ray-ground intersection in uv_to_world.
        bpos = np.array([0.0, -2.0 * py, 1.1 * py])
        blook = np.array([0.0, 0.0, 0.4])
        fwd = blook - bpos; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
        bup = np.cross(right, fwd)
        # Fit by construction: project every pitch corner into the camera frame
        # and take the fovy that keeps them all inside, with 6% margin. Fitting
        # a guessed margin at the centre-spot distance instead kept clipping the
        # NEAR corners, which subtend the widest angles in this view.
        Rb = np.stack([right, bup, -fwd], axis=1)
        need = 0.0
        for cx in (-px, px):
            for cy in (-py, py):
                pc = Rb.T @ (np.array([cx, cy, 0.0]) - bpos)
                need = max(need, abs(pc[0]) / -pc[2] / aspect,
                           abs(pc[1]) / -pc[2])
        bfovy = 2.0 * math.atan(1.06 * need)
        wb.add("camera", name="broadcast", pos=bpos.tolist(),
               xyaxes=[*right.tolist(), *bup.tolist()],
               fovy=math.degrees(bfovy))
        # Everything uv<->world needs for the perspective camera, precomputed.
        # Columns of R are the camera's world-frame x/y/z axes (z looks BACKWARD,
        # MuJoCo convention), so R @ d_cam is camera->world.
        self._bcast = dict(pos=bpos,
                           R=np.stack([right, bup, -fwd], axis=1),
                           tanf=math.tan(bfovy / 2.0), aspect=aspect)
        # Default to the ball camera: it is close enough that the 0.35 m ball
        # and the ants are both clearly visible, which the pitch-fitting topdown
        # is not. Players can cycle to topdown/broadcast.
        self.camera = "playercam"
        # Which seat `playercam` follows. The server points this at the human's
        # slot; with one human that is one render per tick, which is what makes
        # a per-player view affordable at all (four of them cost 112 ms against
        # a 50 ms budget -- see the render-cost note on the chase cameras).
        self.player_view = 0
        if not shadows:
            # The pitch ships 4 lights each rendering an 8192x8192 shadowmap: ~90 ms
            # a frame, a fixed cost that dwarfs the physics step and would cap the
            # stream near 9 FPS. Purely cosmetic from straight overhead.
            for light in self.arena.mjcf_model.find_all("light"):
                light.castshadow = "false"
            self.arena.mjcf_model.visual.quality.offsamples = 0

        # -- per-player CHASE cameras -------------------------------------
        # `mode="track"` translates the camera with the body and leaves its
        # ORIENTATION fixed. That is deliberate and not a shortcut: a camera
        # locked to the ant's heading would spin with its yaw, and an ant yaws
        # constantly -- unwatchable. A fixed downward tilt with the player's
        # attacking direction up the screen is what a third-person sports
        # camera actually does.
        #
        # Because only the POSITION moves, and it moves exactly with the body,
        # `uv_to_world` for a chase view is the broadcast ray maths with the
        # camera origin offset by the tracked player's xy. Nothing else about
        # the click->world path changes.
        self._chase = []
        self._chase_specs = []
        for i, slot in enumerate(SLOTS[:len(self.task.players)]):
            # home attacks +y, away attacks -y (SLOTS order is home, home,
            # away, away). Sit BEHIND the player relative to that direction.
            sgn = 1.0 if slot.startswith("home") else -1.0
            off = np.array([0.0, -sgn * CHASE_BACK, CHASE_UP])
            fwd = np.array([0.0, sgn * CHASE_BACK, -CHASE_UP])
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            spec = wb.add("camera", name=f"chase_{slot}", pos=off.tolist(),
                          xyaxes=[*right.tolist(), *up.tolist()],
                          fovy=CHASE_FOVY)
            self._chase_specs.append(spec)
            tanf = math.tan(math.radians(CHASE_FOVY) / 2.0)
            self._chase.append(dict(off=off,
                                    R=np.stack([right, up, -fwd], axis=1),
                                    tanf=tanf, aspect=self.chase_w / self.chase_h))

        # -- the SHARED chase camera: tracks the BALL ------------------------
        # One render per tick, so it fits the 20 Hz budget where four per-player
        # views do not (measured: a render costs ~20-27 ms almost independently
        # of resolution -- the cost is scene construction and the render-executor
        # round trip, not rasterisation, so shrinking the views buys nothing).
        # Same fixed-orientation `track` trick as the player cams: it follows the
        # ball around the pitch without ever rotating, which keeps the picture
        # readable and keeps click->world a ray with a moving origin.
        bfwd = np.array([0.0, BALLCAM_BACK, -BALLCAM_UP])
        bfwd /= np.linalg.norm(bfwd)
        bright = np.cross(bfwd, [0.0, 0.0, 1.0]); bright /= np.linalg.norm(bright)
        bup2 = np.cross(bright, bfwd)
        self._ballcam_off = np.array([0.0, -BALLCAM_BACK, BALLCAM_UP])
        wb.add("camera", name="ballcam", pos=self._ballcam_off.tolist(),
               xyaxes=[*bright.tolist(), *bup2.tolist()],
               fovy=BALLCAM_FOVY)
        self._ballcam = dict(off=self._ballcam_off,
                             R=np.stack([bright, bup2, -bfwd], axis=1),
                             tanf=math.tan(math.radians(BALLCAM_FOVY) / 2.0),
                             aspect=aspect)

        # -- the creatures wear their own target's colour --------------------
        # Both members of a team shipped the same rgba, so on screen you could
        # see which disc was yours but not which ANT was. Painting every geom of
        # player i with marker hue i makes the pairing unambiguous: your ant and
        # your disc are the same colour, your partner's are a visibly different
        # shade of the same team colour.
        for i, pl in enumerate(self.task.players):
            rgba = marker_rgba(i, alpha=1.0)
            mjcf = pl.walker.mjcf_model
            # The MATERIAL, not just the geoms. Each walker carries its own
            # `self` material (rgba 0.8 0.6 0.4, a tan), and where a geom has a
            # material MuJoCo renders the material -- so painting only
            # `geom.rgba` left every creature washed toward tan. Measured
            # before this fix: the two away players, whose colours are red
            # (255,79,71) and crimson (199,56,70), both rendered as orange
            # (255,153,75) and (254,127,70) -- indistinguishable, and the wrong
            # hue. Both are set now, so there is no rule to get wrong.
            for mat in mjcf.find_all("material"):
                mat.rgba = rgba
            for g in mjcf.find_all("geom"):
                g.rgba = rgba

        # -- target markers: a DISC on the pitch, not a floating sphere ------
        # A sphere at z = 0.35 sits exactly where the ball is and hides it,
        # which is the whole complaint. A thin cylinder lies flush with the
        # turf: it can never occlude anything, and "my target is there" reads
        # instantly because it is drawn on the same plane you are aiming at.
        self._marker_specs = []
        self._dash_specs = []
        for i, slot in enumerate(SLOTS[:len(self.task.players)]):
            self._marker_specs.append(wb.add(
                "geom", name=f"target_{slot}", type="cylinder",
                size=[MARKER_RADIUS, MARKER_HALF_H],
                rgba=marker_rgba(i), contype=0, conaffinity=0, mass=1e-6))
            # The aim line, as REAL GEOMETRY lying on the pitch rather than a
            # 2-D canvas overlay. An overlay is drawn in screen space: constant
            # pixel width at any distance, painted over the creatures instead of
            # behind them, and not actually on the ground plane it claims to
            # describe. A row of flat boxes is perspective-correct, occludes
            # properly, and matches the disc it points at.
            row = [wb.add("geom", name=f"dash_{slot}_{k}", type="box",
                          size=[DASH_LEN, DASH_W, MARKER_HALF_H],
                          rgba=marker_rgba(i, 0.75),
                          contype=0, conaffinity=0, mass=1e-6)
                   for k in range(DASH_N)]
            self._dash_specs.append(row)

    def _rebind(self):
        """dm_control recompiles the model on reset, replacing `physics`.  Every
        handle derived from it (camera id, marker bindings, walker bindings) must be
        re-fetched or the picture and the root poses silently refer to a dead model."""
        ph = self.env.physics
        self.physics = ph
        self.cam_id = next(i for i in range(ph.model.ncam)
                           if (ph.model.camera(i).name or "").endswith("topdown"))
        self.bcast_cam_id = next(i for i in range(ph.model.ncam)
                                 if (ph.model.camera(i).name or "").endswith("broadcast"))
        self._markers = [ph.bind(g) for g in self._marker_specs]
        self._dashes = [[ph.bind(g) for g in row] for row in self._dash_specs]
        self.ballcam_id = next(i for i in range(ph.model.ncam)
                               if (ph.model.camera(i).name or "").endswith("ballcam"))
        self.chase_cam_ids = [
            next(i for i in range(ph.model.ncam)
                 if (ph.model.camera(i).name or "").endswith(f"chase_{slot}"))
            for slot in SLOTS[:len(self.task.players)]]
        self._roots = [ph.bind(p.walker.root_body) for p in self.task.players]
        if self.controller is not None:
            self.controller.bind(self.env)

    def _reset_env(self):
        self.timestep = self.env.reset()
        self._rebind()
        self._goal_latch = False

    # -- geometry ----------------------------------------------------------
    CAMERAS = ("playercam", "ballcam", "topdown", "broadcast")

    def set_camera(self, name):
        if name not in self.CAMERAS:
            raise ValueError(f"unknown camera {name!r}")
        self.camera = name

    def _chase_origin(self, view):
        """Where player `view`'s tracking camera actually is, right now.

        `mode="track"` puts the camera at the tracked body's position plus the
        static offset, keeping orientation fixed -- so the ray maths is the
        broadcast case with a moving origin. Reading the live root position
        (not `model.cam_pos`, which still holds the authored offset) is what
        makes a click land where the player is looking rather than where they
        spawned.
        """
        c = self._chase[view]
        p = np.array(self._roots[view].xpos, dtype=float)
        return np.array([p[0] + c["off"][0], p[1] + c["off"][1], c["off"][2]])

    def uv_to_world(self, u, v, view=None):
        """Click -> world xy. `view=i` uses player i's chase camera."""
        if view is not None:
            c = self._chase[view]
            pos = self._chase_origin(view)
            d_cam = np.array([(u * 2.0 - 1.0) * c["tanf"] * c["aspect"],
                              (1.0 - v * 2.0) * c["tanf"], -1.0])
            d = c["R"] @ d_cam
            d2 = min(d[2], -1e-3)
            t = -pos[2] / d2
            w = pos + t * np.array([d[0], d[1], d2])
            px, py = self.pitch_half
            return (float(np.clip(w[0], -1.05 * px, 1.05 * px)),
                    float(np.clip(w[1], -1.05 * py, 1.05 * py)))
        return self._uv_to_world_fixed(u, v)

    def _ballcam_origin(self):
        bpos, _ = self.task.ball.get_pose(self.physics)
        o = self._ballcam["off"]
        return np.array([float(bpos[0]) + o[0], float(bpos[1]) + o[1], o[2]])

    def _ray_to_ground(self, c, pos, u, v):
        """Shared by every MOVING camera: pixel -> ray -> the z = 0 plane."""
        d_cam = np.array([(u * 2.0 - 1.0) * c["tanf"] * c["aspect"],
                          (1.0 - v * 2.0) * c["tanf"], -1.0])
        d = c["R"] @ d_cam
        d2 = min(d[2], -1e-3)               # a sky click still lands, far away
        w = pos + (-pos[2] / d2) * np.array([d[0], d[1], d2])
        px, py = self.pitch_half
        return (float(np.clip(w[0], -1.05 * px, 1.05 * px)),
                float(np.clip(w[1], -1.05 * py, 1.05 * py)))

    def _uv_to_world_fixed(self, u, v):
        if self.camera == "ballcam":
            return self._ray_to_ground(self._ballcam, self._ballcam_origin(), u, v)
        if self.camera == "playercam":
            i = self.player_view
            return self._ray_to_ground(self._chase[i], self._chase_origin(i), u, v)
        """Normalized click (u, v in [0, 1], origin top-left of the frame) -> world xy.

        topdown: image-right -> world +x, image-UP -> world +y, a pure affine.
        broadcast: cast the pixel ray from the perspective camera onto the z=0
        ground plane. A click at or above the horizon has no ground intersection;
        it is clamped just below so the ray still lands, far, and the result is
        then clamped to the pitch (a sky click means "way over there", not NaN).
        """
        if self.camera == "broadcast":
            c = self._bcast
            d_cam = np.array([(u * 2.0 - 1.0) * c["tanf"] * c["aspect"],
                              (1.0 - v * 2.0) * c["tanf"], -1.0])
            d = c["R"] @ d_cam
            d2 = min(d[2], -1e-3)                 # keep the ray pointing down
            t = -c["pos"][2] / d2
            w = c["pos"] + t * np.array([d[0], d[1], d2])
            px, py = self.pitch_half
            return (float(np.clip(w[0], -1.05 * px, 1.05 * px)),
                    float(np.clip(w[1], -1.05 * py, 1.05 * py)))
        return (float((u * 2.0 - 1.0) * self.half_x),
                float((1.0 - v * 2.0) * self.half_y))

    def world_to_uv(self, x, y, z=0.0, view=None):
        if view is None and self.camera == "playercam":
            view = self.player_view
        if view is None and self.camera == "ballcam":
            c = self._ballcam
            p = c["R"].T @ (np.array([x, y, z]) - self._ballcam_origin())
            if p[2] > -1e-6:
                return (-1.0, -1.0)
            return (float((p[0] / -p[2] / (c["tanf"] * c["aspect"]) + 1.0) * 0.5),
                    float((1.0 - p[1] / -p[2] / c["tanf"]) * 0.5))
        if view is not None:
            c = self._chase[view]
            p = c["R"].T @ (np.array([x, y, z]) - self._chase_origin(view))
            if p[2] > -1e-6:
                return (-1.0, -1.0)
            return (float((p[0] / -p[2] / (c["tanf"] * c["aspect"]) + 1.0) * 0.5),
                    float((1.0 - p[1] / -p[2] / c["tanf"]) * 0.5))
        if self.camera == "broadcast":
            c = self._bcast
            p = c["R"].T @ (np.array([x, y, z]) - c["pos"])
            if p[2] > -1e-6:                       # behind the camera
                return (-1.0, -1.0)
            return (float((p[0] / -p[2] / (c["tanf"] * c["aspect"]) + 1.0) * 0.5),
                    float((1.0 - p[1] / -p[2] / c["tanf"]) * 0.5))
        return (float((x / self.half_x + 1.0) * 0.5), float((1.0 - y / self.half_y) * 0.5))

    # -- match lifecycle ---------------------------------------------------
    def start_match(self, demo_path=None, meta_extra=None):
        self.match_id = uuid.uuid4().hex[:12]
        self.score = [0, 0]
        self.tick = 0
        self._reset_env()
        for c in self.commands:
            # A human keeps whatever they armed in the lobby -- pressing "follow"
            # while waiting and having it silently forgotten at kickoff is the kind
            # of thing that makes a game feel broken. Filled seats are re-armed.
            if c.controller != "human":
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
        # Close the file BEFORE announcing the end. `phase == "ended"` is what every
        # watcher (the clients, the CI self test) waits on before reading the demo,
        # and compressing a 45 s match takes long enough that flipping the flag
        # first hands them a half-written file.
        if self.writer is not None:
            self.last_demo = self.writer.close()
            self.writer = None
        self.phase = PHASE_ENDED
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

    # -- unflip ------------------------------------------------------------
    def unflip(self, p, force=False):
        """Stand player p's creature upright where it is.

        Keeps the ground position and the heading (yaw extracted from the root
        frame), sets the root upright at the drill spawn height, and zeroes the
        root velocity — then physics takes over and the creature drops onto its
        feet. There is no trained get-up policy; this is a game action, the
        digital equivalent of a referee standing your beetle back up.

        Deterministic given the current state, which is what lets a demo replay
        reproduce it: replay_actions re-applies the recorded `unflip` events at
        their recorded ticks and gets bit-identical state back. `force=True` is
        for that replay path — it skips the anti-spam cooldown, which is wall
        clock and therefore meaningless in a resimulation.

        Returns (ok, reason)."""
        if not 0 <= p < self.n_players:
            return False, f"no player {p}"
        now = time.monotonic()
        if not force and now - self._unflip_at[p] < UNFLIP_COOLDOWN:
            wait = UNFLIP_COOLDOWN - (now - self._unflip_at[p])
            return False, f"unflip cooling down ({wait:.1f}s)"
        self._unflip_at[p] = now

        root = self._roots[p]
        m = np.asarray(root.xmat, np.float64).reshape(3, 3)
        yaw = math.atan2(m[1, 0], m[0, 0])
        quat = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
        pos = np.array([root.xpos[0], root.xpos[1], UNFLIP_DROP_Z])
        walker = self.task.players[p].walker
        walker.set_pose(self.physics, position=pos, quaternion=quat)
        walker.set_velocity(self.physics, velocity=np.zeros(3),
                            angular_velocity=np.zeros(3))
        self.physics.forward()
        self._emit("unflip", player=p, slot=SLOTS[p],
                   pos=[float(pos[0]), float(pos[1])], yaw=float(yaw))
        return True, "up"

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

        # The ball's WORLD pose, read once and handed to every player. The skills
        # need it in world coordinates rather than from `ball_ego_position`, which
        # dm_soccer expresses in MuJoCo's inertial frame -- an axis permutation away
        # from the body frame the drills trained in. See GameSkillLayer.act.
        ball = (self.task.ball.get_pose(self.physics)[0],
                self.task.ball.get_velocity(self.physics)[0])

        outs, actions = [], []
        for p in range(self.n_players):
            cmd = self.commands[p]
            skill = cmd.skill if self.phase == PHASE_PLAYING else "idle"
            if self.controller is None:
                out = None
                actions.append(np.zeros(self.act_dim, np.float32))
            else:
                out = self.controller.act(p, obs_list[p], skill, cmd.target, cmd.aim,
                                          ball=ball)
                actions.append(out.action)
            outs.append(out)
            # Show the human where their creature is actually being sent (for
            # `scripted` that is the ball, which is the point of showing it).
            tgt = out.target if out is not None else cmd.target
            self._markers[p].pos = np.array([tgt[0], tgt[1], MARKER_HALF_H])
            self._place_dashes(p, tgt)

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

    def _place_dashes(self, p, tgt):
        """Lay player p's aim line on the pitch, creature -> target.

        Dashes are spaced along the line and rotated to face down it. Surplus
        dashes (a short line does not need all of them) are PARKED BELOW THE
        FLOOR rather than resized: a zero-size geom is a degenerate box and
        MuJoCo's renderer is entitled to do anything with it, whereas a box at
        z = -1 is simply not visible.
        """
        src = np.asarray(self._roots[p].xpos)[:2]
        d = np.asarray(tgt, dtype=float) - src
        dist = float(np.linalg.norm(d))
        row = self._dashes[p]
        if dist < 1e-3:
            for g in row:
                g.pos = np.array([0.0, 0.0, -1.0])
            return
        u = d / dist
        c, s_ = float(u[0]), float(u[1])
        # Rotation about z only: the line lies flat, so the box's local x runs
        # along the line and its local y stays across it.
        quat = np.array([math.cos(math.atan2(s_, c) / 2.0), 0.0, 0.0,
                         math.sin(math.atan2(s_, c) / 2.0)])
        # Start clear of the creature so the line does not stab through it, and
        # stop short of the disc so the two read as separate marks.
        t0, t1 = min(0.55, 0.3 * dist), max(dist - MARKER_RADIUS, 0.0)
        for k, g in enumerate(row):
            t = t0 + DASH_PITCH * (k + 0.5)
            if t > t1:
                g.pos = np.array([0.0, 0.0, -1.0])
                continue
            g.pos = np.array([src[0] + u[0] * t, src[1] + u[1] * t,
                              MARKER_HALF_H])
            g.quat = quat

    # -- rendering ---------------------------------------------------------
    def _aim_moving_cameras(self):
        """Put the tracking cameras where `uv_to_world` says they are.

        MuJoCo's `mode="track"` follows the camera's PARENT body, and these are
        worldbody cameras with no parent -- so they silently never moved, while
        the ray maths assumed they did. The picture and the click coordinates
        then disagreed, which is the worst version of this bug because both
        halves look fine alone. Writing `cam_pos` from the SAME origin function
        the raycast uses makes them one thing by construction.
        """
        cp = self.physics.model.cam_pos
        cp[self.ballcam_id] = self._ballcam_origin()
        for i, cid in enumerate(self.chase_cam_ids):
            cp[cid] = self._chase_origin(i)
        # `cam_pos` is MODEL data; the renderer reads `data.cam_xpos`, which is
        # derived from it by `mj_camlight` (NOT by mj_kinematics -- checked).
        # Without this the camera moves in the model and the picture does not,
        # which is how the first version of this looked correct in every number
        # and wrong in every frame.
        mujoco.mj_camlight(self.physics.model.ptr, self.physics.data.ptr)

    def render(self, view=None):
        self._aim_moving_cameras()
        """`view=None` -> the shared spectator frame; `view=i` -> player i's
        chase camera, at the smaller per-player size."""
        if view is not None:
            # SAME size as the shared frame, deliberately. Rendering per-player
            # views smaller re-sizes the offscreen framebuffer on every tick
            # (dm_control builds a Camera per call), and that mixed-size churn
            # SEGFAULTED the server the first time two humans held seats -- the
            # only moment a second size is ever requested. On EGL a full-size
            # render is 2.2 ms, so the smaller size was buying nothing anyway;
            # it was a leftover from the CPU path where it was meant to buy a
            # 4x saving that measurement later showed does not exist.
            return self.physics.render(camera_id=self.chase_cam_ids[view],
                                       width=self.render_w, height=self.render_h)
        if self.camera == "playercam":
            return self.physics.render(
                camera_id=self.chase_cam_ids[self.player_view],
                width=self.render_w, height=self.render_h)
        cam = {"broadcast": self.bcast_cam_id,
               "ballcam": self.ballcam_id}.get(self.camera, self.cam_id)
        return self.physics.render(camera_id=cam,
                                   width=self.render_w, height=self.render_h)

    # -- client-side rendering feed ----------------------------------------
    # The server streams STATE and the browser draws it, instead of the server
    # rasterising pixels and streaming those. That removes the ceiling the
    # MJPEG path has: a render costs 20-27 ms almost regardless of resolution
    # (the cost is scene construction, not pixels), so per-player views were
    # unaffordable server-side and are free client-side -- each browser draws
    # its own camera on its own GPU. Payload is 90 geoms x 7 floats = 2.5 kB a
    # tick, ~100 kB/s at 40 Hz.

    def scene_description(self):
        """Everything static the browser needs to BUILD the scene, sent once.

        Geometry only -- no poses. MuJoCo geom types map onto three.js
        primitives directly (0 plane, 2 sphere, 3 capsule, 5 cylinder, 6 box),
        which is why this ships primitives rather than a mesh export: the whole
        scene is 90 shapes.
        """
        m = self.physics.model
        geoms = []
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m.ptr, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            geoms.append(dict(
                i=g, name=name, type=int(m.geom_type[g]),
                size=[round(float(x), 5) for x in m.geom_size[g]],
                rgba=[round(float(x), 4) for x in m.geom_rgba[g]],
                body=int(m.geom_bodyid[g]),
                # A geom on the world body with a fixed pose never moves, so the
                # client can skip it in the per-tick update. The target discs
                # are world geoms that DO move, so they are not exempt -- the
                # flag is advisory and the stream still carries every geom.
                static=bool(m.geom_bodyid[g] == 0 and not name.startswith("target_")),
            ))
        px, py = self.pitch_half
        return dict(geoms=geoms, pitch_half=[px, py],
                    n_players=self.n_players, slots=list(SLOTS[:self.n_players]),
                    player_colors=[marker_rgba(i, 1.0) for i in range(self.n_players)],
                    chase=dict(back=CHASE_BACK, up=CHASE_UP, fovy=CHASE_FOVY),
                    ball_radius=float(self.task.ball.geom.size[0]))

    def pose_frame(self):
        """`[tick, t, geom_xpos(3N), geom_xmat(9N)]` as float32 bytes.

        xmat rather than a quaternion: MuJoCo already has the matrix, three.js
        takes one directly, and converting here would be a chance to get a
        handedness convention wrong for no saving worth having.
        """
        d = self.physics.data
        head = np.array([self.tick, self.match_time], dtype=np.float32)
        return (head.tobytes()
                + np.asarray(d.geom_xpos, dtype=np.float32).tobytes()
                + np.asarray(d.geom_xmat, dtype=np.float32).tobytes())

    def snapshot(self):
        """Small JSON-able state for the clients (positions in normalized uv, so the
        browser can draw an overlay on top of the stream without knowing the affine)."""
        players = []
        for p in range(self.n_players):
            # Project at the body's real height: in the broadcast view a
            # ground-plane projection would draw the overlay at the creature's
            # shadow, not the creature. Topdown ignores z.
            xyz = self._roots[p].xpos
            u, v = self.world_to_uv(xyz[0], xyz[1], xyz[2])
            c = self.commands[p]
            tu, tv = self.world_to_uv(c.target[0], c.target[1])
            r, g, b, _ = marker_rgba(p)
            players.append(dict(slot=SLOTS[p], u=u, v=v, tu=tu, tv=tv,
                                # World xy as well as the projected uv: a
                                # client-side renderer places its own camera and
                                # needs the position, not a projection of it.
                                world=[round(float(xyz[0]), 3),
                                       round(float(xyz[1]), 3),
                                       round(float(xyz[2]), 3)],
                                color=f"#{int(255*r):02x}{int(255*g):02x}{int(255*b):02x}",
                                skill=c.skill, controller=c.controller, name=c.name))
        bpos, _ = self.task.ball.get_pose(self.physics)
        bu, bv = self.world_to_uv(bpos[0], bpos[1], bpos[2])
        return dict(phase=self.phase, tick=self.tick, t=round(self.match_time, 2),
                    time_left=round(self.time_left, 1), score=list(self.score),
                    match_id=self.match_id, players=players, camera=self.camera,
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
