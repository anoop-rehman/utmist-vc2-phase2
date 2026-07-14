"""Interactive multi-model human-control mode (inference only).

A human drives one creature like a game character, switching trained skill
policies on the fly:

    Q + left-click  -> arm FOLLOW; click sets a destination, the follow policy
                       drives the creature there.
    W + left-click  -> arm DRIBBLE; click sets a destination, the dribble policy
                       dribbles the ball there. (drill env only; see below)
    left-click      -> (skill already active) retarget to the new point.
    ESC             -> quit.
    (no command)    -> creature sits still (zero action).

The program never stops on its own: after a command completes the active policy
keeps running and the creature holds at the target until the next command.

Two scenes, chosen with `--env` (same UI, same drill-trained models):

  * `--env drill` (default): the drill scene (`InteractivePlayTask`, a flat floor
    + target marker + ball). `DribbleTask` subclasses `FollowTask` and merely adds
    a ball + `ball_ego`, so ONE compiled scene serves both skills; switching skill
    == switching which model runs and which obs vector we build. FOLLOW is the
    DRIBBLE key set minus `ball_ego`, so weights transfer unchanged.
  * `--env soccer`: the real soccer env (`RandomizedPitch`, 40x30, ball/walls/
    goals present but ignored). The follow model drives the creature via the
    `soccer_bridge` obs adapter, which reconstructs the exact drill FOLLOW vector
    from the soccer per-player obs (proprio copied verbatim; `target_ego`/
    `target_ego_future` synthesized from the clicked point). FOLLOW only — dribble
    on soccer needs a different `ball_ego` recompute (see docs/SOCCER_BRIDGE.md).

Observation layout (worm follow -> 41, dribble -> 45; rower 77 / 81).

Heavy deps (`stable_baselines3`, `torch`, `pygame`) are imported lazily so the
env/task logic in this module stays importable without them.
"""

import argparse
import math

import numpy as np
from dm_control import composer

from rower_soccer.drills.dribble import DribbleTask

FOLLOW = "follow"
DRIBBLE = "dribble"

# TEMPORARY feature toggle. Dribble is disabled until a trained dribble policy
# exists; with this False the W key is inert, no dribble checkpoint is required,
# and only the follow policy is loaded. Flip back to True once dribble weights
# are available. (Applies to the drill scene; dribble is never offered on soccer.)
ENABLE_DRIBBLE = False


class InteractivePlayTask(DribbleTask):
    """Adapts the dribble scene for human control.

    Adds a straight-down `topdown` camera (fovy derived from cam_height /
    view_half so floor-plane unprojection is a simple affine), runs an
    effectively infinite episode, and exposes `set_command_target()` for a
    STATIC target (`_target_vel = 0`, so the inherited `after_step` leaves the
    target fixed and `target_ego_future == target_ego`).
    """

    def __init__(self, cam_height=25.0, view_half=10.0, **kwargs):
        super().__init__(**kwargs)
        self._cam_height = float(cam_height)
        self._view_half = float(view_half)
        # Straight-down perspective camera. Its -z (view dir) points to world -z;
        # xyaxes maps image-right -> world +x, image-up -> world +y.
        fovy_deg = 2.0 * math.degrees(math.atan(self._view_half / self._cam_height))
        self._arena.mjcf_model.worldbody.add(
            "camera", name="topdown",
            pos=[0.0, 0.0, self._cam_height],
            xyaxes=[1, 0, 0, 0, 1, 0],
            fovy=fovy_deg)

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        # Start static: creature sits still and the target holds until commanded.
        self._target_vel = np.zeros(2)

    def set_command_target(self, physics, xy):
        """Point the (static) target at world xy and snap the marker there."""
        self._target_xy = np.asarray(xy, dtype=np.float64)
        self._target_vel = np.zeros(2)
        self._target.set_pose_xy(physics, self._target_xy, self._target_height)


def make_interactive_env(random_state=None, **task_kwargs):
    """Mirror of make_dribble_env but instantiating InteractivePlayTask, with an
    effectively infinite episode so we never truncate."""
    task = InteractivePlayTask(**task_kwargs)
    return composer.Environment(
        task=task,
        time_limit=1e9,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)


def build_obs(obs_dict, keys):
    """Flatten the selected observables in sorted-key order — the same contract
    as drills/gym_wrap.py's `_flatten`, restricted to the active skill's keys."""
    return np.concatenate(
        [np.asarray(obs_dict[k], dtype=np.float32).ravel() for k in keys])


def resolve_topdown_camera(physics):
    """Camera id whose name ends in 'topdown' (handles model-prefixing)."""
    model = physics.model
    for i in range(model.ncam):
        name = model.camera(i).name
        if name and name.endswith("topdown"):
            return i
    raise RuntimeError("topdown camera not found")


def pixel_to_world(px, py, w, h, view_half):
    """Affine unprojection for the straight-down camera at z=0.

    Image-right maps to world +x, image-up (top of frame) to world +y, so the
    frame center is the world origin and the frame corners are +/- view_half.
    """
    x = (px / w * 2.0 - 1.0) * view_half
    y = (1.0 - py / h * 2.0) * view_half
    return float(x), float(y)


def skill_layout(obs_dict, keys):
    """Positions (within the flattened `keys` vector) of proprio vs task
    features, matching drills/gym_wrap.py: task keys are the ones NOT prefixed
    by the walker name (no '/'). Returns (obs_dim, proprio_indices, task_indices)."""
    proprio, task, idx = [], [], 0
    for k in keys:
        size = int(np.asarray(obs_dict[k]).size)
        rng = range(idx, idx + size)
        (task if "/" not in k else proprio).extend(rng)
        idx += size
    return idx, proprio, task


def load_latent_policy(path, obs_dim, act_dim, proprio_indices, task_indices, device):
    """Load a `.pt` state dict (exported by warp_port.ppo.export_sb3_compatible)
    into a LatentActorCriticPolicy via load_into_sb3_policy. Validates the
    checkpoint's dimensions against the live env layout and raises a clear error
    on mismatch (e.g. wrong --creature)."""
    import gymnasium as gym
    import torch
    from rower_soccer.models.latent_policy import LatentActorCriticPolicy
    from rower_soccer.warp_port.ppo import load_into_sb3_policy

    sd = torch.load(path, map_location="cpu", weights_only=True)
    ck_obs = int(sd["mlp_extractor"]["p_idx"].numel() + sd["mlp_extractor"]["t_idx"].numel())
    ck_act = int(sd["log_std"].numel())
    ck_z = int(sd["mlp_extractor"]["z_proj.weight"].shape[0])
    if (ck_obs, ck_act) != (obs_dim, act_dim):
        raise ValueError(
            f"checkpoint {path} is dimensioned obs={ck_obs}/act={ck_act}, but the live "
            f"scene needs obs={obs_dim}/act={act_dim}. Wrong --creature? "
            f"(obs=41/act=2 -> worm, obs=77/act=8 -> rower)")

    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
    policy = LatentActorCriticPolicy(
        obs_space, act_space, lr_schedule=lambda _: 0.0,
        extractor_kwargs=dict(proprio_indices=list(proprio_indices),
                              task_indices=list(task_indices), z_dim=ck_z))
    policy.to(device)
    load_into_sb3_policy(policy, path)
    policy.set_training_mode(False)
    return policy


# --- scene adapters --------------------------------------------------------
# Each scene hides the env-specific plumbing so the pygame loop is env-agnostic.
# Contract: attributes `init_obs`, `cam_id`, `act_dim`, `zero_action`,
# `control_hz`, `view_half`, `skills`; methods `obs_vector(obs, skill)`,
# `policy_layout(skill)`, `set_target(world_xy)`, `step(action) -> obs`,
# `render(w, h) -> frame`.

class DrillScene:
    """The flat-floor drill scene (`InteractivePlayTask`). FOLLOW (+ DRIBBLE when
    ENABLE_DRIBBLE)."""

    def __init__(self, creature, cam_height, view_half, match_physics_dt=False,
                 render_shadows=False, device=None):
        self.view_half = float(view_half)
        self._env = make_interactive_env(
            random_state=0, creature_kind=creature,
            cam_height=cam_height, view_half=view_half)
        ts = self._env.reset()
        self.init_obs = ts.observation
        self.cam_id = resolve_topdown_camera(self._env.physics)
        all_keys = sorted(ts.observation.keys())
        self._keys = {DRIBBLE: all_keys, FOLLOW: [k for k in all_keys if k != "ball_ego"]}
        spec = self._env.action_spec()
        self.act_dim = int(spec.shape[0])
        self.zero_action = np.clip(
            np.zeros(spec.shape, np.float32), spec.minimum, spec.maximum).astype(np.float32)
        self.control_hz = int(round(1.0 / self._env.task.control_timestep))
        self.physics_dt = float(self._env.task.physics_timestep)
        self.skills = {FOLLOW} | ({DRIBBLE} if ENABLE_DRIBBLE else set())

    def policy_layout(self, skill):
        return skill_layout(self.init_obs, self._keys[skill])

    def obs_vector(self, obs, skill):
        return build_obs(obs, self._keys[skill])

    def set_target(self, world_xy):
        self._env.task.set_command_target(self._env.physics, world_xy)

    def step(self, action):
        return self._env.step(action).observation

    def render(self, w, h):
        return self._env.physics.render(camera_id=self.cam_id, width=w, height=h)


class SoccerScene:
    """The real soccer env (`RandomizedPitch`) driven by the drill FOLLOW model
    through the `soccer_bridge` obs adapter. FOLLOW only."""

    def __init__(self, creature, cam_height, view_half, match_physics_dt=True,
                 render_shadows=False, device=None):
        from rower_soccer.envs.build import make_soccer_env
        from rower_soccer import soccer_bridge as SB  # lazy: soccer_bridge imports this module

        self.view_half = float(view_half)
        self._SB = SB
        self._target = np.zeros(2)  # world target; set on click

        # One creature, no away team -> no teammate_/opponent_ keys.
        self._env = make_soccer_env(home_team=(creature,), n_away=0, time_limit=1e9)
        task = self._env.task
        arena = task.arena
        wb = arena.mjcf_model.worldbody
        fovy_deg = 2.0 * math.degrees(math.atan(self.view_half / float(cam_height)))
        wb.add("camera", name="topdown", pos=[0.0, 0.0, float(cam_height)],
               xyaxes=[1, 0, 0, 0, 1, 0], fovy=fovy_deg)
        # Rendering perf: the pitch ships 4 lights each casting an 8192x8192
        # shadowmap (~90ms/frame) + 4x MSAA — a fixed per-frame cost that dwarfs
        # the physics step and would cap the UI near ~9 FPS. Shadows/AA are purely
        # cosmetic for this top-down view, so disable them by default: render
        # drops ~100ms -> ~11ms (smooth 40 Hz). --shadows restores them.
        if not render_shadows:
            for light in arena.mjcf_model.find_all("light"):
                light.castshadow = "false"
            arena.mjcf_model.visual.quality.offsamples = 0
        self._marker = wb.add("geom", name="play_target", type="sphere", size=[0.5],
                              rgba=[1, 0.2, 0.2, 1], contype=0, conaffinity=0)
        # The follow policy trained at the drill's physics dt (0.0025, 10 substeps).
        # The soccer Task defaults to 0.005 (5 substeps); match it to training by
        # default so the creature sees the same integration it was optimized on.
        if match_physics_dt:
            task.set_timesteps(control_timestep=0.025, physics_timestep=0.0025)
        self.physics_dt = float(self._env.task.physics_timestep)

        ts = self._env.reset()
        self.init_obs = ts.observation[0]
        self.cam_id = resolve_topdown_camera(self._env.physics)
        self._follow_keys, self._prop_bases, self._task_keys = SB.reference_follow_layout(creature)
        spec = self._env.action_spec()[0]
        self.act_dim = int(spec.shape[0])
        self.zero_action = np.clip(
            np.zeros(spec.shape, np.float32), spec.minimum, spec.maximum).astype(np.float32)
        self.control_hz = int(round(1.0 / self._env.task.control_timestep))
        self.skills = {FOLLOW}  # dribble unsupported on soccer (ball_ego frame differs)

    def _drill_dict(self, obs):
        return self._SB.soccer_to_drill_follow_dict(
            obs, self._target, self._prop_bases, self._task_keys)

    def policy_layout(self, skill):
        assert skill == FOLLOW, "soccer scene supports FOLLOW only"
        return skill_layout(self._drill_dict(self.init_obs), self._follow_keys)

    def obs_vector(self, obs, skill):
        return self._SB.drill_follow_obs(
            obs, self._target, self._follow_keys, self._prop_bases, self._task_keys)

    def set_target(self, world_xy):
        self._target = np.asarray(world_xy, dtype=np.float64)
        # Move the (static, non-colliding) marker geom; recomputed at next step.
        self._env.physics.bind(self._marker).pos = np.array(
            [world_xy[0], world_xy[1], 0.5])

    def step(self, action):
        return self._env.step([action]).observation[0]  # soccer expects a list

    def render(self, w, h):
        return self._env.physics.render(camera_id=self.cam_id, width=w, height=h)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=["drill", "soccer"], default="drill",
                   help="drill floor scene (default) or the real soccer pitch")
    # Default creature is the worm: the only trained checkpoint we have
    # (runs_v2/follow_drill_model.pt) is a worm follow policy. Switch to rower
    # once rower weights exist. TEMPORARY, paired with ENABLE_DRIBBLE.
    p.add_argument("--creature", default="worm")
    p.add_argument("--follow-model", default="runs_v2/follow_drill_model.pt",
                   help=".pt state dict (warp_port.ppo.export_sb3_compatible layout)")
    p.add_argument("--dribble-model", default=None,
                   help="ignored while dribble is disabled / on soccer")
    p.add_argument("--window", type=int, default=800,
                   help="square render/window size in px")
    p.add_argument("--cam-height", type=float, default=None,
                   help="top-down camera height (default 25 drill / 32 soccer)")
    p.add_argument("--view-half", type=float, default=None,
                   help="half the world extent framed (default 10 drill / 22 soccer)")
    p.add_argument("--match-physics-dt", action=argparse.BooleanOptionalAction, default=True,
                   help="soccer: match the drill physics dt (0.0025) the policy trained on "
                        "(default on; pass --no-match-physics-dt to keep soccer's native 0.005)")
    p.add_argument("--shadows", action="store_true",
                   help="soccer: keep the pitch's shadows/MSAA (prettier but ~100ms/frame -> "
                        "laggy). Off by default for a smooth 40 Hz top-down view.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    # Lazy heavy import so the module stays importable for smoke tests.
    import pygame

    # Per-env top-down framing defaults (the pitch is bigger than the floor).
    cam_height = args.cam_height if args.cam_height is not None else (
        32.0 if args.env == "soccer" else 25.0)
    view_half = args.view_half if args.view_half is not None else (
        22.0 if args.env == "soccer" else 10.0)

    SceneCls = SoccerScene if args.env == "soccer" else DrillScene
    scene = SceneCls(creature=args.creature, cam_height=cam_height, view_half=view_half,
                     match_physics_dt=args.match_physics_dt, render_shadows=args.shadows,
                     device=args.device)

    # Load one policy per enabled skill; the loader validates dims vs the scene.
    f_obs, f_prop, f_task = scene.policy_layout(FOLLOW)
    print(f"[obs] env={args.env} creature={args.creature} FOLLOW={f_obs} "
          f"(proprio={len(f_prop)} task={len(f_task)} act={scene.act_dim}) "
          f"skills={sorted(scene.skills)} physics_dt={scene.physics_dt} "
          f"({scene.control_hz}Hz control)", flush=True)
    models = {FOLLOW: load_latent_policy(
        args.follow_model, f_obs, scene.act_dim, f_prop, f_task, args.device)}
    if DRIBBLE in scene.skills:
        d_obs, d_prop, d_task = scene.policy_layout(DRIBBLE)
        models[DRIBBLE] = load_latent_policy(
            args.dribble_model, d_obs, scene.act_dim, d_prop, d_task, args.device)

    W = H = args.window
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    dribble_hint = " W=dribble" if DRIBBLE in scene.skills else ""
    pygame.display.set_caption(
        f"rower_soccer — interactive play [{args.env}] (Q=follow{dribble_hint})")
    clock = pygame.time.Clock()

    armed = None    # skill queued by Q/W, awaiting a click
    active = None   # skill currently driving the creature
    obs = scene.init_obs
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_q:
                    armed = FOLLOW
                elif ev.key == pygame.K_w and DRIBBLE in scene.skills:
                    armed = DRIBBLE
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                scene.set_target(pixel_to_world(ev.pos[0], ev.pos[1], W, H, scene.view_half))
                if armed is not None:
                    active = armed
                    armed = None

        if active is None:
            action = scene.zero_action
        else:
            action, _ = models[active].predict(
                scene.obs_vector(obs, active), deterministic=True)

        obs = scene.step(action)

        frame = scene.render(W, H)
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(scene.control_hz)

    pygame.quit()


if __name__ == "__main__":
    main()
