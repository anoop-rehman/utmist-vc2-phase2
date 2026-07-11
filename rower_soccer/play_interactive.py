"""Interactive multi-model human-control mode (inference only).

One rower stands in the drill arena on an effectively infinite episode; a human
drives it like a game character, switching trained skill policies on the fly:

    Q + left-click  -> arm FOLLOW; click sets a destination, the follow policy
                       drives the rower there.
    W + left-click  -> arm DRIBBLE; click sets a destination, the dribble policy
                       dribbles the ball there.
    left-click      -> (skill already active) retarget to the new point.
    ESC             -> quit.
    (no command)    -> rower sits still (zero action).

The program never stops on its own: after a command completes the active policy
keeps running and the rower holds at the target until the next command.

Design insight: `DribbleTask` subclasses `FollowTask` and merely adds a ball +
`ball_ego` observable, so ONE compiled scene (the dribble scene) serves both
skills. Switching skill == switching which model runs and which observation
vector we build; the physics scene never changes. Each model was trained on a
flat obs vector in sorted-key order (see `drills/gym_wrap.py`); FOLLOW is exactly
the DRIBBLE key set minus `ball_ego`, so trained weights transfer unchanged.

Observation layout (rower, verified against the live scene):
    FOLLOW  : 11 keys (all except ball_ego)  -> 77
    DRIBBLE : 12 keys (all)                   -> 81   (+ ball_ego, dim 4)

Heavy deps (`stable_baselines3`, `torch`, `pygame`) are imported lazily inside
`main()` so the env/task logic in this module stays importable without them.
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
# are available.
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
        # Start static: rower sits still and the target holds until commanded.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Default creature is the worm: the only trained checkpoint we have
    # (runs_v2/follow_drill_model.pt) is a worm follow policy. Switch to rower
    # once rower weights exist. TEMPORARY, paired with ENABLE_DRIBBLE.
    p.add_argument("--creature", default="worm")
    p.add_argument("--follow-model", default="runs_v2/follow_drill_model.pt",
                   help=".pt state dict (warp_port.ppo.export_sb3_compatible layout)")
    p.add_argument("--dribble-model", default=None,
                   help="ignored while dribble is disabled (ENABLE_DRIBBLE=False)")
    p.add_argument("--window", type=int, default=800,
                   help="square render/window size in px")
    p.add_argument("--cam-height", type=float, default=25.0)
    p.add_argument("--view-half", type=float, default=10.0,
                   help="half the world-space extent framed by the top-down camera (m)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    # Lazy heavy imports so the module stays importable for smoke tests.
    import pygame

    env = make_interactive_env(
        random_state=0, creature_kind=args.creature,
        cam_height=args.cam_height, view_half=args.view_half)
    ts = env.reset()
    cam_id = resolve_topdown_camera(env.physics)

    # Key layouts off the LIVE scene (sorted order == training order).
    all_keys = sorted(ts.observation.keys())
    keys = {
        DRIBBLE: all_keys,
        FOLLOW: [k for k in all_keys if k != "ball_ego"],
    }
    spec = env.action_spec()
    act_dim = int(spec.shape[0])
    f_obs, f_prop, f_task = skill_layout(ts.observation, keys[FOLLOW])
    print(f"[obs] FOLLOW={f_obs} (proprio={len(f_prop)} task={len(f_task)} act={act_dim}) "
          f"dribble={'on' if ENABLE_DRIBBLE else 'DISABLED'}", flush=True)

    # Follow policy from a .pt state dict (load_into_sb3_policy path). The loader
    # validates the checkpoint dims against this scene and errors on mismatch.
    models = {
        FOLLOW: load_latent_policy(
            args.follow_model, f_obs, act_dim, f_prop, f_task, args.device),
    }
    if ENABLE_DRIBBLE:
        d_obs, d_prop, d_task = skill_layout(ts.observation, keys[DRIBBLE])
        models[DRIBBLE] = load_latent_policy(
            args.dribble_model, d_obs, act_dim, d_prop, d_task, args.device)

    zero_action = np.clip(
        np.zeros(spec.shape, dtype=np.float32),
        spec.minimum, spec.maximum).astype(np.float32)

    W = H = args.window
    hz = int(round(1.0 / env.task.control_timestep))

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    caption = "rower_soccer — interactive play (Q=follow" + (
        " W=dribble)" if ENABLE_DRIBBLE else ", dribble disabled)")
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()

    armed = None    # skill queued by Q/W, awaiting a click
    active = None   # skill currently driving the rower
    obs = ts.observation
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
                elif ev.key == pygame.K_w and ENABLE_DRIBBLE:
                    armed = DRIBBLE
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                wx, wy = pixel_to_world(ev.pos[0], ev.pos[1], W, H, args.view_half)
                env.task.set_command_target(env.physics, (wx, wy))
                if armed is not None:
                    active = armed
                    armed = None

        if active is None:
            action = zero_action
        else:
            obs_vec = build_obs(obs, keys[active])
            action, _ = models[active].predict(obs_vec, deterministic=True)

        ts = env.step(action)
        obs = ts.observation

        frame = env.physics.render(camera_id=cam_id, width=W, height=H)
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(hz)

    pygame.quit()


if __name__ == "__main__":
    main()
