"""Render a Warp env's state. Warp is ground truth; MuJoCo is only the camera.

The drills train in mujoco_warp and, as of this module, they are also SCORED and
RENDERED there. There is no dm_control transfer eval in the training loop any more.

That decision removes a whole class of lie. Warp and MuJoCo CPU do not agree --
mujoco_warp resolves contacts ~6.7x softer on byte-identical parameters, and the
worm topples chaotically, so the two diverge exponentially from identical states
(0.4 m apart within 0.6 s, open-loop, same actions). Every eval number we ever
took on the CPU drill was therefore grading a policy under physics it had never
trained in, and every video showed a body behaving differently from the one being
optimised. Scoring in Warp makes the eval report the thing we are actually
training.

Physics never leaves Warp. This module builds a SEPARATE, render-only MjModel --
same scene, plus a visible target marker -- copies Warp's qpos into it, and calls
mujoco.Renderer. No stepping, no solver, no contacts: one mj_forward to place the
geoms, then a picture. The render model's creature/ball qpos addresses are
identical to the physics model's by construction (scene.py appends the marker
last), so what you see is exactly the state that was simulated.
"""

import mujoco
import numpy as np

from rower_soccer.warp_port.scene import build_creature_scene, BallSpec


class WarpRenderer:
    """Draws world `w` of a Warp env. Reused across evals; builds its model once."""

    def __init__(self, creature_xml, has_ball, width=640, height=480,
                 distance=6.0, elevation=-20.0, azimuth=110.0,
                 topdown=False, view_half=12.0, cam_height=25.0,
                 base_xml=None, ball=None):
        self.model, self.meta = build_creature_scene(
            creature_xml, ball=(ball or BallSpec()) if has_ball else None,
            target_marker=True, base_xml=base_xml,
            topdown_cam=topdown, view_half=view_half, cam_height=cam_height)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.has_ball = has_ball

        # The target marker's free joint is the LAST one in the model.
        self.tgt_qpos = int(self.model.jnt_qposadr[self.model.njnt - 1])
        # Everything before it is the physics model's qpos, verbatim.
        self.n_phys_qpos = self.tgt_qpos

        # topdown: render from the fixed 'topdown' camera (id), so the view never
        # moves and clicks map to fixed world coords. Otherwise a tracking free cam.
        self.topdown = topdown
        if topdown:
            self.topdown_cam_id = int(mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "topdown"))
        self.cam = mujoco.MjvCamera()
        self.cam.distance = distance
        self.cam.elevation = elevation
        self.cam.azimuth = azimuth

    def frame(self, env, w=0, target_height=0.5):
        """Copy world `w` of `env` into the render model and return an RGB frame."""
        q = self.data.qpos
        q[:self.n_phys_qpos] = env.qpos[w, :self.n_phys_qpos].detach().cpu().numpy()

        # Not every env has a target. The tracking env (track_env.py) chases a
        # reference gait, not a point on the pitch, so park the marker far below
        # the floor where it cannot be seen rather than requiring every env to
        # invent a target_xy just to be renderable.
        if getattr(env, "target_xy", None) is not None:
            tx, ty = env.target_xy[w].detach().cpu().numpy()
            q[self.tgt_qpos:self.tgt_qpos + 3] = [tx, ty, target_height]
        else:
            q[self.tgt_qpos:self.tgt_qpos + 3] = [0.0, 0.0, -100.0]
        q[self.tgt_qpos + 3:self.tgt_qpos + 7] = [1.0, 0.0, 0.0, 0.0]

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if self.topdown:
            # Fixed top-down camera: the view never moves, so screen->world is stable.
            self.renderer.update_scene(self.data, camera=self.topdown_cam_id)
        else:
            # Track the creature rather than staring at the origin: at +/-10 m bounds
            # a fixed camera loses it entirely.
            self.cam.lookat[:] = self.data.xpos[self.meta.root_body]
            self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()


def eval_video(env, ac, path, renderer, fps=40, deterministic=True,
               record_joints=False):
    """Run ONE deterministic episode in the Warp env `env`, render it, and return
    (ep_reward, final_fitness). Both come from Warp -- the sim being trained in.

    deterministic=True takes the action distribution's MEAN. Never sample here: the
    action head's log_std is exploration noise, and at our entropy floor it is 0.30
    on a +/-1 action, which would swamp exactly the fine control an eval is meant to
    measure.

    record_joints additionally returns the world-0 joint trajectory [T, nu], which
    is what rower_soccer.tools.style grades. It is taken from THIS rollout rather
    than a second one so the style number costs no extra simulation and describes
    exactly the episode in the video.
    """
    import imageio
    import numpy as np
    import torch

    obs = env.reset()
    ep_rew, done, frames, q = 0.0, False, [], []
    while not done:
        with torch.no_grad():
            d = ac.dist(obs.float())
            a = (d.mean if deterministic else d.sample()).clamp(-1, 1)
        obs, r, done = env.step(a)
        ep_rew += float(r[0])
        frames.append(renderer.frame(env, w=0))
        if record_joints:
            q.append(env.qpos[0, env.jq].detach().float().cpu().numpy().copy())

    fitness = float(env.fitness()[0]) if hasattr(env, "fitness") else float("nan")
    with imageio.get_writer(path, fps=fps, quality=7) as wr:
        for f in frames:
            wr.append_data(f)
    if record_joints:
        return ep_rew, fitness, np.stack(q)
    return ep_rew, fitness
