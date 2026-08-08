"""`rower_soccer.skills` — the high-level SkillController (WS3).

The 2022 DeepMind football paper puts a mid-level layer between the low-level
motor controller and the game: per-skill experts that emit a latent motor
intention `z`, which one shared frozen decoder turns into torques. This package
is that layer, made runnable inside the CPU soccer env.

    given (skill_id, target_xy) + a game observation
      -> rebuild the drill's exact observation vector
      -> run that skill's expert head  -> z
      -> push z through the shared frozen decoder -> joint torques

Used by the play server today (a human picks skill + target) and, next sprint, by
BC and self-play (a policy picks them instead). Nothing in the API distinguishes
the two callers.

Quick start
-----------
    from rower_soccer.skills import SkillController, SoccerFrameSource
    from rower_soccer.skills.soccer import make_skill_soccer_env

    env = make_skill_soccer_env(home=("ant",))
    src = SoccerFrameSource(env)
    ctrl = SkillController("ant")
    ctrl.set_command("follow", target_xy=(12.0, -4.0))

    ts = env.reset()
    while not ts.last():
        out = ctrl.act(src.frame(ts, 0))     # SkillOutput(action, z, ...)
        ts = env.step([out.action])

See `demo_follow_soccer.py` for a runnable version with mid-episode skill and
target switching, and `api.py` for the data contract WS4 builds against.
"""

from rower_soccer.skills.api import (CheckpointMismatch, ObservationContractError,
                                     PlayerFrame, SkillCommand, SkillError,
                                     SkillOutput, SkillUnavailable, UnknownSkill,
                                     ego3_to_world, to_ego_xy, vec_to_ego3,
                                     world_to_ego3)
from rower_soccer.skills.contract import CreatureContract, contract_for
from rower_soccer.skills.controller import (MODE_AUTO, MODE_MEAN, MODE_NOISE,
                                            SkillController, SkillControllerPool)
from rower_soccer.skills.policy import (clear_policy_cache, load_policy,
                                        resolve_checkpoint)
from rower_soccer.skills.registry import (DEFAULT_TARGET_CLIP, PROPRIO_V1, SKILLS,
                                          SkillSpec, available_skills, get_spec,
                                          list_skills, register_skill)

__all__ = [
    # controller
    "SkillController", "SkillControllerPool",
    "MODE_AUTO", "MODE_MEAN", "MODE_NOISE",
    # data contract
    "PlayerFrame", "SkillCommand", "SkillOutput",
    # registry
    "SKILLS", "SkillSpec", "PROPRIO_V1", "DEFAULT_TARGET_CLIP",
    "register_skill", "get_spec", "list_skills", "available_skills",
    # creature contract
    "CreatureContract", "contract_for",
    # checkpoints
    "load_policy", "resolve_checkpoint", "clear_policy_cache",
    # geometry
    "to_ego_xy", "world_to_ego3", "vec_to_ego3", "ego3_to_world",
    # errors
    "SkillError", "UnknownSkill", "SkillUnavailable", "CheckpointMismatch",
    "ObservationContractError",
]


def __getattr__(name):
    # `SoccerFrameSource` pulls dm_control; keep it out of the import path for
    # callers that only want the registry or the data types.
    if name in ("SoccerFrameSource", "make_skill_soccer_env",
                "match_drill_timesteps"):
        from rower_soccer.skills import soccer
        return getattr(soccer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
