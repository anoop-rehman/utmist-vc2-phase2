"""The skill registry — one entry per skill, no adapter code per skill.

A `SkillSpec` is a config record: which observation fields, in which order, and
which checkpoint per creature. Everything else (obs assembly, proprio/task index
derivation, checkpoint validation, caching, deterministic action) is generic.

Adding `dribble` when WS1 lands it is therefore:

    SKILLS["dribble"] = replace(SKILLS["dribble"],
                                checkpoints={"ant": "runs_v2/dribble_ant_v1/best.pt"})

or, from outside this file, `register_skill(SkillSpec(...), replace=True)`.

The `fields` tuple must be the training env's exact concatenation order,
including where the task block sits. It usually is NOT "proprio then task":
`warp_port/dribble_env.py` emits `ball_ego` FIRST because it replicates
dm_control's sorted-key order and "ball_ego" sorts before "creature/...". Getting
this wrong is not a silent failure — `policy.load_policy` compares the derived
proprio/task index arrays against the checkpoint's own `p_idx`/`t_idx` buffers
and refuses to load on any difference.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional, Tuple

from rower_soccer.skills.api import SkillUnavailable, UnknownSkill
from rower_soccer.skills.fields import PROPRIO, TASK, field_width, get_field

__all__ = ["SkillSpec", "SKILLS", "PROPRIO_V1", "DEFAULT_TARGET_CLIP",
           "register_skill", "get_spec", "list_skills", "available_skills"]

# The shared low-level decoder's input contract, in warp drill-env order.
# Every skill that rides the frozen decoder MUST use this exact block, or the
# decoder is being fed a permuted version of its own input.
PROPRIO_V1: Tuple[str, ...] = (
    "bodies_pos",
    "body_height",
    "joints_pos",
    "joints_vel",
    "sensors_accelerometer",
    "sensors_gyro",
    "sensors_velocimeter",
    "touch_sensors",
    "world_zaxis",
)

# Metres. A commanded point further away than this is re-aimed to a waypoint on
# the same bearing (see `fields._target_ego`), because the expert never saw a
# distant target and its response to one is undefined.
#
# 3.2 m is not a guess: it is the top of `follow_ant_v1`'s `spawn_dist` range
# (1.07-3.22 m, from the run's config.json). The drill spawns its target inside
# that band and the creature chases it, so `|target_ego|` in training lived
# almost entirely in [0, 3.2]. The pitch is 96 x 72 m — a human clicking the far
# corner is ~15x outside it. Measured: at 3 m the ant arrives within 0.04 m; the
# first version of this constant was 10 m and the ant made almost no progress
# toward a 12 m target. Raise it only with evidence, and remember that raising it
# is the same as asking the expert to extrapolate.
DEFAULT_TARGET_CLIP = 3.2

# Where a target comes from.
TARGET_COMMAND = "command"   # the human's clicked point
TARGET_BALL = "ball"         # the ball's current world position (scripted chase)
TARGET_NONE = "none"

# What kind of policy runs.
KIND_LATENT = "latent"       # expert head -> z -> shared decoder, from a checkpoint
KIND_ZERO = "zero"           # emits zero torque; needs no checkpoint at all


@dataclass(frozen=True)
class SkillSpec:
    """Config for one skill."""

    skill_id: str
    fields: Tuple[str, ...] = ()
    #: creature kind -> checkpoint path. Relative paths resolve via
    #: `policy.resolve_checkpoint` ($VC2_CHECKPOINT_ROOT, then the repo root,
    #: then the parent checkout when running inside a git worktree).
    checkpoints: Mapping[str, str] = field(default_factory=dict)
    kind: str = KIND_LATENT
    target_source: str = TARGET_COMMAND
    #: Reuse another skill's checkpoints (the `scripted` chase runs the follow
    #: expert with the ball as its target — same weights, different target).
    weights_from: Optional[str] = None
    doc: str = ""

    # -- derived layout ----------------------------------------------------
    def proprio_fields(self) -> Tuple[str, ...]:
        return tuple(n for n in self.fields if get_field(n).role == PROPRIO)

    def task_fields(self) -> Tuple[str, ...]:
        return tuple(n for n in self.fields if get_field(n).role == TASK)

    def layout(self, contract):
        """(obs_dim, proprio_indices, task_indices) for this creature.

        Indices are absolute positions in the assembled vector, in field order —
        the same convention as `LatentExtractor`'s `p_idx`/`t_idx`, so they can be
        compared to a checkpoint's buffers element by element.
        """
        proprio, task, i = [], [], 0
        for name in self.fields:
            spec = get_field(name)
            w = field_width(name, contract)
            (proprio if spec.role == PROPRIO else task).extend(range(i, i + w))
            i += w
        return i, proprio, task

    def needs_target(self) -> bool:
        return self.target_source == TARGET_COMMAND

    def checkpoint_for(self, creature: str) -> str:
        """Checkpoint path for `creature`, following `weights_from`."""
        src = self
        seen = {self.skill_id}
        while src.weights_from is not None and creature not in src.checkpoints:
            nxt = get_spec(src.weights_from)
            if nxt.skill_id in seen:
                raise SkillUnavailable(
                    f"weights_from cycle at '{src.skill_id}'")
            seen.add(nxt.skill_id)
            src = nxt
        path = src.checkpoints.get(creature)
        if not path:
            known = sorted(src.checkpoints) or "none — not trained yet"
            raise SkillUnavailable(
                f"skill '{self.skill_id}' has no checkpoint for creature "
                f"'{creature}'. Creatures it does have: {known}. Pass "
                f"checkpoints={{'{self.skill_id}': '/path/to/best.pt'}} to "
                "SkillController, or add one with register_skill().")
        return path

    def is_available(self, creature: str) -> bool:
        if self.kind == KIND_ZERO:
            return True
        try:
            self.checkpoint_for(creature)
            return True
        except SkillUnavailable:
            return False


# --- the registry ----------------------------------------------------------

_FOLLOW_FIELDS = PROPRIO_V1 + ("target_ego", "target_ego_future")

# The v3 contract: worm_env_base's proprio-FIRST layout with 3-D task vectors
# (`_target_obs3` / `_ball_ego6`). Every *_ant_v3 checkpoint and the
# follow_ant_final line trained on it; p_idx is [0..64] for all of them, which
# `load_policy` verifies against each checkpoint's own buffers.
_FOLLOW_FIELDS_V3 = PROPRIO_V1 + ("target_ego3", "target_ego3_future")

SKILLS: Dict[str, SkillSpec] = {}


def register_skill(spec: SkillSpec, replace_existing: bool = False) -> SkillSpec:
    if spec.skill_id in SKILLS and not replace_existing:
        raise ValueError(
            f"skill '{spec.skill_id}' already registered; pass "
            "replace_existing=True to override")
    for name in spec.fields:
        get_field(name)          # raises on an unknown field name
    SKILLS[spec.skill_id] = spec
    return spec


def get_spec(skill_id: str) -> SkillSpec:
    try:
        return SKILLS[skill_id]
    except KeyError:
        raise UnknownSkill(
            f"unknown skill '{skill_id}'. Registered: {sorted(SKILLS)}") from None


def list_skills() -> Tuple[str, ...]:
    return tuple(sorted(SKILLS))


def available_skills(creature: str) -> Tuple[str, ...]:
    """Skills that can actually run for this creature right now."""
    return tuple(sorted(s for s, spec in SKILLS.items() if spec.is_available(creature)))


register_skill(SkillSpec(
    skill_id="follow",
    fields=_FOLLOW_FIELDS_V3,
    # follow_ant_final_frozen: the follow policy retrained ON the published
    # frozen decoder (runs_v2/_decoder_ant_final.pt) — the same decoder every
    # v3 drill checkpoint holds, so all four skills in this registry now share
    # one motor controller byte-for-byte. best.pt scored 589.9 on the warp eval
    # at the 3.0 m/s curriculum cap.
    #
    # The v1 lesson below still applies VERBATIM to any new checkpoint: probe
    # the symmetric-spawn fixed point and measure in the CPU soccer env before
    # trusting it in a match. The historical v1 analysis follows.
    #
    # v1: final.pt, NOT best.pt — and the difference is large.
    #
    # `best.pt` is whichever checkpoint scored highest on the deterministic WARP
    # eval (train_follow_warp.py:414): here the 55.8M-step weights, warp fitness
    # 0.981. `final.pt` is the 147M-step ones, warp fitness 0.997. Measured in
    # the CPU soccer env on the mean action, six 45-s episodes retargeting 4 m
    # every 10 s:
    #
    #     final.pt   upright 99.9%   mean end-of-leg distance 0.56 m   1/6 episodes ended a leg > 1 m out
    #     best.pt    upright 77.4%                            1.44 m   5/6
    #
    # `best.pt` also carries an outright pathology: from a left-right SYMMETRIC
    # state — the drill's canonical spawn, target dead ahead on the body x-axis —
    # its mean action is symmetric too, and the ant stands still forever (2.94 m
    # of a 3 m target after 15 s). Rotating the spawn yaw by 0.3 rad breaks the
    # symmetry and it walks (0.20 m); jittering the joints does not, because the
    # target bearing is what has to break. `final.pt` shows no such fixed point
    # at any perturbation tried.
    #
    # The lesson generalises: "best" means best on the metric the trainer
    # happened to log, in the simulator it happened to run. Measure a new
    # checkpoint in the env the game uses before pinning it here.
    checkpoints={"ant": "runs_v2/follow_ant_final_frozen/best.pt"},
    doc="Walk to a commanded world point and hold there. Trained on the shared "
        "frozen decoder, like every other skill here.",
))

register_skill(SkillSpec(
    skill_id="scripted",
    # Same fields as follow, ALWAYS: weights_from="follow" means this runs the
    # follow checkpoint, and the loader rejects a fields tuple the checkpoint
    # was not trained on (task 4-wide vs 6-wide, caught by the test suite when
    # follow moved to the v3 contract and this line initially did not).
    fields=_FOLLOW_FIELDS_V3,
    weights_from="follow",
    target_source=TARGET_BALL,
    doc="Naive chase-the-ball baseline for filling an unclaimed player slot: the "
        "follow expert, retargeted at the ball's world position every tick. It "
        "needs no NEW training — it reuses whatever locomotion checkpoint the "
        "creature already has. If not even that exists, use 'idle'.",
))

register_skill(SkillSpec(
    skill_id="idle",
    fields=(),
    kind=KIND_ZERO,
    target_source=TARGET_NONE,
    doc="Zero torque. The only skill that needs no checkpoint of any kind, so a "
        "slot is always fillable and a controller is always constructible.",
))

# --- planned skills --------------------------------------------------------
# Registered now, with the field order their training env emits, so WS4 can build
# its key bindings against the final skill_ids. They have no checkpoints, so
# `available_skills()` omits them and asking for one raises `SkillUnavailable`
# naming the missing weights. When WS1 lands a run, add the path — that is the
# whole integration.
#
# dribble's order is NOT proprio-first: `warp_port/dribble_env.py` replicates
# dm_control's sorted-key order, where "ball_ego" sorts ahead of "creature/*".
# If a future dribble trains on the proprio-first `worm_env_base` layout instead,
# this tuple changes and nothing else does — and `load_policy` will reject the
# checkpoint loudly if the tuple is stale.

register_skill(SkillSpec(
    skill_id="dribble",
    # v3 contract: proprio-FIRST (worm_env_base), unlike v1's ball-first
    # sorted-key order. Confirmed against the checkpoint: dribble_ant_v3's
    # p_idx is [0..64], t_idx [65..76] = ball_ego(6) | target_ego3(3) |
    # target_ego3_future(3). dribble_env._task_obs concatenates exactly
    # [ball_ego, _target_obs3()].
    fields=PROPRIO_V1 + ("ball_ego", "target_ego3", "target_ego3_future"),
    # Trained post ball-fix (985cff7): every earlier dribble run that passed
    # --ball-radius trained on the WRONG (0.35 m) ball in the wrong arena, so
    # v1's checkpoints and its best-vs-final measurements are void. best.pt =
    # warp fitness 0.965, ball moved 7.0 m/ep with the target moving 98% of the
    # episode.
    checkpoints={"ant": "runs_v2/dribble_ant_v3/best.pt"},
    doc="Drive the ball to a commanded world point. Trained on the FROZEN "
        "shared decoder against the r=0.15 dm_control-proportioned ball.",
))

register_skill(SkillSpec(
    skill_id="kick",
    # kick_env._task_obs: ball_ego6 | _xy_ego3(target) | _dir_ego3(cmd_dir);
    # t_idx [65..76] confirmed against kick_ant_v3/best.pt. The target field is
    # the UNSHRUNK strike variant — kick trained targets out to 6 m.
    fields=PROPRIO_V1 + ("ball_ego", "strike_target_ego3", "cmd_dir_ego3"),
    checkpoints={"ant": "runs_v2/kick_ant_v3/best.pt"},
    doc="Strike the ball so it comes to rest at the commanded point (reward-"
        "kind 'point', w_arrive 3.0). Warp best fitness 0.544, 0.9 strikes/ep.",
))

register_skill(SkillSpec(
    skill_id="shoot",
    # shoot_env._task_obs: ball_ego6 | goal_mid_ego3 | post_left(2) | post_right(2);
    # t_idx [65..77] confirmed against shoot_ant_v3/best.pt. The commanded
    # target IS the goal mouth's ground centre; the goal fields derive mouth
    # height and post offsets from it (fields.SHOOT_GOAL_*). Trained on the +x
    # goal only — mirror before building obs for a -x shot (see _post_ego).
    fields=PROPRIO_V1 + ("ball_ego", "goal_mid_ego3",
                         "post_left_ego", "post_right_ego"),
    checkpoints={"ant": "runs_v2/shoot_ant_v3/best.pt"},
    doc="Kick at the goal mouth. Warp best fitness 0.944, ~1 goal/episode at "
        "the end of training.",
))
