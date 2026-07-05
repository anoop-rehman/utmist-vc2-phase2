# Interface Contracts (FROZEN — changes require all three workstreams to agree)

These are the integration boundaries between WS1 (low-level control), WS2 (data
collection + BC), and WS3 (RL finetune). Code implementing them lives at the
paths given. Proxies (BoxHead) implement the same interfaces, which is what
makes proxy-built infrastructure transfer to the real creatures unchanged.

## 1. Command (the "controller input") — `rower_soccer/envs/commands.py`

```python
Command(a_cmd: int, r_cmd: int, kick: bool)
# a_cmd ∈ {-1,0,+1}: S/coast/W   (backward / none / forward acceleration intent)
# r_cmd ∈ {-1,0,+1}: D/none/A    (clockwise / none / counter-clockwise)
# kick: burst intent; accepted only when KickState.ready
```

- Kick: `KickState` (same file) — cooldown 2.0s, active burst 0.4s. Effective
  kick signal + cooldown fraction come from `KickState.tick(dt)` /
  `.cooldown_fraction`. UI shows cooldown; env enforces it.
- MultiDiscrete encoding for BC/RL: `command_to_multidiscrete` → `[a+1, r+1, k]`,
  i.e. `gymnasium.spaces.MultiDiscrete([3,3,2])`.
- Timing: high level (human keys, BC, RL) acts every **0.1s**; low level acts
  every **0.025s** (the env control step). The env layer holds a command for 4
  low-level steps.

## 2. LowLevelController — `rower_soccer/controllers.py`

```python
class LowLevelController(Protocol):
    creature_kind: str                      # "rower" | "worm" | "boxhead" | ...
    def reset(self) -> None: ...
    def act(self, proprio_obs: np.ndarray, command: Command) -> np.ndarray:
        """proprio obs (see §3) + current command -> actuator vector in [-1,1].
        Called every 0.025s. Stateful (may hold RNN state / smoothing)."""
```

- WS1 delivers: `RowerController`, `WormController` (PPO policies wrapped in
  this interface), checkpoint format: torch `.pt` + a `meta.json` with
  {creature_kind, obs_layout, obs_dim, act_dim, control_dt, git_sha}.
- WS2/WS3 use today: `BoxHeadPassthroughController` — BoxHead's native action
  space is (accelerate, rotate, jump); mapping: a_cmd→accelerate, r_cmd→rotate,
  kick burst→jump channel. Zero training required.
- Loader: `load_controller(checkpoint_dir) -> LowLevelController` (same file).

## 3. Low-level proprio observation (WS1-internal, but the wrapper is shared)

Per creature kind, defined by WS1 in `meta.json: obs_layout`. Baseline layout:
joints_pos, joints_vel, root z-height, root orientation matrix (9), gyro (3),
velocimeter (3), accelerometer (3), prev_action, command_obs (4, from
`commands.command_obs`), egocentric ball position (3) and velocity (3).
WS2/WS3 never construct this directly — they call the controller.

## 4. High-level observation (omniscient) — `rower_soccer/envs/obs.py`

`hl_obs(physics, player_index, env_meta) -> np.ndarray` — built from physics
state, identical for humans' recorded frames, BC training, and RL rollouts.
Egocentric to the controlled player, all positions in the player's yaw frame,
distances normalized by pitch diagonal:

| block | dims |
|---|---|
| self: yaw-frame velocity (2), yaw rate (1), facing-vs-ball angle sin/cos (2), kick cooldown fraction (1), own creature kind one-hot (2) | 8 |
| ball: relative pos (2), velocity (2), speed (1) | 5 |
| teammate: rel pos (2), vel (2), facing sin/cos (2), kind one-hot (2) | 8 |
| opponents ×2 (same layout as teammate) | 16 |
| goals: attack goal rel pos (2), own goal rel pos (2) | 4 |
| pitch: rel pos of pitch center (2), pitch half-size (2) | 4 |
| game: score diff (1), time remaining fraction (1) | 2 |
| **total** | **47** |

Mirror augmentation: `mirror_obs(obs)` and `mirror_command(md)` (same file)
implement the left/right field flip (y-negation in yaw frame + r_cmd swap).

## 5. Trajectory data format (WS2 produces, BC consumes) — one file per episode

`data/demos/<session>/<episode_id>.npz` with:

```
hl_obs      float32 [T, 4, 47]   # per tick (0.1s), per player slot 0..3
command     int8    [T, 4, 3]    # MultiDiscrete values actually applied
kick_ready  bool    [T, 4]
ball_xy     float32 [T, 2]       # pitch frame, for stats/debug
player_xy   float32 [T, 4, 2]
score       int8    [T, 2]
meta.json (sidecar or npz field): {slots: ["human:alice", "human:bob",
  "scripted:chase", "bc:v3", ...], teams: [[0,1],[2,3]],
  creature_kinds: ["rower","worm","rower","worm"], sim_speed: 0.5,
  pitch_size: [40,30], control_dt: 0.025, hl_dt: 0.1, git_sha, timestamp}
```

Slot order: [home0, home1, away0, away1]. Player 0/2 = rower, 1/3 = worm
(kind list is authoritative). Only slots with `human:` prefixes are used as
BC targets by default.

## 6. High-level policy checkpoint (WS2 BC delivers, WS3 consumes/produces)

`checkpoints/hl/<name>/`: `policy.pt` (torch state_dict), `meta.json`
{obs_dim: 47, action_space: [3,3,2], arch: [256,256], creature_kind_conditioned:
true, role: "shared" | "rower" | "worm", trained_on: ..., git_sha}.
BC and RL use the same network class: `rower_soccer/hl_policy.py: HLPolicy`
(MLP 256×256, three categorical heads). WS3's KL anchor loads the BC checkpoint
through this same loader — do not fork the class.

## 7. Env factory — `rower_soccer/envs/build.py`

`make_soccer_env(home_team=("rower","worm"), away_team=..., time_limit=45.0,
disable_walker_contacts=False, terminate_on_goal=True)`. Proxy teams:
`("boxhead","boxhead")` (WS2/WS3 today). The command-level wrapper
`envs/hl_soccer.py: HLSoccerEnv` (WS3 builds, WS2 reuses the stepping loop)
owns: command holding (4:1), KickState per player, hl_obs construction,
scripted-bot slots, and reward plumbing.

## Anti-drift rules

1. Don't change dims/orders/dtypes in §4–§6 without a group decision — BC data
   already collected would be invalidated.
2. Additive evolution only: append fields, never reorder.
3. Every checkpoint/dataset carries `git_sha` — refuse to load on contract
   mismatch (obs_dim check at minimum).
4. Proxy weights never ship: anything trained on BoxHead is for pipeline
   validation only.
