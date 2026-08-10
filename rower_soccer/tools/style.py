"""Does the policy still MOVE like the evolved creature?

    python -m rower_soccer.tools.style score --checkpoint runs_v2/x/final.pt
    python -m rower_soccer.tools.style score --reference          # sanity ceiling

Why this file exists
--------------------
follow_rower_npmp (frozen NPMP prior) scored fitness 0.960 and rows. The
from-scratch control, follow_rower_baseline, scored 0.950 and does NOT row: its
limbs stay folded against the torso and the body just translates. Two policies,
the same number, completely different motions -- because follow fitness is
exp(-c*||player - target||), which measures ARRIVING and is structurally blind to
HOW. Nothing in the codebase could tell those two apart, which means we could not
have detected a style regression, and an automated search pointed at fitness
would happily optimise its way to more twitching.

So this scores the gait itself, against the reference the NPMP tracker was
trained on (runs_v2/rower_ref_gait.npz).

The measurement problem
-----------------------
A task policy carries no reference and no phase clock: it starts at a random yaw,
chases a moving target, and turns whenever the target does. So the score cannot
compare trajectories sample-by-sample the way track_env's joint error does --
there is no correspondence to compare against. It has to be phase-invariant.

But phase-invariance applied PER JOINT would be a hole big enough to drive the
whole result through: eight joints each free to pick their own phase offset means
eight independent oscillators score exactly like one coordinated stroke. Arm
timing IS the gait. So the shape term searches for a SINGLE circular shift shared
by every joint, and grades each joint at that common shift. Relative phase
between joints therefore survives the invariance, and a body whose arms sweep the
right arcs in the wrong order is correctly marked down.

Four sub-scores, combined as a geometric mean so that any one of them going to
zero takes the total with it (the same reasoning as track_env's product reward --
lying on its side waving the arms correctly should score ~0, not 3/4):

  amp    per-joint amplitude ratio vs the reference. This is the term that
         catches the twitcher, whose amplitudes are a few degrees against a
         reference that sweeps +/-74.
  freq   per-joint dominant frequency ratio. Right arc traced at half speed is
         not the same gait. Graded per joint, not globally, because the paddles
         genuinely run at 2x the arms.
  shape  waveform correlation at the best COMMON circular shift, per above.
         Amplitude-invariant by construction (it correlates normalised signals),
         so it is orthogonal to `amp` rather than double-counting it.
  pose   per-joint MEAN angle vs the reference: not how far a joint swings, but
         where it sits while swinging. The other three terms are all invariant
         to a constant offset, which leaves posture entirely ungraded -- and
         posture is most of what the eye reads. follow_rower_baseline holds both
         shoulders jammed at their stops, 73 and 85 degrees from the reference,
         which is the "arms folded against the torso" look; the tracking policy
         is within 8.5 degrees on every joint. It is also the only term that
         sees the two paddle joints properly, because those do not oscillate at
         all -- the evolved creature pins them at OPPOSITE stops (+74/-74), so
         amplitude says "still, correct" for a policy that lets them hang
         neutral.

Joints are weighted by the reference's own `track_weight` (phase-average
coherence), so a joint whose reference is not reproducible cannot dominate the
score -- the same weighting the tracking reward uses. The two joints with no
reference OSCILLATION are still graded, just not on every term: `freq` and
`shape` need a waveform and would be meaningless, so they are scored on `amp`
(as stillness) and `pose` (as where they are pinned). Nothing is left ungraded --
an ungraded joint is a free parameter for any search process to exploit.
"""

import argparse
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_PATH = os.path.join(REPO, "runs_v2", "rower_ref_gait.npz")
ROWER_XML = os.path.join(REPO, "creature_configs", "two_arm_rower_scaled.xml")

PHASE_BINS = 64
SETTLE_S = 2.0        # drop the launch transient before scoring
FMIN = 0.1            # ignore near-DC when picking a dominant frequency
MIN_REF_AMP = np.deg2rad(2.0)   # below this the reference joint is "not moving"
# Offset tolerance for the pose term, exp(-|dmean| / POSE_TOL). Deliberately
# loose: a follow policy turns and corrects, and steering a two-armed body
# legitimately biases its joints away from the straight-line reference, so a
# tight tolerance would grade task competence as bad style. At 30 deg the
# tracking policy's worst joint still scores 0.75 while the baseline's folded
# shoulders (73-85 deg off) score below 0.09 -- the gross posture difference
# survives the leniency, which is the point.
POSE_TOL = np.deg2rad(30.0)
REF_TILE = 12         # cycles to tile the 1-cycle reference to for its FFT


# ----------------------------------------------------------------- primitives
def dominant_freq(x, dt, fmin=FMIN):
    """Peak of the windowed power spectrum, ignoring near-DC."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if n < 8 or not np.isfinite(x).all() or x.std() < 1e-9:
        return 0.0
    power = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, dt)
    power[freqs < fmin] = 0.0
    return float(freqs[power.argmax()])


def _amp(x, lo=2.5, hi=97.5):
    """Half peak-to-peak, percentile-trimmed.

    Plain ptp/2 would let one blown contact -- which mujoco_warp does produce --
    set the amplitude of an entire episode. Trimming costs ~0.3% on a clean
    sinusoid and bounds that failure.
    """
    x = np.asarray(x, float)
    return float(np.percentile(x, hi) - np.percentile(x, lo)) / 2.0


def _ratio(a, b, eps=1e-9):
    """Symmetric ratio in (0, 1]: 1 when equal, 0.5 at 2x either way."""
    a, b = abs(float(a)), abs(float(b))
    if a < eps and b < eps:
        return 1.0
    return min(a, b) / max(a, b, eps)


def _phase_average(x, dt, freq, bins=PHASE_BINS):
    """Fold a signal onto one cycle at `freq`. x: [T, J] -> [bins, J] or None.

    Empty bins are filled by circular interpolation rather than failing. They are
    the normal case, not an edge case: sampling at 40 Hz a 0.89 Hz gait gives 45
    samples per cycle, and -- because every sample lands at phase i/45 exactly --
    those 45 distinct phases never populate more than 45 of the bins no matter
    how many cycles are averaged. Returning None there silently zeroed the shape
    term for a perfectly good signal.
    """
    if freq <= 0:
        return None
    t = np.arange(len(x)) * dt
    idx = np.clip((((t * freq) % 1.0) * bins).astype(int), 0, bins - 1)
    out = np.zeros((bins, x.shape[1]))
    cnt = np.bincount(idx, minlength=bins).astype(float)
    np.add.at(out, idx, x)
    filled = cnt > 0
    if filled.sum() < 4:
        return None
    out[filled] /= cnt[filled, None]
    if not filled.all():
        # Circular interpolation over the filled bin centres. Wrapping one filled
        # bin around each end makes phase 0 and phase 1 the same point, so a gap
        # straddling the seam interpolates the short way round.
        c = np.nonzero(filled)[0]
        xp = np.concatenate([c - bins, c, c + bins]).astype(float)
        fp = np.concatenate([out[c], out[c], out[c]], axis=0)
        miss = np.nonzero(~filled)[0]
        for j in range(x.shape[1]):
            out[miss, j] = np.interp(miss.astype(float), xp, fp[:, j])
    return out


def _normalize(c):
    """Zero-mean, unit-std per column; flat columns become exact zeros."""
    c = c - c.mean(0, keepdims=True)
    s = c.std(0, keepdims=True)
    return np.where(s > 1e-9, c / np.maximum(s, 1e-12), 0.0)


def _best_common_shift(a, b, w):
    """Circular shift of `b` maximising the w-weighted correlation with `a`.

    a, b: [bins, J] already normalised. Returns (shift, per_joint_corr).

    One shift for all joints, deliberately -- see the module docstring. Searching
    a shift per joint would make eight independent oscillators indistinguishable
    from one coordinated stroke, which is the exact distinction this file exists
    to measure.
    """
    bins = a.shape[0]
    best, best_shift, best_corr = -np.inf, 0, None
    for s in range(bins):
        corr = (a * np.roll(b, s, axis=0)).mean(0)      # [J], Pearson per joint
        score = float((corr * w).sum())
        if score > best:
            best, best_shift, best_corr = score, s, corr
    return best_shift, best_corr


# --------------------------------------------------------------------- scoring
def load_reference(path=REF_PATH):
    z = np.load(path, allow_pickle=True)
    ref = np.asarray(z["ref_qpos"], float)              # [K, J], exactly one cycle
    dt = float(z["dt_ctrl"])
    names = [str(s) for s in z["joint_names"]]
    coh = np.asarray(z["track_weight"], float)

    amp = np.array([_amp(ref[:, j]) for j in range(ref.shape[1])])
    mean = ref.mean(0)
    # One cycle is far too short to resolve a spectrum, so tile it. The signal is
    # periodic by construction, and the paddles' 2x harmonic is what we need to
    # resolve -- at one cycle the FFT has ~2 usable bins.
    tiled = np.tile(ref, (REF_TILE, 1))
    freq = np.array([dominant_freq(tiled[:, j], dt) for j in range(ref.shape[1])])

    live = amp > MIN_REF_AMP
    if not live.any():
        raise SystemExit("reference has no moving joints")

    # Joints the reference holds STILL are scored, not skipped. The rower has two
    # (seg0_to_1, seg0_to_8) and follow_rower_baseline swings them through 64 and
    # 69 degrees -- a quarter of the body moving in a way the evolved creature
    # never does. Dropping them outright, which is what a coherence weight of zero
    # did, leaves a hole worth 2 of 8 joints that any search process would find
    # and exploit: keep the graded joints tidy, thrash the ungraded ones.
    #
    # Their coherence is 0/0 and carries no information, so they take the median
    # weight of the joints that do move, and are graded on stillness (below).
    w_shape = coh * live                       # shape needs a waveform to correlate
    if w_shape.sum() <= 0:
        raise SystemExit("reference joints have no usable coherence")
    w_shape = w_shape / w_shape.sum()

    w = np.where(live, coh, np.median(coh[live]))
    w = w / w.sum()
    return dict(ref=ref, dt=dt, names=names, coh=coh, amp=amp, freq=freq,
                mean=mean, live=live, w=w, w_shape=w_shape,
                cycle=_normalize(_resample_cycle(ref)))


def _resample_cycle(ref, bins=PHASE_BINS):
    """Resample a one-cycle reference [K, J] onto `bins` phase bins."""
    k = len(ref)
    src = np.arange(k) / k
    dst = np.arange(bins) / bins
    return np.stack([np.interp(dst, src, ref[:, j], period=1.0)
                     for j in range(ref.shape[1])], axis=1)


def style_score(q, dt, R):
    """Score one rollout's joint trajectory `q` [T, J] against reference dict R."""
    J = q.shape[1]
    w, ws, live = R["w"], R["w_shape"], R["live"]

    amp = np.array([_amp(q[:, j]) for j in range(J)])
    freq = np.array([dominant_freq(q[:, j], dt) for j in range(J)])
    mean = q.mean(0)
    # Graded on every joint, live or still: posture is exactly as meaningful for
    # a joint the reference pins at a stop as for one that sweeps.
    s_pose = np.exp(-np.abs(mean - R["mean"]) / POSE_TOL)

    s_amp = np.empty(J)
    s_freq = np.empty(J)
    for j in range(J):
        if live[j]:
            s_amp[j] = _ratio(amp[j], R["amp"][j])
            s_freq[j] = _ratio(freq[j], R["freq"][j])
        else:
            # Reference-still joint: grade how still the policy keeps it. At or
            # under the stillness tolerance scores 1; 64 degrees of thrash scores
            # ~0.03. It carries no frequency or waveform -- "how fast does a joint
            # oscillate that should not oscillate" has no answer -- so it is
            # weighted into `amp` only, and dropped from freq and shape (w_shape).
            s_amp[j] = MIN_REF_AMP / max(MIN_REF_AMP, amp[j])
            s_freq[j] = 1.0    # unused: freq is weighted by w_shape (live only)

    # Fold the rollout at the gait frequency the WEIGHTED-DOMINANT joints show,
    # not at the reference's -- folding at the reference rate would import the
    # answer and let a policy running at the wrong speed still look shaped.
    f_gait = float(freq[live].dot(ws[live]) / max(ws[live].sum(), 1e-9))
    roll_cycle = _phase_average(q, dt, f_gait)
    s_shape = np.zeros(J)
    shift = 0
    if roll_cycle is not None:
        # Still joints are excluded here: a flat reference normalises to exact
        # zeros, so its correlation is 0 by construction, not by any fault of the
        # policy. They are already fully accounted for by s_amp above.
        shift, corr = _best_common_shift(R["cycle"], _normalize(roll_cycle), ws)
        s_shape = np.where(live, np.clip(corr, 0.0, 1.0), 0.0)

    parts = {"amp": float((s_amp * w).sum()),          # all joints: stillness counts
             "freq": float((s_freq * ws).sum()),       # live joints only
             "shape": float((s_shape * ws).sum()),     # live joints only
             "pose": float((s_pose * w).sum())}        # all joints
    # Geometric mean: any one term at zero takes the total with it.
    total = float(np.prod([max(parts[k], 0.0)
                           for k in ("amp", "freq", "shape", "pose")]) ** 0.25)
    return dict(style=total, **parts, gait_hz=f_gait, shift=int(shift),
                per_joint=dict(amp=amp, freq=freq, mean=mean, s_amp=s_amp,
                               s_freq=s_freq, s_shape=s_shape, s_pose=s_pose))


def report(res, R, label=""):
    print(f"\n=== style {label} ===")
    print(f"{'joint':<12}{'amp':>9}{'ref':>8}{'s_amp':>7}"
          f"{'f':>6}{'ref':>6}{'s_frq':>7}{'s_shp':>7}"
          f"{'mean':>8}{'ref':>8}{'s_pose':>8}")
    print("-" * 84)
    pj = res["per_joint"]
    for j, n in enumerate(R["names"]):
        mark = "  (ref still)" if not R["live"][j] else ""
        print(f"{n:<12}{np.rad2deg(pj['amp'][j]):>8.1f}d"
              f"{np.rad2deg(R['amp'][j]):>7.1f}d{pj['s_amp'][j]:>7.3f}"
              f"{pj['freq'][j]:>6.2f}{R['freq'][j]:>6.2f}{pj['s_freq'][j]:>7.3f}"
              f"{pj['s_shape'][j]:>7.3f}"
              f"{np.rad2deg(pj['mean'][j]):>7.1f}d{np.rad2deg(R['mean'][j]):>7.1f}d"
              f"{pj['s_pose'][j]:>8.3f}{mark}")
    print("-" * 84)
    print(f"amp {res['amp']:.3f} * freq {res['freq']:.3f} * shape {res['shape']:.3f}"
          f" * pose {res['pose']:.3f}"
          f"  ->  STYLE {res['style']:.3f}    (gait {res['gait_hz']:.2f} Hz)")
    return res


# -------------------------------------------------------------------- rollouts
def rollout_joints(env, ac, settle_s=SETTLE_S, control_dt=0.025, deterministic=True):
    """Run one full episode in `env` and return joint angles [n_worlds, T, J]."""
    import torch
    obs = env.reset()
    q, done = [], False
    while not done:
        with torch.no_grad():
            d = ac.dist(obs.float())
            a = (d.mean if deterministic else d.sample()).clamp(-1, 1)
        obs, _, done = env.step(a)
        q.append(env.qpos[:, env.jq].detach().float().cpu().numpy().copy())
    q = np.stack(q, axis=1)                              # [n, T, J]
    skip = int(settle_s / control_dt)
    return q[:, skip:, :]


def _build_policy(env, ckpt, device="cuda"):
    import torch
    from rower_soccer.warp_port.ppo import ActorCritic, _flatten_checkpoint
    sd = _flatten_checkpoint(torch.load(ckpt, map_location=device, weights_only=True))
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(),
                     z_dim=16, state_dependent_std="log_std_net.weight" in sd).to(device)
    own = ac.state_dict()
    # p_idx/t_idx describe THIS env's obs layout; never take them from the file.
    buffers = {"mlp_extractor.p_idx", "mlp_extractor.t_idx"}
    missing = [k for k in own if k not in buffers and k not in sd]
    if missing:
        raise SystemExit(f"checkpoint {ckpt} is missing {missing}; wrong body/env?")
    for k, v in own.items():
        if k not in buffers:
            if sd[k].shape != v.shape:
                raise SystemExit(f"{k}: checkpoint {tuple(sd[k].shape)} vs "
                                 f"env {tuple(v.shape)}; wrong creature?")
            v.copy_(sd[k].to(v.device))
    ac.load_state_dict(own)
    ac.eval()
    return ac


def score_checkpoint(ckpt, env_kind="follow", xml=ROWER_XML, worlds=32,
                     ref=REF_PATH, seed=0, episode_secs=None):
    os.environ.setdefault("MUJOCO_GL", "egl")
    R = load_reference(ref)
    if env_kind == "follow":
        from rower_soccer.warp_port.follow_env import WarpFollowEnv, CONTROL_DT
        env = WarpFollowEnv(num_worlds=worlds, creature_xml=xml, seed=seed,
                            episode_seconds=episode_secs or 15.0,
                            target_speed_range=(0.07, 0.6), spawn_dist_range=(1.07, 3.22))
    elif env_kind == "track":
        from rower_soccer.warp_port.track_env import WarpTrackEnv
        from rower_soccer.warp_port.follow_env import CONTROL_DT
        env = WarpTrackEnv(num_worlds=worlds, creature_xml=xml, ref_path=ref,
                           episode_seconds=episode_secs or 10.0, seed=seed, rsi=False)
    else:
        raise SystemExit(f"unknown env {env_kind}")

    ac = _build_policy(env, ckpt)
    q = rollout_joints(env, ac, control_dt=CONTROL_DT)
    per_world = [style_score(q[i], CONTROL_DT, R) for i in range(q.shape[0])]
    agg = _aggregate(per_world, R)
    return agg, R, per_world


def _aggregate(per_world, R):
    """Mean over worlds, keeping the spread -- n=1 comparisons have burned us."""
    keys = ("style", "amp", "freq", "shape", "pose", "gait_hz")
    out = {k: float(np.mean([p[k] for p in per_world])) for k in keys}
    out.update({f"{k}_std": float(np.std([p[k] for p in per_world])) for k in keys})
    out["shift"] = int(np.median([p["shift"] for p in per_world]))
    out["per_joint"] = {k: np.mean([p["per_joint"][k] for p in per_world], axis=0)
                        for k in per_world[0]["per_joint"]}
    out["n_worlds"] = len(per_world)
    return out


# ------------------------------------------------------------------------ cli
def selftest(ref=REF_PATH, cycles=12, seed=0):
    """Synthetic signals engineered to fool one term each.

    A metric that only scored the reference at 1.0 would prove nothing -- the
    interesting question is what it REFUSES. Each case below is a plausible way a
    policy could look right on some axis while not rowing, and names the term that
    is supposed to catch it.
    """
    rng = np.random.default_rng(seed)
    R = load_reference(ref)
    base = np.tile(R["ref"], (cycles, 1))
    dt, J = R["dt"], R["ref"].shape[1]
    K = len(R["ref"])
    cases = []

    cases.append(("reference (ceiling)", base, "all ~1.0"))

    # The twitcher: right body, tiny fast motion. This is follow_rower_baseline.
    tw = np.deg2rad(1.5) * rng.standard_normal((len(base), J))
    cases.append(("twitcher (small fast noise)", tw, "amp -> 0"))

    # Motionless. A policy that locks its joints and slides.
    cases.append(("frozen limbs", np.zeros((len(base), J)), "amp -> 0"))

    # Correct amplitudes and frequencies, but each joint on its OWN random phase.
    # Eight independent oscillators instead of one stroke -- the case that a
    # per-joint phase search would score as perfect. `shape` must catch it.
    sc = np.stack([np.roll(base[:, j], int(rng.integers(0, K))) for j in range(J)], 1)
    cases.append(("phase-scrambled joints", sc, "shape drops, amp/freq stay 1"))

    # Whole gait rigidly time-shifted: still one coordinated stroke, just starting
    # elsewhere in the cycle. MUST still score ~1 -- this is the invariance the
    # task policy actually needs, since it has no phase clock.
    cases.append(("global phase shift", np.roll(base, K // 3, axis=0), "all ~1.0"))

    # Right shape, half speed. Correct arcs traced at the wrong rate.
    slow = np.repeat(base, 2, axis=0)
    cases.append(("half frequency", slow, "freq -> ~0.5"))

    # Right motion, quarter amplitude: a timid version of the real gait.
    cases.append(("quarter amplitude", base * 0.25, "amp -> ~0.25"))

    # A perfect stroke on every graded joint, while the joints the reference holds
    # still are thrashed through 60 degrees. This is the reward-hacking shape the
    # old zero-weighting invited, and follow_rower_baseline does a version of it
    # for real (64 and 69 degrees on those two joints).
    thrash = base.copy()
    ph = np.arange(len(base)) * dt * 2 * np.pi * 0.89
    for j in np.nonzero(~R["live"])[0]:
        thrash[:, j] = np.deg2rad(60.0) * np.sin(ph)
    cases.append(("thrashes reference-still joints", thrash, "amp penalised"))

    # The exact stroke, at the exact rate, held 50 degrees away from where the
    # creature holds it -- arms folded in rather than extended. amp/freq/shape are
    # all offset-invariant, so before `pose` existed this scored a clean 1.0.
    cases.append(("right gait, folded posture", base + np.deg2rad(50.0),
                  "pose -> ~0.19, others 1"))

    print(f"{'case':<30}{'style':>8}{'amp':>8}{'freq':>8}{'shape':>8}"
          f"{'pose':>8}   expected")
    print("-" * 86)
    out = {}
    for name, q, expect in cases:
        r = style_score(q, dt, R)
        out[name] = r
        print(f"{name:<30}{r['style']:>8.3f}{r['amp']:>8.3f}"
              f"{r['freq']:>8.3f}{r['shape']:>8.3f}{r['pose']:>8.3f}   {expect}")
    print("-" * 86)

    ok = True
    def chk(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and bool(cond)

    live = R["live"]
    def live_mean(name, key):
        v = out[name]["per_joint"][key]
        return float(v[live].mean())

    chk(out["reference (ceiling)"]["style"] > 0.95, "reference scores > 0.95")
    chk(out["global phase shift"]["style"] > 0.95,
        "a rigid phase shift is NOT penalised (task policies have no phase clock)")
    chk(out["reference (ceiling)"]["style"]
        - out["twitcher (small fast noise)"]["style"] > 0.7,
        "twitcher sits > 0.7 below the reference")
    chk(out["frozen limbs"]["style"] < 0.05, "frozen limbs score < 0.05")
    # Inter-joint timing is graded by the SHAPE term, so that is what this asserts.
    # The total moves less (0.999 -> ~0.73) because a geometric mean over three
    # terms takes the cube root of a single term's collapse. That is deliberate --
    # the headline number stays interpretable and the breakdown shows where the
    # loss is -- but it means asserting on the total here would be testing the
    # aggregation, not the coordination sensitivity this case exists to prove.
    chk(out["phase-scrambled joints"]["shape"]
        < 0.5 * out["reference (ceiling)"]["shape"],
        "phase-scrambled joints lose >50% of SHAPE (inter-joint timing counts)")
    chk(out["phase-scrambled joints"]["amp"] > 0.95
        and out["phase-scrambled joints"]["freq"] > 0.95,
        "  ...while amp and freq still read as correct (the terms are independent)")
    chk(out["reference (ceiling)"]["style"] - out["phase-scrambled joints"]["style"]
        > 0.2, "  ...and the total still separates them by > 0.2")
    chk(abs(out["half frequency"]["freq"] - 0.5) < 0.1, "half frequency -> freq ~0.5")
    # Asserted on the LIVE joints: the aggregate legitimately sits higher, because
    # scaling a still joint by 0.25 leaves it still, which is correct behaviour.
    chk(abs(live_mean("quarter amplitude", "s_amp") - 0.25) < 0.05,
        "quarter amplitude -> live-joint amp ~0.25")
    chk(out["thrashes reference-still joints"]["amp"]
        < 0.8 * out["reference (ceiling)"]["amp"],
        "thrashing reference-still joints costs >20% of amp (no ungraded joints)")
    fold = out["right gait, folded posture"]
    chk(fold["pose"] < 0.3, "a 50 deg posture offset collapses pose (< 0.3)")
    chk(min(fold["amp"], fold["freq"], fold["shape"]) > 0.95,
        "  ...while amp/freq/shape stay ~1, being offset-invariant by design")
    chk(out["reference (ceiling)"]["style"] - fold["style"] > 0.3,
        "  ...and posture alone moves the total by > 0.3")
    print("\nSELFTEST " + ("PASSED" if ok else "FAILED"))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="adversarial checks that the score discriminates")
    s = sub.add_parser("score", help="score a checkpoint (or the reference)")
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--reference", action="store_true",
                   help="score the reference gait against itself: the ceiling "
                        "sanity check, must come out at 1.000")
    s.add_argument("--env", default="follow", choices=["follow", "track"])
    s.add_argument("--creature-xml", default=ROWER_XML)
    s.add_argument("--ref", default=REF_PATH)
    s.add_argument("--worlds", type=int, default=32)
    s.add_argument("--episode-secs", type=float, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--label", default="")
    a = p.parse_args()

    if a.cmd == "selftest":
        raise SystemExit(0 if selftest() else 1)

    R = load_reference(a.ref)
    if a.reference:
        # Tile the one-cycle reference out to a realistic episode length and score
        # it as if it were a rollout. Exercises the whole path -- FFT, folding,
        # common-shift search -- so a bug that inflates scores shows up here.
        q = np.tile(R["ref"], (12, 1))
        res = style_score(q, R["dt"], R)
        report(res, R, a.label or "REFERENCE (ceiling)")
        return
    if not a.checkpoint:
        raise SystemExit("pass --checkpoint or --reference")
    agg, R, _ = score_checkpoint(a.checkpoint, env_kind=a.env,
                                 xml=a.creature_xml, worlds=a.worlds,
                                 ref=a.ref, seed=a.seed,
                                 episode_secs=a.episode_secs)
    res = dict(agg, per_joint=agg["per_joint"])
    report(res, R, a.label or f"{os.path.basename(a.checkpoint)} [{a.env}]")
    print(f"  over {agg['n_worlds']} worlds: style {agg['style']:.3f} "
          f"+/- {agg['style_std']:.3f}")


if __name__ == "__main__":
    main()
