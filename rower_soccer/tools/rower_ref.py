"""Turn the ported Sims trajectory into a tracking reference for the scaled rower.

    python -m rower_soccer.tools.rower_ref build
    python -m rower_soccer.tools.rower_ref replay

`build` phase-averages the evolved gait into one canonical cycle and resamples
it to the control rate at the Froude-correct frequency. `replay` puppets the
scaled rower through that reference with no learning, so the joint mapping and
axis signs can be checked before any training compute is spent.

Two things this file exists to get right, both of which silently broke a 400M
step run (`npmp_rower_track`, which never rose off the floor):

1. THE FROUDE FACTOR IS AGAINST THE UNITY CREATURE, NOT THE BLUEPRINT.
   run_sims_brain simulates blueprint geometry (2x Unity lengths) at 2x gravity
   precisely so that time runs 1:1 with Unity -- see its GRAVITY constant. So
   the 0.533 Hz it reports is ALREADY the real creature's frequency in real
   seconds, and the real creature is the Unity-scale one (blueprint / 2). The
   factor is sqrt(L_unity / L_rower) = 1.681, giving 0.897 Hz. Taking the
   blueprint as the creature applies the 2x a second time and yields 1.78 Hz --
   twice what physics allows, and a reference no policy can ever reach.

2. THE BODY MUST BE ABLE TO DRIVE IT. Torque for a sinusoid goes as
   I*A*(2*pi*f)^2, so a 2x frequency error is a 4x torque error. `check` reports
   the margin per joint; run it whenever the gait, the body or gear_scale moves.

Joint angles are dimensionless, so amplitudes transfer across the scale change
untouched -- only the time axis is rescaled. That is also why this tracks
kinematics rather than the recorded torques, which were produced by Unity-style
velocity servos at 2x blueprint scale and are unit-incompatible with the
scaled rower's torque actuators.
"""

import argparse
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAJ_IN = os.path.join(REPO, "runs_v2", "sims_brain_rower_traj.npz")
REF_OUT = os.path.join(REPO, "runs_v2", "rower_ref_gait.npz")
VIDEO_OUT = os.path.join(REPO, "videos", "npmp_ref_replay.mp4")
ROWER_XML = os.path.join(REPO, "creature_configs", "two_arm_rower_scaled.xml")
BLUEPRINT_XML = os.path.join(REPO, "creature_configs", "two_arm_rower_blueprint.xml")

CTRL_HZ = 40.0        # our control rate
PHASE_BINS = 64       # resolution of the canonical cycle
SETTLE_S = 3.0        # drop the launch transient before averaging
FMIN = 0.1            # ignore near-DC when picking a dominant frequency


def _bbox_len(xml_path):
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return float((d.xpos[1:].max(0) - d.xpos[1:].min(0)).max())


def froude_factor():
    """Frequency multiplier taking the evolved gait onto the scaled rower."""
    from rower_soccer.tools.run_sims_brain import LENGTH_RATIO
    l_unity = _bbox_len(BLUEPRINT_XML) / LENGTH_RATIO
    l_rower = _bbox_len(ROWER_XML)
    return np.sqrt(l_unity / l_rower), l_unity, l_rower


def dominant_freq(x, dt):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    power = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, dt)
    power[freqs < FMIN] = 0.0
    return float(freqs[power.argmax()])


def _phase_average(seg, dt, freq, bins=PHASE_BINS):
    t = np.arange(len(seg)) * dt
    idx = np.clip((((t * freq) % 1.0) * bins).astype(int), 0, bins - 1)
    cycle = np.zeros((bins, seg.shape[1]))
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            return None
        cycle[b] = seg[sel].mean(0)
    return cycle


def _refine_freq(seg, dt, f0, span=0.06, n=401):
    """Sharpen the period estimate by maximising phase-average coherence.

    An FFT peak is only accurate to the bin width (1/T = 0.033 Hz here), and
    over ~14 cycles that error smears the phase-average enough to shave several
    degrees off every stroke. The true period is the one whose phase-average
    retains the most energy, so scan for it directly. Amplitude-normalised per
    joint, otherwise the two highest-amplitude joints alone decide the answer.
    """
    scale = seg.std(0)
    scale[scale < 1e-9] = 1.0
    best, best_f = -np.inf, f0
    for f in np.linspace(f0 * (1 - span), f0 * (1 + span), n):
        cycle = _phase_average(seg, dt, f)
        if cycle is None:
            continue
        energy = float((cycle.var(0) / scale ** 2).sum())
        if energy > best:
            best, best_f = energy, f
    return best_f


LIMIT_MARGIN = np.deg2rad(1.0)   # stay just inside the stop, not exactly on it


def _clip_to_limits(ref, names, xml):
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    out = ref.copy()
    clipped = np.zeros(len(names))
    for i, n in enumerate(names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0 or not m.jnt_limited[jid]:
            continue
        lo, hi = m.jnt_range[jid]
        lo, hi = lo + LIMIT_MARGIN, hi - LIMIT_MARGIN
        out[:, i] = np.clip(ref[:, i], lo, hi)
        clipped[i] = np.abs(out[:, i] - ref[:, i]).max()
    return out, clipped


def build(traj_in=TRAJ_IN, out=REF_OUT, ctrl_hz=CTRL_HZ, froude=True):
    z = np.load(traj_in, allow_pickle=True)
    jd = np.deg2rad(z["joint_deg"])
    dt = float(z["dt"])
    names = [str(s) for s in z["joint_names"]]

    # The four arm joints carry the gait, so the period is estimated from them.
    # The two paddle effectors do not cycle at all in steady state: their
    # dangling child reference shoves them at a constant 130 deg/s, so after the
    # launch transient they sit pinned at opposite stops (+75.8 and -76.1 deg).
    arm = [i for i, n in enumerate(names) if n in
           ("seg0_to_4", "seg4_to_5", "seg0_to_6", "seg6_to_7")]
    f_fft = dominant_freq(jd[:, arm].sum(1), dt)

    # Phase-average into one canonical cycle, skipping the launch transient.
    i0 = int(round(SETTLE_S / dt))
    seg = jd[i0:]
    f_src = _refine_freq(seg, dt, f_fft)

    factor, l_unity, l_rower = froude_factor() if froude else (1.0, None, None)
    f_tgt = f_src * factor

    cycle = _phase_average(seg, dt, f_src)
    if cycle is None:
        raise RuntimeError("empty phase bin; need a longer trajectory")

    # Resample the canonical cycle onto the control grid at the target rate.
    n_ctrl = int(round(ctrl_hz / f_tgt))
    src_phase = np.arange(PHASE_BINS) / PHASE_BINS
    dst_phase = np.arange(n_ctrl) / n_ctrl
    wrapped = np.vstack([cycle, cycle[:1]])                  # periodic closure
    ref = np.stack([np.interp(dst_phase, np.append(src_phase, 1.0), wrapped[:, j])
                    for j in range(jd.shape[1])], axis=1)

    # Clip into the body's joint range. MuJoCo limits are soft constraints, so
    # the source sim overshot its own +/-75 deg stops by up to 15 deg -- the
    # evolved gait drives hard into them and the solver lets it bulge through.
    # An unclipped reference would ask the policy to hold a pose outside the
    # stop forever, which no torque achieves; it would sit in permanent limit
    # contact and eat the tracking penalty every step. Clipping is also the
    # faithful reading: the creature really was pinned AT the stop.
    ref, clipped = _clip_to_limits(ref, names, ROWER_XML)

    # How much of each joint's motion is repeatable gait rather than chaos:
    # variance surviving the phase-average, over raw variance. Every joint here
    # scores >= 0.75, so the evolved gait is highly periodic and all eight are
    # worth tracking. Kept as a per-joint reward weight anyway, so a future gait
    # with a genuinely chaotic joint down-weights it automatically instead of
    # scoring the policy on noise it cannot reproduce.
    coherence = cycle.var(0) / np.maximum(seg.var(0), 1e-12)
    coherence = np.clip(coherence, 0.0, 1.0)

    np.savez_compressed(
        out, ref_qpos=ref, phase=dst_phase, joint_names=np.asarray(names),
        freq_src=f_src, freq_tgt=f_tgt, froude_factor=factor, track_weight=coherence,
        ctrl_hz=ctrl_hz, dt_ctrl=1.0 / ctrl_hz, cycles_averaged=len(seg) * dt * f_src,
    )

    print(f"source gait      : {f_src:.4f} Hz  (fft {f_fft:.4f}, refined)  "
          f"({len(seg)*dt*f_src:.1f} cycles averaged)")
    if froude:
        print(f"froude factor    : {factor:.4f}x   "
              f"(unity creature {l_unity:.3f} m -> rower {l_rower:.3f} m)")
    print(f"target gait      : {f_tgt:.4f} Hz  ({n_ctrl} ctrl steps/cycle @ {ctrl_hz:g} Hz)")
    print(f"\n{'joint':<12}{'raw amp':>9}{'ref amp':>9}{'coherence':>11}{'clipped':>11}")
    print("-" * 53)
    for i, n in enumerate(names):
        raw = np.rad2deg(np.ptp(seg[:, i])) / 2
        cur = np.rad2deg(np.ptp(ref[:, i])) / 2
        print(f"{n:<12}{raw:>8.0f}d{cur:>8.0f}d{coherence[i]:>11.3f}"
              f"{np.rad2deg(clipped[i]):>10.1f}d")
    print(f"\nwrote {out}")
    return out


def check(ref=REF_OUT, xml=ROWER_XML):
    """Can this body actually drive this reference? tau = I*A*(2*pi*f)^2."""
    import mujoco
    z = np.load(ref, allow_pickle=True)
    r = z["ref_qpos"]
    names = [str(s) for s in z["joint_names"]]
    f_tgt = float(z["freq_tgt"])

    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    full = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, full)

    print(f"{'joint':<12}{'amp(deg)':>10}{'f(Hz)':>8}{'gear':>10}{'required':>10}{'margin':>9}")
    print("-" * 59)
    worst = np.inf
    for i, n in enumerate(names):
        # each joint at its own dominant rate (paddles run at 2x the arms)
        f_i = max(dominant_freq(r[:, i], 1.0 / float(z["ctrl_hz"])), f_tgt)
        amp = np.ptp(r[:, i]) / 2
        dof = m.jnt_dofadr[m.actuator_trnid[i, 0]]
        gear = abs(m.actuator_gear[i, 0])
        req = full[dof, dof] * amp * (2 * np.pi * f_i) ** 2
        margin = gear / req if req > 0 else np.inf
        worst = min(worst, margin)
        print(f"{n:<12}{np.rad2deg(amp):>10.1f}{f_i:>8.3f}{gear:>10.4f}{req:>10.4f}"
              f"{margin:>8.2f}x")
    verdict = "OK" if worst >= 1.0 else "UNDER-ACTUATED"
    print(f"\nworst margin: {worst:.2f}x  -> {verdict}")
    if worst < 1.0:
        print("  regenerate the body with a larger --gear-scale "
              "(rower_soccer/tools/unity2mujoco.py)")
    return worst


def replay(ref=REF_OUT, xml=ROWER_XML, out=VIDEO_OUT, cycles=4.0, fps=50):
    """Puppet the rower through the reference. No physics, no learning."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import imageio

    z = np.load(ref, allow_pickle=True)
    r = z["ref_qpos"]
    names = [str(s) for s in z["joint_names"]]
    ctrl_hz = float(z["ctrl_hz"])

    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    # Reference column i must land on the joint of the SAME NAME, not position i.
    qadr = []
    for n in names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise KeyError(f"reference joint {n!r} absent from {xml}")
        qadr.append(m.jnt_qposadr[jid])
    qadr = np.asarray(qadr)

    n_frames = int(round(cycles * len(r) * fps / ctrl_hz))
    renderer = mujoco.Renderer(m, height=480, width=640)
    frames = []
    for k in range(n_frames):
        idx = int(round(k * ctrl_hz / fps)) % len(r)
        d.qpos[qadr] = r[idx]
        mujoco.mj_forward(m, d)          # kinematics only; never mj_step
        renderer.update_scene(d)
        frames.append(renderer.render())

    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimwrite(out, frames, fps=fps, quality=8)
    print(f"wrote {out}  ({n_frames} frames, {cycles:g} cycles, "
          f"{n_frames/fps:.1f} s at {fps} fps)")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="phase-average the gait into a reference")
    b.add_argument("--traj", default=TRAJ_IN)
    b.add_argument("--out", default=REF_OUT)
    b.add_argument("--ctrl-hz", type=float, default=CTRL_HZ)
    b.add_argument("--no-froude", action="store_true",
                   help="keep the source frequency (the stately 0.53 Hz version)")

    c = sub.add_parser("check", help="torque margin of the body against the reference")
    c.add_argument("--ref", default=REF_OUT)
    c.add_argument("--xml", default=ROWER_XML)

    r = sub.add_parser("replay", help="kinematic replay video, no learning")
    r.add_argument("--ref", default=REF_OUT)
    r.add_argument("--xml", default=ROWER_XML)
    r.add_argument("--out", default=VIDEO_OUT)
    r.add_argument("--cycles", type=float, default=4.0)

    a = p.parse_args()
    if a.cmd == "build":
        build(a.traj, a.out, a.ctrl_hz, froude=not a.no_froude)
    elif a.cmd == "check":
        check(a.ref, a.xml)
    else:
        replay(a.ref, a.xml, a.out, a.cycles)


if __name__ == "__main__":
    main()
