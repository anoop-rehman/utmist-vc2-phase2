"""Their policy, our env: does the gap live in the environment or in training?

Section 7 of `M2E_VALIDATION.md` established that our ants fall exactly as often
as theirs (32%) and score 6.5x less often, timing out instead. That compared two
DIFFERENT training runs, so it cannot distinguish:

  (a) our env is slower -- the same policy would travel less far in it, or
  (b) our training produced a worse gait in an env that is fine.

Loading their `epoch_0107` weights into our network and running them in our env
separates the two, because it holds the policy fixed and changes only the
simulator. Their weights in their env score 42.6%. If their weights in our env
also score ~42%, the env is exonerated and (b) is the answer. If they score ~7%,
the env is the problem and (a) is.

The weight mapping is one-to-one -- every tensor matches in shape -- which is
itself a check on the port: a network that had drifted structurally could not be
loaded at all.

    python -m rower_soccer.competevo_port.cross_eval --episodes 384
"""

import argparse
import collections
import pickle

import numpy as np
import torch

from rower_soccer.competevo_port.dev_env import CONTROL_DT, RunToGoalDevEnv
from rower_soccer.competevo_port.dev_ppo import DevActorCritic

# Their `lib/rl/core/running_norm.py` defaults to clip=5.0 and no call site ever
# overrides it. Ours defaults to 10.0. Their weights were trained under 5, so a
# cross-evaluation MUST use 5 or it is not running their policy.
THEIR_CLIP = 5.0

# theirs -> ours. Their MLPs are `affine_layers.k`; ours is an nn.Sequential
# with Tanh between, so Linear k sits at index 2k.
DIRECT = {
    "scale_state_log_std": "scale_log_std",
    "control_action_log_std": "control_log_std",
    "scale_state_mean.weight": "scale_mean.weight",
    "scale_state_mean.bias": "scale_mean.bias",
    "control_action_mean.weight": "control_mean.weight",
    "control_action_mean.bias": "control_mean.bias",
}


def remap(their_sd):
    """Returns (our_state_dict_fragment, notes). Raises on anything unmapped so
    a silently-dropped tensor cannot be mistaken for a successful load."""
    out, unmapped = {}, []
    for k, v in their_sd.items():
        if k in DIRECT:
            out[DIRECT[k]] = v.reshape(-1) if v.ndim == 2 and v.shape[0] == 1 else v
        elif k.endswith((".n", ".mean", ".var")):
            out[k] = v.reshape(1) if k.endswith(".n") else v
        elif k.endswith(".std"):
            continue  # ours derives std from var at forward time
        elif ".affine_layers." in k:
            head, rest = k.split(".affine_layers.")
            idx, param = rest.split(".")
            out[f"{head}.{int(idx) * 2}.{param}"] = v
        else:
            unmapped.append(k)
    if unmapped:
        raise KeyError(f"unmapped tensors from their checkpoint: {unmapped}")
    return out


def load_their_pair(ckpt_dir, ckpt, device):
    acs = []
    for i in (0, 1):
        path = f"{ckpt_dir}/agent_{i}/epoch_{ckpt:04d}.p"
        with open(path, "rb") as f:
            cp = pickle.load(f)
        assert cp["running_state"] is None, \
            "their checkpoint carries an external obs normalizer; port it first"
        ac = DevActorCritic().to(device)
        frag = remap(cp["policy_dict"])
        missing, unexpected = ac.load_state_dict(frag, strict=False)
        # Only the critic may be missing: their policy checkpoint has no critic
        # in it, and the critic is not consulted by `mean_action`.
        bad = [m for m in missing if not (m.startswith("vf") or
                                          m.startswith("value_net"))]
        assert not bad and not unexpected, (bad, unexpected)
        for norm in (ac.scale_norm, ac.control_norm):
            norm.clip = THEIR_CLIP
        ac.eval()
        acs.append(ac)
    return acs


def load_our_pair(path, device):
    blob = torch.load(path, map_location="cpu")
    acs = []
    for key in ("ac_0", "ac_1"):
        ac = DevActorCritic().to(device)
        ac.load_state_dict(blob[key])
        ac.eval()
        acs.append(ac)
    return acs


@torch.no_grad()
def rollout(env, acs, target_games):
    """Endings plus the two travel numbers, over whole episodes."""
    endings = collections.Counter()
    peak_v, travel, ep_len = [], [], []
    wins = np.zeros(env.n_agents)
    games = 0

    obs = env.reset()
    env.reset_win_stats()
    # NOT the post-reset position: the design step that follows every reset ends
    # with `_apply_designs` writing qpos0, so the position an execution rollout
    # actually starts from is the one AFTER that step. Capturing it at reset
    # measures displacement from a pose the episode never occupied.
    start_x = env._agent_com_x().clone()
    run_peak = torch.zeros(env.n, env.n_agents, device=env.device,
                           dtype=env.dtype)
    # How often does the 5-vs-10 clip difference actually bite? Counted on the
    # control tower's input, which is the one that drives the legs.
    over5 = over10 = seen = 0

    while games < target_games:
        o = obs.float()
        a = torch.stack([acs[i].mean_action(o[:, i])
                         for i in range(env.n_agents)], dim=1)
        norm = acs[0].control_norm
        if float(norm.n) > 0:
            z = (o[:, :, 21:] - norm.mean) / (norm.var.sqrt() + 1e-8)
            over5 += int((z.abs() > 5).sum())
            over10 += int((z.abs() > 10).sum())
            seen += int(z.numel())

        obs, _, done, info = env.step(a.to(env.dtype))
        run_peak = torch.maximum(run_peak, info["forward"])
        if bool(info["was_design"].any()):
            start_x = torch.where(info["was_design"].unsqueeze(-1),
                                  env._agent_com_x(), start_x)

        if bool(done.any()):
            idx = done.nonzero(as_tuple=True)[0]
            won = info["winner"][idx].any(-1)
            fell = info["fell"][idx].any(-1)
            trunc = info["truncated"][idx]
            for w, f, t in zip(won.tolist(), fell.tolist(), trunc.tolist()):
                endings["goal" if w else "fell" if f
                        else "timeout" if t else "other"] += 1
            wins += info["winner"][idx].sum(0).cpu().numpy()
            # `com_x` is read in `terms()` before the auto-reset, so this is the
            # position at the moment the episode ended, not after respawn.
            # `move_sign` is per AGENT, not per world -- it broadcasts.
            signed = env.move_sign * (info["com_x"][idx] - start_x[idx])
            travel.extend(signed.max(-1).values.float().cpu().tolist())
            peak_v.extend(run_peak[idx].max(-1).values.float().cpu().tolist())
            ep_len.extend(env.last_len[idx].float().cpu().tolist())
            run_peak[idx] = 0.0
            games += len(idx)

    return {"endings": endings, "games": games, "wins": wins / max(games, 1),
            "travel": np.array(travel), "peak_v": np.array(peak_v),
            "ep_len": np.array(ep_len),
            "clip_bite": (over5 / max(seen, 1), over10 / max(seen, 1))}


def report(name, r):
    n = r["games"]
    e = r["endings"]
    print(f"\n{name}   {n} games")
    for k in ("goal", "fell", "timeout", "other"):
        if e[k]:
            print(f"    {k:8s} {e[k]:5d}  {100 * e[k] / n:5.1f}%")
    print(f"    episode length   {r['ep_len'].mean():6.1f} of 500")
    print(f"    travel toward goal (best agent)  mean {r['travel'].mean():6.2f} m"
          f"   median {np.median(r['travel']):6.2f} m"
          f"   p90 {np.percentile(r['travel'], 90):6.2f} m")
    print(f"    peak forward speed               mean {r['peak_v'].mean():6.2f} m/s"
          f"   p90 {np.percentile(r['peak_v'], 90):6.2f} m/s")
    print(f"    win rate {np.round(r['wins'], 4)}  summed {r['wins'].sum():.4f}")
    o5, o10 = r["clip_bite"]
    print(f"    control-obs components beyond 5 sigma {100 * o5:.3f}%"
          f"   beyond 10 sigma {100 * o10:.3f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--their-ckpt-dir",
                   default="/workspace/competevo/tmp/run-to-goal-devants-v0/"
                           "20260810_211247/models")
    p.add_argument("--our-policies",
                   default="runs/competevo_port/m2e_validation/policies.pt")
    p.add_argument("--ckpt", type=int, default=107)
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--episodes", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = RunToGoalDevEnv(num_worlds=args.worlds, use_gpu=(device == "cuda"),
                          seed=args.seed)

    print("goal line at |x| =", float(env.goal_x.abs().max()),
          " control dt =", CONTROL_DT)

    theirs = load_their_pair(args.their_ckpt_dir, args.ckpt, device)
    print(f"loaded their epoch_{args.ckpt:04d} weights into our network "
          f"(clip set to {THEIR_CLIP}, theirs)")
    report("THEIR policy in OUR env", rollout(env, theirs, args.episodes))

    ours = load_our_pair(args.our_policies, device)
    report("OUR policy in OUR env", rollout(env, ours, args.episodes))


if __name__ == "__main__":
    main()
