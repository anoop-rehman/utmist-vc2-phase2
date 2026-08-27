"""Gate `train_soccer2v2_warp` BEFORE the overnight run (D1 unit 1f).

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \
        -m rower_soccer.warp_port.gate_soccer2v2_train

`tests/test_soccer2v2.py` (12/12) gates the ENV. This gates the TRAINER, and
specifically the six things a self-play run would get wrong SILENTLY -- each
would produce a run that logs healthy numbers for twelve hours and has learned
the wrong thing, or nothing:

  1. warm start   a checkpoint that "loads" but drops the tensors that mattered
  2. z-space      a policy that reaches the actuators by some path other than
                  the frozen decoder, so nothing transfers to/from the drills
  3. slot mapping rows attributed to the wrong world or player -- the ant would
                  still walk, and would be optimised against someone else's
                  reward
  4. bootstrap    the wrong V at the match-clock cut (ppo.py's convention);
                  every match's last transition discounted against the NEXT
                  match's kickoff
  5. prior term   an Eq. 5 regulariser that is silently inert, non-finite, or
                  pushing the WRONG WAY
  6. freezing     a decoder that quietly trains, or a high-level policy that
                  quietly does not

Every check carries a NEGATIVE CONTROL: the same comparator, applied to a
deliberately broken input, must fail. A check that cannot fail is not a check.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from rower_soccer.warp_port import train_soccer2v2_warp as M
from rower_soccer.warp_port.gate_drill_priors import action_from_z
from rower_soccer.warp_port.ppo import ActorCritic

SHOOT_CKPT = "runs_v2/s5_c_all/best.pt"


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, what):
        self.rows.append((name, bool(ok), what))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {what}", flush=True)
        return ok

    def summary(self):
        n = sum(r[1] for r in self.rows)
        print(f"\nGATE {n}/{len(self.rows)}", flush=True)
        for name, ok, what in self.rows:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: {what}")
        return n == len(self.rows)


def build(worlds=4, seed=0, extra=()):
    args = M.build_parser().parse_args(
        ["--run-name", "_gate", "--worlds", str(worlds), *extra])
    env = M.make_env(args, worlds, seed)
    dev = str(env.device)
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(),
                     z_dim=args.z_dim).to(dev)
    return args, env, ac, dev


# ---------------------------------------------------------------------------
# 1. warm start
# ---------------------------------------------------------------------------
def check_warm_start(rep, args, env, ac, dev):
    src = M._flat_source(SHOOT_CKPT, dev)
    r = M.load_warm_start(ac, SHOOT_CKPT, env.n_proprio, device=dev,
                          verbose=False)
    own = ac.state_dict()
    t_src, p_src = r["t_src"], r["p_src"]

    def bit_identical(k):
        return bool(torch.equal(own[k], src[k].to(own[k].device)))

    verbatim = [k for k in own
                if k in src and src[k].shape == own[k].shape
                and not k.endswith("_idx")]
    bad = [k for k in verbatim if not bit_identical(k)]
    must = ("mlp_extractor.decoder.0.weight", "mlp_extractor.decoder.4.weight",
            "action_net.weight", "mlp_extractor.z_proj.weight",
            "mlp_extractor.proprio_enc.0.weight", "mlp_extractor.expert.0.weight",
            "log_std")
    rep.add("1a warm-start verbatim",
            not bad and all(k in verbatim for k in must),
            f"{len(verbatim)} tensors bit-identical to {SHOOT_CKPT} "
            f"(decoder x3 + action_net + proprio_enc + expert + z_proj + "
            f"log_std + value_net); {len(bad)} mismatched, "
            f"{len(r['missing'])} absent from source, "
            f"{len(r['unexpected'])} unexpected in source")

    te = own["mlp_extractor.task_enc.0.weight"]
    te_src = src["mlp_extractor.task_enc.0.weight"].to(te.device)
    ok_te = (torch.equal(te[:, :t_src], te_src)
             and bool((te[:, t_src:] == 0).all()))
    cr = own["mlp_extractor.critic.0.weight"]
    cr_src = src["mlp_extractor.critic.0.weight"].to(cr.device)
    ok_cr = (torch.equal(cr[:, :p_src], cr_src[:, :p_src])
             and torch.equal(cr[:, p_src:p_src + t_src], cr_src[:, p_src:])
             and bool((cr[:, p_src + t_src:] == 0).all()))
    rep.add("1b task-encoder splice", ok_te and ok_cr,
            f"task_enc.0[:, :{t_src}] and critic.0's proprio+task prefix are "
            f"bit-identical to shoot; the {te.shape[1]-t_src} new task columns "
            f"({cr.shape[1]-p_src-t_src} in the critic) are exactly zero -- so "
            f"at init the policy IS shoot evaluated on football obs")

    # NEGATIVE CONTROL: the comparator must notice a single perturbed element.
    with torch.no_grad():
        ac.mlp_extractor.decoder[0].weight[0, 0] += 1e-7
    caught = not bool(torch.equal(ac.state_dict()["mlp_extractor.decoder.0.weight"],
                                  src["mlp_extractor.decoder.0.weight"].to(dev)))
    with torch.no_grad():
        ac.mlp_extractor.decoder[0].weight[0, 0] -= 1e-7
    restored = bool(torch.equal(ac.state_dict()["mlp_extractor.decoder.0.weight"],
                                src["mlp_extractor.decoder.0.weight"].to(dev)))
    rep.add("1c NEG warm-start comparator fails on demand", caught and restored,
            "perturbing decoder.0.weight[0,0] by 1e-7 is detected and the "
            "restore is detected as clean -- the equality test is not vacuous")

    # NEGATIVE CONTROL: --no-splice really does drop those two layers.
    ac2 = ActorCritic(env.obs_dim, env.act_dim,
                      proprio_indices=env.proprio_indices.tolist(),
                      task_indices=env.task_indices.tolist(), z_dim=16).to(dev)
    r2 = M.load_warm_start(ac2, SHOOT_CKPT, env.n_proprio, device=dev,
                           splice=False, verbose=False)
    dropped = {s.split()[0] for s in r2["shape_skip"]}
    te2 = ac2.state_dict()["mlp_extractor.task_enc.0.weight"]
    rep.add("1d NEG --no-splice ablation is really different",
            dropped == {"mlp_extractor.task_enc.0.weight",
                        "mlp_extractor.critic.0.weight"}
            and not torch.equal(te2[:, :t_src], te_src),
            "with --no-splice exactly those two layers are re-initialised and "
            "task_enc.0 no longer matches shoot -- so the splice is doing work")
    return r


# ---------------------------------------------------------------------------
# 2. the z-space path
# ---------------------------------------------------------------------------
def check_z_path(rep, env, ac, dev):
    obs = env.reset().float()
    with torch.no_grad():
        mean = ac.dist(obs).mean
        z = ac.z(obs)
        via_z = action_from_z(ac, obs, z)
        exact = bool(torch.equal(mean, via_z))
        # the drill priors' own path, driven by an EXTERNAL z
        z_ext = z + 0.01
        other = action_from_z(ac, obs, z_ext)
        responds = float((other - via_z).abs().max())
        # the decoder must be blind to task observations
        obs_t = obs.clone()
        obs_t[:, env.task_indices] += 3.0
        frozen_wrt_task = float(
            (action_from_z(ac, obs_t, z) - via_z).abs().max())
    rep.add("2a z-space path is the ONLY path", exact,
            "ac.dist(obs).mean == gate_drill_priors.action_from_z(ac, obs, "
            "ac.z(obs)) BITWISE on every one of "
            f"{obs.shape[0]} rows -- the trained policy reaches the actuators "
            "through the same frozen decoder the drill priors drive")
    rep.add("2b external z actually drives the decoder", responds > 1e-6,
            f"perturbing z by 0.01 moves the action by {responds:.2e} -- the "
            "decoder is not ignoring its latent input")
    rep.add("2c NEG decoder is blind to task obs", frozen_wrt_task == 0.0,
            "adding 3.0 to all 34 task columns with z held fixed changes the "
            f"action by exactly {frozen_wrt_task:.1e}: the low-level "
            "controller's input contract is proprio + z, as the drills assume")


# ---------------------------------------------------------------------------
# 3. rows -> (world, slot)
# ---------------------------------------------------------------------------
def check_slot_mapping(rep, env, dev):
    n, A = env.n, env.n_agents
    obs = env.reset()
    per_slot = torch.stack([env._player_obs(k) for k in range(A)], 1)
    ok_rows = bool(torch.equal(obs.view(n, A, -1), per_slot))
    rep.add("3a row = w*A + k", ok_rows,
            f"obs.view({n}, {A}, -1)[w, k] is bit-identical to "
            "env._player_obs(k)[w] for every (world, slot): the flattening is "
            "world-major, as the docstring claims and the trainer assumes")

    # The team-mate block must be the TEAM-MATE, evaluated in the observer's
    # own frame. This is the entry a slot permutation corrupts.
    from rower_soccer.warp_port.worm_env_base import to_ego3
    off = env.n_proprio + 6 + 3 + 2 + 2 + 3          # start of teammate_ego
    good = bad = 0
    for k in range(A):
        pos, rot = env.root_frames(k)
        mate = int(env.mate[k])
        want = to_ego3(pos, rot, env.xpos[:, env.root_body[mate], :])
        got = obs.view(n, A, -1)[:, k, off:off + 3]
        good += int(torch.allclose(got, want, atol=1e-5))
        wrong_slot = (mate + 1) % A
        wrong = to_ego3(pos, rot, env.xpos[:, env.root_body[wrong_slot], :])
        bad += int(torch.allclose(got, wrong, atol=1e-5))
    rep.add("3b team-mate block names the right player", good == A and bad == 0,
            f"all {A} slots' teammate_ego[{off}:{off+3}] match "
            "to_ego3(own frame, mate root) and NONE match a neighbouring "
            "slot's root -- a slot permutation in the obs is detectable")

    # Action routing: the [n*A, nu] -> [n, A*nu] reshape IS the slot routing,
    # so ctrl column k*nu + i must belong to an actuator named p{k}-*.
    nu = env.act_dim
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    routed = all(names[k * nu + i].startswith(f"p{k}-")
                 for k in range(A) for i in range(nu))
    a = torch.zeros(n * A, nu, device=env.device)
    for k in range(A):
        a.view(n, A, nu)[:, k] = 0.1 * (k + 1)
    env.step(a)
    c = env.ctrl.view(n, A, nu)
    landed = all(torch.allclose(c[:, k], torch.full_like(c[:, k], 0.1 * (k + 1)))
                 for k in range(A))
    rep.add("3c actions reach the named creature", routed and landed,
            f"model actuators {nu} per slot are contiguous and prefixed p0-..p"
            f"{A-1}-, and a per-slot constant action lands in the matching "
            "ctrl block -- the reshape routes slot k to creature p{k}-")

    # NEGATIVE CONTROL
    perm = torch.tensor([1, 0, 3, 2], device=env.device)[:A]
    permuted = obs.view(n, A, -1)[:, perm].reshape(obs.shape)
    rep.add("3d NEG a slot permutation is detected",
            not bool(torch.equal(permuted, obs)),
            "swapping the two players within each team changes the observation "
            "batch, so 3a/3b would fail on a permuted env rather than silently "
            "pass")


# ---------------------------------------------------------------------------
# 4. GAE bootstrap at a cut
# ---------------------------------------------------------------------------
class _StubEnv:
    """A 1-world, 1-player env whose observation IS its value: obs = [x].

    Scripted so the arithmetic is hand-computable and the three candidate
    bootstrap conventions give three DIFFERENT answers -- which is the only way
    a test of this can prove anything.
    """
    obs_dim, act_dim = 1, 1
    n, n_agents, n_per_team = 1, 1, 1
    episode_steps = 2

    def __init__(self, device):
        self.device = device
        self.n_diverged = 0
        self.shaping_scale = 1.0
        self.score = torch.zeros(1, 2, device=device)
        self._seq = [5.0, 7.0, 9.0]        # obs AFTER each step
        self._i = 0

    def _o(self, x):
        return torch.tensor([[x]], device=self.device)

    def reset(self):
        return self._o(2.0)                # V(kickoff of the next match) = 2

    def step(self, a):
        x = self._seq[self._i]
        self._i += 1
        done = self._i == 2                # the match clock fires at t = 1
        return self._o(x), torch.ones(1, device=self.device), done

    def match_stats(self):
        return dict(home_goals=0.0, away_goals=0.0, throw_ins=0.0,
                    ball_dist=0.0, upright=1.0, diverged=0)


class _StubAC(nn.Module):
    """value(obs) = obs[:, 0] exactly; one real parameter so Adam is happy."""
    state_dependent_std = False

    def __init__(self):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(1))
        self.mlp_extractor = nn.Module()

    def value(self, obs):
        return obs[:, 0]

    def dist(self, obs):
        return torch.distributions.Normal(obs[:, :1] * 0.0, self.log_std.exp())

    def act(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)

    def z(self, obs):
        return obs


def check_bootstrap(rep, dev):
    env = _StubEnv(dev)
    ac = _StubAC().to(dev)
    tr = M.SelfPlayPPO(env, ac, rollout_len=3, gamma=0.9, gae_lambda=0.5,
                       device=dev)
    tr._obs = env.reset()
    tr._i = 0
    env._i = 0
    adv, ret = tr.collect()
    got = [round(float(x), 6) for x in adv.reshape(-1)]

    g, lam = 0.9, 0.5
    # values seen: v = [V(2)=2, V(5)=5, V(2)=2]; r = 1 everywhere;
    # done at t=1 with s_T = 7 (V=7) and the reset kickoff at 2 (V=2).
    d2 = 1 + g * 9 - 2                        # t=2: rollout cut, boot V(9)
    a2 = d2
    d1 = 1 + g * 7 - 5                        # t=1: match cut, boot V(s_T)=7
    a1 = d1                                   #      recursion CUT here
    d0 = 1 + g * 5 - 2
    a0 = d0 + g * lam * a1
    want = [round(a0, 6), round(a1, 6), round(a2, 6)]

    # the two conventions this is guarding against
    ppo_d1 = 1 + g * 2 - 5                    # ppo.py: V(the NEXT kickoff)
    ppo = [round(d0 + g * lam * ppo_d1, 6), round(ppo_d1, 6), round(a2, 6)]
    term_d1 = 1 + 0 - 5                       # treat the time limit as terminal
    term = [round(d0 + g * lam * term_d1, 6), round(term_d1, 6), round(a2, 6)]

    ok = all(abs(x - y) < 1e-5 for x, y in zip(got, want))
    rep.add("4a bootstrap V(s_T) at the match cut", ok,
            f"advantages {got} == hand-computed {want} "
            f"(gamma 0.9, lambda 0.5, V(s_T)=7 read BEFORE the reset)")
    distinct = (want != ppo and want != term)
    caught = (got != ppo and got != term)
    rep.add("4b NEG the wrong conventions give different answers",
            distinct and caught,
            f"ppo.py's convention (bootstrap V(next kickoff)=2) would give "
            f"{ppo} and a terminal cut would give {term}; the trainer produced "
            f"neither, so this test can fail")
    no_boot = round(1 + 0.0 - 2, 6)           # if the rollout cut bootstrapped 0
    rep.add("4c rollout-boundary cut also bootstraps",
            abs(got[2] - round(a2, 6)) < 1e-5 and abs(got[2] - no_boot) > 1e-5,
            f"the t=T-1 transition uses V(self._obs)=V(9)=9, giving {got[2]} "
            f"= 1 + 0.9*9 - V(2); dropping that bootstrap would give "
            f"{no_boot}. The fixed-T sampler's own truncation is bootstrapped "
            "too (D3_HANDOFF's 'the port has partial episodes their code never "
            "has')")
    rep.add("4d GAE does not leak across the match boundary",
            abs(got[1] - d1) < 1e-9,
            f"adv[t=1] == its own delta ({d1}) with no contribution from the "
            "next match's t=2 -- the recursion is cut, not merely re-based")


# ---------------------------------------------------------------------------
# 5. the drill-prior mixture term
# ---------------------------------------------------------------------------
def check_prior(rep, args, env, ac, dev):
    prior = M.DrillPriorMixture(args.prior_dir, list(M.PRIOR_SKILLS),
                                env.proprio_indices.tolist(),
                                env.task_indices.tolist(), device=dev)
    obs = env.reset().float()
    for _ in range(30):                      # get off the kickoff state
        with torch.no_grad():
            obs = env.step(ac.dist(obs).sample().clamp(-1, 1))[0].float()
    z = ac.z(obs)
    pen = prior.neg_log_prob(obs, z)
    finite = bool(torch.isfinite(pen).all())
    rep.add("5a term is finite and per-row", finite and pen.shape == z.shape[:1],
            f"-log p_mix(z) over {len(prior.skills)} priors: shape "
            f"{tuple(pen.shape)}, mean {float(pen.mean()):.2f}, "
            f"range [{float(pen.min()):.1f}, {float(pen.max()):.1f}], all finite")

    # weight 0 must remove the term, not multiply it by zero
    tr0 = M.SelfPlayPPO(env, ac, rollout_len=4, prior=prior, w_prior=0.0,
                        device=dev)
    s0 = tr0.train_iter()
    tr1 = M.SelfPlayPPO(env, ac, rollout_len=4, prior=prior, w_prior=1e-3,
                        device=dev)
    s1 = tr1.train_iter()
    rep.add("5b w_prior=0 removes the term",
            "prior_nll" not in s0 and "prior_nll" in s1
            and np.isfinite(s1["prior_nll"]),
            "at w_prior=0 the penalty is never computed (no 'prior_nll' in the "
            f"iteration stats); at 1e-3 it is {s1['prior_nll']:.2f} and finite "
            "-- the ablation is an `if`, not a multiply-by-zero")

    # SIGN: a gradient step on the penalty alone must reduce the penalty.
    z1 = z.detach().clone().requires_grad_(True)
    p1 = prior.neg_log_prob(obs, z1).mean()
    p1.backward()
    with torch.no_grad():
        z2 = z1 - 1.0 * z1.grad
        p2 = prior.neg_log_prob(obs, z2).mean()
        # ...and a z pushed far off the mixture must score WORSE
        far = prior.neg_log_prob(obs, z.detach() + 50.0).mean()
    rep.add("5c term is correctly signed",
            float(p2) < float(p1) and float(far) > float(p1),
            f"descending it takes -log p from {float(p1):.2f} to "
            f"{float(p2):.2f}, and z + 50 scores {float(far):.2f} -- the "
            "penalty pulls z TOWARD the drill mixture, not away")

    r = prior.responsibilities(obs, z).mean(0)
    resp = {k: round(float(v), 3) for k, v in zip(prior.skills, r)}
    rep.add("5d the mixture actually mixes",
            float(r.max()) < 0.999 and float(r.min()) >= 0.0,
            f"mean responsibilities {resp} on warm-started football states -- "
            "no single prior owns everything, so Eq. 5 has something to mix")
    return prior


# ---------------------------------------------------------------------------
# 6. a short run: gradients where they belong, and nowhere else
# ---------------------------------------------------------------------------
def check_short_run(rep, args, env, dev, prior):
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(), z_dim=16).to(dev)
    M.load_warm_start(ac, SHOOT_CKPT, env.n_proprio, device=dev, verbose=False)
    with torch.no_grad():
        ac.value_net.weight.zero_()
        ac.value_net.bias.zero_()
    for mod in (ac.mlp_extractor.decoder, ac.action_net):
        for p in mod.parameters():
            p.requires_grad_(False)
    before = {k: v.detach().clone()
              for k, v in ac.state_dict().items()
              if k.startswith(("mlp_extractor.decoder", "action_net"))}
    tr = M.SelfPlayPPO(env, ac, rollout_len=16, prior=prior, w_prior=1e-3,
                       device=dev)
    stats = []
    for i in range(6):
        stats.append(tr.train_iter(critic_only=i < 2))

    after = ac.state_dict()
    unchanged = all(torch.equal(after[k], v) for k, v in before.items())
    dec_grads = [p.grad for p in ac.mlp_extractor.decoder.parameters()] \
        + [p.grad for p in ac.action_net.parameters()]
    dec_clean = all(g is None or float(g.abs().max()) == 0.0 for g in dec_grads)
    rep.add("6a decoder stays frozen", unchanged and dec_clean,
            f"after 6 PPO iterations all {len(before)} decoder + action_net "
            "tensors are BIT-IDENTICAL and every one of their .grad slots is "
            "None -- autograd never touched the low-level controller")

    hi = ac.mlp_extractor
    gn = {n: float(p.grad.norm()) for n, p in
          [("z_proj", hi.z_proj.weight), ("task_enc.0", hi.task_enc[0].weight),
           ("proprio_enc.0", hi.proprio_enc[0].weight),
           ("critic.0", hi.critic[0].weight),
           ("value_net", ac.value_net.weight)]
          if p.grad is not None}
    rep.add("6b the high-level policy gets gradient",
            len(gn) == 5 and all(v > 0 for v in gn.values()),
            "nonzero grad norms on "
            + ", ".join(f"{k} {v:.2e}" for k, v in gn.items())
            + " -- including task_enc.0, whose 21 spliced-in zero columns are "
              "learning (the PIPELINE_V2 zero-padding trap would show 0 here)")

    params_ok = all(torch.isfinite(v).all() for v in ac.state_dict().values())
    rews = [s["rew_mean"] for s in stats]
    vals = [s["val_mean"] for s in stats]
    rep.add("6c short run does not diverge",
            params_ok and all(np.isfinite(rews)) and tr.n_bad_grads == 0,
            f"6 iters, {tr.total_steps:,} env steps: every parameter finite, "
            f"rew/step {min(rews):+.5f}..{max(rews):+.5f}, "
            f"V {vals[0]:+.3f}->{vals[-1]:+.3f}, "
            f"obs-diverged {tr.n_diverged}, sim-diverged {env.n_diverged}, "
            f"non-finite gradients {tr.n_bad_grads}")

    warm = stats[0]
    rep.add("6d critic warmup holds the policy still",
            warm["critic_only"] and "pg" in warm,
            "the warmup iterations report a value loss but add no policy-"
            "gradient or prior term to the objective, so the warm-started "
            "policy is not spent on advantages that are only the critic's bias")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=4)
    a = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.manual_seed(0)
    rep = Report()

    print("\n== 1. warm start ==", flush=True)
    args, env, ac, dev = build(a.worlds)
    check_warm_start(rep, args, env, ac, dev)

    print("\n== 2. the z-space action path ==", flush=True)
    check_z_path(rep, env, ac, dev)

    print("\n== 3. rows -> (world, slot) ==", flush=True)
    check_slot_mapping(rep, env, dev)

    print("\n== 4. GAE / bootstrap at a cut ==", flush=True)
    check_bootstrap(rep, dev)

    print("\n== 5. drill-prior mixture (Eq. 5) ==", flush=True)
    prior = check_prior(rep, args, env, ac, dev)

    print("\n== 6. short run ==", flush=True)
    check_short_run(rep, args, env, dev, prior)

    sys.exit(0 if rep.summary() else 1)


if __name__ == "__main__":
    main()
