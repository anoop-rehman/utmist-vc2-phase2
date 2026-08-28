"""Gate: the port's policy must START trainable, and one Adam step must move
every tower.

    cd /workspace/utmist-vc2-phase2
    PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_policy_init

CPU-only, no MuJoCo stepping, seconds. Run it before any training arm.

Why it exists
-------------
Every earlier gate on the policy (`gate_dense_policy.py`, 8/8) begins by
loading THEIR checkpoint with `strict=True`. That overwrites all 54 policy
tensors, so those gates measure the port's forward pass and are structurally
blind to the port's own INITIALISATION -- and they take no gradient step, so
they are blind to whether the parameters can move at all.

`runs/t2a_port/port_s1` trained for 1,000 epochs with `IndexLinear` allocated
as `torch.zeros` and never initialised, because the port copied their
allocation (`design_opt/models/jsmlp.py:14-15`) and dropped the
`reset_parameters()` call on the line below it. A zero `IndexLinear` is dead,
not merely small:

    a1 = tanh(W1 x + b1) = tanh(0) = 0          W1 = b1 = 0
    dL/dW3 = delta3 a2^T = 0                    a2 = tanh(W2 a1 + b2) = 0
    delta2 = W3^T delta3 * (1 - a2^2) = 0       W3 = 0

so only the last layer's bias ever receives a gradient, the GNN feeding the
stack receives none, and the policy is `linear.b[body_index]` -- a constant per
body type, independent of the observation, forever. Measured: 48 of the 53
trainable policy tensors in `port_s1` are bit-identical between `epoch_0100.p`
and `epoch_1000.p` -- 16 of each tower's 17. The five that moved are the two
`log_std`s and the three `ind_mlp.linear.b`s.

The three checks below are the ones that would have caught it at epoch 0.
"""
import math
import types

import torch

from rower_soccer.t2a_port.train_t2a import Trainer

TOWERS = ("skel", "attr", "control")


def build(zero_init=False):
    args = types.SimpleNamespace(
        cfg="hopper_gpu_s2", run="gate_init_tmp", outdir="/tmp",
        seed=0, batch_steps=1024, min_worlds=4, max_worlds=8, eval_worlds=4,
        epochs=0, device="cpu", backend="cpu", fp32=False, save_interval=1000,
        mempool_mb=-1, stop_file="", batch_design=None)
    if zero_init:
        # The negative control: put the bug back, the way it was written.
        import rower_soccer.t2a_port.dense_policy as dp
        orig = dp.IndexLinear.reset_parameters
        dp.IndexLinear.reset_parameters = lambda self: None
        try:
            tr = Trainer(args)
        finally:
            dp.IndexLinear.reset_parameters = orig
        return tr
    return Trainer(args)


def synthetic(tr, stage, G=16):
    """A block of `G` graphs of `N` nodes with random observations. The gate
    needs a gradient, not physics, so the observation is noise -- a real batch
    would only make the same check slower."""
    N = 4
    F_ = tr.state_dim
    g = torch.Generator().manual_seed(11)
    obs = torch.randn(G, N, F_, generator=g, dtype=tr.dtype)
    adj = torch.zeros(G, N, N, dtype=tr.dtype)
    for i in range(1, N):                       # a chain, root first
        adj[:, i, i - 1] = 1.0
        adj[:, i - 1, i] = 1.0
    ind = torch.arange(N).unsqueeze(0).expand(G, -1).clone()
    return obs, adj, ind


def main():
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
        if not ok:
            fails.append(name)

    print("check 1: a freshly-initialised policy has no dead tensor")
    tr = build()
    # Two families are zero AT INIT BY DESIGN and must be excluded, or the
    # check would be one nobody could make pass: the three `ind_mlp.linear.b`
    # that `rescale_linear` multiplies by 0, and any `*_action_log_std` whose
    # cfg value is literally 0 (`hopper_gpu_s2.yml:33`, `control_log_std: 0`).
    ps = tr.cfg["policy_specs"]
    expect_zero = {"skel.ind_mlp.linear.b", "attr.ind_mlp.linear.b",
                   "control.ind_mlp.linear.b"}
    for nm, key in (("control_action_log_std", "control_log_std"),
                    ("attr_action_log_std", "attr_log_std")):
        if float(ps[key]) == 0.0:
            expect_zero.add(nm)
    n_par = sum(1 for _ in tr.policy.named_parameters())
    dead = [n for n, p in tr.policy.named_parameters()
            if float(p.abs().max()) == 0.0 and n not in expect_zero]
    check("every policy tensor is nonzero at init "
          "(bar the ones their init sets to exactly 0: the three rescaled "
          "head biases and any log_std whose cfg value is 0)",
          not dead,
          f"dead: {dead}" if dead else
          f"{n_par} tensors, {len(expect_zero)} zero by design")

    print("check 2: their `rescale_linear` is applied, per tower")
    for t in TOWERS:
        mlp = getattr(tr.policy, t).ind_mlp
        w0 = mlp.affine_layers[0].W
        fan_in = w0.shape[1] * w0.shape[2]
        bound = math.sqrt(6.0 / ((1 + 5.0) * fan_in))    # kaiming_uniform, a=sqrt(5)
        hi = float(mlp.affine_layers[0].W.abs().max())
        lw = float(mlp.linear.W.abs().max())
        lb = float(mlp.linear.b.abs().max())
        # kaiming bound on the hidden layers, 0.1x that shape on the head,
        # and an exactly-zero head bias -- `jsmlp.py:54-56`.
        check(f"{t}: hidden W within kaiming bound, head W rescaled 0.1x, "
              f"head b zeroed",
              hi <= bound * 1.001 and hi > 0.5 * bound and lw > 0 and lb == 0.0,
              f"bound {bound:.4f} hidden_max {hi:.4f} head_max {lw:.4f} "
              f"head_b {lb:g}")

    print("check 3: the action DEPENDS ON THE OBSERVATION at init")
    for t, stage in zip(TOWERS, ("skel_trans", "attr_trans", "execution")):
        obs, adj, ind = synthetic(tr, stage)
        with torch.no_grad():
            h = tr.policy._stage_head(stage, obs, adj, ind)
        spread = float(h.std(0).max())
        check(f"{stage}: head varies across observations",
              spread > 1e-9, f"max std over graphs {spread:.3e}")

    print("check 4: ONE Adam step moves every parameter of every tower")

    def one_step(trainer):
        opt = torch.optim.Adam(trainer.policy.parameters(), lr=1e-3)
        before = {n: p.detach().clone()
                  for n, p in trainer.policy.named_parameters()}
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for stage in ("skel_trans", "attr_trans", "execution"):
            obs, adj, ind = synthetic(trainer, stage)
            act, _ = trainer.policy.act(stage, obs, adj, ind,
                                        generator=torch.Generator().manual_seed(3))
            lp = trainer.policy.log_prob(stage, obs, adj, ind, act.detach())
            adv = torch.randn(lp.shape[0], generator=torch.Generator().manual_seed(5),
                              dtype=trainer.dtype)
            loss = loss + (-(lp * adv).mean())          # a PPO surrogate at ratio 1
        loss.backward()
        gnorm = {t: 0.0 for t in TOWERS}
        for n, p in trainer.policy.named_parameters():
            if p.grad is not None:
                for t in TOWERS:
                    if n.startswith(t + "."):
                        gnorm[t] += float(p.grad.pow(2).sum())
        opt.step()
        delta = {n: float((p.detach() - before[n]).abs().max())
                 for n, p in trainer.policy.named_parameters()}
        return {t: math.sqrt(v) for t, v in gnorm.items()}, delta

    gnorm, delta = one_step(tr)
    for t in TOWERS:
        frozen = [n for n, d in delta.items()
                  if n.startswith(t + ".") and d == 0.0]
        check(f"{t}: grad norm > 0 and no parameter frozen",
              gnorm[t] > 0 and not frozen,
              f"|g| {gnorm[t]:.4e}  frozen {len(frozen)}"
              + (f" {frozen}" if frozen else ""))

    print("negative control: restore the bug (zero-init IndexLinear); "
          "checks 1, 3 and 4 must all fail")
    trz = build(zero_init=True)
    dead_z = [n for n, p in trz.policy.named_parameters()
              if float(p.abs().max()) == 0.0]
    obs, adj, ind = synthetic(trz, "execution")
    with torch.no_grad():
        spread_z = float(trz.policy._stage_head("execution", obs, adj, ind).std(0).max())
    gnorm_z, delta_z = one_step(trz)
    frozen_z = sorted(n for n, d in delta_z.items() if d == 0.0)
    check("control: the zero-init policy IS dead "
          "(tensors at zero, head constant, towers frozen)",
          len(dead_z) >= 18 and spread_z == 0.0 and len(frozen_z) >= 45,
          f"zero tensors {len(dead_z)}/54  head std {spread_z:.1e}  "
          f"frozen after one step {len(frozen_z)}/54  "
          f"moved: {[n for n in delta_z if delta_z[n] > 0]}")

    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILURES: ' + str(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
