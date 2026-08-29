"""D3 M3 E1 smoke: does design+control training START on our ant, and are the
gradients sensible?

Deliberately NOT a training run. E1's real run is 3 seeds x ~100 epochs and
comes after this converter is trusted; a separate agent is running E0 on their
ant on the same GPU. What this answers is narrower and is the only thing that
should be answered before the converter is committed:

  1. `Transform2ActAgent` builds on `ant_competevo` -- the three GNN heads
     (skeleton, attribute, control) size themselves to a 13-body graph with an
     attr_design width of 5 and 8 actuators, not to their 5-body ant.
  2. A PPO iteration completes: sampling produces whole episodes, the design
     stages run, the execution stage returns reward.
  3. The gradients are finite, non-zero, and reach ALL THREE heads. A design
     head that receives no gradient would train a fixed body while looking
     exactly like a working run -- which is the failure this checks for.
  4. Two iterations change the parameters, i.e. the optimiser is actually
     stepping.

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/smoke_e1_train.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/Transform2Act")
os.chdir("/workspace/Transform2Act")

import torch  # noqa: E402
import yaml  # noqa: E402

from design_opt.agents.transform2act_agent import Transform2ActAgent  # noqa: E402
from design_opt.utils.config import Config  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main(epochs=2, batch=4000, threads=8):
    torch.set_default_dtype(torch.float64)
    files = "design_opt/cfg/ant_competevo.yml"
    cfg_dict = yaml.safe_load(open(files))
    cfg_dict["min_batch_size"] = batch
    cfg_dict["mini_batch_size"] = batch
    cfg_dict["eval_batch_size"] = batch
    cfg_dict["max_epoch_num"] = epochs
    cfg = Config("ant_competevo_smoke", tmp=True, cfg_dict=cfg_dict)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64, device=device,
                               seed=cfg.seed, num_threads=threads, training=True,
                               checkpoint=0)
    env = agent.env
    print(f"\nE1 SMOKE -- design+control training on the converted CompetEvo ant "
          f"({device})")
    check("env built on our ant", len(env.robot.bodies) == 13
          and env.attr_design_dim == 5,
          f"{len(env.robot.bodies)} bodies, attr_design_dim "
          f"{env.attr_design_dim}, sim_obs_dim {env.sim_obs_dim}, "
          f"{env.model.nu} actuators, xml {env.model_xml_file}")

    # `Transform2ActPolicy` prefixes every parameter with its head
    # (`skel_*`, `attr_*`, `control_*`), so these three sets are disjoint --
    # which is what makes "this head got a gradient" a real statement rather
    # than "the network got a gradient".
    heads = {"skeleton": "skel_", "attribute": "attr_", "control": "control_"}
    before = {k: [p.detach().clone() for n, p in agent.policy_net.named_parameters()
                  if n.startswith(v)] for k, v in heads.items()}
    for k, v in heads.items():
        check(f"{k} head has parameters", len(before[k]) > 0,
              f"{sum(p.numel() for p in before[k])} params")

    # One PPO iteration, with the gradients captured after the first backward.
    grads = {}
    orig = agent.optimizer_policy.step

    def spy(*a, **kw):
        if not grads:
            for k, v in heads.items():
                gs = [p.grad for n, p in agent.policy_net.named_parameters()
                      if n.startswith(v) and p.grad is not None]
                grads[k] = (len(gs),
                            float(sum(float(g.norm()) ** 2 for g in gs) ** 0.5),
                            all(bool(torch.isfinite(g).all()) for g in gs))
        return orig(*a, **kw)

    agent.optimizer_policy.step = spy
    for epoch in range(epochs):
        agent.optimize(epoch)
    agent.optimizer_policy.step = orig

    for k in heads:
        n, norm, finite = grads.get(k, (0, 0.0, False))
        check(f"{k} head receives a finite, non-zero gradient",
              n > 0 and finite and norm > 0,
              f"{n} tensors, ||g|| {norm:.4e}")

    after = {k: [p.detach().clone() for n, p in agent.policy_net.named_parameters()
                 if n.startswith(v)] for k, v in heads.items()}
    for k in heads:
        d = max((float((a - b).abs().max()) for a, b in zip(after[k], before[k])),
                default=0.0)
        check(f"{k} head parameters actually moved", d > 0, f"max|dw| {d:.3e}")

    print(f"\n  (this is a SMOKE, not a result: {epochs} epochs at "
          f"min_batch_size={batch}, against E1's real {yaml.safe_load(open(files))['min_batch_size']} "
          f"x {yaml.safe_load(open(files))['max_epoch_num']} epochs. No reward "
          f"number from it means anything.)")
    print()
    if FAIL:
        print(f"SMOKE FAILED: {FAIL}")
        sys.exit(1)
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
