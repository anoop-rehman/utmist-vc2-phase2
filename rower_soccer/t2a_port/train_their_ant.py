"""D3 M3 E0: run THEIR Transform2Act on THEIR ant, with a stop file.

Why not `train_t2a.py` (our GPU port): it has no ant path. See
`gate_their_ant.py`'s docstring for the four specific places it is hopper-only
(`hopper.xml` hardcoded, `sim_obs_dim = 5` against the ant's 13, a planar
`(height, ang)` root in `batched_exec_env.sim_obs`, and `index_base` computed
as `max_nchild + 1 = 3` where `ant.py:32` hardcodes 5). `results/` in the
reference contains only hopper runs; the port has never been run on an ant.

Why not their `design_opt/train.py` verbatim: it runs to `cfg.max_epoch_num`
with no way to stop it, and under NVIDIA MPS killing a CUDA client can corrupt
the live survivors -- which has destroyed two runs on this project. This is
their loop, unchanged, plus a between-epoch stop-file check so a run can be
ended without a signal. `--stop-file` matches `train_t2a.py`'s flag.

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
           CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python \
        /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/train_their_ant.py \
        --cfg ant_e0_s1 --num_threads 15 --stop-file /tmp/stop_ant_e0_s1
"""

import argparse
import os
import sys

sys.path.append("/workspace/Transform2Act")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--num_threads", type=int, default=20)
    p.add_argument("--gpu_index", type=int, default=0)
    p.add_argument("--epoch", default="0")
    p.add_argument("--stop-file", default=None,
                   help="touch this path to end the run cleanly after the "
                        "epoch in flight. Never send a signal: under MPS that "
                        "can take out unrelated CUDA clients.")
    args = p.parse_args()

    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    cfg = Config(args.cfg, tmp=False)
    device = (torch.device("cuda", index=args.gpu_index)
              if torch.cuda.is_available() else torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    start_epoch = int(args.epoch) if args.epoch.isnumeric() else args.epoch
    agent = Transform2ActAgent(cfg=cfg, dtype=dtype, device=device,
                               seed=cfg.seed, num_threads=args.num_threads,
                               training=True, checkpoint=start_epoch)
    agent.logger.info(f"E0: cfg {args.cfg}  seed {cfg.seed}  "
                      f"num_threads {args.num_threads}  "
                      f"max_epoch_num {cfg.max_epoch_num}  "
                      f"save_model_interval {cfg.save_model_interval}  "
                      f"stop_file {args.stop_file}")

    for epoch in range(start_epoch, cfg.max_epoch_num):
        agent.optimize(epoch)
        agent.save_checkpoint(epoch)
        torch.cuda.empty_cache()
        if args.stop_file and os.path.exists(args.stop_file):
            agent.logger.info(f"stop file {args.stop_file} present -- "
                              f"stopping cleanly after epoch {epoch}")
            # Save whatever the last epoch produced so the stop is not a loss.
            cp = "%s/epoch_%04d.p" % (cfg.model_dir, epoch + 1)
            if not os.path.exists(cp):
                import pickle
                from khrylib.utils.torch import to_cpu
                with to_cpu(agent.policy_net, agent.value_net):
                    pickle.dump({"policy_dict": agent.policy_net.state_dict(),
                                 "value_dict": agent.value_net.state_dict(),
                                 "running_state": agent.running_state,
                                 "loss_iter": agent.loss_iter,
                                 "best_rewards": agent.best_rewards,
                                 "epoch": epoch}, open(cp, "wb"))
                agent.logger.info(f"saved {cp}")
            break
    else:
        agent.logger.info("training done!")


if __name__ == "__main__":
    main()
