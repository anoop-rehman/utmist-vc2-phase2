"""D3 M3 E1.1: plain-MLP PPO **inside the Transform2Act ant env**.

The baseline for "is the GNN controller as good as ordinary PPO". Published
Ant numbers are NOT the baseline: `gym/envs/mujoco/ant.py` pays a +1.0/step
survive bonus and charges `0.5*sum(a^2)` control plus a contact cost, while
`design_opt/envs/ant.py:153-165` pays no survive bonus, charges
`1e-4*mean(a^2)` and no contact cost. Same name, different objective. So the
baseline is run in THEIR env, on OUR ant, with the same reward, the same
episode structure and the same step budget as the GNN arm; only the policy
architecture differs.

What is held identical to the GNN arm:

  * the env: `design_opt/envs/ant.py` on `assets/mujoco_envs/ant_competevo.xml`
    via the same cfg, so the same `done_condition` (max_ang 60, max_nsteps
    1000), the same reward, the same `init_height: false` reset;
  * the episode structure: `skel_transform_nsteps` (5) skeleton steps and one
    attribute step are STEPPED, with `env_specs.force_identity_design: true`
    forcing them to the identity so the body is not edited. Not skipped --
    skipping would change episode length and strip the stage flag's meaning
    from the observation;
  * the observation CONTENT: the MLP is fed the same per-body matrix the GNN
    receives (`attr_fixed | sim_obs | design_params`, 22 columns x nbody),
    flattened, plus the one-hot stage flag. The graph edges and body indices
    carry no extra information once the morphology is frozen;
  * gamma (0.995) and GAE lambda (0.95), read from the cfg, so the two arms
    optimise the SAME discounted objective;
  * the batch size (`min_batch_size`, 50,000 env steps), the number of PPO
    epochs per batch (10), the minibatch size (2048) and the clip epsilon
    (0.2), read from the cfg;
  * the step budget: `max_epoch_num` x `min_batch_size`.

What differs, deliberately, and is the thing under test:

  * the policy is a plain MLP over the flattened observation with a
    state-independent log-std, and the action space is the 8 actuators
    directly, rather than a per-body GNN emitting one scalar per node;
  * the learning rate is the published PPO-MuJoCo 3e-4 for both actor and
    critic (`--policy-lr`, `--value-lr` to change), not their 5e-5 policy lr.
    Their 5e-5 is tuned for their GNN; handing the MLP the same number would
    be tuning the baseline down.

What the MLP arm does NOT put in its PPO buffer: the six design steps. They
carry zero reward and an action that is discarded, so training a value
function on them would be fitting noise. They are still STEPPED, and they
still count against the step budget, so the budget comparison is honest. Six
steps of ~500-1000 is under 1%.

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
           CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    cd /workspace/Transform2Act && source env-gpu.sh
    setsid nohup .venv-gpu/bin/python .../t2a_port/train_e11_mlp.py \
        --cfg ant_e11_mlp_s1 --num-threads 12 \
        --stop-file /tmp/stop_e11_mlp_s1 &

**CPU only.** The MLP is ~90k parameters; a CUDA context would cost more than
it saves and would take VRAM from the GNN arm. `--device` exists but defaults
to cpu.
"""

import argparse
import json
import math
import multiprocessing
import os
import sys
import time

sys.path.append("/workspace/Transform2Act")
os.chdir("/workspace/Transform2Act")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

N_STAGES = 3


# --------------------------------------------------------------------------
# observation
# --------------------------------------------------------------------------
def flat_obs(all_obs):
    """`AntEnv._get_obs` returns [node_features, edges, stage, num_nodes,
    body_index]. With the morphology frozen, `edges`, `num_nodes` and
    `body_index` are constants, so the MLP loses nothing by taking the node
    features flattened plus the one-hot stage."""
    node = np.asarray(all_obs[0], dtype=np.float64).reshape(-1)
    stage = int(np.asarray(all_obs[2]).reshape(-1)[0])
    onehot = np.zeros(N_STAGES)
    onehot[stage] = 1.0
    return np.concatenate([node, onehot])


class RunningNorm:
    """Welford mean/var over observations, frozen during a rollout and updated
    between batches. Frozen matters: if the statistics moved inside a rollout
    the stored log-probs would not correspond to the observations the update
    replays, exactly the bug `topology_census.census` was fixed for."""

    def __init__(self, dim, clip=10.0):
        self.n = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)
        self.clip = clip

    def std(self):
        v = self.m2 / max(self.n - 1, 1)
        return np.sqrt(np.maximum(v, 1e-8))

    def __call__(self, x):
        if self.n < 2:
            return np.clip(x, -self.clip, self.clip)
        return np.clip((x - self.mean) / self.std(), -self.clip, self.clip)

    def update_from(self, n, s, ss):
        """Merge a worker's (count, sum, sumsq) in, Chan-Golub-LeVeque style."""
        if n == 0:
            return
        b_mean = s / n
        b_m2 = ss - n * b_mean ** 2
        delta = b_mean - self.mean
        tot = self.n + n
        self.mean = self.mean + delta * n / tot
        self.m2 = self.m2 + b_m2 + delta ** 2 * self.n * n / tot
        self.n = tot

    def state(self):
        return {"n": self.n, "mean": self.mean.tolist(), "m2": self.m2.tolist()}

    def load(self, d):
        self.n = d["n"]
        self.mean = np.asarray(d["mean"])
        self.m2 = np.asarray(d["m2"])


# --------------------------------------------------------------------------
# nets -- SB3/CleanRL-shaped: two hidden layers, tanh, orthogonal init,
# state-independent log-std.
# --------------------------------------------------------------------------
def mlp(sizes, out_gain):
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        gain = out_gain if i == len(sizes) - 2 else math.sqrt(2)
        nn.init.orthogonal_(lin.weight, gain)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hdims, log_std):
        super().__init__()
        self.net = mlp([obs_dim] + hdims + [act_dim], 0.01)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std)

    def dist(self, obs):
        mu = self.net(obs)
        return torch.distributions.Normal(mu, self.log_std.exp())

    def select_action(self, obs, mean_action=False):
        d = self.dist(obs)
        a = d.mean if mean_action else d.sample()
        return a, d.log_prob(a).sum(-1)


class Critic(nn.Module):
    def __init__(self, obs_dim, hdims):
        super().__init__()
        self.net = mlp([obs_dim] + hdims + [1], 1.0)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# --------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------
class Trainer:
    def __init__(self, args):
        from design_opt.envs.ant import AntEnv
        from design_opt.utils.config import Config

        self.args = args
        self.cfg = Config(args.cfg, tmp=False)
        assert self.cfg.env_specs.get('force_identity_design', False), (
            f"{args.cfg} must set env_specs.force_identity_design: true -- "
            "the MLP arm has no design head, so an unforced design stage "
            "would edit the body with a zero action's worth of nothing on "
            "some steps and not others.")
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.env = AntEnv(self.cfg, agent=None)
        self.nbody = len(self.env.robot.bodies)
        self.act_width = self.env.control_action_dim + self.env.attr_design_dim + 1

        # which body rows carry an actuator -- fixed, because the body is
        # frozen. `action_to_control` looks actuators up BY NAME.
        names = list(self.env.model.actuator_names)
        self.act_rows = [i for i, b in enumerate(self.env.robot.bodies)
                         if i > 0 and b.get_actuator_name() in names]
        assert len(self.act_rows) == self.env.model.nu

        obs_dim = flat_obs(self.env.reset()).shape[0]
        self.obs_dim = obs_dim
        self.act_dim = self.env.model.nu
        hdims = [int(x) for x in args.hdims.split(",")]
        self.actor = Actor(obs_dim, self.act_dim, hdims, args.log_std)
        self.critic = Critic(obs_dim, hdims)
        self.norm = RunningNorm(obs_dim)
        self.opt_p = torch.optim.Adam(self.actor.parameters(), lr=args.policy_lr)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=args.value_lr)

        self.gamma = self.cfg.gamma
        self.lam = self.cfg.tau
        self.clip = self.cfg.clip_epsilon
        self.batch = args.batch or self.cfg.min_batch_size
        self.mini = args.mini_batch or self.cfg.mini_batch_size
        self.optim_epochs = args.optim_epochs or self.cfg.num_optim_epoch
        self.out = (f"/workspace/Transform2Act/results/{args.cfg}"
                    + (f"_{args.tag}" if args.tag else ""))
        os.makedirs(self.out, exist_ok=True)
        self.log_path = os.path.join(self.out, "log.jsonl")

    # ---- rollout ---------------------------------------------------------
    def run_design_stages(self, state):
        """5 skeleton steps + 1 attribute step, identity-forced by the env."""
        zero = np.zeros((self.nbody, self.act_width))
        n = 0
        while self.env.if_use_transform_action() != 2:
            state, _, done, info = self.env.step(zero)
            n += 1
            if done:
                return None, n
            if n > self.cfg.skel_transform_nsteps + 4:
                raise RuntimeError("design stages did not terminate")
        return state, n

    def sample_worker(self, pid, queue, min_steps, mean_action):
        if pid > 0:
            torch.manual_seed(self.cfg.seed * 7919 + pid * 104729 + self.epoch)
            self.env.np_random.seed((self.cfg.seed * 131 + pid * 977
                                     + self.epoch * 31) % (2 ** 31 - 1))
        obs_b, act_b, lp_b, rew_b, mask_b = [], [], [], [], []
        ep_rets, ep_lens = [], []
        raw_n, raw_s, raw_ss = 0, np.zeros(self.obs_dim), np.zeros(self.obs_dim)
        design_steps = 0
        with torch.no_grad():
            while len(rew_b) < min_steps:
                state = self.env.reset()
                state, nd = self.run_design_stages(state)
                design_steps += nd
                if state is None:
                    continue
                ret, n = 0.0, 0
                while True:
                    raw = flat_obs(state)
                    raw_n += 1
                    raw_s += raw
                    raw_ss += raw ** 2
                    o = self.norm(raw)
                    ot = torch.as_tensor(o, dtype=torch.float64).unsqueeze(0)
                    a, lp = self.actor.select_action(ot, mean_action)
                    a = a.numpy()[0]
                    full = np.zeros((self.nbody, self.act_width))
                    full[self.act_rows, 0] = a
                    state, r, done, info = self.env.step(full)
                    obs_b.append(o)
                    act_b.append(a)
                    lp_b.append(float(lp.item()))
                    rew_b.append(float(r))
                    mask_b.append(0.0 if done else 1.0)
                    ret += r
                    n += 1
                    if done:
                        break
                ep_rets.append(ret)
                ep_lens.append(n)
        out = dict(obs=np.asarray(obs_b), act=np.asarray(act_b),
                   lp=np.asarray(lp_b), rew=np.asarray(rew_b),
                   mask=np.asarray(mask_b), ep_rets=ep_rets, ep_lens=ep_lens,
                   raw=(raw_n, raw_s, raw_ss), design_steps=design_steps)
        if queue is not None:
            queue.put([pid, out])
        else:
            return out

    def sample(self, nthreads, mean_action=False):
        per = int(math.floor(self.batch / nthreads))
        queue = multiprocessing.Queue()
        outs = [None] * nthreads
        procs = []
        for i in range(nthreads - 1):
            p = multiprocessing.Process(target=self.sample_worker,
                                        args=(i + 1, queue, per, mean_action))
            p.start()
            procs.append(p)
        outs[0] = self.sample_worker(0, None, per, mean_action)
        for _ in range(nthreads - 1):
            pid, o = queue.get()
            outs[pid] = o
        for p in procs:
            p.join()
        return outs

    # ---- update ----------------------------------------------------------
    def gae(self, rew, mask, val):
        adv = np.zeros_like(rew)
        last = 0.0
        for t in reversed(range(len(rew))):
            nv = val[t + 1] if t + 1 < len(rew) else 0.0
            delta = rew[t] + self.gamma * nv * mask[t] - val[t]
            last = delta + self.gamma * self.lam * mask[t] * last
            adv[t] = last
        return adv, adv + val

    def update(self, outs):
        obs = np.concatenate([o["obs"] for o in outs])
        act = np.concatenate([o["act"] for o in outs])
        lp_old = np.concatenate([o["lp"] for o in outs])
        advs, rets = [], []
        with torch.no_grad():
            for o in outs:
                v = self.critic(torch.as_tensor(o["obs"],
                                                dtype=torch.float64)).numpy()
                a, r = self.gae(o["rew"], o["mask"], v)
                advs.append(a)
                rets.append(r)
        adv = np.concatenate(advs)
        ret = np.concatenate(rets)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(obs, dtype=torch.float64)
        act_t = torch.as_tensor(act, dtype=torch.float64)
        lp_t = torch.as_tensor(lp_old, dtype=torch.float64)
        adv_t = torch.as_tensor(adv, dtype=torch.float64)
        ret_t = torch.as_tensor(ret, dtype=torch.float64)

        n = obs_t.shape[0]
        for _ in range(self.optim_epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.mini):
                idx = perm[s:s + self.mini]
                d = self.actor.dist(obs_t[idx])
                lp = d.log_prob(act_t[idx]).sum(-1)
                ratio = (lp - lp_t[idx]).exp()
                a = adv_t[idx]
                l1 = ratio * a
                l2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a
                loss_p = -torch.min(l1, l2).mean()
                loss_p = loss_p - self.args.ent_coef * d.entropy().sum(-1).mean()
                self.opt_p.zero_grad()
                loss_p.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),
                                         self.args.max_grad_norm)
                self.opt_p.step()

                v = self.critic(obs_t[idx])
                loss_v = ((v - ret_t[idx]) ** 2).mean()
                self.opt_v.zero_grad()
                loss_v.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),
                                         self.args.max_grad_norm)
                self.opt_v.step()
        return float(loss_v.item())

    # ---- loop ------------------------------------------------------------
    def train(self):
        args = self.args
        total_steps = 0
        best = -1e18
        n_epochs = (args.max_epoch if args.max_epoch is not None
                    else self.cfg.max_epoch_num)
        for epoch in range(n_epochs):
            self.epoch = epoch
            if args.anneal_lr:
                frac = 1.0 - epoch / n_epochs
                for g in self.opt_p.param_groups:
                    g["lr"] = args.policy_lr * frac
                for g in self.opt_v.param_groups:
                    g["lr"] = args.value_lr * frac
            t0 = time.time()
            outs = self.sample(args.num_threads)
            t_sample = time.time() - t0
            t1 = time.time()
            loss_v = self.update(outs)
            t_update = time.time() - t1

            exec_steps = sum(len(o["rew"]) for o in outs)
            design_steps = sum(o["design_steps"] for o in outs)
            total_steps += exec_steps + design_steps
            rets = [r for o in outs for r in o["ep_rets"]]
            lens = [l for o in outs for l in o["ep_lens"]]
            for o in outs:
                self.norm.update_from(*o["raw"])

            row = dict(epoch=epoch, exec_R_eps=float(np.mean(rets)),
                       exec_R_eps_max=float(np.max(rets)),
                       ep_len=float(np.mean(lens)), n_eps=len(rets),
                       exec_steps=exec_steps, design_steps=design_steps,
                       total_steps=total_steps, loss_v=loss_v,
                       log_std=float(self.actor.log_std.mean().item()),
                       T_sample=t_sample, T_update=t_update)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{epoch}\tT_sample {t_sample:.2f}\tT_update {t_update:.2f}"
                  f"\texec_R_eps {row['exec_R_eps']:.2f}"
                  f"\tep_len {row['ep_len']:.1f}\teps {len(rets)}"
                  f"\tsteps {total_steps}\t{args.cfg}", flush=True)

            if (epoch + 1) % args.save_interval == 0 or epoch == 0:
                self.save(epoch)
            if row["exec_R_eps"] > best:
                best = row["exec_R_eps"]
                self.save("best")
            if args.stop_file and os.path.exists(args.stop_file):
                print(f"stop file present -- stopping after epoch {epoch}",
                      flush=True)
                self.save(epoch)
                break

    def save(self, tag):
        name = (f"epoch_{tag:04d}.p" if isinstance(tag, int) else f"{tag}.p")
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "norm": self.norm.state(),
                    "args": vars(self.args)},
                   os.path.join(self.out, name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--num-threads", type=int, default=12)
    p.add_argument("--max-epoch", type=int, default=None)
    p.add_argument("--policy-lr", type=float, default=3e-4,
                   help="published PPO-MuJoCo default (SB3/CleanRL), not the "
                        "GNN arm's 5e-5.")
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--hdims", default="64,64")
    p.add_argument("--log-std", type=float, default=0.0)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--stop-file", default=None)
    p.add_argument("--mini-batch", type=int, default=None,
                   help="override mini_batch_size. The GNN arm uses the "
                        "cfg's 2048; published PPO-MuJoCo uses 64.")
    p.add_argument("--optim-epochs", type=int, default=None)
    p.add_argument("--anneal-lr", action="store_true",
                   help="linear decay to 0 over the run, as CleanRL's "
                        "ppo_continuous_action.py does by default.")
    p.add_argument("--tag", default=None,
                   help="suffix for the results directory, so two "
                        "hyperparameter settings of one cfg do not collide.")
    p.add_argument("--batch", type=int, default=None,
                   help="override min_batch_size. FOR SMOKE TESTS ONLY -- the "
                        "real run must use the cfg's 50,000 so the step "
                        "budget matches the GNN arm.")
    args = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    Trainer(args).train()


if __name__ == "__main__":
    main()
