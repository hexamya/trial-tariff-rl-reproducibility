"""PPO-Lagrangian tariff designer for the monthly constrained decision problem.

Policy: categorical over the candidate day patterns (``patterns.candidate_patterns``, the planners'
action set). Logits are built from pattern features shared across patterns (High / Low in each block,
run lengths, single or double band; ``pattern_features``) plus a bias per pattern, so what is learned
about, say, a High block at 17-20 h carries over to every pattern that has it; at initialisation half the
probability is on the all-Normal day, so exploration starts from a plain schedule. Masks enforce the
hard rules: day-to-day compatibility with the previous day's trailing run,
the month's remaining High and Low half-hours and, in London, the weekly limit of days with both bands.
With these masks the trial's schedule rules are never broken, and the CMDP constraints left are revenue
(both cases) and the segment bill cap (London), both episode-level excesses (percentage points).

Observation: the environment's observation (day-ahead information only) plus, in London, the worst
segment's month-to-date bill change over days whose consumption is complete (posted day - 2).

Training: PPO with clipped ratio and GAE (gamma = 1 over the month), separate critics for the reward and
for each constraint cost; the reward advantage is normalised, the cost advantages (percentage points) are
only centred, following the PPO-Lagrangian baseline of Ray, Achiam and Amodei (2019) (scaling rare, near-zero costs
by their own spread would turn noise into a large signal), and they are combined as
(A_r - sum lambda_k A_k) / (1 + sum lambda_k); the multipliers follow a PID controller on the batch mean excess above a threshold
(Stooke, Achiam and Abbeel 2020, "Responsive safety in reinforcement learning by PID Lagrangian methods").
Episodes draw a training month and a training ensemble member at random. ``lagrangian=False`` gives the
unconstrained PPO baseline.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from ttrl.env import london as ld
from ttrl.env.tariff import HH_PER_BLOCK, HIGH, LOW, slot_prices
from ttrl.s3 import harness as hs
from ttrl.s3 import patterns as pt
from ttrl.s3.planners import iso_weeks


@dataclass
class PPOConfig:
    iterations: int = 400
    episodes_per_iter: int = 32
    epochs: int = 8
    minibatch: int = 256
    lr: float = 3e-4
    clip: float = 0.2
    gae_lambda: float = 0.95
    entropy: float = 0.01
    entropy_final: float | None = None   # linear annealing of the entropy bonus to this value (None: constant)
    value_coef: float = 0.5
    hidden: int = 128
    reward_scale: float = 1.0
    cost_scale: float = 100.0            # excess fraction -> percentage points
    cost_threshold: float = 0.05         # tolerated batch-mean excess (pp) per constraint
    pid: tuple = (0.5, 0.02, 0.0)        # KP, KI, KD
    lambda_max: float | None = None      # cap on each multiplier (None: no cap)
    lagrangian: bool = True
    seed: int = 0


# ------------------------------------------------------------------ features and masks
def pattern_features(patterns: np.ndarray) -> np.ndarray:
    """(P, F) features: High in block b, Low in block b, High / Low run length one-hot, has High, has Low,
    both."""
    from ttrl.env.tariff import BLOCKS_PER_DAY, runs
    P = len(patterns)
    hi, lo = (patterns == HIGH).astype(float), (patterns == LOW).astype(float)
    hl, ll = np.zeros((P, BLOCKS_PER_DAY)), np.zeros((P, BLOCKS_PER_DAY))
    for j, p in enumerate(patterns):
        for band, _, n in runs(p):
            (hl if band == HIGH else ll)[j, n - 1] = 1.0
    flags = np.c_[hi.any(1), lo.any(1), hi.any(1) & lo.any(1)].astype(float)
    return np.c_[hi, lo, hl, ll, flags]



class Features:
    """Agent observation and action mask from an environment (London or ERCOT)."""

    def __init__(self, env, patterns: np.ndarray):
        self.P = patterns
        self.compat = pt.compatibility(patterns)
        self.hh_high = (patterns == HIGH).sum(1) * HH_PER_BLOCK
        self.hh_low = (patterns == LOW).sum(1) * HH_PER_BLOCK
        self.both = (patterns == HIGH).any(1) & (patterns == LOW).any(1)
        self.london = hs.is_london(env)
        self.constraints = ["revenue", "bill_segment"] if self.london else ["revenue"]
        if self.london:
            self.weeks = iso_weeks(env.inputs.days["tday"].to_numpy())
            self._k = None
        self.obs_size = env.obs_size + (1 if self.london else 0)

    def mask(self, env) -> np.ndarray:
        t, rules = env.t, env.cfg.rules
        prev = pt.trailing_state(env.blocks[t - 1]) if t > 0 else 0
        m = self.compat[prev].copy()
        used = env.blocks[:t]
        m &= self.hh_high + int((used == HIGH).sum()) * HH_PER_BLOCK <= rules.max_high_hh_per_month
        m &= self.hh_low + int((used == LOW).sum()) * HH_PER_BLOCK <= rules.max_low_hh_per_month
        if self.london:
            w = self.weeks[env.day_idx]
            done = ((used == HIGH).any(1) & (used == LOW).any(1))
            if int(done[w[:t] == w[t]].sum()) >= rules.max_double_days_per_week:
                m &= ~self.both
        m[0] = True                                       # all-Normal is always allowed
        return m

    def obs(self, env, obs: np.ndarray) -> np.ndarray:
        if not self.london:
            return obs
        k = max(0, env.t - 1)                             # days whose consumption is complete: < t - 1
        if self._k is None or env.t == 0 or k < self._k or self._month != (env.month, id(env)):
            H = env.inputs.households.shape[0]
            self._tariff, self._flat, self._k, self._month = np.zeros(H), np.zeros(H), 0, (env.month, id(env))
        while self._k < k:                                # running non-responding bills, one day at a time
            q = np.nan_to_num(env.inputs.households[:, env.day_idx[self._k]])
            self._tariff += q @ slot_prices(env.blocks[self._k])
            self._flat += q.sum(1) * ld.PRICE_FLAT
            self._k += 1
        seg = 0.0
        if k > 0:
            ok = self._flat > 0
            b = np.full(len(self._flat), np.nan)
            b[ok] = self._tariff[ok] / self._flat[ok] - 1
            seg = float(np.max(ld.segment_bill_means(b, env.segments)))
        return np.r_[obs, seg * 10].astype(np.float32)


# ------------------------------------------------------------------ networks
class RunningNorm:
    def __init__(self, n: int):
        self.mean, self.var, self.count = np.zeros(n), np.ones(n), 1e-4

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


def mlp(n_in: int, n_out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out))


class ActorCritic(nn.Module):
    """Policy logits = g(obs) . phi(pattern) + bias(pattern); separate reward and cost critics."""

    def __init__(self, n_obs: int, phi: np.ndarray, n_cost: int, hidden: int):
        super().__init__()
        self.register_buffer("phi", torch.as_tensor(phi, dtype=torch.float32))
        self.pi = mlp(n_obs, phi.shape[1], hidden)
        nn.init.normal_(self.pi[-1].weight, std=0.01)
        nn.init.zeros_(self.pi[-1].bias)
        bias = torch.zeros(phi.shape[0])
        bias[0] = float(np.log(max(phi.shape[0] - 1, 1)))   # half the initial probability on all-Normal
        self.bias = nn.Parameter(bias)
        self.v = mlp(n_obs, 1, hidden)
        self.vc = mlp(n_obs, max(n_cost, 1), hidden)

    def dist(self, x: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
        logits = (self.pi(x) @ self.phi.T + self.bias).masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)


# ------------------------------------------------------------------ PID Lagrangian
@dataclass
class PIDLagrangian:
    kp: float
    ki: float
    kd: float
    threshold: float
    integral: float = 0.0
    prev: float = 0.0
    value: float = 0.0

    def update(self, cost: float) -> float:
        err = cost - self.threshold
        self.integral = max(0.0, self.integral + self.ki * err)
        deriv = max(0.0, cost - self.prev)
        self.prev = cost
        self.value = max(0.0, self.kp * err + self.integral + self.kd * deriv)
        return self.value


# ------------------------------------------------------------------ agent
@dataclass
class Agent:
    """Trained policy with its observation normaliser; ``act`` follows the harness's policy protocol
    (deterministic: the most likely allowed pattern)."""
    net: ActorCritic
    norm: RunningNorm
    features: Features
    patterns: np.ndarray
    name: str = "PPO-Lagrangian"
    stochastic: bool = False

    def reset(self, env) -> None:
        pass

    def act(self, env, obs) -> np.ndarray:
        x = torch.as_tensor(self.norm(self.features.obs(env, obs))[None], dtype=torch.float32)
        m = torch.as_tensor(self.features.mask(env)[None])
        with torch.no_grad():
            d = self.net.dist(x, m)
            a = int(d.sample()) if self.stochastic else int(d.probs.argmax())
        return self.patterns[a].copy()


def episode_costs(info: dict, names: list[str], scale: float) -> np.ndarray:
    return np.array([scale * float(info["cost"].get(k, 0.0)) for k in names])


def collect(env, agent: Agent, months: list[str], members: np.ndarray, n: int, rng: np.random.Generator,
            reward_scale: float, cost_scale: float):
    feats = agent.features
    buf = {k: [] for k in ("obs", "mask", "act", "logp", "rew", "done")}
    ep_costs, ep_rewards, raw = [], [], []
    for _ in range(n):
        obs, _ = env.reset(options={"month": str(rng.choice(months)), "member": int(rng.choice(members))})
        done, total = False, 0.0
        while not done:
            o = feats.obs(env, obs)
            m = feats.mask(env)
            x = torch.as_tensor(agent.norm(o)[None], dtype=torch.float32)
            with torch.no_grad():
                d = agent.net.dist(x, torch.as_tensor(m[None]))
                a = d.sample()
                lp = float(d.log_prob(a))
            obs, r, done, _, info = env.step(agent.patterns[int(a)])
            buf["obs"].append(o)
            buf["mask"].append(m)
            buf["act"].append(int(a))
            buf["logp"].append(lp)
            buf["rew"].append(r * reward_scale)
            buf["done"].append(done)
            total += r
        c = episode_costs(info, feats.constraints, cost_scale)
        ep_costs.append(c)
        ep_rewards.append(total)
        raw.append({"reward": total, **{f"cost_{k}": v for k, v in zip(feats.constraints, c)}})
    out = {k: np.asarray(v) for k, v in buf.items()}
    out["ep_costs"] = np.asarray(ep_costs)
    return out, pd.DataFrame(raw)


def gae(rew: np.ndarray, val: np.ndarray, done: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Undiscounted GAE within episodes (terminal value 0)."""
    adv = np.zeros_like(rew)
    last = 0.0
    for t in range(len(rew) - 1, -1, -1):
        nxt = 0.0 if done[t] else val[t + 1]
        delta = rew[t] + nxt - val[t]
        last = delta + lam * (0.0 if done[t] else last)
        adv[t] = last
    return adv, adv + val


def train(env, months: list[str], members: np.ndarray, patterns: np.ndarray, cfg: PPOConfig,
          log_every: int = 10, label: str = "", eval_every: int = 0, eval_members: np.ndarray | None = None
          ) -> tuple[Agent, pd.DataFrame]:
    """Train on ``months`` x ``members``. With ``eval_every``, the deterministic policy is also run on the
    same training months with ``eval_members`` (training members) every that many iterations (columns
    ``det_*``), to follow learning without touching held-out data."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    feats = Features(env, patterns)
    K = len(feats.constraints)
    net = ActorCritic(feats.obs_size, pattern_features(patterns), K, cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    agent = Agent(net, RunningNorm(feats.obs_size), feats, patterns,
                  name="PPO-Lagrangian" if cfg.lagrangian else "PPO (unconstrained)")
    pids = [PIDLagrangian(*cfg.pid, cfg.cost_threshold) for _ in range(K)]
    hist, t0 = [], time.time()
    for it in range(cfg.iterations):
        batch, eps = collect(env, agent, months, members, cfg.episodes_per_iter, rng, cfg.reward_scale, cfg.cost_scale)
        agent.norm.update(batch["obs"])
        x = torch.as_tensor(agent.norm(batch["obs"]), dtype=torch.float32)
        mask = torch.as_tensor(batch["mask"])
        act = torch.as_tensor(batch["act"])
        old_lp = torch.as_tensor(batch["logp"], dtype=torch.float32)
        done = batch["done"]
        with torch.no_grad():
            v = net.v(x).squeeze(-1).numpy()
            vc = net.vc(x).numpy()
        adv_r, ret_r = gae(batch["rew"], v, done, cfg.gae_lambda)
        # episode costs are paid at the last step of each episode
        cost_rew = np.zeros((len(done), K))
        cost_rew[np.flatnonzero(done)] = batch["ep_costs"]
        adv_c, ret_c = np.zeros_like(cost_rew), np.zeros_like(cost_rew)
        for k in range(K):
            adv_c[:, k], ret_c[:, k] = gae(cost_rew[:, k], vc[:, k], done, cfg.gae_lambda)
        lam = np.array([p.update(float(batch["ep_costs"][:, k].mean())) for k, p in enumerate(pids)]) \
            if cfg.lagrangian else np.zeros(K)
        if cfg.lambda_max is not None:
            lam = np.minimum(lam, cfg.lambda_max)
        ent = cfg.entropy if cfg.entropy_final is None else \
            cfg.entropy + (cfg.entropy_final - cfg.entropy) * it / max(cfg.iterations - 1, 1)
        a_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        a_c = adv_c - adv_c.mean(0)
        adv = (a_r - (a_c * lam).sum(1)) / (1.0 + lam.sum())
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_r_t = torch.as_tensor(ret_r, dtype=torch.float32)
        ret_c_t = torch.as_tensor(ret_c, dtype=torch.float32)
        n = len(adv)
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, cfg.minibatch):
                i = perm[s:s + cfg.minibatch]
                d = net.dist(x[i], mask[i])
                lp = d.log_prob(act[i])
                ratio = torch.exp(lp - old_lp[i])
                pg = -torch.min(ratio * adv_t[i], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_t[i]).mean()
                vl = ((net.v(x[i]).squeeze(-1) - ret_r_t[i]) ** 2).mean() + ((net.vc(x[i]) - ret_c_t[i]) ** 2).mean()
                loss = pg + cfg.value_coef * vl - ent * d.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        row = {"iteration": it, "steps": n, "reward": float(eps["reward"].mean()),
               **{f"cost_{k}": float(eps[f"cost_{k}"].mean()) for k in feats.constraints},
               **{f"violated_{k}": float((eps[f"cost_{k}"] > 0).mean()) for k in feats.constraints},
               **{f"lambda_{k}": float(lam[j]) for j, k in enumerate(feats.constraints)},
               "seconds": time.time() - t0}
        if eval_every and (it % eval_every == 0 or it == cfg.iterations - 1):
            r = hs.evaluate(env, agent, months, eval_members if eval_members is not None else members[:3])
            row.update({"det_reward": float(r["reward"].mean()), "det_violated": float(r["violated"].mean()),
                        **{f"det_cost_{k}": float(cfg.cost_scale * r[f"cost_{k}"].mean()) for k in feats.constraints}})
        hist.append(row)
        if log_every and (it % log_every == 0 or it == cfg.iterations - 1):
            print(f"{label} it {it}: " + ", ".join(f"{k} {v:.3g}" for k, v in row.items() if k not in ("iteration", "steps")),
                  flush=True)
    return agent, pd.DataFrame(hist)


def save_agent(agent: Agent, cfg: PPOConfig, path) -> None:
    torch.save({"state": agent.net.state_dict(), "norm": (agent.norm.mean, agent.norm.var, agent.norm.count),
                "config": asdict(cfg), "name": agent.name, "patterns": agent.patterns}, path)


def load_agent(path, env) -> Agent:
    d = torch.load(path, weights_only=False)
    cfg = PPOConfig(**d["config"])
    feats = Features(env, d["patterns"])
    net = ActorCritic(feats.obs_size, pattern_features(d["patterns"]), len(feats.constraints), cfg.hidden)
    net.load_state_dict(d["state"])
    norm = RunningNorm(feats.obs_size)
    norm.mean, norm.var, norm.count = d["norm"]
    return Agent(net, norm, feats, d["patterns"], name=d["name"])
