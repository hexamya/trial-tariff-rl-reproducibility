"""Tests for the PPO-Lagrangian agent on synthetic environments."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ttrl.env import london_env as le  # noqa: E402
from ttrl.env import tariff as tf  # noqa: E402
from ttrl.s3 import harness as hs  # noqa: E402
from ttrl.s3 import patterns as pt  # noqa: E402
from ttrl.s3 import rl  # noqa: E402
from test_s3 import _arrays, _inputs  # noqa: E402


def test_gae_and_pid():
    rew = np.array([1.0, 2.0, 3.0, 1.0, 1.0])
    done = np.array([False, False, True, False, True])
    val = np.zeros(5)
    adv, ret = rl.gae(rew, val, done, lam=1.0)
    assert ret.tolist() == [6.0, 5.0, 3.0, 2.0, 1.0]          # undiscounted returns within each episode
    p = rl.PIDLagrangian(0.5, 0.1, 0.0, threshold=0.05)
    assert p.update(0.0) == 0.0                                # below threshold: no penalty
    lams = [p.update(1.05) for _ in range(5)]
    assert all(b > a for a, b in zip(lams, lams[1:]))          # integral grows while violated
    for _ in range(50):
        last = p.update(0.0)
    assert last < lams[-1]


def test_masks_keep_every_schedule_inside_the_rules():
    inp = _inputs()
    env = le.LondonTariffEnv(inp, _arrays(), le.LondonConfig(months=["2013-01"], members=np.arange(4)), seed=0)
    P = pt.candidate_patterns()
    feats = rl.Features(env, P)
    rng = np.random.default_rng(0)
    for _ in range(3):
        obs, _ = env.reset(options={"month": "2013-01", "member": 0})
        done = False
        while not done:
            m = feats.mask(env)
            assert m[0] and len(feats.obs(env, obs)) == feats.obs_size
            a = rng.choice(np.flatnonzero(m))
            obs, _, done, _, info = env.step(P[a])
            assert not info["invalid"]
        assert info["cost"]["high_hours"] == 0 and info["cost"]["low_hours"] == 0 and info["cost"]["double_days"] == 0


def test_short_training_run_and_agent_roundtrip(tmp_path):
    inp = _inputs()
    env = le.LondonTariffEnv(inp, _arrays(), le.LondonConfig(months=["2013-01", "2013-02"], members=np.arange(4)), seed=0)
    P = pt.candidate_patterns()
    cfg = rl.PPOConfig(iterations=2, episodes_per_iter=2, epochs=1, minibatch=32, hidden=16, seed=1)
    agent, hist = rl.train(env, ["2013-01"], np.arange(4), P, cfg, log_every=0)
    assert len(hist) == 2 and {"reward", "cost_revenue", "cost_bill_segment", "lambda_revenue"} <= set(hist.columns)
    r = hs.evaluate(env, agent, ["2013-02"], [0, 1])
    assert (r["invalid_days"] == 0).all()
    path = tmp_path / "agent.pt"
    rl.save_agent(agent, cfg, path)
    back = rl.load_agent(path, env)
    r2 = hs.evaluate(env, back, ["2013-02"], [0, 1])
    assert np.allclose(r2["reward"], r["reward"])


def test_pattern_features_and_initial_policy():
    P = pt.candidate_patterns()
    phi = rl.pattern_features(P)
    assert phi.shape == (len(P), 35) and (phi[0] == 0).all()
    j = next(i for i, p in enumerate(P) if p.tolist() == [0, 0, 0, 0, 0, 1, 1, 0])
    assert phi[j, 5] == 1 and phi[j, 6] == 1 and phi[j, 16 + 1] == 1 and phi[j, 32] == 1 and phi[j, 34] == 0
    net = rl.ActorCritic(10, phi, 2, 16)
    d = net.dist(torch.zeros(1, 10), torch.ones(1, len(P), dtype=torch.bool))
    assert abs(float(d.probs[0, 0].detach()) - 0.5) < 0.05


def test_incremental_segment_feature_matches_direct_computation():
    from ttrl.env import london as ld
    inp = _inputs()
    env = le.LondonTariffEnv(inp, _arrays(), le.LondonConfig(months=["2013-01"], members=np.arange(4)), seed=0)
    P = pt.candidate_patterns()
    feats = rl.Features(env, P)
    rng = np.random.default_rng(2)
    obs, _ = env.reset(options={"month": "2013-01", "member": 0})
    for _ in range(12):
        obs, *_ = env.step(P[rng.choice(np.flatnonzero(feats.mask(env)))])
    f = feats.obs(env, obs)[-1] / 10
    k = env.t - 1
    b = ld.bill_changes(env.blocks[:k], inp.households[:, env.day_idx[:k]], np.zeros((k, 48)))
    assert f == pytest.approx(float(np.max(ld.segment_bill_means(b, env.segments))), rel=1e-6)
