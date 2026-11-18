import numpy as np

from genesis import ppo


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    p = ppo.init_params(7, 4, (8, 8), seed=1)
    for k in p:  # larger weights so the check is not trivial
        p[k] = p[k] + 0.3 * rng.standard_normal(p[k].shape)
    N = 32
    obs = rng.standard_normal((N, 7)); act = rng.integers(0, 4, N)
    old_logp = np.log(np.full(N, 0.25)) + 0.05 * rng.standard_normal(N)
    adv = rng.standard_normal(N); ret = rng.standard_normal(N)
    args = (obs, act, old_logp, adv, ret, 10.0, 0.05, 0.5)  # large clip → smooth loss
    _, g, _ = ppo.ppo_loss_and_grads(p, *args)
    eps = 1e-6
    for k in p:
        for idx in [tuple(rng.integers(0, s) for s in p[k].shape) for _ in range(3)]:
            orig = p[k][idx]
            p[k][idx] = orig + eps; lp, _, _ = ppo.ppo_loss_and_grads(p, *args)
            p[k][idx] = orig - eps; lm, _, _ = ppo.ppo_loss_and_grads(p, *args)
            p[k][idx] = orig
            num = (lp - lm) / (2 * eps)
            assert abs(num - g[k][idx]) < 1e-6 + 1e-4 * abs(num), (k, idx, num, g[k][idx])


def test_init_is_random_and_seeded():
    a = ppo.init_params(42, 6, (64, 64), seed=3)
    b = ppo.init_params(42, 6, (64, 64), seed=3)
    c = ppo.init_params(42, 6, (64, 64), seed=4)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["pi.W0"], c["pi.W0"])
