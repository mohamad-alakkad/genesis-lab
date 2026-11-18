import math

import numpy as np

from genesis.world import C, OBS_DIM, VecEnv, World


def make(n=32, fam="train", seed0=100):
    w = World(n, fam)
    w.reset(np.ones(n, bool), np.arange(seed0, seed0 + n))
    return w


def test_deterministic_replay():
    a, b = make(), make()
    rng = np.random.default_rng(0)
    for _ in range(300):
        act = rng.integers(0, 6, 32)
        ra = a.step(act); rb = b.step(act)
        assert np.array_equal(a.observe(), b.observe())
        assert all(np.array_equal(x, y) for x, y in zip(ra, rb))


def test_env_streams_independent_of_batch():
    single = World(1, "train"); single.reset(np.ones(1, bool), np.array([105]))
    batch = make(); rng = np.random.default_rng(1)
    for _ in range(200):
        act = rng.integers(0, 6, 32)
        single.step(act[5:6]); batch.step(act)
        assert np.allclose(single.observe()[0], batch.observe()[5], atol=0, rtol=0)


def test_observation_bounds_and_layout():
    w = make()
    rng = np.random.default_rng(2)
    for _ in range(200):
        w.step(rng.integers(0, 6, 32))
        o = w.observe()
        assert o.shape == (32, OBS_DIM) and np.isfinite(o).all()
        assert o.min() >= 0.0 and o.max() <= 1.0
        assert np.allclose(o[:, -6:].sum(1), 1.0)


def test_no_tunnelling_through_walls_or_obstacles():
    w = make(64)
    rc = C["creature_radius"]
    for _ in range(600):
        w.step(np.zeros(64, np.int64))  # always full speed forward
        assert (w.pos[:, 0] >= rc - 1e-9).all() and (w.pos[:, 0] <= C["width"] - rc + 1e-9).all()
        assert (w.pos[:, 1] >= rc - 1e-9).all() and (w.pos[:, 1] <= C["height"] - rc + 1e-9).all()
        d = np.linalg.norm(w.obs_pos - w.pos[:, None], axis=2)
        # overlap allowed only transiently by stacked obstacles; never deep inside one
        assert np.all(~w.obs_on | (d > w.obs_r + rc - 0.16))
        w.energy[:] = 1.0  # keep alive


def test_energy_accounting_and_starvation():
    w = World(1, "train"); w.reset(np.ones(1, bool), np.array([7]))
    w.food_on[:] = False; w.vent_on[:] = False; w.f = dict(w.f, p_food=0, p_obs=0, p_vent=0)
    w.vent_present[:] = False
    e0 = w.energy[0]
    w.step(np.array([5]))
    assert math.isclose(e0 - w.energy[0], C["energy_decay"], rel_tol=1e-12)
    e1 = w.energy[0]; w.step(np.array([0]))
    assert math.isclose(e1 - w.energy[0], C["energy_decay"] + C["move_cost"], rel_tol=1e-12)
    for _ in range(1000):
        r, dead, trunc = w.step(np.array([5]))
        if dead[0]:
            assert r[0] == C["reward_death"]
            break
    assert dead[0]


def test_eating_reward_and_respawn():
    w = World(1, "train"); w.reset(np.ones(1, bool), np.array([9]))
    w.f = dict(w.f, p_food=0, p_obs=0, p_vent=0); w.vent_on[:] = False; w.vent_present[:] = False
    w.food_pos[0, 0] = w.pos[0] + 0.1 * np.array([math.cos(w.heading[0]), math.sin(w.heading[0])])
    w.energy[:] = 0.5
    r, _, _ = w.step(np.array([5]))
    assert r[0] == C["reward_food"]
    assert math.isclose(w.energy[0], 0.5 + C["food_gain"] - C["energy_decay"], rel_tol=1e-12)
    assert np.linalg.norm(w.food_pos[0, 0] - w.pos[0]) > 1.0  # respawned away


def test_hazard_drain_and_penalty():
    w = World(1, "train"); w.reset(np.ones(1, bool), np.array([11]))
    w.f = dict(w.f, p_food=0, p_obs=0, p_vent=0); w.food_on[:] = False
    w.vent_present[0, 0] = True; w.vent_on[0, 0] = True; w.vent_pos[0, 0] = w.pos[0]; w.vent_r[0, 0] = 0.8
    w.vent_timer[0, 0] = 50
    e0 = w.energy[0]; r, _, _ = w.step(np.array([5]))
    assert math.isclose(r[0], C["reward_hazard_tick"]) and w.in_hazard[0]
    assert math.isclose(e0 - w.energy[0], C["energy_decay"] + C["hazard_drain"], rel_tol=1e-12)


def test_truncation_at_step_cap():
    w = World(1, "train"); w.reset(np.ones(1, bool), np.array([13]))
    for t in range(C["max_ticks"]):
        w.energy[:] = 1.0
        _, dead, trunc = w.step(np.array([5]))
    assert trunc[0] and not dead[0]


def test_vecenv_autoreset_and_seed_ranges():
    v = VecEnv(8, seed=3)
    for _ in range(400):
        _, _, _, _, info = v.step(np.full(8, 5))
    assert (v.w.t < C["max_ticks"]).all()
    s = v._seeds(); assert (s < C["train_seed_max"]).all()
    from genesis.evalonly.suite import eval_seeds
    assert eval_seeds("standard", 10).min() >= C["eval_seed_base"] > C["train_seed_max"]
