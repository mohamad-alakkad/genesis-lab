import math

import numpy as np

from genesis.world import C, HUNT, OBS_DIM, REST, VecEnv, World


def make(n=32, fam="train", seed0=100):
    w = World(n, fam)
    w.reset(np.ones(n, bool), np.arange(seed0, seed0 + n))
    return w


def quiet(w):
    w.f = dict(w.f, p_food=0, p_obs=0, p_vent=0, p_alert=0)
    w.vent_on[:] = False; w.vent_present[:] = False; w.pr_on[:] = False
    return w


def test_deterministic_replay():
    a, b = make(), make()
    rng = np.random.default_rng(0)
    for _ in range(400):
        act = rng.integers(0, 8, 32)
        ra = a.step(act); rb = b.step(act)
        assert np.array_equal(a.observe(), b.observe())
        assert all(np.array_equal(x, y) for x, y in zip(ra, rb))


def test_env_streams_independent_of_batch():
    single = World(1, "train"); single.reset(np.ones(1, bool), np.array([105]))
    batch = make(); rng = np.random.default_rng(1)
    for _ in range(300):
        act = rng.integers(0, 8, 32)
        single.step(act[5:6]); batch.step(act)
        assert np.array_equal(single.observe()[0], batch.observe()[5])


def test_observation_bounds_and_layout():
    w = make(); rng = np.random.default_rng(2)
    for _ in range(300):
        w.step(rng.integers(0, 8, 32))
        o = w.observe()
        assert o.shape == (32, OBS_DIM) and np.isfinite(o).all()
        assert o.min() >= -1.0 and o.max() <= 1.0
        assert np.allclose(o[:, -8:].sum(1), 1.0)


def test_no_tunnelling_and_predator_outside_shelters():
    w = make(64); rc = C["creature_radius"]; pr = C["predator_radius"]
    for _ in range(800):
        w.step(np.full(64, 3))  # dash forward always
        assert (w.pos[:, 0] >= rc - 1e-9).all() and (w.pos[:, 0] <= C["width"] - rc + 1e-9).all()
        assert (w.pos[:, 1] >= rc - 1e-9).all() and (w.pos[:, 1] <= C["height"] - rc + 1e-9).all()
        d = np.linalg.norm(w.obs_pos - w.pos[:, None], axis=2)
        assert np.all(~w.obs_on | (d > w.obs_r + rc - 0.25))
        ds = np.linalg.norm(w.sh_pos - w.pr_pos[:, None], axis=2)
        assert np.all(~w.sh_on | ~w.pr_on[:, None] | (ds > w.sh_r + pr - 0.25))
        w.energy[:] = 1.0; w.stamina[:] = 1.0


def test_momentum_and_energy_accounting():
    w = quiet(World(1, "train")); w.reset(np.ones(1, bool), np.array([7])); quiet(w)
    w.food_on[:] = False; w.food_timer[:] = 10**6
    e0 = w.energy[0]; w.step(np.array([7]))
    assert math.isclose(e0 - w.energy[0], C["energy_decay_rest"], rel_tol=1e-12)
    w.step(np.array([0]))
    assert math.isclose(w.v[0], C["action_v"][0] * C["accel_v"], rel_tol=1e-12)  # accelerates, no teleport
    for _ in range(30):
        w.step(np.array([0]))
    assert abs(w.v[0] - C["action_v"][0]) < 1e-3
    for _ in range(3000):
        _, dead, _ = w.step(np.array([7]))
        if dead[0]:
            break
    assert dead[0] and w.ep["caught"][0] == 0


def test_stamina_exhaustion_and_rest_recovery():
    w = quiet(World(1, "train")); w.reset(np.ones(1, bool), np.array([8])); quiet(w)
    for _ in range(100):
        w.step(np.array([3])); w.energy[:] = 1.0
    assert w.stamina[0] < C["exhausted_stamina"] and w.v[0] < C["action_v"][0]
    s = w.stamina[0]
    for _ in range(20):
        w.step(np.array([7]))
    assert w.stamina[0] > s + 0.1


def test_eating_reward_and_regrowth_delay():
    w = quiet(World(1, "train")); w.reset(np.ones(1, bool), np.array([9])); quiet(w)
    w.food_pos[0, 0] = w.pos[0]
    w.energy[:] = 0.5
    r, _, _ = w.step(np.array([7]))
    assert r[0] == C["reward_food"] and not w.food_on[0, 0]
    for _ in range(C["regrow_min"] - 2):
        w.step(np.array([7]))
    assert not w.food_on[0, 0]
    for _ in range(C["regrow_rand"] + 2):
        w.step(np.array([7]))
    assert w.food_on[0, 0]


def test_shelter_protects_and_predator_catches_in_open():
    w = World(2, "standard"); w.reset(np.ones(2, bool), np.array([11, 12]))
    w.f = dict(w.f, p_food=0, p_obs=0, p_vent=0, p_alert=0)
    w.vent_on[:] = False; w.obs_on[:] = False
    w.sh_pos[0, 0] = w.pos[0]; w.sh_r[0, 0] = 1.0; w.sh_on[0, 0] = True  # env 0: creature inside a shelter
    w.sh_on[1] = False
    for e in range(2):
        w.pr_pos[e] = w.pos[e] + np.array([1.4, 0.0]); w.pr_mode[e] = HUNT
    caught_any = np.zeros(2, bool)
    for _ in range(60):
        _, dead, _ = w.step(np.array([7, 7]))
        caught_any |= w.ep["caught"] > 0
        if dead[1]:
            break
    assert not caught_any[0] and caught_any[1]


def test_predator_fatigue_ends_hunt():
    w = World(1, "standard"); w.reset(np.ones(1, bool), np.array([13]))
    w.pr_fatigue[:] = C["predator_fatigue_max"]; w.pr_mode[:] = HUNT
    w.step(np.array([7]))
    assert w.pr_mode[0] == REST


def test_truncation_at_step_cap():
    w = quiet(World(1, "no_predator")); w.reset(np.ones(1, bool), np.array([13])); quiet(w)
    for _ in range(C["max_ticks"]):
        w.energy[:] = 1.0
        _, dead, trunc = w.step(np.array([7]))
    assert trunc[0] and not dead[0]


def test_vecenv_autoreset_and_seed_ranges():
    v = VecEnv(8, seed=3)
    for _ in range(400):
        v.step(np.full(8, 7))
    s = v._seeds(); assert (s < C["train_seed_max"]).all()
    from genesis.evalonly.suite import eval_seeds
    assert eval_seeds("standard", 10).min() >= C["eval_seed_base"] > C["train_seed_max"]
