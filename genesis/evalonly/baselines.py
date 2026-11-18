"""EVALUATION-ONLY baselines. GENESIS_HEURISTIC_MARKER — must never be imported by export or the web build."""
import numpy as np

from genesis.world import C, N_ACTIONS, N_CH

R = C["n_rays"]
MID = R // 2
S = R * N_CH  # start of self features
ENERGY, STAMINA, IN_SHELTER, HEARD = S, S + 1, S + 4, S + 12


def random_policy(rng):
    def act(obs):
        return rng.integers(0, N_ACTIONS, obs.shape[0])
    return act


def _rays(obs):
    r = obs[:, :S].reshape(-1, R, N_CH)
    return r[..., 0], r[..., 1], r[..., 2], r[..., 3], r[..., 4]  # water channel unused by baselines


def _steer(best, fast):
    off = best - MID
    straight = np.where(fast, 3, 0)
    return np.where(off == 0, straight, np.where(off < 0, np.where(off < -1, 4, 1), np.where(off > 1, 5, 2)))


def _forage(obs):
    solid, food, haz, shel, pred = _rays(obs)
    score = 2.0 * food - 1.5 * haz - 2.0 * (solid > 0.85) - 0.02 * np.abs(np.arange(R) - MID)
    score = np.where(food > 0, score, score - 5.0)
    act = _steer(score.argmax(1), np.zeros(len(obs), bool))
    blocked = (solid[:, MID - 1:MID + 2] > 0.88).any(1) | (haz[:, MID - 1:MID + 2] > 0.7).any(1)
    return np.where((food > 0).any(1), act, np.where(blocked, 5, 0)).astype(np.int64)


def forager_policy(obs):
    """Greedy seed-seeker that ignores the predator."""
    return _forage(obs)


def survivor_policy(obs):
    """Scripted survival: flee to a sensed shelter when the predator is sensed, rest inside, else forage."""
    solid, food, haz, shel, pred = _rays(obs)
    threat = (obs[:, HEARD] > 0) | (pred > 0).any(1)
    in_sh = obs[:, IN_SHELTER] > 0.5
    see_sh = (shel > 0).any(1)
    flee_sh = _steer((shel - 2.0 * (solid > 0.9)).argmax(1), np.ones(len(obs), bool))
    away = _steer((-pred - 0.5 * (solid > 0.85)).argmax(1), np.ones(len(obs), bool))
    tired = obs[:, STAMINA] < 0.25
    act = _forage(obs)
    act = np.where(threat & ~in_sh, np.where(see_sh, flee_sh, away), act)
    act = np.where(threat & in_sh, 7, act)
    act = np.where(~threat & tired & (obs[:, ENERGY] > 0.4), 7, act)
    return act.astype(np.int64)
