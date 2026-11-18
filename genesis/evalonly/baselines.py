"""EVALUATION-ONLY baselines. GENESIS_HEURISTIC_MARKER — must never be imported by export or the web build."""
import numpy as np

from genesis.world import C, N_ACTIONS

R = C["n_rays"]
MID = R // 2


def random_policy(rng):
    def act(obs):
        return rng.integers(0, N_ACTIONS, obs.shape[0])
    return act


def heuristic_policy(obs):
    """Greedy: steer to the most attractive sensed food ray, away from vents and close solids."""
    rays = obs[:, : R * 3].reshape(-1, R, 3)
    solid, food, haz = rays[..., 0], rays[..., 1], rays[..., 2]
    score = 2.0 * food - 1.5 * haz - 2.0 * (solid > 0.85) - 0.02 * np.abs(np.arange(R) - MID)
    score = np.where(food > 0, score, score - 5.0)
    best = score.argmax(1)
    any_food = (food > 0).any(1)
    off = best - MID
    act = np.where(off == 0, 0, np.where(off < 0, np.where(off < -1, 3, 1), np.where(off > 1, 4, 2)))
    # explore when no food is visible: forward unless blocked, turn right if blocked
    blocked = (solid[:, MID - 1:MID + 2] > 0.88).any(1) | (haz[:, MID - 1:MID + 2] > 0.7).any(1)
    explore = np.where(blocked, 4, 0)
    return np.where(any_food, act, explore).astype(np.int64)
