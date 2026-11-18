"""Held-out evaluation suite. Seeds come from the reserved range >= eval_seed_base (never trained on)."""
from __future__ import annotations

import numpy as np

from genesis.world import C, World
from genesis.evalonly.baselines import heuristic_policy, random_policy

FAMILIES = ["standard", "unseen_layout", "perturbed", "harder"]
FAMILY_OFFSET = {f: i * 100_000 for i, f in enumerate(FAMILIES)}


def eval_seeds(family, n):
    return C["eval_seed_base"] + FAMILY_OFFSET[family] + np.arange(n, dtype=np.int64)


def run_episodes(act_fn, family, n, max_ticks=None):
    w = World(n, family)
    w.reset(np.ones(n, bool), eval_seeds(family, n))
    done = np.zeros(n, bool)
    res = {k: np.zeros(n) for k in ("ticks", "food", "hazard", "dist", "contact", "survived")}
    cap = max_ticks or C["max_ticks"]
    for _ in range(cap):
        a = act_fn(w.observe())
        _, dead, trunc = w.step(a)
        fin = (dead | trunc) & ~done
        if fin.any():
            res["ticks"][fin] = w.t[fin]; res["food"][fin] = w.ep_food[fin]; res["hazard"][fin] = w.ep_hazard[fin]
            res["dist"][fin] = w.ep_dist[fin]; res["contact"][fin] = w.ep_contact[fin]; res["survived"][fin] = trunc[fin]
            done |= fin
        w.events.clear()
        if done.all():
            break
    return res


def summarize(res):
    out = {}
    for k, v in res.items():
        m = float(v.mean()); h = float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        out[k] = {"mean": round(m, 4), "ci95": [round(m - h, 4), round(m + h, 4)]}
    return out


def evaluate_policy(act_fn, families=("standard",), n=100):
    return {f: summarize(run_episodes(act_fn, f, n)) for f in families}


def baselines(families=FAMILIES, n=100, seed=12345):
    rng = np.random.default_rng(seed)
    return {"random": evaluate_policy(random_policy(rng), families, n),
            "heuristic": evaluate_policy(heuristic_policy, families, n)}
