"""Held-out evaluation suite. Seeds come from the reserved range >= eval_seed_base (never trained on)."""
from __future__ import annotations

import numpy as np

from genesis.world import C, World
from genesis.evalonly.baselines import forager_policy, random_policy, survivor_policy

FAMILIES = ["standard", "unseen_layout", "night", "relentless", "one_shelter", "perturbed", "no_predator", "drought", "fissure_heavy", "lure_heavy"]
FAMILY_OFFSET = {f: i * 100_000 for i, f in enumerate(FAMILIES)}


def eval_seeds(family, n, start=0):
    return C["eval_seed_base"] + FAMILY_OFFSET[family] + start + np.arange(n, dtype=np.int64)


def run_episodes(act_fn, family, n, max_ticks=None, start=0):
    """start > 0 selects a disjoint block of held-out seeds (e.g. for unbiased final verdicts after best-checkpoint selection)."""
    w = World(n, family)
    w.reset(np.ones(n, bool), eval_seeds(family, n, start))
    done = np.zeros(n, bool)
    keys = ("ticks", "food", "survived", "caught", "hazard", "dist", "shelter_when_threat", "shelter_at_night",
            "shelter_frac", "rest_frac", "dash_when_threat", "threat_frac")
    res = {k: np.full(n, np.nan) for k in keys}
    cap = max_ticks or C["max_ticks"]
    for _ in range(cap):
        _, dead, trunc = w.step(act_fn(w.observe()))
        fin = (dead | trunc) & ~done
        if fin.any():
            e = w.ep; T = np.maximum(w.t, 1)
            res["ticks"][fin] = w.t[fin]; res["food"][fin] = e["food"][fin]; res["survived"][fin] = trunc[fin]
            res["caught"][fin] = e["caught"][fin]; res["hazard"][fin] = e["hazard"][fin]; res["dist"][fin] = e["dist"][fin]
            res["shelter_frac"][fin] = (e["shelter"] / T)[fin]; res["rest_frac"][fin] = (e["rest"] / T)[fin]
            res["threat_frac"][fin] = (e["threat"] / T)[fin]
            with np.errstate(invalid="ignore", divide="ignore"):
                res["shelter_when_threat"][fin] = np.where(e["threat"] > 0, e["shelter_threat"] / e["threat"], np.nan)[fin]
                res["dash_when_threat"][fin] = np.where(e["threat"] > 0, e["dash_threat"] / e["threat"], np.nan)[fin]
                res["shelter_at_night"][fin] = np.where(e["night"] > 0, e["shelter_night"] / e["night"], np.nan)[fin]
            done |= fin
        w.events.clear()
        if done.all():
            break
    return res


def summarize(res):
    out = {}
    for k, v in res.items():
        v = v[~np.isnan(v)]
        if len(v) == 0:
            out[k] = {"mean": None, "ci95": [None, None], "n": 0}; continue
        m = float(v.mean()); h = float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        out[k] = {"mean": round(m, 4), "ci95": [round(m - h, 4), round(m + h, 4)], "n": int(len(v))}
    return out


def evaluate_policy(act_fn, families=("standard",), n=100):
    return {f: summarize(run_episodes(act_fn, f, n)) for f in families}


def baselines(families=FAMILIES, n=100, seed=12345):
    rng = np.random.default_rng(seed)
    return {"random": evaluate_policy(random_policy(rng), families, n),
            "forager": evaluate_policy(forager_policy, families, n),
            "survivor": evaluate_policy(survivor_policy, families, n)}
