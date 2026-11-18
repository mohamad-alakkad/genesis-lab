"""Trajectory-level failure analysis (evaluation-only). python -m genesis.evalonly.trace <ckpt_dir> [episodes]
Decomposes defensive failure into: perception (was the threat sensed in time?), response (shelter entry),
persistence (staying while hunted), and outcome (what happened after leaving)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from genesis import ckpt, ppo
from genesis.evalonly.baselines import survivor_policy
from genesis.evalonly.suite import eval_seeds
from genesis.world import C, HUNT, N_CH, SEARCH, World

S = N_CH * C["n_rays"]


def policy_from(d):
    from genesis.evalreport import load_policy
    return load_policy(Path(d))


def record(act, n, family="standard"):
    w = World(n, family); w.reset(np.ones(n, bool), eval_seeds(family, n))
    T = C["max_ticks"]
    rec = {k: np.zeros((T, n), dtype=np.float32) for k in ("heard", "seen", "shelter_seen", "in_sh", "mode", "pounce", "dpred", "dshel", "alive", "act", "v", "caught")}
    alive = np.ones(n, bool)
    for t in range(T):
        o = w.observe()
        a = act(o)
        rays = o[:, :S].reshape(n, -1, N_CH)
        rec["heard"][t] = o[:, S + 12] > 0; rec["seen"][t] = (rays[..., 4] > 0).any(1); rec["shelter_seen"][t] = (rays[..., 3] > 0).any(1)
        rec["in_sh"][t] = w.in_shelter; rec["mode"][t] = np.where(w.pr_on, w.pr_mode, -1); rec["pounce"][t] = w.pr_pounce
        rec["dpred"][t] = np.linalg.norm(w.pr_pos - w.pos, axis=1)
        dsh = np.linalg.norm(w.sh_pos - w.pos[:, None], axis=2) - w.sh_r
        rec["dshel"][t] = np.where(w.sh_on, dsh, np.inf).min(1)
        rec["alive"][t] = alive; rec["act"][t] = a; rec["v"][t] = w.v
        _, dead, trunc = w.step(a)
        rec["caught"][t] = alive & (w.ep["caught"] > 0) & dead
        alive &= ~(dead | trunc)
        w.events.clear()
        if not alive.any():
            for k in rec: rec[k] = rec[k][: t + 1]
            break
    return rec


def analyze(rec):
    T, n = rec["alive"].shape
    out = {}
    # --- captures: what did Vela know, and when?
    caps = []
    for e in range(n):
        ts = np.nonzero(rec["caught"][:, e])[0]
        if not len(ts):
            continue
        tc = ts[0]; lo = max(0, tc - 80)
        cue = (rec["heard"][lo:tc + 1, e] > 0) | (rec["seen"][lo:tc + 1, e] > 0)
        first = np.nonzero(cue)[0]
        lead = (tc - (lo + first[0])) if len(first) else -1
        t_cue = lo + first[0] if len(first) else tc
        was_in = rec["in_sh"][lo:tc, e].max() > 0
        exits = np.nonzero(np.diff(rec["in_sh"][lo:tc + 1, e]) < 0)[0]
        caps.append({"lead": int(lead), "dshel_at_cue": float(rec["dshel"][t_cue, e]), "shelter_visible_at_cue": bool(rec["shelter_seen"][t_cue, e]),
                     "in_shelter_last80": bool(was_in), "exited_then_caught": bool(len(exits)), "ticks_exit_to_catch": int(tc - (lo + exits[-1])) if len(exits) else None,
                     "dashed_after_cue": float((rec["act"][t_cue:tc + 1, e] == 3).mean())})
    out["captures"] = len(caps)
    if caps:
        L = np.array([c["lead"] for c in caps])
        out["capture_no_cue_in_80_ticks"] = float(np.mean(L < 0))
        out["capture_cue_lead_ticks_median"] = float(np.median(L[L >= 0])) if (L >= 0).any() else None
        out["capture_cue_lead_lt_10"] = float(np.mean((L >= 0) & (L < 10)))
        reach = np.array([c["dshel_at_cue"] for c in caps])
        out["capture_shelter_within_2u_at_cue"] = float(np.mean(reach < 2.0))
        out["capture_shelter_visible_at_cue"] = float(np.mean([c["shelter_visible_at_cue"] for c in caps]))
        out["capture_after_leaving_shelter"] = float(np.mean([c["exited_then_caught"] for c in caps]))
        out["capture_dash_frac_after_cue"] = float(np.mean([c["dashed_after_cue"] for c in caps]))
    # --- threat bouts: predator hunting/searching within 6 u
    bouts = []
    for e in range(n):
        thr = ((rec["mode"][:, e] == HUNT) | (rec["mode"][:, e] == SEARCH)) & (rec["dpred"][:, e] < 6) & (rec["alive"][:, e] > 0)
        t = 0
        while t < T:
            if thr[t]:
                t0 = t
                while t < T and thr[t]: t += 1
                seg = slice(t0, t)
                sensed = ((rec["heard"][seg, e] > 0) | (rec["seen"][seg, e] > 0))
                ins = rec["in_sh"][seg, e] > 0
                bouts.append({"len": t - t0, "sensed": bool(sensed.any()), "sensed_frac": float(sensed.mean()), "entered": bool(ins.any() and not ins[0]),
                              "started_inside": bool(ins[0]), "inside_frac": float(ins.mean()), "dshel0": float(rec["dshel"][t0, e]),
                              "left_while_threat": bool(ins.any() and (np.diff(ins.astype(int)) < 0).any())})
            t += 1
    out["threat_bouts"] = len(bouts)
    if bouts:
        B = bouts
        out["bout_sensed_at_all"] = float(np.mean([b["sensed"] for b in B]))
        out["bout_sensed_frac_mean"] = float(np.mean([b["sensed_frac"] for b in B]))
        out["bout_entered_shelter"] = float(np.mean([b["entered"] for b in B]))
        out["bout_entered_given_shelter_within_2u"] = float(np.mean([b["entered"] for b in B if b["dshel0"] < 2.0])) if any(b["dshel0"] < 2.0 for b in B) else None
        out["bout_left_while_threat_given_inside"] = float(np.mean([b["left_while_threat"] for b in B if b["inside_frac"] > 0])) if any(b["inside_frac"] > 0 for b in B) else None
        out["bout_inside_frac_mean"] = float(np.mean([b["inside_frac"] for b in B]))
    return out


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    res = {"policy": analyze(record(policy_from(sys.argv[1]), n)), "survivor": analyze(record(survivor_policy, n))}
    print(json.dumps(res, indent=1))
