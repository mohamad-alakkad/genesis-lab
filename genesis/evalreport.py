"""`evaluate`: held-out suite (all families) for one checkpoint vs baselines → docs/reports/eval-*.json/.md."""
from __future__ import annotations

import json
from pathlib import Path

from genesis import ckpt, ppo
from genesis.evalonly import suite

REPORTS = ckpt.ROOT / "docs" / "reports"
METRICS = ("ticks", "food", "caught", "survived", "shelter_when_threat", "shelter_at_night", "rest_frac", "dash_when_threat")


def load_policy(d: Path):
    p, _, _, mu, var = ckpt.unpack((d / "model.npz").read_bytes())
    norm = ppo.RunningNorm(len(mu)); norm.mean = mu; norm.var = var
    return lambda o: ppo.policy_logits(p, norm(o)).argmax(1)


def baselines(n, families):
    bp = REPORTS / "baselines.json"
    if bp.exists():
        b = json.loads(bp.read_text())
        if b.get("episodes") == n and all(f in b["random"] for f in families):
            return b
    b = suite.baselines(families, n)
    b["episodes"] = n
    REPORTS.mkdir(parents=True, exist_ok=True)
    ckpt.atomic_write(bp, json.dumps(b, indent=1).encode())
    return b


def _beats(a, b):  # CI-separated improvement
    return a["ci95"][0] > b["ci95"][1]


def evaluate_checkpoint(ckpt_id, run_id, n, families):
    fams = [f for f in families.split(",") if f]
    run_dir = ckpt.ROOT / "runs" / run_id
    if ckpt_id:
        d = ckpt.ckpt_dir(run_dir, ckpt_id)
    else:
        d, _, _ = ckpt.find_loadable(run_dir)
    ok, why, man = ckpt.verify_dir(d) if d else (False, "no checkpoint", None)
    if not ok:
        print(f"cannot evaluate: {why}"); return 5
    pol = suite.evaluate_policy(load_policy(d), fams, n)
    b = baselines(n, fams)
    res = {"schema": "genesis.eval/1", "checkpoint_id": man["checkpoint_id"], "suite": "heldout-v1",
           "seeds_range": f"eval_seed_base + family offset + [0, {n})", "episodes_per_family": n, "deterministic": True,
           "families": {f: {"policy": pol[f], **{k: b[k][f] for k in ("random", "forager", "survivor")}} for f in fams}}
    wins = {f: _beats(pol[f]["ticks"], b["random"][f]["ticks"]) and _beats(pol[f]["ticks"], b["survivor"][f]["ticks"]) for f in fams}
    verdict = "learned" if all(wins.values()) else ("partially" if any(wins.values()) else "not_learned")
    res["verdict"] = verdict; res["beats_random_and_survivor_on_survival"] = wins
    REPORTS.mkdir(parents=True, exist_ok=True)
    name = man["checkpoint_id"].replace("/", "-")
    ckpt.atomic_write(REPORTS / f"eval-{name}.json", json.dumps(res, indent=1).encode())
    md = [f"# Evaluation — {man['checkpoint_id']}", "", f"Verdict: **{verdict}** · {n} deterministic episodes per family · held-out seeds", "",
          "| family | metric | policy | random | forager | survivor |", "|---|---|---|---|---|---|"]
    for f in fams:
        for m in METRICS:
            c = lambda x: f"{x[m]['mean']} [{x[m]['ci95'][0]}, {x[m]['ci95'][1]}]"
            md.append(f"| {f} | {m} | {c(pol[f])} | {c(b['random'][f])} | {c(b['forager'][f])} | {c(b['survivor'][f])} |")
    ckpt.atomic_write(REPORTS / f"eval-{name}.md", ("\n".join(md) + "\n").encode())
    for f in fams:
        print(f"{f:14s} survival {pol[f]['ticks']['mean']:7.1f} (rand {b['random'][f]['ticks']['mean']:.0f}, survivor {b['survivor'][f]['ticks']['mean']:.0f}) "
              f"caught {pol[f]['caught']['mean']:.2f} food {pol[f]['food']['mean']:.1f}")
    print(f"verdict: {verdict} → docs/reports/eval-{name}.md")
    return 0
