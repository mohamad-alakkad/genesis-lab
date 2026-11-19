"""Generalization evaluation across families (fresh held-out seeds), parallel.
python -m genesis.familyeval <conditions.json> <episodes> <seed_start> <out_name> [families,comma,separated] [workers]
Each run's *best* checkpoint (selected on another seed block) is evaluated on every family; arms pool their seeds."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from genesis import ckpt
from genesis.experiment import ci, fetch_run

ROOT = ckpt.ROOT
FAMS = ["standard", "relentless", "night", "drought", "fissure_heavy", "lure_heavy", "one_shelter", "unseen_layout", "perturbed", "no_predator"]
KEYS = ("ticks", "caught", "food", "shelter_when_threat", "dash_when_threat", "survived")


def job(args):
    d, family, n, start = args
    from genesis.evalreport import load_policy
    from genesis.evalonly.suite import run_episodes
    res = run_episodes(load_policy(Path(d)), family, n, start=start)
    return {k: [float(x) for x in res[k]] for k in KEYS}


def main(argv):
    cpath, n, start, name = Path(argv[0]), int(argv[1]), int(argv[2]), argv[3]
    fams = argv[4].split(",") if len(argv) > 4 and argv[4] else FAMS
    workers = int(argv[5]) if len(argv) > 5 else 4
    conds = json.loads(cpath.read_text())["tokens"]
    tmp = Path(tempfile.mkdtemp(prefix="genesis-fam-"))
    t0 = time.time()
    try:
        runs = {}
        for tok, c in conds.items():
            got = fetch_run("genesis-store-" + c["run"], tmp / tok)
            if not got:
                continue
            rd, latest = got
            st = json.loads((rd / "ckpt" / latest["checkpoint_id"].split("/")[-1] / "state.json").read_text())
            best = (st.get("best") or {}).get("checkpoint_id") or latest["checkpoint_id"]
            runs[c["run"]] = {"dir": str(rd / "ckpt" / best.split("/")[-1]), "checkpoint": best, "arm": c["run"].rsplit("-s", 1)[0]}
        jobs = [(r["dir"], f, n, start) for r in runs.values() for f in fams]
        with Pool(workers) as pool:
            out = pool.map(job, jobs)
        raw = {}
        for (d, f, _, _), res in zip(jobs, out):
            run = next(k for k, v in runs.items() if v["dir"] == d)
            raw.setdefault(run, {})[f] = res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    arms = sorted({r["arm"] for r in runs.values()})
    table = {}
    for arm in arms:
        members = [k for k, v in runs.items() if v["arm"] == arm]
        for f in fams:
            pooled = {k: sum((raw[m][f][k] for m in members), []) for k in KEYS}
            table.setdefault(arm, {})[f] = {k: ci(v) for k, v in pooled.items()} | {"seeds": len(members), "episodes": len(pooled["ticks"])}
    comps = []
    if len(arms) == 2:
        a, b = arms
        for f in fams:
            x, y = table[a][f]["ticks"], table[b][f]["ticks"]
            sep = x["ci95"][0] > y["ci95"][1] or y["ci95"][0] > x["ci95"][1]
            comps.append({"family": f, a: x["mean"], b: y["mean"], "separated": bool(sep)})
    report = {"schema": "genesis.familyeval/1", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "conditions": str(cpath), "episodes_per_run_family": n,
              "seeds": f"fresh held-out block: eval_seed_base + family offset + {start}..{start + n - 1}; best checkpoints were selected on block 0..99",
              "runs": {k: {"checkpoint": v["checkpoint"], "arm": v["arm"]} for k, v in runs.items()}, "arms": table, "comparisons": comps,
              "wall_seconds": round(time.time() - t0)}
    rep = ROOT / "docs" / "reports"; rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    md = [f"# {name}", "", report["seeds"] + f"; {n} episodes per run per family; arms pool their seeds.", "",
          "| family | arm | survival ticks (95% CI) | caught | seeds eaten | sheltered when threatened | dash when threatened |", "|---|---|---|---|---|---|---|"]
    for f in fams:
        for arm in arms:
            r = table[arm][f]
            md.append(f"| {f} | {arm} | {r['ticks']['mean']} {r['ticks']['ci95']} | {r['caught']['mean']} | {r['food']['mean']} | "
                      f"{(r['shelter_when_threat'] or {}).get('mean')} | {(r['dash_when_threat'] or {}).get('mean')} |")
    if comps:
        md += ["", "## Arm comparison (survival, CI-separated?)", *[f"- {c['family']}: " + ", ".join(f"{k} {v}" for k, v in c.items() if k not in ("family", "separated")) + (" — separated" if c["separated"] else " — overlapping") for c in comps]]
    (rep / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1:])
