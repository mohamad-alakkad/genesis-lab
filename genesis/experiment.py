"""Controlled-experiment analysis from published stores only.
python -m genesis.experiment [conditions.json] [episodes]  →  docs/reports/gen4-results.json + .md
For every run: training curve at matched steps (eval_history) + a fresh held-out evaluation of its best and final
checkpoints (same seeds for every arm, deterministic argmax, the run's own input mask)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from genesis import ckpt, persist
from genesis.evalreport import load_policy
from genesis.evalonly.suite import run_episodes

ROOT = ckpt.ROOT
KEYS = ("ticks", "caught", "food", "shelter_when_threat", "dash_when_threat")


def fetch_run(branch, dst):
    r = subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "-q", "--depth", "1", "--branch", branch,
                        persist.config()[0], str(dst)], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return None
    latest = json.loads((dst / "latest.json").read_text())
    return dst / "runs" / latest["run_id"], latest


def ci(v):
    v = np.asarray([x for x in v if x == x], float)
    if len(v) == 0:
        return None
    m = float(v.mean()); h = float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return {"mean": round(m, 4), "ci95": [round(m - h, 4), round(m + h, 4)], "n": int(len(v))}


def main(argv):
    cpath = Path(argv[0]) if argv else ROOT / "cloud" / "conditions.json"
    n = int(argv[1]) if len(argv) > 1 else 100
    start = int(argv[2]) if len(argv) > 2 else 50000  # fresh held-out block, disjoint from the seeds used to pick "best"
    conds = json.loads(cpath.read_text())
    tmp = Path(tempfile.mkdtemp(prefix="genesis-exp-"))
    runs, arms = {}, {}
    try:
        for tok, c in conds["tokens"].items():
            got = fetch_run("genesis-store-" + c["run"], tmp / tok)
            if not got:
                runs[c["run"]] = {"status": "not published"}; continue
            rd, latest = got
            ev = [json.loads(l) for l in (rd / "eval_history.jsonl").read_text().splitlines() if l.strip()]
            st_last = json.loads((rd / "ckpt" / latest["checkpoint_id"].split("/")[-1] / "state.json").read_text())
            best_id = (st_last.get("best") or {}).get("checkpoint_id")
            info = {"token": tok, "steps": int(st_last["global_step"]), "terminal": st_last.get("terminal"),
                    "curve": [{k: e.get(k) for k in ("step", "score", "caught", "shelter_when_threat")} for e in ev]}
            for label, cid in (("best", best_id), ("final", latest["checkpoint_id"])):
                d = rd / "ckpt" / cid.split("/")[-1] if cid else None
                if d is None or not d.exists():
                    continue
                res = run_episodes(load_policy(d), "standard", n, start=start)
                info[label] = {"checkpoint": cid, **{k: ci(res[k]) for k in KEYS}, "raw_ticks": [float(x) for x in res["ticks"]]}
            runs[c["run"]] = info
            arm = c["run"].rsplit("-s", 1)[0]
            arms.setdefault(arm, []).append(c["run"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # arm summaries (episodes pooled over seeds; same held-out seeds for every arm → paired by seed)
    summary = {}
    for arm, rs in arms.items():
        for label in ("best", "final"):
            pooled = {k: [] for k in KEYS}
            ok = [runs[r] for r in rs if label in runs[r]]
            if not ok:
                continue
            for r in ok:
                for k in KEYS:
                    if k == "ticks":
                        pooled[k] += r[label]["raw_ticks"]
            summary.setdefault(arm, {})[label] = {"seeds": len(ok), "ticks": ci(pooled["ticks"]),
                                                  **{k: round(float(np.mean([r[label][k]["mean"] for r in ok if r[label][k]])), 4) for k in KEYS if k != "ticks"}}
    # matched-step comparison from training curves (common step range across all runs)
    common = min((runs[r]["curve"][-1]["step"] for r in runs if runs[r].get("curve")), default=0)
    at = {}
    for arm, rs in arms.items():
        vals = []
        for r in rs:
            cur = [e for e in runs[r].get("curve", []) if e["step"] <= common]
            if cur:
                vals.append(cur[-1]["score"])
        at[arm] = {"matched_step": common, "score_mean_over_seeds": round(float(np.mean(vals)), 1) if vals else None, "seeds": len(vals)}
    verdict = []
    for g in ("g99", "g997"):
        a, b = summary.get(f"gen4-mem-{g}", {}).get("best"), summary.get(f"gen4-nomem-{g}", {}).get("best")
        if a and b:
            sep = a["ticks"]["ci95"][0] > b["ticks"]["ci95"][1]
            verdict.append(f"gamma {g[1:].replace('g', '')}: memory {a['ticks']['mean']} vs masked {b['ticks']['mean']} ticks → "
                           + ("memory better (CIs separated)" if sep else "no CI-separated memory effect"))
    for m in ("mem", "nomem"):
        a, b = summary.get(f"gen4-{m}-g997", {}).get("best"), summary.get(f"gen4-{m}-g99", {}).get("best")
        if a and b:
            sep = a["ticks"]["ci95"][0] > b["ticks"]["ci95"][1]
            verdict.append(f"{m}: gamma .997 {a['ticks']['mean']} vs .99 {b['ticks']['mean']} ticks → " + ("longer horizon better (CIs separated)" if sep else "no CI-separated horizon effect"))
    out = {"schema": "genesis.experiment/1", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "experiment": conds.get("experiment"),
           "evaluation": f"held-out family 'standard', {n} episodes per checkpoint, fresh seeds 3e9+{start}..{start + n - 1} (disjoint from the seeds used to select 'best'; same for all arms), deterministic argmax, run's own input mask",
           "reference": {"scripted survivor": 724.0, "sensory hider memory 0": 724.0, "sensory hider memory 150": 933.3, "gen3 best": 688.7},
           "arms": summary, "matched_steps": at, "verdict": verdict,
           "runs": {k: {kk: vv for kk, vv in v.items() if kk != "raw_ticks"} for k, v in runs.items()}}
    for r in out["runs"].values():
        for label in ("best", "final"):
            if label in r:
                r[label].pop("raw_ticks", None)
    rep = ROOT / "docs" / "reports"; rep.mkdir(parents=True, exist_ok=True)
    (rep / "gen4-results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# gen4 results", "", out["evaluation"], "", "| arm | label | seeds | survival ticks (95% CI) | caught | seeds eaten | sheltered when threatened |", "|---|---|---|---|---|---|---|"]
    for arm, d in sorted(summary.items()):
        for label, s in d.items():
            md.append(f"| {arm} | {label} | {s['seeds']} | {s['ticks']['mean']} {s['ticks']['ci95']} | {s['caught']} | {s['food']} | {s['shelter_when_threat']} |")
    md += ["", "## Verdict (pre-registered rule: CI-separated survival)", *[f"- {v}" for v in verdict], "",
           "Reference: scripted survivor 724 ticks; sensory hider with 150-tick memory 933; gen3 best 689."]
    (rep / "gen4-results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1:])
