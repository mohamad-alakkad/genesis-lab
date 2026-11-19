"""Build web/public/research.json for the Research Console from real records only:
store-branch eval histories, segment histories, configs, manifests and docs/reports.
python -m genesis.research [--lineages gen1:s1,s2 gen2:s1,s2,s3,s4 gen3:s1,s2,s3,s4]"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from genesis import ckpt, persist

ROOT = ckpt.ROOT
OUT = ROOT / "web" / "public" / "research.json"
BRANCH = {"gen1": "genesis-store-s{seed}"}


def _branch(lineage, seed):
    return BRANCH.get(lineage, "genesis-store-" + lineage + "-s{seed}").format(seed=seed)


def read_store(branch):
    remote = persist.config()[0]
    tmp = Path(tempfile.mkdtemp(prefix="genesis-rj-"))
    try:
        r = subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "-q", "--depth", "1", "--branch", branch, remote,
                            str(tmp / "s")], capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            return None
        s = tmp / "s"
        latest = json.loads((s / "latest.json").read_text())
        rd = s / "runs" / latest["run_id"]
        rows = lambda n: [json.loads(l) for l in (rd / n).read_text().splitlines() if l.strip()] if (rd / n).exists() else []
        cfg = json.loads((rd / "config.json").read_text()) if (rd / "config.json").exists() else {}
        hb = json.loads((rd / "heartbeat.json").read_text()) if (rd / "heartbeat.json").exists() else {}
        ce = json.loads((rd / "CLEAN_EXIT").read_text()) if (rd / "CLEAN_EXIT").exists() else None
        man = json.loads((rd / "ckpt" / latest["checkpoint_id"].split("/")[-1] / "manifest.json").read_text())
        hist = [h for h in rows("history.jsonl") if h.get("event") == "end"]
        return {"run_id": latest["run_id"], "branch": branch, "published_at": latest["published_at"],
                "latest_checkpoint": latest["checkpoint_id"], "global_step": man["global_step"],
                "world_spec_version": man["world_spec_version"], "code_commit": man["code_commit"],
                "wall_clock_trained_s": man.get("wall_clock_trained_s"), "eval": rows("eval_history.jsonl"),
                "segments": [{k: h.get(k) for k in ("segment", "end", "end_reason", "steps_start", "steps_end", "peak_rss_mb")} for h in hist],
                "steps_per_sec": hb.get("steps_per_sec"), "clean_exit": ce,
                "config": {k: cfg.get(k) for k in ("n_envs", "n_steps", "epochs", "minibatch", "lr", "gamma", "lam", "clip", "ent_coef",
                                                   "vf_coef", "hidden", "total_steps", "eval_every_steps", "eval_episodes", "seed")}}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv):
    specs = argv or ["gen1:1,2", "gen2:1,2,3,4", "gen3:1,2,3,4"]
    lineages = {}
    for spec in specs:
        lin, seeds = spec.split(":")
        runs = [r for r in (read_store(_branch(lin, s)) for s in seeds.split(",")) if r]
        if runs:
            lineages[lin] = runs
    if not argv:  # controlled experiments: cloud/experiments/<lineage>.json (one per generation)
        for cpath in sorted((ROOT / "cloud" / "experiments").glob("gen*.json")):
            conds = json.loads(cpath.read_text())["tokens"]
            runs = [r for r in (read_store("genesis-store-" + c["run"]) for c in conds.values()) if r]
            if runs:
                lineages[cpath.stem] = runs
    reports = {}
    for p in sorted((ROOT / "docs" / "reports").glob("*.json")):
        try:
            reports[p.stem] = json.loads(p.read_text())
        except Exception:
            pass
    out = {"schema": "genesis.research/1", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source": "store branches on github.com/mohamad-alakkad/genesis-lab + docs/reports (no synthetic values)",
           "lineages": lineages, "reports": reports,
           "world_versions": {"gen1": "1.0.0", "gen2": "2.0.0", "gen3": "2.1.0", "gen4": "2.2.0", "gen5": "3.0.0", "gen6": "3.0.0"}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"research.json: {', '.join(f'{k}×{len(v)}' for k, v in lineages.items())}, reports {list(reports)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1:])
