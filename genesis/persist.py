"""External checkpoint store: a git branch holding only the latest verified checkpoints (one
orphan commit, force-pushed). The pushed commit is the commit point; latest.json is inside it.
Verification: the remote branch must resolve to exactly the local commit hash (git content hash),
and a fetch re-checks every SHA-256 in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from genesis import ckpt

ROOT = ckpt.ROOT
WARNING = "Cross-environment recovery NOT guaranteed: no external checkpoint store configured."


def config():
    cfg = {}
    p = ROOT / "genesis.json"
    if p.exists():
        cfg = json.loads(p.read_text()).get("persist", {})
    remote = os.environ.get("GENESIS_STORE_REMOTE", cfg.get("remote"))
    branch = os.environ.get("GENESIS_STORE_BRANCH", cfg.get("branch", "checkpoints"))
    if remote == "origin":  # use the clone's own origin (works through the cloud sandbox git proxy)
        r = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True)
        remote = r.stdout.strip() or None
    return remote, branch


def _redact(s):
    import re
    return re.sub(r"//[^/@\s]+@", "//***@", s)


def _git(args, cwd, timeout=180):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {_redact(r.stderr.strip()[-300:])}")
    return r.stdout.strip()


def publish(run_dir: Path, run_id: str, retries=3, log=print):
    """Publish latest verified checkpoints + history + heartbeat. Returns dict or raises."""
    remote, branch = config()
    if not remote:
        return None
    store = run_dir.parent / f"_store-{branch}"
    if not (store / ".git").exists():
        store.mkdir(parents=True, exist_ok=True)
        _git(["init", "-q"], store)
        _git(["config", "user.name", "genesis-trainer"], store)
        _git(["config", "user.email", "genesis-trainer@localhost"], store)
    lg = ckpt.read_last_good(run_dir)
    if not lg:
        raise RuntimeError("nothing verified to publish")
    d = ckpt.ckpt_dir(run_dir, lg)
    ok, why, man = ckpt.verify_dir(d)
    if not ok:
        raise RuntimeError(f"refusing to publish unverified checkpoint: {why}")
    # rebuild store tree: latest 3 verified + milestones of this run
    for p in store.iterdir():
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    dst_run = store / "runs" / run_id
    keep = []
    for c in sorted((run_dir / "ckpt").iterdir()):
        okc, _, mc = ckpt.verify_dir(c)
        if okc:
            keep.append((c, mc))
    sel = keep[-3:] + [(c, mc) for c, mc in keep[:-3] if mc.get("milestones")]
    for c, _ in sel:
        shutil.copytree(c, dst_run / "ckpt" / c.name)
    for n in ("history.jsonl", "heartbeat.json", "eval_history.jsonl", "config.json"):
        if (run_dir / n).exists():
            shutil.copy2(run_dir / n, dst_run / n)
    (dst_run / "LAST_GOOD").write_text(lg + "\n")
    mbytes = (d / "manifest.json").read_bytes()
    latest = {"checkpoint_id": lg, "run_id": run_id, "manifest_sha256": hashlib.sha256(mbytes).hexdigest(),
              "published_at": ckpt.now_iso()}
    (store / "latest.json").write_text(json.dumps(latest, indent=1))
    (store / "README.md").write_text("GENESIS checkpoint store (machine-written; latest verified checkpoints only).\n")
    _git(["checkout", "-q", "--orphan", f"pub{int(time.time())}"], store)
    _git(["add", "-A"], store)
    _git(["commit", "-q", "-m", f"checkpoint {lg}"], store)
    head = _git(["rev-parse", "HEAD"], store)
    last = None
    for i in range(retries):
        try:
            _git(["push", "-q", "-f", remote, f"HEAD:refs/heads/{branch}"], store)
            rem = _git(["ls-remote", remote, f"refs/heads/{branch}"], store, timeout=60).split()
            if rem and rem[0] == head:
                # drop old local branches/objects
                for b in _git(["branch", "--format=%(refname:short)"], store).split():
                    if b != _git(["rev-parse", "--abbrev-ref", "HEAD"], store):
                        _git(["branch", "-q", "-D", b], store)
                return {"checkpoint_id": lg, "commit": head, "published_at": latest["published_at"]}
            last = f"remote head {rem[:1]} != {head}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"publish failed after {retries} tries: {last}")


def fetch(run_dir_parent: Path, log=print):
    """Fetch the remote store into runs/. Returns (run_id, checkpoint_id) of the remote latest or None."""
    remote, branch = config()
    if not remote:
        return None
    tmp = Path(tempfile.mkdtemp(prefix="genesis-fetch-"))
    try:
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", branch, remote, str(tmp / "s")],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            log(f"fetch: no remote store available ({r.stderr.strip()[-200:]})")
            return None
        s = tmp / "s"
        latest = json.loads((s / "latest.json").read_text())
        run_id = latest["run_id"]
        src = s / "runs" / run_id
        d = ckpt.ckpt_dir(src, latest["checkpoint_id"])
        if hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest() != latest["manifest_sha256"]:
            log("fetch: remote manifest hash mismatch with latest.json")
        dst = run_dir_parent / run_id
        (dst / "ckpt").mkdir(parents=True, exist_ok=True)
        for c in (src / "ckpt").iterdir():
            okc, why, _ = ckpt.verify_dir(c)
            if not okc:
                log(f"fetch: remote checkpoint {c.name} invalid ({why}); skipped")
                continue
            if not (dst / "ckpt" / c.name).exists():
                shutil.copytree(c, dst / "ckpt" / c.name)
        for n in ("history.jsonl", "eval_history.jsonl", "config.json"):
            if (src / n).exists() and not (dst / n).exists():
                shutil.copy2(src / n, dst / n)
        if not (dst / "heartbeat.json").exists() and (src / "heartbeat.json").exists():
            shutil.copy2(src / "heartbeat.json", dst / "heartbeat.remote.json")
        if not (dst / "LAST_GOOD").exists():
            ckpt.atomic_write(dst / "LAST_GOOD", (latest["checkpoint_id"] + "\n").encode())
        return run_id, latest["checkpoint_id"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
