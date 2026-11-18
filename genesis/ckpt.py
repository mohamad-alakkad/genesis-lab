"""Checkpoint write → verify → promote, manifest, LAST_GOOD, and verified loading with fallback."""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from genesis import ppo
from genesis.world import SPEC_VERSION

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def code_commit():
    try:
        c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain", "--", "genesis"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        return c or "none", bool(d)
    except Exception:
        return "none", True


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def pack(params, adam, norm):
    arrays = {f"p/{k}": v for k, v in params.items()}
    arrays.update({f"m/{k}": v for k, v in adam.m.items()})
    arrays.update({f"v/{k}": v for k, v in adam.v.items()})
    arrays["norm/mean"] = norm.mean; arrays["norm/var"] = norm.var
    buf = io.BytesIO(); np.savez(buf, **arrays)
    return buf.getvalue()


def unpack(data):
    z = np.load(io.BytesIO(data))
    params = {k[2:]: z[k] for k in z.files if k.startswith("p/")}
    m = {k[2:]: z[k] for k in z.files if k.startswith("m/")}
    v = {k[2:]: z[k] for k in z.files if k.startswith("v/")}
    return params, m, v, z["norm/mean"], z["norm/var"]


def arch_signature(params):
    return {k: list(v.shape) for k, v in sorted(params.items())}


def write_checkpoint(run_dir: Path, *, run_id, step, params, adam, norm, state: dict, config: dict,
                     probe_obs: np.ndarray, milestones=(), parent=None, eval_summary=None):
    """Returns the manifest dict. Only after load-test + manifest is LAST_GOOD updated."""
    d = run_dir / "ckpt" / f"{step:010d}"
    d.mkdir(parents=True, exist_ok=True)
    model = pack(params, adam, norm)
    atomic_write(d / "model.npz", model)
    st = dict(state); st["adam_t"] = adam.t; st["norm_count"] = norm.count; st["arch"] = arch_signature(params)
    atomic_write(d / "state.json", json.dumps(st, indent=1).encode())
    # load-test: fresh deserialization → forward pass equals live policy
    p2, _, _, mu, var = unpack((d / "model.npz").read_bytes())
    x_live = np.clip((probe_obs - norm.mean) / np.sqrt(norm.var + 1e-8), -10, 10)
    x_new = np.clip((probe_obs - mu) / np.sqrt(var + 1e-8), -10, 10)
    if not np.allclose(ppo.policy_logits(params, x_live), ppo.policy_logits(p2, x_new), atol=1e-9, rtol=0):
        raise RuntimeError("checkpoint load-test failed")
    files = [{"path": n, "sha256": sha256_file(d / n), "size": (d / n).stat().st_size} for n in ("model.npz", "state.json")]
    commit, dirty = code_commit()
    prev = read_last_good(run_dir)
    manifest = {
        "schema": "genesis.manifest/1", "run_id": run_id, "checkpoint_id": f"{run_id}/{step:010d}",
        "parent": parent, "global_step": int(step), "wall_clock_trained_s": round(state.get("wall_trained_s", 0.0), 1),
        "created_at": now_iso(), "files": files,
        "config_hash": "sha256:" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "world_spec_version": SPEC_VERSION, "code_commit": commit, "code_dirty": dirty,
        "dependencies": {"python": platform.python_version(), "numpy": np.__version__},
        "verified": True, "last_known_good": prev, "milestones": list(milestones),
        "eval_summary": eval_summary,
    }
    atomic_write(d / "manifest.json", json.dumps(manifest, indent=1).encode())
    atomic_write(run_dir / "LAST_GOOD", (manifest["checkpoint_id"] + "\n").encode())
    return manifest


def read_last_good(run_dir: Path):
    p = run_dir / "LAST_GOOD"
    return p.read_text().strip() if p.exists() else None


def ckpt_dir(run_dir: Path, checkpoint_id: str) -> Path:
    return run_dir / "ckpt" / checkpoint_id.split("/")[-1]


def verify_dir(d: Path):
    """Return (ok, reason, manifest)."""
    mp = d / "manifest.json"
    if not mp.exists():
        return False, "no manifest", None
    try:
        m = json.loads(mp.read_text())
    except Exception as e:
        return False, f"manifest unreadable: {e}", None
    if not m.get("verified"):
        return False, "manifest not verified", m
    for f in m["files"]:
        p = d / f["path"]
        if not p.exists() or p.stat().st_size != f["size"] or sha256_file(p) != f["sha256"]:
            return False, f"checksum mismatch: {f['path']}", m
    try:
        unpack((d / "model.npz").read_bytes())
    except Exception as e:
        return False, f"load failed: {e}", m
    return True, "ok", m


def find_loadable(run_dir: Path, log=print):
    """LAST_GOOD first, then walk older checkpoints. Returns (dir, manifest, fallbacks list)."""
    fallbacks = []
    cands = []
    lg = read_last_good(run_dir)
    if lg:
        cands.append(ckpt_dir(run_dir, lg))
    root = run_dir / "ckpt"
    if root.exists():
        cands += sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    seen = set()
    for d in cands:
        if d in seen:
            continue
        seen.add(d)
        ok, why, m = verify_dir(d)
        if ok:
            return d, m, fallbacks
        fallbacks.append(f"{d.name}: {why}")
        log(f"checkpoint {d.name} rejected ({why}); falling back")
    return None, None, fallbacks


def prune(run_dir: Path, keep_last=3, keep=()):
    root = run_dir / "ckpt"
    if not root.exists():
        return
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    protected = set(keep)
    for d in dirs:
        mp = d / "manifest.json"
        if mp.exists():
            try:
                if json.loads(mp.read_text()).get("milestones"):
                    protected.add(d.name)
            except Exception:
                pass
    verified = [d for d in dirs if verify_dir(d)[0]]
    protected |= {d.name for d in verified[-keep_last:]}
    for d in dirs:
        if d.name not in protected and verified and d < verified[-1]:
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
