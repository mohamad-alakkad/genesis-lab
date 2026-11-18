"""GENESIS CLI: python -m genesis <train|resume|stop|status|evaluate|export|diagnose|publish|fetch>"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from genesis import ckpt, persist
from genesis.world import SPEC_VERSION

ROOT = ckpt.ROOT
RUNS = ROOT / "runs"


def _iso_ts(s):
    import calendar
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def current_run():
    p = RUNS / "CURRENT"
    return p.read_text().strip() if p.exists() else None


def set_current(run_id):
    RUNS.mkdir(exist_ok=True)
    try:
        ckpt.atomic_write(RUNS / "CURRENT", (run_id + "\n").encode())
    except OSError:
        pass  # convenience pointer only; concurrent trainers may race on it


def _parse_sets(sets):
    out = {}
    for s in sets or []:
        k, v = s.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _prev_end(run_dir: Path, manifest):
    ce = run_dir / "CLEAN_EXIT"
    if ce.exists():
        d = json.loads(ce.read_text())
        return {"reason": d.get("end_reason"), "confirmed": True, "lost_time_s": 0.0}
    hb = None
    for n in ("heartbeat.json", "heartbeat.remote.json"):
        if (run_dir / n).exists():
            hb = json.loads((run_dir / n).read_text()); break
    if hb is None:
        return {"reason": "unknown(no heartbeat)", "confirmed": False, "lost_time_s": 0.0}
    lost = max(0.0, hb.get("updated_ts", 0) - _iso_ts(manifest["created_at"]))
    return {"reason": "inferred_kill", "confirmed": False, "lost_time_s": round(lost, 1),
            "gap_s": round(time.time() - hb.get("updated_ts", time.time()), 1),
            "last_heartbeat_step": hb.get("global_step")}


def _launch_detached(argv_extra, run_id, timeout_s=180):
    from genesis import jobguard
    import psutil
    avail = psutil.virtual_memory().available / 2**20
    if avail < 3072:
        print(f"resource guard: only {avail:.0f} MB free system RAM (< 2 GB budget + 1 GB margin); refusing to start")
        return 6
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "trainer.log"
    if log.exists() and log.stat().st_size > 5 * 2**20:
        os.replace(log, run_dir / "trainer.log.1")
    lg_before = ckpt.read_last_good(run_dir)
    t0 = time.time()
    p = jobguard.launch([sys.executable, "-m", "genesis", *argv_extra, "--foreground"], detach=True,
                        log_path=str(log), cwd=str(ROOT))
    hb_ok = ck_ok = False
    while time.time() - t0 < timeout_s:
        if p.poll() is not None:
            print(f"trainer exited early with code {p.returncode}; see {log}")
            return 1
        hbp = run_dir / "heartbeat.json"
        if hbp.exists():
            try:
                hb = json.loads(hbp.read_text()); hb_ok = hb.get("pid") == p.pid
            except Exception:
                pass
        lg = ckpt.read_last_good(run_dir)
        if lg and (lg != lg_before) and ckpt.verify_dir(ckpt.ckpt_dir(run_dir, lg))[0]:
            ck_ok = True
        if hb_ok and ck_ok:
            print(f"trainer running detached: pid {p.pid}, run {run_id}, first verified checkpoint {lg}")
            return 0
        time.sleep(2)
    print(f"launch not confirmed within {timeout_s}s (heartbeat={hb_ok}, checkpoint={ck_ok}); trainer pid {p.pid} left running; see {log}")
    return 1


# --------------------------------------------------------------------------- commands
def cmd_train(a):
    from genesis.trainer import DEFAULT_CONFIG
    cfg = dict(DEFAULT_CONFIG)
    if a.config:
        cfg.update(json.loads(Path(a.config).read_text()))
    cfg.update(_parse_sets(a.set))
    if a.seed is not None:
        cfg["seed"] = a.seed
    run_id = a.run or time.strftime("r%Y%m%d-%H%M%S") + f"-s{cfg['seed']}"
    run_dir = RUNS / run_id
    if (run_dir / "LAST_GOOD").exists():
        print(f"run {run_id} already has checkpoints; use resume"); return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt.atomic_write(run_dir / "config.json", json.dumps(cfg, indent=1).encode())
    set_current(run_id)
    if not a.foreground:
        return _launch_detached(["resume", "--run", run_id] + (["--minutes", str(a.minutes)] if a.minutes else []), run_id)
    return _run_foreground(run_id, None, cfg, a)


def _run_foreground(run_id, ck, cfg, a, parent=None, prev_end=None):
    from genesis.trainer import Trainer
    if a.minutes:
        cfg = dict(cfg, segment_minutes=a.minutes)
    t = Trainer(run_id, cfg, ck, parent=parent, log=lambda m: print(m, flush=True))
    state, reason = t.run(segments=a.segments or 1, until_steps=a.until, prev_end=prev_end)
    print(f"trainer ended: state={state} reason={reason} step={t.state['global_step']}")
    return 6 if reason in ("resource_limit", "host_pressure") else (1 if state == "failed" else 0)


def cmd_resume(a):
    run_id = a.run or current_run()
    fresh = run_id is not None and (RUNS / run_id / "config.json").exists() and not (RUNS / run_id / "ckpt").exists()
    if run_id is None or (not fresh and not (RUNS / run_id / "ckpt").exists()):
        r = persist.fetch(RUNS)
        if r:
            run_id = run_id or r[0]
            print(f"fetched remote checkpoint {r[1]}")
    if run_id is None:
        print("no run found locally or remotely; start one with: python -m genesis train"); return 3
    run_dir = RUNS / run_id
    cfgp = run_dir / "config.json"
    d, man, fb = ckpt.find_loadable(run_dir)
    if d is None and not fresh:
        r = persist.fetch(RUNS)
        if r and r[0] == run_id:
            d, man, fb2 = ckpt.find_loadable(run_dir); fb += fb2
    if fb:
        print("integrity fallbacks: " + "; ".join(fb[:5]))
    if d is None:
        if (cfgp.exists() and not (run_dir / "ckpt").exists()) or a.new:
            pass  # brand-new run created by `train`
        elif not fb:
            print(f"no checkpoint for {run_id}"); return 3
        else:
            print("no valid checkpoint remains after integrity checks"); return 5
    if not cfgp.exists() and d is not None:
        cfg = json.loads((d / "state.json").read_text()).get("config")
        if cfg is None:
            from genesis.trainer import DEFAULT_CONFIG
            cfg = dict(DEFAULT_CONFIG)
        ckpt.atomic_write(cfgp, json.dumps(cfg, indent=1).encode())
    cfg = json.loads(cfgp.read_text())
    cfg.update(_parse_sets(a.set))
    ckpt.atomic_write(cfgp, json.dumps(cfg, indent=1).encode())
    set_current(run_id)
    if d is not None:
        st = json.loads((d / "state.json").read_text())
        if st.get("terminal") and not a.new:
            print(f"run {run_id} already finished: {st['terminal']} at step {man['global_step']}; nothing to resume")
            return 0
        hbr = run_dir / "heartbeat.remote.json"
        if hbr.exists() and not (run_dir / "heartbeat.json").exists() and not (run_dir / "CLEAN_EXIT").exists():
            age = time.time() - json.loads(hbr.read_text()).get("updated_ts", 0)
            if age < 900:
                print(f"another trainer published a heartbeat {age:.0f} s ago without a clean exit; refusing to fork the run")
                return 7
        from genesis import ppo
        from genesis.world import OBS_DIM, N_ACTIONS
        want = ckpt.arch_signature(ppo.init_params(OBS_DIM, N_ACTIONS, cfg["hidden"], 0))
        if man["world_spec_version"] != SPEC_VERSION or st.get("arch") != want:
            print(f"incompatible checkpoint (spec {man['world_spec_version']} vs {SPEC_VERSION} or architecture); use a new run")
            return 4
    if not a.foreground:
        extra = ["resume", "--run", run_id]
        for k in ("segments", "until", "minutes"):
            if getattr(a, k):
                extra += [f"--{k}", str(getattr(a, k))]
        return _launch_detached(extra, run_id)
    prev = _prev_end(run_dir, man) if d is not None else None
    if prev and fb:
        prev["fallbacks"] = fb
    if prev:
        print(f"previous end: {prev['reason']} ({'confirmed' if prev['confirmed'] else 'inferred'}), lost {prev['lost_time_s']} s; resuming at step {man['global_step']}")
    return _run_foreground(run_id, d, cfg, a, prev_end=prev)


def cmd_stop(a):
    run_id = a.run or current_run()
    if not run_id:
        print("no current run"); return 3
    (RUNS / run_id / "STOP").write_text("stop\n")
    print(f"stop requested for {run_id}; the trainer checkpoints and exits within one iteration")
    return 0


def _pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower()
    except Exception:
        return False


def cmd_status(a):
    remote, branch = persist.config()
    if a.remote:
        for br in a.remote.split(","):
            print(persist.remote_summary(br))
        return 0
    p = RUNS / "status.json"
    lines = []
    if not p.exists():
        lines.append("state: no local run status")
        st = None
    else:
        st = json.loads(p.read_text())
        now = time.time()
        hb_age = now - st["heartbeat_ts"]
        state = st["state"]
        hbp = RUNS / st["run_id"] / "heartbeat.json"
        pid = json.loads(hbp.read_text()).get("pid") if hbp.exists() else None
        if state == "running" and (hb_age > 3 * st["heartbeat_s"] or (pid and not _pid_alive(pid))):
            state = "interrupted (inferred)"
        st["state_effective"] = state
        st["heartbeat_age_s"] = round(hb_age)
        st["last_local_ckpt_age_s"] = round(now - st["last_local_ckpt_ts"]) if st.get("last_local_ckpt_ts") else None
        pub = st.get("last_remote_publish")
        st["last_remote_publish_age_s"] = round(now - _iso_ts(pub["published_at"])) if pub else None
        tot = st.get("total_steps") or 1
        lines.append(f"state: {state} · run {st['run_id']} · step {st['global_step']:,} / {tot:,} ({100*st['global_step']/tot:.1f}%) · {st['steps_per_sec']} steps/s")
        if st.get("end_reason"):
            lines.append(f"end reason: {st['end_reason']}")
        lines.append(f"heartbeat {st['heartbeat_age_s']} s ago · local ckpt {st['last_local_ckpt_age_s']} s ago · remote publish "
                     + (f"{st['last_remote_publish_age_s']} s ago ({pub['checkpoint_id']})" if pub else "never"))
        if pub and state.startswith("running"):
            lines.append(f"max progress at risk on hard kill: {st['last_remote_publish_age_s']} s of training")
        if st.get("publish_error"):
            lines.append(f"publish error: {st['publish_error'][:160]}")
        el, eb = st.get("eval_latest"), st.get("eval_best")
        if el:
            lines.append(f"held-out (standard, argmax): survival {el['score']} ticks {el['score_ci95']} · food {el['food_mean']} · caught {el['caught']} · sheltered-under-threat {el['shelter_when_threat']} @ {el['step']:,}")
        if eb:
            lines.append(f"best: survival {eb['score']} ticks, caught {eb['caught']} ({eb['checkpoint_id']})")
        bp = ROOT / "docs" / "reports" / "baselines.json"
        if bp.exists():
            b = json.loads(bp.read_text())
            lines.append("baselines (standard survival ticks): " + " · ".join(
                f"{k} {v['standard']['ticks']['mean']}" for k, v in b.items() if isinstance(v, dict) and "standard" in v))
        tr = st.get("train_recent") or {}
        lines.append(f"training episodes (recent, stochastic): food {tr.get('food')} ticks {tr.get('ticks')}")
        i = st["interruptions"]
        lines.append(f"interruptions: {i['count']} · lost {i['lost_time_s']} s · last end: {i['last_end_reason']}")
        r = st["resources"]
        lines.append(f"RAM: {r['rss_mb']} MB now, peak {r['peak_rss_mb']} MB, limit {r['limit_mb']} MB ({r['mechanism']})")
    if not remote:
        lines.append(persist.WARNING)
    else:
        lines.append(f"external store: {branch} @ {remote.split('@')[-1]}")
    ci = ROOT / "runs" / "ci_status.json"
    if ci.exists():
        c = json.loads(ci.read_text())
        lines.append(f"cloud (GitHub Actions): {c.get('summary', '?')} (checked {c.get('checked_at', '?')})")
    rec = "nothing — training healthy; check again next session"
    if st is None:
        rec = "python -m genesis train"
    elif st["state_effective"].startswith("interrupted"):
        rec = "python -m genesis resume"
    elif st["state_effective"] in ("completed", "plateaued", "budget_exhausted"):
        rec = "python -m genesis evaluate"
    elif st["state_effective"] == "paused":
        rec = f"python -m genesis resume (paused: {st.get('end_reason')})"
    elif st["state_effective"] == "failed":
        rec = "python -m genesis diagnose"
    lines.append(f"next: {rec}")
    if a.json:
        print(json.dumps(st, indent=1)[:4000])
    else:
        print("\n".join(lines[:25]))
    return 0


def cmd_diagnose(a):
    import shutil
    findings = []
    crit = False
    p = RUNS / "status.json"
    if not p.exists():
        print("no status"); return 0
    st = json.loads(p.read_text())
    d = st.get("diag") or {}
    if d.get("nan"):
        findings.append("CRITICAL NaN/inf in weights"); crit = True
    if d.get("entropy0") and d.get("entropy") is not None and d["entropy"] < 0.05 * d["entropy0"]:
        findings.append(f"entropy collapse: {d['entropy']} vs initial {d['entropy0']}")
    if (st["resources"]["peak_rss_mb"] or 0) > 1200:
        findings.append(f"memory near budget: peak {st['resources']['peak_rss_mb']} MB")
    if st.get("end_reason") in ("resource_limit", "host_pressure"):
        findings.append(f"run stopped by guard: {st['end_reason']}")
    if st["state"] == "running" and time.time() - st["heartbeat_ts"] > 3 * st["heartbeat_s"]:
        findings.append("stale heartbeat while state is running")
    if st.get("publish_error"):
        findings.append(f"publish failing: {st['publish_error'][:150]}")
    if shutil.disk_usage(ROOT).free < 2**30:
        findings.append("CRITICAL disk free < 1 GB"); crit = True
    run_dir = RUNS / st["run_id"]
    lg = ckpt.read_last_good(run_dir)
    if lg:
        ok, why, _ = ckpt.verify_dir(ckpt.ckpt_dir(run_dir, lg))
        if not ok:
            findings.append(f"CRITICAL LAST_GOOD invalid: {why}"); crit = True
    eh = run_dir / "eval_history.jsonl"
    if eh.exists():
        rows = [json.loads(l) for l in eh.read_text().splitlines() if l.strip()]
        if len(rows) >= 6 and max(r["score"] for r in rows[-5:]) <= rows[-6]["score"] + 5 and rows[-1]["survived"] < 0.5:
            findings.append("flat held-out score over the last 5 evals")
    print("\n".join(findings) if findings else "no findings")
    return 11 if crit else (10 if findings else 0)


def cmd_publish(a):
    run_id = a.run or current_run()
    if not run_id:
        print("no current run"); return 3
    if not persist.config()[0]:
        print(persist.WARNING); return 1
    r = persist.publish(RUNS / run_id, run_id)
    print(f"published {r['checkpoint_id']} commit {r['commit'][:10]}")
    return 0


def cmd_fetch(a):
    r = persist.fetch(RUNS)
    if not r:
        print("nothing fetched"); return 3
    print(f"fetched {r[1]}")
    return 0


def cmd_evaluate(a):
    from genesis.evalreport import evaluate_checkpoint
    return evaluate_checkpoint(a.ckpt, a.run or current_run(), a.episodes, a.families)


def cmd_export(a):
    from genesis.export import export
    return export(a.ckpt, a.run or current_run(), a.milestones)


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="genesis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(s):
        s.add_argument("--run"); s.add_argument("--foreground", action="store_true")
        s.add_argument("--segments", type=int); s.add_argument("--until", type=int)
        s.add_argument("--minutes", type=float); s.add_argument("--set", action="append")
    s = sub.add_parser("train"); common(s); s.add_argument("--config"); s.add_argument("--seed", type=int)
    s = sub.add_parser("resume"); common(s); s.add_argument("--new", action="store_true")
    s = sub.add_parser("stop"); s.add_argument("--run")
    s = sub.add_parser("status"); s.add_argument("--json", action="store_true")
    s.add_argument("--remote", help="comma-separated store branches, e.g. genesis-store-gen2-s1,genesis-store-gen2-s2")
    sub.add_parser("diagnose")
    s = sub.add_parser("publish"); s.add_argument("--run")
    sub.add_parser("fetch")
    s = sub.add_parser("evaluate"); s.add_argument("--run"); s.add_argument("--ckpt"); s.add_argument("--episodes", type=int, default=100)
    s.add_argument("--families", default="standard,unseen_layout,night,relentless,one_shelter,perturbed,no_predator")
    s = sub.add_parser("export"); s.add_argument("--run"); s.add_argument("--ckpt"); s.add_argument("--milestones", action="store_true")
    a = ap.parse_args(argv)
    return {"train": cmd_train, "resume": cmd_resume, "stop": cmd_stop, "status": cmd_status, "diagnose": cmd_diagnose,
            "publish": cmd_publish, "fetch": cmd_fetch, "evaluate": cmd_evaluate, "export": cmd_export}[a.cmd](a)
