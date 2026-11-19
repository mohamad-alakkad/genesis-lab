"""Segmented PPO trainer. Plain Python, no LLM in the loop. Runs in the foreground of its own
(job-limited) process; the CLI launcher detaches it."""
from __future__ import annotations

import json
import os
import signal
import socket
import time
import uuid
from pathlib import Path

import numpy as np
import psutil

from genesis import ckpt, persist, ppo
from genesis.world import C, N_ACTIONS, OBS_DIM, SPEC_VERSION, VecEnv

RUNS = ckpt.ROOT / "runs"
DEFAULT_CONFIG = {
    "n_envs": 256, "n_steps": 128, "epochs": 4, "minibatch": 2048, "lr": 3e-4, "gamma": 0.997, "lam": 0.95,
    "clip": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5, "hidden": [128, 128],
    "total_steps": 60_000_000, "segment_minutes": 60, "ckpt_minutes": 10, "first_ckpt_s": 45,
    "publish_minutes": 30, "heartbeat_s": 30, "eval_every_steps": 2_000_000, "eval_episodes": 100,
    "target_survival": 0.9, "plateau_evals": 10, "plateau_eps": 15.0, "seed": 1,
    "rss_soft_mb": 1400, "rss_hard_mb": 1600, "host_min_avail_frac": 0.10,
    "milestone_steps": {"early": 2_000_000}, "obs_mask": [], "train_family": "train", "reward_mode": "food", "life_reward": 0.01,
}


def tree_rss_mb():
    p = psutil.Process()
    tot = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            tot += c.memory_info().rss
        except psutil.Error:
            pass
    return tot / 2**20


def _jdump(path: Path, obj):
    ckpt.atomic_write(path, json.dumps(obj, indent=1).encode())


def _append(path: Path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n"); f.flush(); os.fsync(f.fileno())


class Trainer:
    def __init__(self, run_id: str, config: dict, resume_from: Path | None, parent=None, log=print):
        self.log = log
        self.run_id = run_id
        self.dir = RUNS / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg = config
        self.stop_flag = None
        self.peak_rss = 0.0
        self.parent = parent
        rng = np.random.default_rng(cfg["seed"])
        self.env = VecEnv(cfg["n_envs"], cfg.get("train_family", "train"), seed=cfg["seed"] + 1000)
        self.act_rng = np.random.default_rng(cfg["seed"] + 2000)
        if resume_from is None:
            self.params = ppo.init_params(OBS_DIM, N_ACTIONS, cfg["hidden"], seed=cfg["seed"])
            self.adam = ppo.Adam(self.params, cfg["lr"])
            self.norm = ppo.RunningNorm(OBS_DIM)
            self.state = {"global_step": 0, "episodes": 0, "segments": 0, "wall_trained_s": 0.0, "updates": 0,
                          "best": None, "evals": 0, "since_best": 0, "consec_target": 0, "init_seed": cfg["seed"],
                          "interruptions": 0, "lost_time_s": 0.0, "milestones_done": []}
        else:
            p, m, v, mu, var = ckpt.unpack((resume_from / "model.npz").read_bytes())
            st = json.loads((resume_from / "state.json").read_text())
            self.params = p
            self.adam = ppo.Adam(self.params, cfg["lr"]); self.adam.m = m; self.adam.v = v; self.adam.t = st["adam_t"]
            self.norm = ppo.RunningNorm(OBS_DIM); self.norm.mean = mu; self.norm.var = var; self.norm.count = st["norm_count"]
            self.state = {k: v for k, v in st.items() if k not in ("adam_t", "norm_count", "arch", "rng", "config")}
            if "rng" in st:
                self.env.seed_rng.bit_generator.state = st["rng"]["seed"]
                self.act_rng.bit_generator.state = st["rng"]["act"]
        self.probe_obs = VecEnv(16, "standard", seed=7).obs()  # fixed probe for load tests
        self.last_ckpt_t = 0.0; self.last_pub_t = 0.0; self.last_hb = 0.0
        self.last_ckpt_id = ckpt.read_last_good(self.dir); self.last_pub = None
        self.sps = 0.0
        self.recent = {"food": [], "ticks": []}
        self.diag = {"entropy0": None, "entropy": None, "v_loss_med": None, "nan": False}

    def mask(self, obs):
        idx = self.cfg.get("obs_mask") or []
        if not idx:
            return obs
        obs = obs.copy(); obs[:, idx] = 0.0
        return obs

    # ------------------------------------------------------------------ io
    def checkpoint(self, milestones=(), eval_summary=None):
        st = dict(self.state)
        st["config"] = self.cfg
        st["rng"] = {"seed": self.env.seed_rng.bit_generator.state, "act": self.act_rng.bit_generator.state}
        t0 = time.time()
        m = ckpt.write_checkpoint(self.dir, run_id=self.run_id, step=self.state["global_step"], params=self.params,
                                  adam=self.adam, norm=self.norm, state=st, config=self.cfg, probe_obs=self.probe_obs,
                                  milestones=milestones, parent=self.parent, eval_summary=eval_summary)
        self.last_ckpt_t = time.time(); self.last_ckpt_id = m["checkpoint_id"]
        keep = [str(b["checkpoint_id"]).split("/")[-1] for b in [self.state.get("best") or {}] if b]
        ckpt.prune(self.dir, keep=keep)
        self.state["ckpt_overhead_s"] = self.state.get("ckpt_overhead_s", 0.0) + (time.time() - t0)
        return m

    def publish(self):
        try:
            r = persist.publish(self.dir, self.run_id, log=self.log)
            self.last_pub_t = time.time()
            if r:
                self.last_pub = r; self.state["last_publish"] = r; self.state.pop("publish_error", None)
        except Exception as e:  # noqa: BLE001 — failure goes to status, training continues
            self.state["publish_error"] = str(e)[-300:]
            self.last_pub_t = time.time()
            self.log(f"publish failed: {e}")

    def heartbeat(self, seg_idx, state="running", end_reason=None):
        rss = tree_rss_mb(); self.peak_rss = max(self.peak_rss, rss)
        hb = {"run_id": self.run_id, "host_id": socket.gethostname(), "session_id": self.session, "pid": os.getpid(),
              "global_step": self.state["global_step"], "wall_time_s": round(self.state["wall_trained_s"], 1),
              "updated_at": ckpt.now_iso(), "updated_ts": time.time(), "last_local_ckpt": self.last_ckpt_id,
              "last_local_ckpt_ts": self.last_ckpt_t, "last_remote_publish": self.last_pub,
              "steps_per_sec": round(self.sps), "rss_mb": round(rss, 1), "segment_index": seg_idx}
        _jdump(self.dir / "heartbeat.json", hb)
        remote, _ = persist.config()
        status = {"schema": "genesis.status/1", "state": state, "run_id": self.run_id, "end_reason": end_reason,
                  "global_step": self.state["global_step"], "steps_per_sec": round(self.sps),
                  "heartbeat_ts": time.time(), "heartbeat_s": self.cfg["heartbeat_s"],
                  "last_local_ckpt": self.last_ckpt_id, "last_local_ckpt_ts": self.last_ckpt_t,
                  "last_remote_publish": self.state.get("last_publish"), "publish_error": self.state.get("publish_error"),
                  "eval_latest": self.state.get("eval_latest"), "eval_best": self.state.get("best"),
                  "interruptions": {"count": self.state.get("interruptions", 0), "lost_time_s": round(self.state.get("lost_time_s", 0.0)),
                                    "last_end_reason": self.state.get("prev_end_reason")},
                  "resources": {"rss_mb": round(rss, 1), "peak_rss_mb": round(self.peak_rss, 1),
                                "limit_mb": int(os.environ.get("GENESIS_MEM_LIMIT_MB", "0")),
                                "mechanism": os.environ.get("GENESIS_MEM_MECHANISM", "none")},
                  "train_recent": {k: round(float(np.mean(v)), 3) if v else None for k, v in self.recent.items()},
                  "diag": self.diag, "total_steps": self.cfg["total_steps"],
                  "persistence_warning": None if remote else persist.WARNING}
        _jdump(self.dir / "status.json", status)  # per-run status (concurrent trainers share runs/)
        try:
            _jdump(RUNS / "status.json", status)  # convenience copy for single-run use; never fatal
        except OSError:
            pass
        self.last_hb = time.time()
        return rss

    # ------------------------------------------------------------------ eval
    def evaluate(self):
        from genesis.evalonly.suite import evaluate_policy  # eval module stays out of export paths
        p, norm, mask = self.params, self.norm, self.mask
        res = evaluate_policy(lambda o: ppo.policy_logits(p, norm(mask(o))).argmax(1), ("standard",), self.cfg["eval_episodes"])["standard"]
        s = {"suite": "heldout-standard", "step": self.state["global_step"], "score": res["ticks"]["mean"],
             "score_ci95": res["ticks"]["ci95"], "food_mean": res["food"]["mean"], "survived": res["survived"]["mean"],
             "caught": res["caught"]["mean"], "shelter_when_threat": res["shelter_when_threat"]["mean"],
             "hazard": res["hazard"]["mean"], "episodes": self.cfg["eval_episodes"]}
        _append(self.dir / "eval_history.jsonl", s)
        self.state["eval_latest"] = s; self.state["evals"] += 1
        best = self.state.get("best")
        is_best = best is None or s["score"] > best["score"] + 1e-9
        if best is None or s["score"] > best["score"] + self.cfg["plateau_eps"]:
            self.state["since_best"] = 0
        else:
            self.state["since_best"] += 1
        self.state["consec_target"] = self.state["consec_target"] + 1 if s["survived"] >= self.cfg["target_survival"] else 0
        return s, is_best

    # ------------------------------------------------------------------ main loop
    def run(self, segments=1, until_steps=None, prev_end=None):
        cfg = self.cfg
        self.session = uuid.uuid4().hex[:8]
        clean = self.dir / "CLEAN_EXIT"
        if clean.exists():
            clean.unlink()
        if prev_end:
            self.state["prev_end_reason"] = prev_end.get("reason")
            if not prev_end.get("confirmed"):
                self.state["interruptions"] = self.state.get("interruptions", 0) + 1
                self.state["lost_time_s"] = self.state.get("lost_time_s", 0.0) + prev_end.get("lost_time_s", 0.0)
        def on_signal(sig, frame):
            self.stop_flag = f"signal:{sig}"
        signal.signal(signal.SIGINT, on_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, on_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, on_signal)
        until = min(until_steps or cfg["total_steps"], cfg["total_steps"])
        if self.state["global_step"] == 0 and "step0" not in self.state["milestones_done"]:
            s, _ = self.evaluate()
            self.state["milestones_done"].append("step0")
            self.checkpoint(milestones=["step0"], eval_summary=s)
        final_state, end_reason = "paused", "segment_done"
        for seg in range(segments):
            self.state["segments"] += 1
            seg_idx = self.state["segments"]
            seg_start = time.time(); steps_start = self.state["global_step"]
            _append(self.dir / "history.jsonl", {"segment": seg_idx, "event": "start", "host_id": socket.gethostname(),
                                                  "pid": os.getpid(), "start": ckpt.now_iso(), "steps_start": steps_start,
                                                  "recovered_from": self.last_ckpt_id, "prev_end": prev_end})
            prev_end = None
            self.heartbeat(seg_idx)
            first_ckpt_due = seg_start + cfg["first_ckpt_s"]
            first_done = False
            final_state, end_reason = self._segment(seg_idx, seg_start, first_ckpt_due, until)
            terminal = final_state in ("completed", "plateaued", "budget_exhausted", "failed")
            if terminal:
                self.state["terminal"] = {"state": final_state, "reason": end_reason}
            m = self.checkpoint(milestones=["final"] if terminal and final_state != "failed" else [])
            if not terminal:
                self.publish()
            _append(self.dir / "history.jsonl", {"segment": seg_idx, "event": "end", "end": ckpt.now_iso(),
                                                  "end_reason": end_reason, "end_confirmed": True,
                                                  "steps_start": steps_start, "steps_end": self.state["global_step"],
                                                  "lost_time_s": 0, "peak_rss_mb": round(self.peak_rss, 1),
                                                  "checkpoint": m["checkpoint_id"]})
            if end_reason != "segment_done":
                break
        if final_state == "paused" and end_reason == "segment_done":
            final_state = "paused"
        _jdump(clean, {"end_reason": end_reason, "state": final_state, "step": self.state["global_step"],
                       "at": ckpt.now_iso(), "at_ts": time.time(), "checkpoint": self.last_ckpt_id})
        self.heartbeat(self.state["segments"], state=final_state, end_reason=end_reason)
        self.publish()  # remote snapshot includes CLEAN_EXIT, heartbeat and history
        return final_state, end_reason

    def _segment(self, seg_idx, seg_start, first_ckpt_due, until):
        cfg = self.cfg
        E, T = cfg["n_envs"], cfg["n_steps"]
        obs = self.env.obs()
        stop_file = self.dir / "STOP"
        first_done = False
        total_mem = psutil.virtual_memory().total
        while True:
            t_it = time.time()
            # ---- rollout
            b_obs = np.zeros((T, E, OBS_DIM)); b_act = np.zeros((T, E), np.int64); b_logp = np.zeros((T, E))
            b_rew = np.zeros((T, E)); b_done = np.zeros((T, E)); b_val = np.zeros((T + 1, E))
            for t in range(T):
                obs = self.mask(obs)
                self.norm.update(obs)
                x = self.norm(obs)
                logits = ppo.policy_logits(self.params, x)
                lp = ppo.log_softmax(logits)
                g = self.act_rng.gumbel(size=lp.shape)
                a = (lp + g).argmax(1)
                v = ppo.mlp_forward(self.params, "v", x)[0][:, 0]
                obs2, r, dead, trunc, info = self.env.step(a)
                if cfg.get("reward_mode", "food") == "life":  # objective = staying alive: +life_reward per tick survived, nothing else
                    r = np.where(dead, 0.0, cfg.get("life_reward", 0.01))
                b_obs[t] = x; b_act[t] = a; b_logp[t] = lp[np.arange(E), a]; b_rew[t] = r; b_val[t] = v
                b_done[t] = dead
                if trunc.any():  # bootstrap truncated episodes with the value of the terminal-ish state
                    b_rew[t] += cfg["gamma"] * v * trunc
                    b_done[t] = np.maximum(b_done[t], trunc)
                if info is not None:
                    self.state["episodes"] += len(info["food"])
                    for k in ("food", "ticks"):
                        self.recent[k] = (self.recent[k] + list(map(float, info[k])))[-200:]
                obs = obs2
            b_val[T] = ppo.mlp_forward(self.params, "v", self.norm(self.mask(obs)))[0][:, 0]
            # ---- GAE
            adv = np.zeros((T, E)); last = np.zeros(E)
            for t in range(T - 1, -1, -1):
                nonterm = 1.0 - b_done[t]
                delta = b_rew[t] + cfg["gamma"] * b_val[t + 1] * nonterm - b_val[t]
                last = delta + cfg["gamma"] * cfg["lam"] * nonterm * last
                adv[t] = last
            ret = adv + b_val[:T]
            N = T * E
            f_obs, f_act, f_logp = b_obs.reshape(N, -1), b_act.reshape(N), b_logp.reshape(N)
            f_adv, f_ret = adv.reshape(N), ret.reshape(N)
            frac = 1.0 - self.state["global_step"] / cfg["total_steps"]
            self.adam.lr = cfg["lr"] * max(frac, 0.05)
            for _ in range(cfg["epochs"]):
                perm = self.act_rng.permutation(N)
                for s in range(0, N, cfg["minibatch"]):
                    idx = perm[s:s + cfg["minibatch"]]
                    a_ = f_adv[idx]; a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                    _, grads, stats = ppo.ppo_loss_and_grads(self.params, f_obs[idx], f_act[idx], f_logp[idx], a_,
                                                             f_ret[idx], cfg["clip"], cfg["ent_coef"], cfg["vf_coef"])
                    ppo.clip_grads(grads, cfg["max_grad_norm"])
                    self.adam.step(self.params, grads)
            self.state["updates"] += 1
            self.state["global_step"] += N
            dt = time.time() - t_it
            self.state["wall_trained_s"] += dt
            self.sps = N / dt
            # diagnostics
            self.diag["entropy"] = round(float(stats["entropy"]), 4)
            if self.diag["entropy0"] is None:
                self.diag["entropy0"] = self.diag["entropy"]
            vm = self.diag["v_loss_med"]
            self.diag["v_loss_med"] = round(float(stats["v_loss"]) if vm is None else 0.95 * vm + 0.05 * float(stats["v_loss"]), 5)
            self.diag["nan"] = bool(any(not np.all(np.isfinite(v)) for v in self.params.values()))
            if self.diag["nan"]:
                return "failed", "nan"
            gs = self.state["global_step"]
            now = time.time()
            # ---- eval / milestones
            if gs // cfg["eval_every_steps"] > (gs - N) // cfg["eval_every_steps"]:
                s, is_best = self.evaluate()
                ms = []
                for name, at in cfg["milestone_steps"].items():
                    if gs >= at and name not in self.state["milestones_done"]:
                        ms.append(name); self.state["milestones_done"].append(name)
                if "middle" not in self.state["milestones_done"] and gs >= cfg["total_steps"] // 2:
                    ms.append("middle"); self.state["milestones_done"].append("middle")
                if is_best:
                    ms.append("best")
                    self.state["best"] = dict(s, checkpoint_id=f"{self.run_id}/{gs:010d}")
                self.checkpoint(milestones=ms, eval_summary=s)
                first_done = True
                if self.state["consec_target"] >= 2:
                    return "completed", "target_reached"
                if self.state["since_best"] >= cfg["plateau_evals"]:
                    return "plateaued", "plateau"
            # ---- checkpoint cadence
            if (not first_done and now >= first_ckpt_due) or now - self.last_ckpt_t >= cfg["ckpt_minutes"] * 60:
                self.checkpoint(); first_done = True
            if now - self.last_pub_t >= cfg["publish_minutes"] * 60 and first_done:
                self.publish()
            # ---- heartbeat + watchdog
            if now - self.last_hb >= cfg["heartbeat_s"]:
                rss = self.heartbeat(seg_idx)
            else:
                rss = tree_rss_mb(); self.peak_rss = max(self.peak_rss, rss)
            if rss >= cfg["rss_hard_mb"]:
                return "paused", "resource_limit"
            if rss >= cfg["rss_soft_mb"]:
                return "paused", "resource_limit"
            if psutil.virtual_memory().available < cfg["host_min_avail_frac"] * total_mem:
                return "paused", "host_pressure"
            if stop_file.exists():
                stop_file.unlink()
                return "paused", "stop_file"
            if self.stop_flag:
                return "paused", "signal"
            if gs >= until:
                return ("budget_exhausted" if gs >= cfg["total_steps"] else "paused"), (
                    "budget_exhausted" if gs >= cfg["total_steps"] else "until_reached")
            if now - seg_start >= cfg["segment_minutes"] * 60:
                return "paused", "segment_done"
