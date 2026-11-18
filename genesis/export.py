"""`export`: web policy bundle (genesis.policy/1) from a verified checkpoint. Policy weights only."""
from __future__ import annotations

import hashlib
import json

from genesis import ckpt
from genesis.world import C, OBS_DIM, SPEC_VERSION

OUT = ckpt.ROOT / "web" / "public" / "policies"


def bundle(d, milestone):
    ok, why, man = ckpt.verify_dir(d)
    if not ok:
        raise RuntimeError(f"refusing to export unverified checkpoint: {why}")
    p, _, _, mu, var = ckpt.unpack((d / "model.npz").read_bytes())
    L = sum(1 for k in p if k.startswith("pi.W"))
    layers = [{"w": p[f"pi.W{k}"].tolist(), "b": p[f"pi.b{k}"].tolist(), "activation": "tanh" if k < L - 1 else "linear"}
              for k in range(L)]
    b = {"schema": "genesis.policy/1", "world_spec_version": man["world_spec_version"], "milestone": milestone,
         "source_checkpoint_id": man["checkpoint_id"], "training_step": man["global_step"],
         "observation": {"size": OBS_DIM, "layout": "see WORLD_SPEC §Observations"}, "actions": C["actions"],
         "normalization": {"mean": mu.tolist(), "var": var.tolist(), "clip": 10.0, "epsilon": 1e-8},
         "network": {"layers": layers, "output": "logits"}, "selection": "argmax",
         "value_network": {"layers": [{"w": p[f"v.W{k}"].tolist(), "b": p[f"v.b{k}"].tolist(),
                                       "activation": "tanh" if k < L - 1 else "linear"} for k in range(L)],
                           "output": "state value V(s) (critic, training-time estimate of discounted return)"},
         "hyperparameters_hidden": [len(p[f"pi.b{k}"]) for k in range(L - 1)],
         "input_mask": json.loads((d / "state.json").read_text()).get("config", {}).get("obs_mask") or [],
         "eval_summary": man.get("eval_summary")}
    b["checksum"] = hashlib.sha256(json.dumps(b, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return b


def export(ckpt_id, run_id, milestones):
    run_dir = ckpt.ROOT / "runs" / run_id
    OUT.mkdir(parents=True, exist_ok=True)
    targets = []
    if milestones:
        for c in sorted((run_dir / "ckpt").iterdir()):
            ok, _, m = ckpt.verify_dir(c)
            for ms in (m or {}).get("milestones", []) if ok else []:
                targets.append((c, ms))
    else:
        d = ckpt.ckpt_dir(run_dir, ckpt_id) if ckpt_id else ckpt.find_loadable(run_dir)[0]
        targets.append((d, "final"))
    seen = {}
    for d, ms in targets:
        seen[ms] = d  # later checkpoints win for repeated milestone names (e.g. best)
    for ms, d in seen.items():
        b = bundle(d, ms)
        ckpt.atomic_write(OUT / f"policy-{ms}.json", json.dumps(b, separators=(",", ":")).encode())
        print(f"policy-{ms}.json ← {b['source_checkpoint_id']}")
    return 0
