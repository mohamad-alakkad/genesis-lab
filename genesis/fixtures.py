"""Parity fixtures for the JS port: python -m genesis.fixtures [--policy <ckpt_dir>]"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from genesis import ckpt, ppo
from genesis.world import World

ROOT = ckpt.ROOT
OUT = ROOT / "web" / "test" / "fixtures"
FAMS = ["standard", "perturbed", "harder", "unseen_layout"]
EVENTS = {100: ("insert_obstacle", 0.6, [8.0, 5.0]), 200: ("wake_vent", 0), 300: ("move_food", 0, [3.0, 3.0]),
          350: ("insert_obstacle", 0.5, [12.0, 2.5])}


def world_fixture(k, steps=600):
    fam = FAMS[k % len(FAMS)]
    seed = 4_000_000_000 + k
    w = World(1, fam); one = np.ones(1, bool)
    w.reset(one, np.array([seed]))
    rng = np.random.default_rng(k)
    acts = rng.integers(0, 6, steps)
    obs, energy, pos, accepted = [w.observe()[0].tolist()], [], [], {}
    for t in range(steps):
        if t in EVENTS:
            ev = EVENTS[t]
            if ev[0] == "insert_obstacle":
                accepted[t] = bool(w.insert_obstacle(one, r=ev[1], at=ev[2])[0])
            elif ev[0] == "wake_vent":
                w.wake_vent(one, ev[1]); accepted[t] = True
            else:
                accepted[t] = bool(w.move_food(one, ev[1], ev[2])[0])
        w.step(acts[t:t + 1])
        obs.append(w.observe()[0].tolist()); energy.append(float(w.energy[0])); pos.append(w.pos[0].tolist())
        w.events.clear()
    return {"family": fam, "seed": seed, "actions": acts.tolist(), "events": {str(t): list(e) for t, e in EVENTS.items()},
            "accepted": {str(t): v for t, v in accepted.items()}, "obs": obs, "energy": energy, "pos": pos}


def policy_fixture(d, n=1000):
    p, _, _, mu, var = ckpt.unpack((d / "model.npz").read_bytes())
    norm = ppo.RunningNorm(len(mu)); norm.mean = mu; norm.var = var
    w = World(50, "standard"); w.reset(np.ones(50, bool), 4_100_000_000 + np.arange(50))
    rng = np.random.default_rng(0); rows = []
    while len(rows) < n:
        o = w.observe(); rows += o.tolist(); w.step(rng.integers(0, 6, 50)); w.events.clear()
    O = np.array(rows[:n]); logits = ppo.policy_logits(p, norm(O))
    return {"source": d.name, "obs": O.tolist(), "logits": logits.tolist(), "argmax": logits.argmax(1).tolist()}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "genesis" / "world_constants.json", ROOT / "web" / "src" / "sim" / "world_constants.json")
    for k in range(20):
        (OUT / f"world-{k:02d}.json").write_text(json.dumps(world_fixture(k)))
    if "--policy" in sys.argv:
        d = Path(sys.argv[sys.argv.index("--policy") + 1])
        (OUT / "policy.json").write_text(json.dumps(policy_fixture(d)))
    print(f"fixtures written to {OUT}")
