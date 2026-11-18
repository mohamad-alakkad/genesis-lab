import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from genesis import ckpt, ppo
from genesis.world import N_ACTIONS, OBS_DIM

ROOT = Path(__file__).resolve().parent.parent


def _write(tmp, step):
    p = ppo.init_params(OBS_DIM, N_ACTIONS, (16, 16), seed=step)
    return ckpt.write_checkpoint(tmp, run_id="t", step=step, params=p, adam=ppo.Adam(p, 1e-3), norm=ppo.RunningNorm(OBS_DIM),
                                 state={"global_step": step}, config={"a": 1}, probe_obs=np.zeros((4, OBS_DIM)))


def test_write_verify_and_manifest_schema(tmp_path):
    m = _write(tmp_path, 100)
    d = ckpt.ckpt_dir(tmp_path, m["checkpoint_id"])
    assert ckpt.verify_dir(d)[0]
    r = subprocess.run([sys.executable, str(ROOT / ".claude/skills/genesis-lab/scripts/verify_manifest.py"),
                        str(d / "manifest.json"), "--dir", str(d)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert ckpt.read_last_good(tmp_path) == "t/0000000100"


def test_corruption_falls_back_to_previous(tmp_path):
    _write(tmp_path, 100); m2 = _write(tmp_path, 200)
    f = ckpt.ckpt_dir(tmp_path, m2["checkpoint_id"]) / "model.npz"
    data = bytearray(f.read_bytes()); data[100] ^= 0xFF; f.write_bytes(bytes(data))
    d, man, fb = ckpt.find_loadable(tmp_path, log=lambda *_: None)
    assert man["global_step"] == 100 and fb and "checksum" in fb[0]


def test_truncated_manifest_rejected(tmp_path):
    m = _write(tmp_path, 100)
    mp = ckpt.ckpt_dir(tmp_path, m["checkpoint_id"]) / "manifest.json"
    mp.write_text(mp.read_text()[:50])
    assert not ckpt.verify_dir(mp.parent)[0]


def test_export_never_imports_baselines():
    src = (ROOT / "genesis" / "export.py").read_text(encoding="utf-8")
    assert "evalonly" not in src and "baselines" not in src
