import json

from genesis import trainer


def test_trainer_runs_checkpoints_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "RUNS", tmp_path)
    monkeypatch.setenv("GENESIS_STORE_REMOTE", "")
    cfg = dict(trainer.DEFAULT_CONFIG, n_envs=16, n_steps=32, minibatch=256, total_steps=1024, eval_every_steps=512,
               eval_episodes=4, hidden=[16, 16])
    t = trainer.Trainer("smoke", cfg, None, log=lambda *_: None)
    state, reason = t.run(segments=1)
    assert state == "budget_exhausted" and t.state["global_step"] >= 1024
    run = tmp_path / "smoke"
    assert (run / "CLEAN_EXIT").exists() and (run / "LAST_GOOD").exists()
    from genesis import ckpt
    d, man, fb = ckpt.find_loadable(run)
    assert man["global_step"] == t.state["global_step"] and "final" in man["milestones"] and not fb
    st = json.loads((d / "state.json").read_text())
    assert st["terminal"]["state"] == "budget_exhausted"
    t2 = trainer.Trainer("smoke", dict(cfg, total_steps=2048), d, log=lambda *_: None)
    assert t2.state["global_step"] == man["global_step"]
