#!/usr/bin/env bash
# Blocks up to 9 minutes; prints one line per seed. Prints ALL_DONE when every cycle has exited.
cd "$(git rev-parse --show-toplevel)"
end=$((SECONDS+540))
while [ $SECONDS -lt $end ]; do
  alive=0
  for f in runs/cycle-*.pid; do kill -0 "$(cat $f)" 2>/dev/null && alive=1; done
  [ $alive -eq 0 ] && break
  sleep 20
done
for f in runs/cycle-*.log; do echo "$f: $(grep -E 'EXIT|trainer ended|previous end' $f | tail -2 | tr '\n' ' ')"; done
python3 - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("runs").glob("gen*-s*/heartbeat.json")):
    h = json.loads(p.read_text()); print(p.parent.name, "step", h["global_step"], "sps", h["steps_per_sec"], "rss", h["rss_mb"], "pub", (h.get("last_remote_publish") or {}).get("checkpoint_id"))
PY
[ $alive -eq 0 ] && echo ALL_DONE || echo STILL_RUNNING
