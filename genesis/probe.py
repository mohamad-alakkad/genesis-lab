"""Heartbeat-only session-lifetime probe: writes ts+pid every 60 s, exits after 24 h."""
import json, os, time
os.makedirs("runs/probe", exist_ok=True)
t0 = time.time()
while time.time() - t0 < 24 * 3600:
    tmp = "runs/probe/heartbeat.json.tmp"
    with open(tmp, "w") as f:
        json.dump({"pid": os.getpid(), "started": t0, "updated": time.time(),
                   "alive_h": round((time.time() - t0) / 3600, 3)}, f)
    os.replace(tmp, "runs/probe/heartbeat.json")
    time.sleep(60)
