#!/usr/bin/env bash
# One GENESIS training cycle in a Claude Code Cloud sandbox. Plain commands, no LLM decisions.
# usage: bash cloud/cycle.sh <minutes> [seeds...]   e.g. bash cloud/cycle.sh 50 1 2
set -u
MIN=${1:-50}; shift || true
SEEDS=${*:-1}
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LINEAGE=${GENESIS_LINEAGE:-gen2}
python3 -m pip install -q numpy psutil pytest >/dev/null 2>&1 || pip install -q numpy psutil pytest >/dev/null 2>&1
mkdir -p runs
python3 - > runs/cloud_env.json <<'PY'
import json, os, platform, shutil, psutil
print(json.dumps({"cpus": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
  "mem_total_mb": psutil.virtual_memory().total // 2**20, "mem_avail_mb": psutil.virtual_memory().available // 2**20,
  "disk_free_gb": round(shutil.disk_usage(".").free / 2**30, 1), "python": platform.python_version(), "os": platform.platform()}))
PY
cat runs/cloud_env.json
python3 -m pytest -q -x tests 2>&1 | tail -1
for S in $SEEDS; do
  ( export GENESIS_STORE_BRANCH=genesis-store-$LINEAGE-s$S
    RUN=$LINEAGE-s$S
    python3 -m genesis resume --run $RUN --foreground --minutes $MIN --set publish_minutes=8 --set ckpt_minutes=4
    rc=$?
    if [ $rc -eq 3 ]; then
      python3 -m genesis train --run $RUN --seed $S --foreground --minutes $MIN --set publish_minutes=8 --set ckpt_minutes=4 --set total_steps=${GENESIS_TOTAL_STEPS:-60000000}
      rc=$?
    fi
    echo "EXIT $rc" ) > runs/cycle-s$S.log 2>&1 &
  echo $! > runs/cycle-s$S.pid
done
echo "STARTED seeds: $SEEDS for $MIN min"
