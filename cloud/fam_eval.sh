#!/usr/bin/env bash
# Start the cross-family generalization evaluation in the background (cloud). usage: bash cloud/fam_eval.sh <out_name> [episodes]
cd "$(git rev-parse --show-toplevel)"
python3 -m pip install -q numpy psutil >/dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p runs
nohup python3 -m genesis.familyeval cloud/conditions.json "${2:-100}" 50000 "$1" "" 4 > runs/fam_eval.log 2>&1 &
echo "STARTED $1"
