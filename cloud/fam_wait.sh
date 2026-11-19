#!/usr/bin/env bash
# Blocks up to 9 minutes; prints DONE when docs/reports/<out_name>.md exists, FAILED on error, else STILL_RUNNING.
cd "$(git rev-parse --show-toplevel)"
end=$((SECONDS+540))
while [ $SECONDS -lt $end ]; do
  [ -f "docs/reports/$1.md" ] && { echo DONE; exit 0; }
  grep -q "Traceback" runs/fam_eval.log 2>/dev/null && { tail -5 runs/fam_eval.log; echo FAILED; exit 1; }
  sleep 20
done
echo STILL_RUNNING
