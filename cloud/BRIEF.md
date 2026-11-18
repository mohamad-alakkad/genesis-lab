# GENESIS cloud training cycle — instructions for the cloud session

You are only an operator. Training is a plain Python process; you make no training decisions,
do not edit code, do not read logs beyond what the scripts print.

1. `bash cloud/cycle.sh <MINUTES> <SEEDS>` (values given in the trigger prompt).
2. Repeat `bash cloud/wait.sh` (each call blocks ≤ 9 min) until it prints `ALL_DONE`.
3. Print `cat runs/cloud_env.json` and the last `wait.sh` output as your final message. Stop.

Rules: never push to `main`; no secrets; no network access other than git/pip; if a command fails
twice, stop and report the exact error line. Checkpoints are published automatically to the
`genesis-store-s<seed>` branches by the trainer.
